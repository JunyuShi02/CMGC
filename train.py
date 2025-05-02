import numpy as np
import time
import torch
from torch import nn
from torch.utils.data import DataLoader
from timm.scheduler import StepLRScheduler, CosineLRScheduler

from h36m_dataset import H36M3dDataset
from GCN import CMGC


def MPJPE_loss_fn(pred_seq, target_seq):
    """ function to calculate MPJPE loss
    Args:
        pred_seq (Torch.float): [B, T, N, 3]
        target_seq (Torch.float): [B, T, N, 3]
    """
    pred_seq = pred_seq[-1]
    joint_loss = torch.mean(torch.norm(pred_seq - target_seq, dim=-1))

    parents_idx = [8, 0, 1, 2, 8, 4, 5, 6, 8, 8, 9, 10, 8, 12, 13, 14, 14, 8, 17, 18, 19, 19]
    pred_parents = pred_seq[:, :, parents_idx]
    target_parents = target_seq[:, :, parents_idx]
    pred_bone = (pred_parents + pred_seq) / 2
    target_bone = (target_parents + target_seq) / 2
    bone_loss = torch.mean(torch.norm(pred_bone - target_bone, dim=-1))

    loss = joint_loss + 0.5 * bone_loss
    return loss


def train_h36m3d(opt):

    batch_size = opt.batch_size
    lr = opt.lr
    model = CMGC(in_dims=opt.len_input + opt.len_output, hid_dims=256,
                  out_dims=opt.len_input + opt.len_output, drop_rate=0.5,
                  num_blocks=12, num_nodes=66, opt=opt)
    model.to(opt.device)

    # optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    # optimizer = torch.optim.AdamW(
    #     filter(lambda x: x.requires_grad, model.parameters()), lr=opt.lr, weight_decay=0.997
    # )
    # scheduler = StepLRScheduler(optimizer, decay_t=2, decay_rate=0.96, w    armup_t=0)
    scheduler = CosineLRScheduler(optimizer, t_initial=opt.epoch)

    # Define dataloader
    train_dataset = H36M3dDataset(opt=opt, dir_path=opt.dir_path, state=0)
    test_dataset = H36M3dDataset(opt=opt, dir_path=opt.dir_path, state=2)
    train_loader = DataLoader(
        dataset=train_dataset, batch_size=batch_size,
        shuffle=True, pin_memory=True, drop_last=True
    )
    test_loader = DataLoader(
        dataset=test_dataset, batch_size=opt.test_batch_size,
        shuffle=False, pin_memory=True, drop_last=True
    )
    print(">>> Total parameters: {:.2f}M".format(sum(p.numel() for p in model.parameters()) / 1000000.0))

    if opt.model_path is not None:
        # Load previous parameters
        model.load_state_dict(torch.load(opt.model_path))
        optimizer.load_state_dict(torch.load(opt.optimizer_path))
        start_epoch = opt.start_epoch
        print('>>> Parameters loaded, start from epoch {}\n'.format(start_epoch))
    else:
        start_epoch = 0

    best_MPJPE, best_epoch = np.inf, None
    for epoch in range(int(start_epoch), opt.epoch):
        model.train()
        print(f'>>> Epoch{epoch}')
        start_time = time.time()
        for data in train_loader:
            # data: [B, len_input + len_output, 96]
            history_seq, future_seq = data[:, :opt.len_input], data[:, opt.len_input:]
            history_seq, future_seq = history_seq.float().to(opt.device), future_seq.float().to(opt.device)

            # Remove the statical joints -> history_seq: [B, len_input, 66]
            joint_to_ignore = np.array([0, 1, 6, 11, 16, 20, 23, 24, 28, 31])
            dim_to_ignore = np.concatenate((joint_to_ignore * 3, joint_to_ignore * 3 + 1, joint_to_ignore * 3 + 2))
            dim_to_use = np.setdiff1d(np.arange(96), dim_to_ignore)
            history_seq, future_seq = history_seq[:, :, dim_to_use], future_seq[:, :, dim_to_use]

            # repeat the last frame of the history sequence to obtain the initial state of prediction
            X_ids = list(range(opt.len_input)) + [opt.len_input - 1] * opt.len_output
            input_seq, target_seq = history_seq[:, X_ids], future_seq

            # Forward
            list_pred_seq = model(input_seq.transpose(1, 2).contiguous())

            # Keep the prediction part
            for i in range(len(list_pred_seq)):
                list_pred_seq[i] = list_pred_seq[i][:, -opt.len_output:, :]
                list_pred_seq[i] = list_pred_seq[i].view(opt.batch_size, opt.len_output, -1, 3)

            # Calculate the loss function
            target_seq = target_seq.view(opt.batch_size, opt.len_output, -1, 3)
            loss = MPJPE_loss_fn(list_pred_seq, target_seq)

            # Backward
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
            optimizer.step()

        scheduler.step(epoch)

        # Validate
        MPJPE, MPJPE_per_frame = valid_h36m3d(model, test_loader, opt)
        MPJPE_per_frame = np.round(MPJPE_per_frame.cpu().numpy(), 2)
        print('>>> MPJPE of valid data:{:.2f} \n>>> {}'.format(MPJPE, MPJPE_per_frame))

        if MPJPE < best_MPJPE:
            best_MPJPE, best_epoch = MPJPE, epoch
            if epoch > 20 and opt.is_save is True:
                torch.save(model.state_dict(), opt.ckpt_path + '/model_MPJPE{:.2f}_epoch{}.pth'.format(MPJPE, epoch))
                torch.save(optimizer.state_dict(), opt.ckpt_path + '/optimizer_MPJPE{:.2f}_epoch{}.pth'.format(MPJPE, epoch))
        print('>>> Time Spend at this epoch:{:.2f}s\n'.format(time.time() - start_time))

    print('>>> Finish! \n>>> Best MPJPE:{:.2f}, Best epoch:{}\n'.format(best_MPJPE, best_epoch))


def valid_h36m3d(model, loader, opt):

    model.eval()
    num_seq = 0
    MPJPE = 0
    MPJPE_per_frame = 0
    with torch.no_grad():
        for data in loader:
            # data: [B, len_input + len_output, 96]
            history_seq, future_seq = data[:, :opt.len_input], data[:, opt.len_input:]
            history_seq, future_seq = history_seq.float().to(opt.device), future_seq.float().to(opt.device)

            # Remove the statical joints -> input_seq: [B, len_input, 66]
            # Repeat the last frame of the history sequence to obtain the initial state of prediction
            joint_to_ignore = np.array([0, 1, 6, 11, 16, 20, 23, 24, 28, 31])
            dim_to_ignore = np.concatenate((joint_to_ignore * 3, joint_to_ignore * 3 + 1, joint_to_ignore * 3 + 2))
            dim_to_use = np.setdiff1d(np.arange(96), dim_to_ignore)
            X_ids = list(range(opt.len_input)) + [opt.len_input - 1] * opt.len_output
            input_seq, target_seq = history_seq[:, X_ids], future_seq
            input_seq = input_seq[:, :, dim_to_use]

            # Forward
            list_pred_seq = model(input_seq.transpose(1, 2).contiguous())
            pred_seq = list_pred_seq[-1]

            # Keep the prediction part
            pred_seq = pred_seq[:, -opt.len_output:, :]
            pred_seq_all_joints = target_seq.detach().clone()

            # In h36m3d, some different numbers represent the same joints
            joint_to_ignore = np.array([16, 20, 23, 24, 28, 31])
            dim_to_ignore = np.concatenate((joint_to_ignore * 3, joint_to_ignore * 3 + 1, joint_to_ignore * 3 + 2))
            joint_equal = np.array([13, 19, 22, 13, 27, 30])
            dim_to_equal = np.concatenate((joint_equal * 3, joint_equal * 3 + 1, joint_equal * 3 + 2))
            pred_seq_all_joints[:, :, dim_to_use] = pred_seq
            pred_seq_all_joints[:, :, dim_to_ignore] = pred_seq_all_joints[:, :, dim_to_equal]

            # Calculate the MPJPE
            target_seq = target_seq.view(opt.test_batch_size, opt.len_output, -1, 3)
            pred_seq_all_joints = pred_seq_all_joints.view(opt.test_batch_size, opt.len_output, -1, 3)
            MPJPE += torch.sum(
                torch.mean(torch.mean(torch.norm(pred_seq_all_joints - target_seq, dim=-1), dim=-1), dim=-1)
            )
            MPJPE_per_frame += torch.sum(
                torch.mean(torch.norm(pred_seq_all_joints - target_seq, dim=-1), dim=-1), dim=0
            )
            num_seq += pred_seq.shape[0]

        MPJPE /= num_seq
        MPJPE_per_frame /= num_seq

    return MPJPE, MPJPE_per_frame
