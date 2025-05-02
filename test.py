import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from h36m_dataset import H36M3dDataset
from GCN import CMGC

def test_h36m3d(opt):

    model = CMGC(in_dims=opt.len_input + opt.len_output, hid_dims=256,
                  out_dims=opt.len_input + opt.len_output, drop_rate=0.5,
                  num_blocks=12, num_nodes=66, opt=opt)
    model.to(opt.device)

    # Define dataloader
    actions = [
        "walking", "eating", "smoking", "discussion", "directions", "greeting", "phoning", "posing",
        "purchases", "sitting", "sittingdown", "takingphoto", "waiting", "walkingdog", "walkingtogether"
    ]
    test_loaders = {}
    for action in actions:
        test_dataset = H36M3dDataset(opt=opt, dir_path=opt.dir_path, state=-1, actions=[action])
        test_loader = DataLoader(
            dataset=test_dataset, batch_size=opt.test_batch_size,
            shuffle=False, pin_memory=True, drop_last=False
        )
        test_loaders[action] = test_loader

    print(">>> Total parameters: {:.2f}M".format(sum(p.numel() for p in model.parameters()) / 1000000.0))
    model.load_state_dict(torch.load(opt.model_path))

    MPJPE, MPJPE_per_frame, MPJPE_per_frame_action = run_test_h36m3d(model, test_loaders, opt, actions)
    print(">>> Average ", np.round(MPJPE_per_frame.cpu().numpy(), 2))
    print(">>> Average action ", np.round(MPJPE_per_frame_action.cpu().numpy(), 2))
    print(">>> MPJPE: ", MPJPE.item())


def run_test_h36m3d(model, loaders, opt, actions):

    with torch.no_grad():

        model.eval()
        num_samples = 0
        total_MPJPE = 0.  # per frame
        total_action_MPJPE = 0.

        for action in actions:

            num_action_samples = 0
            action_MPJPE = 0.
            loader = loaders[action]

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
                B, _, _ = target_seq.shape
                target_seq = target_seq.view(B, opt.len_output, -1, 3)
                pred_seq_all_joints = pred_seq_all_joints.view(B, opt.len_output, -1, 3)

                batch_MPJPE = torch.sum(
                    torch.mean(torch.norm(pred_seq_all_joints - target_seq, dim=-1), dim=-1), dim=0
                )
                num_samples += B
                num_action_samples += B
                total_MPJPE += batch_MPJPE
                action_MPJPE += batch_MPJPE

            action_MPJPE /= num_action_samples
            print("Action: ", action)
            print("MPJPE ", np.round(action_MPJPE.cpu().numpy(), 2))

            total_action_MPJPE += action_MPJPE

        total_action_MPJPE /= len(actions)
        total_MPJPE /= num_samples

    return torch.mean(total_MPJPE), total_MPJPE, total_action_MPJPE
