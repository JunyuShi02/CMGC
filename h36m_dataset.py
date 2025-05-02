import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
from torch.utils.data import Dataset

from data_utils import expmap2xyz_torch


class H36M3dDataset(Dataset):
    """ Dataset of Human3.6M
    Args:
        opt (class): Option
        dir_path (str): Path of data directory
        state (int): 0 for train, 1 for valid, 2 for test
    """
    def __init__(self, opt, dir_path, state, actions=None):
        super().__init__()

        self.dir_path = dir_path
        self.state = state
        self.len_input = opt.len_input
        self.len_output = opt.len_output
        self.all_motion_seq = []
        self.data_ids = []

        # define actions, subjects
        if actions is None:
            self.actions = [
                "walking", "eating", "smoking", "discussion", "directions", "greeting", "phoning", "posing",
                "purchases", "sitting", "sittingdown", "takingphoto", "waiting", "walkingdog", "walkingtogether"
            ]
        else:
            self.actions = actions
        self.subjects = [['S1', 'S6', 'S7', 'S8', 'S9'], ['S11'], ['S5']][state]

        # load train/valid/test data
        self.load_data(opt)

    def load_data(self, opt):
        i = 0
        for subject in tqdm(self.subjects):
            for action in self.actions:
                for number in ['1', '2']:
                    file_path = self.dir_path + '/' + subject + '/' + action + '_' + number + '.txt'
                    motion_seq = np.array(pd.read_csv(file_path, header=None))  # load data
                    seq_len, N = motion_seq.shape
                    spl_ids = range(0, seq_len, opt.sample_rate)  # spl: sample
                    num_frames = len(spl_ids)
                    motion_seq = torch.from_numpy(motion_seq[spl_ids]).float().to(opt.device)

                    # transform to xyz, calling 'expmap2xyz_torch'
                    motion_seq[:, 0:6] = 0
                    motion_seq_3d = expmap2xyz_torch(motion_seq, device=opt.device)  # [_, 32, 3]
                    self.all_motion_seq.append(motion_seq_3d.view(num_frames, -1).cpu().data.numpy())

                    # select sequences for training/validating
                    if self.state in [0, 1, -1]:
                        frame_ids = list(range(0, num_frames - (opt.len_input + opt.len_output) + 1, opt.skip_rate))
                        self.data_ids.extend(zip([i] * len(frame_ids), frame_ids))

                    # randomly select sequences for testing
                    else:
                        rand_setting = np.random.RandomState(1234567890)
                        ids = None
                        for _ in range(0, 128):
                            rand_idx = rand_setting.randint(16, num_frames - 150)
                            ids_ = np.arange(rand_idx + 50 - opt.len_input, rand_idx + 50 + opt.len_output)
                            ids = ids_ if ids is None else np.vstack((ids, ids_))
                        frame_ids = ids[:, 0]
                        self.data_ids.extend(zip([i] * len(frame_ids), frame_ids))

                    i += 1

    def __len__(self):
        return len(self.data_ids)

    def __getitem__(self, idx):
        i, start_frame = self.data_ids[idx]
        seq_ids = np.arange(start_frame, start_frame + self.len_input + self.len_output)
        data = self.all_motion_seq[i][seq_ids]
        return data
