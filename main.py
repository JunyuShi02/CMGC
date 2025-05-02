import argparse
from train import train_h36m3d
from test import test_h36m3d


def main():

    parser = argparse.ArgumentParser()

    # model options
    parser.add_argument('--lr', type=float, default=0.0013,
                        help='learning rate')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='batch size')
    parser.add_argument('--test_batch_size', type=int, default=512,
                        help='test batch size')
    parser.add_argument('--epoch', type=int, default=50,
                        help='number of epoch')

    # training options
    parser.add_argument('--len_input', type=int,
                        help='length of input sequence')
    parser.add_argument('--len_output', type=int,
                        help='length of output sequence')
    parser.add_argument('--dct_n', type=int, help='dimension of DCT coefficient')
    parser.add_argument('--state', type=str, default='train', choices=['train', 'test'],
                        help='train or test')
    parser.add_argument('--skip_rate', type=int, default=1,
                        help='sample interval of neighbor sequences')
    parser.add_argument('--device', type=str, default='cuda:0',
                        help='device to use')
    parser.add_argument('--sample_rate', type=int, default=2,
                        help='sample interval of a consecutive sequence')

    # other options
    parser.add_argument('--dir_path', type=str,
                        help='path to data directory')
    parser.add_argument('--ckpt_path', type=str, default='checkpoint/',
                        help='path to save')
    parser.add_argument('--is_save', type=bool, default=True,
                        help='save or not')
    parser.add_argument('--model_path', type=str, default=None,
                        help='path to previous model. Required when retraining')
    parser.add_argument('--optimizer_path', type=str, default=None,
                        help='path to previous optimizer. Required when retraining')
    parser.add_argument('--start_epoch', type=int, default=0,
                        help='needed when retrain')
    args = parser.parse_args()

    class Option:
        # training options
        len_input = args.len_input
        len_output = args.len_output
        dct_n = args.dct_n
        sample_rate = args.sample_rate
        skip_rate = args.skip_rate
        device = args.device
        # model options
        batch_size = args.batch_size
        test_batch_size = args.test_batch_size
        lr = args.lr
        epoch = args.epoch
        is_save = args.is_save
        model_path = args.model_path
        optimizer_path = args.optimizer_path
        start_epoch = args.start_epoch
        dir_path = args.dir_path
        ckpt_path = args.ckpt_path

    if args.state == 'train':
        train_h36m3d(opt=Option)
    else:
        test_h36m3d(opt=Option)


if __name__ == "__main__":
    main()