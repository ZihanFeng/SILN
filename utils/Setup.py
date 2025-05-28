import torch
import logging
import random
import numpy as np


def setup(args):
    setup_logging()
    setup_info(args)
    setup_device(args)
    setup_seed(args)

def setup_device(args):
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    args.n_gpu = torch.cuda.device_count()


def setup_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)


def setup_logging():
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S',
                        level=logging.INFO)
    logger = logging.getLogger(__name__)
    return logger


def trans_to_cuda(variable):
    device = torch.device("cuda:{}".format(0) if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        return variable.to(device)
    else:
        return variable

def setup_info(args):
    """
    GitHub imposes a maximum file size limit of 25 MB, 
    making it challenging to upload complete preprocessed datasets.
    """
    if args.dataset == 'twitter':
        setup_twitter(args)
    elif args.dataset == 'douban':
        setup_douban(args)
    elif args.dataset == 'android':
        setup_android(args)
    elif args.dataset == 'christian':
        setup_christian(args)
    # elif args.dataset == 'WeiboM':
    #     setup_WeiboM(args)
    # elif args.dataset == 'WeiboL':
    #     setup_WeiboL(args)
    else:
        print(' Data_Path ERROR ! ')

    args.cascade_train_path = 'dataset/' + args.dataset + '/cascade_train.json'
    args.cascade_valid_path = 'dataset/' + args.dataset + '/cascade_valid.json'
    args.cascade_test_path = 'dataset/' + args.dataset + '/cascade_test.json'
    args.graphPath = 'dataset/' + args.dataset + '/graph.npz'
    args.dis_path = 'dataset/' + args.dataset + '/distance.npy'


def setup_twitter(args):
    args.user_num = 12627 + 2
    args.dim = 128
    args.n_warmup_steps = 1000
    args.window_size = 3


def setup_douban(args):
    args.user_num = 12232 + 2
    args.dim = 64
    args.n_warmup_steps = 500
    args.window_size = 3


def setup_android(args):
    args.user_num = 2927 + 2
    args.dim = 64
    args.n_warmup_steps = 200
    args.window_size = 2


def setup_christian(args):
    args.user_num = 1651 + 2
    args.dim = 64
    args.n_warmup_steps = 200
    args.window_size = 2
