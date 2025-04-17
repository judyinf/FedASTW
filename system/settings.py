"""
Settings for the experiment.
including model, dataset, partition,logs, and other settings.
"""
import math
import time
import os
import argparse

import numpy as np
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
from fedlab.contrib.dataset.partitioned_cifar10 import PartitionedCIFAR10

from model import ToyCifarNet

def get_settings(args):
    """
    Args:
        args: arguments from command line.
    Returns:
        model: model for the experiment.
        dataset: dataset for the experiment.
        datasize_list: list of datasize for each client.
        gen_test_loader: test loader for the experiment.
    """
    if args.dataset == "cifar10":
        model = ToyCifarNet()
        # model = resnet18()
        # model = vgg11_bn(bn=False, num_class=10)
        if args.partition == "dirichlet":
            dataset = PartitionedCIFAR10(
                root="./datasets/cifar10/",
                path="./datasets/Dirichlet_cifar_{}_{}_{}".format(args.dir, args.num_clients, args.dseed),
                dataname="cifar10",
                num_clients=args.num_clients,
                preprocess=args.preprocess,
                balance=None,
                partition="dirichlet",
                dir_alpha=args.dir,
                transform=transforms.Compose(
                    [
                        # transforms.ToPILImage(),
                        transforms.RandomCrop(32, padding=4),
                        transforms.RandomHorizontalFlip(),
                        transforms.ToTensor(),
                        transforms.Normalize(
                            (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
                        ),
                    ]
                ),
            )

        datasize_list = [len(dataset.get_dataset(i, "train")) for i in range(args.num_clients)]
        
        test_data = torchvision.datasets.CIFAR10(
            root="./datasets/cifar10/",
            train=False,
            transform=transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize(
                        (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
                    ),
                ]
            ),
        )
        gen_test_loader = DataLoader(test_data, num_workers=4, batch_size=1024)

    else :
        assert False, "dataset not supported"
    
    return model, dataset, datasize_list, gen_test_loader

def get_logs(args):
    run_time = time.strftime("%m-%d-%H-%M-%S")
    if args.partition == "dirichlet":
        data_log = "{}_{}_{}_{}".format(
            args.dataset, args.partition, args.dir, args.dseed
        )
    else:
        data_log = "{}_{}_{}".format(args.dataset, args.partition, args.dseed)
    
    dir = "./{}-logs/{}/Run{}_N{}_K{}_S{}_BS{}_EP{}_LLR{}_T{}_H{}_A{}".format(
        args.dataset,
        data_log,
        args.seed,
        args.num_clients,
        args.K,
        args.sampler_name,
        args.batch_size,
        args.epochs,
        args.lr,
        args.com_round,
        args.agnostic,
        args.all_clients,
    )

    if args.method == "fedavg":
        log = "Setting_{}_GLR{}_{}".format(args.method, args.glr, run_time)
    elif args.method == "fedastw":
        log = "Setting_{}_GLR{}_M{}".format(
            args.method, args.glr, args.mode
        )
        if args.alpha != 1:
            log += "_alpha{}".format(args.alpha)
            
        if args.mode == 1:
            log += "_F{}-{}".format(args.a, args.b)
        
        if args.tw == True:
            log += "_TW_{:.2f}".format(args.tw_a)
        
        log += "_{}".format(run_time)

    path = os.path.join(dir, log)
    return path


def get_heterogeneity(args, datasize):
    """
    Set the batch size and epoch according to datasize.
    Args:
        args: arguments from command line.
        datasize: size of the dataset.
    Returns:
        batch_size: batch size for the experiment.
        eps: eps for the experiment.
    """
    if args.agnostic == 1:
        eps = np.random.randint(2, 6)
        batch_size = np.random.randint(10, datasize) if datasize > 10 else datasize
        # print("size {} - batch {} - ep {}".format(datasize, batch_size, eps))
        return batch_size, eps
    else:
        return args.batch_size, args.epochs


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-method", type=str, default="None")

    parser.add_argument("-num_clients", type=int)
    parser.add_argument("-com_round", type=int)

    # sampling
    parser.add_argument("-sample_ratio", type=float)
    parser.add_argument("-sampler_name", type=str, default="uniform")

    # aggregation
    parser.add_argument("-all_clients", type=bool, default=True) # all clients in each round

    # local solver
    parser.add_argument("-optim", type=str)
    parser.add_argument("-batch_size", type=int)
    parser.add_argument("-epochs", type=int)
    parser.add_argument("-lr", type=float)
    parser.add_argument("-glr", type=float)
    parser.add_argument("-agnostic", type=float, default=0) # 0: no agnostic, 1: agnostic(choose batch size and epochs according to datasize)
    # parser.add_argument("-local_momentum", type=float, default=0)

    # data & reproduction
    parser.add_argument("-dataset", type=str, default="synthetic")
    parser.add_argument("-partition", type=str, default="dirichlet")  # dirichlet, pathological
    parser.add_argument("-dir", type=float, default=0.1)
    parser.add_argument("-preprocess", type=bool, default=False)
    parser.add_argument("-seed", type=int, default=0)  # run seed
    parser.add_argument("-dseed", type=int, default=0)  # data seed

    # evaluation  
    parser.add_argument("-freq", type=int, default=1) 
    
    # fedaware
    parser.add_argument("-startup", type=int, default=0)
    parser.add_argument("-alpha", type=float, default=1) # momentum coefficient

    # fedastw
    parser.add_argument("-mode", type=int, default=1) 
    # mode (int): Mode of computation
            #  0: layer_sync; 
            #  1: layer_async with constant freq; 
            #  2: layer_async with adaptive freq.
    parser.add_argument("-thresh", type=float, default=0.0) # threshold used in mode 2
    parser.add_argument("-a",type=int,default=0) # freq numerator(deep layers)
    parser.add_argument("-b",type=int,default=0) # freq denominator (deep+shallow layers)
    
    parser.add_argument("-tw", action="store_true")
    parser.add_argument("-tw_a", type=float,default=math.e/2)
    
    # nontation
    parser.add_argument("-notation", type=str, default=None) # notation for the experiment")
    return parser.parse_args()    