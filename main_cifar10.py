# -*- coding: utf-8 -*-
from __future__ import division

""" 
Trains a ResNeXt Model on Cifar10 and Cifar 100. Implementation as defined in:
Xie, S., Girshick, R., Dollár, P., Tu, Z., & He, K. (2016). 
Aggregated residual transformations for deep neural networks. 
arXiv preprint arXiv:1611.05431.
"""

import argparse
import os
import numpy as np

import torch
import torch.nn.functional as F
import torchvision.datasets as dset
import torchvision.transforms as transforms
from models_cifar10 import CifarResNeXt
from sdim import SDIM
from utils import get_dataset, cal_parameters


def clean_state_dict(state_dict):
    # see https://discuss.pytorch.org/t/solved-keyerror-unexpected-key-module-encoder-embedding-weight-in-state-dict/1686/3
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        assert k.startswith('module.')
        name = k[7:]  # remove `module.`
        new_state_dict[name] = v
    # load params
    return new_state_dict


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Trains ResNeXt on CIFAR',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # Positional arguments
    parser.add_argument('--data_path', type=str, default='data', help='Root for the Cifar dataset.')
    parser.add_argument('--dataset', type=str, default='cifar10', choices=['cifar10', 'cifar100'], help='Choose between Cifar10/100.')

    # Optimization options
    parser.add_argument('--epochs', type=int, default=300, help='Number of epochs to train.')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size.')
    parser.add_argument('--learning_rate', '-lr', type=float, default=0.1, help='The Learning Rate.')
    parser.add_argument('--momentum', type=float, default=0.9, help='Momentum.')
    parser.add_argument('--decay', type=float, default=0.0005, help='Weight decay (L2 penalty).')
    parser.add_argument('--test_batch_size', type=int, default=100)
    parser.add_argument('--schedule', type=int, nargs='+', default=[150, 225],
                        help='Decrease learning rate at these epochs.')
    parser.add_argument('--gamma', type=float, default=0.1, help='LR is multiplied by gamma on schedule.')

    # Checkpoints
    parser.add_argument('--save',  type=str, default='./logs', help='Folder to save checkpoints.')
    parser.add_argument('--load',  type=str, default='./logs', help='Checkpoint path to resume / test.')

    # Architecture
    parser.add_argument('--depth', type=int, default=29, help='Model depth.')
    parser.add_argument('--cardinality', type=int, default=8, help='Model cardinality (group).')
    parser.add_argument('--base_width', type=int, default=64, help='Number of channels in each group.')
    parser.add_argument('--widen_factor', type=int, default=4, help='Widen factor. 4 -> 64, 8 -> 128, ...')

    # SDIM parameters
    parser.add_argument("--mi_units", type=int,
                        default=512, help="output size of 1x1 conv network for mutual information estimation")
    parser.add_argument("--rep_size", type=int,
                        default=64, help="size of the global representation from encoder")
    parser.add_argument("--encoder_name", type=str, default='SDIM_ResNeXt29_8x64d',
                        help="encoder name: resnet#")

    # Acceleration
    parser.add_argument('--n_gpu', type=int, default=1, help='0 = CPU.')
    parser.add_argument('--prefetch', type=int, default=2, help='Pre-fetching threads.')

    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')

    args = parser.parse_args()  # So error if typo

    use_cuda = not args.no_cuda and torch.cuda.is_available()

    args.device = torch.device("cuda" if use_cuda else "cpu")

    mean = [x / 255 for x in [125.3, 123.0, 113.9]]
    std = [x / 255 for x in [63.0, 62.1, 66.7]]

    train_transform = transforms.Compose(
        [transforms.RandomHorizontalFlip(),
         transforms.RandomCrop(32, padding=4),
         transforms.ToTensor(),
         transforms.Normalize(mean, std)])

    test_transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize(mean, std)])

    if args.dataset == 'cifar10':
        train_data = dset.CIFAR10(args.data_path, train=True, transform=train_transform, download=True)
        test_data = dset.CIFAR10(args.data_path, train=False, transform=test_transform, download=True)
        args.n_classes = 10
    else:
        train_data = dset.CIFAR100(args.data_path, train=True, transform=train_transform, download=True)
        test_data = dset.CIFAR100(args.data_path, train=False, transform=test_transform, download=True)
        args.n_classes = 100
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, shuffle=True,
                                               num_workers=args.prefetch, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=args.test_batch_size, shuffle=False,
                                              num_workers=args.prefetch, pin_memory=True)

    # Init checkpoints
    if not os.path.isdir(args.save):
        os.makedirs(args.save)

    # Init model, criterion, and optimizer
    classifier = CifarResNeXt(args.cardinality, args.depth, args.n_classes, args.base_width, args.widen_factor).to(args.device)
    print('# classifier parameters: ', cal_parameters(classifier))

    save_name = 'ResNeXt{}_{}x{}d.pth'.format(args.depth, args.cardinality, args.base_width)
    classifier.load_state_dict(clean_state_dict(torch.load(os.path.join(args.save, save_name))))

    sdim = SDIM(disc_classifier=classifier, rep_size=args.rep_size, mi_units=args.mi_units, n_classes=args.n_classes).to(args.device)

    optimizer = torch.optim.Adam(filter(lambda param: param.requires_grad is True, sdim.parameters()),
                                 lr=args.learning_rate)

    if use_cuda and args.n_gpu > 1:
        sdim = torch.nn.DataParallel(sdim, device_ids=list(range(args.n_gpu)))

    best_train_loss = np.inf
    best_accuracy = 0.

    # train function (forward, backward, update)
    def train():
        sdim.train()
        loss_list = []
        mi_list = []
        nll_list = []
        margin_list = []

        for batch_idx, (x, y) in enumerate(train_loader):
            x, y = x.to(args.device), y.to(args.device)

            # backward
            optimizer.zero_grad()

            loss, mi_loss, nll_loss, ll_margin = sdim.eval_losses(x, y)
            loss.backward()
            optimizer.step()

            loss_list.append(loss.item())
            mi_list.append(mi_loss.item())
            nll_list.append(nll_loss.item())
            margin_list.append(ll_margin.item())

        print('loss: {:.4f}, mi: {:.4f}, nll: {:.4f}, ll_margin: {:.4f}'.format(
            np.mean(loss_list),
            np.mean(mi_list),
            np.mean(nll_list),
            np.mean(margin_list)
        ))
        return np.mean(loss_list)

    # test function (forward only)
    def test():
        sdim.eval()
        acc_list = []
        for batch_idx, (x, y) in enumerate(test_loader):
            x, y = x.to(args.device), y.to(args.device)

            preds = sdim(x).argmax(dim=1)
            acc = (preds == y).float().mean()
            acc_list.append(acc.item())

        test_accuracy = np.mean(acc_list)
        print('Test accuracy: {:.4f}'.format(test_accuracy))
        return test_accuracy

    for epoch in range(args.epochs):
        print('===> Epoch: {}'.format(epoch + 1))
        train_loss = train()
        test_accuracy = test()

        if train_loss > best_train_loss:
            best_accuracy = test_accuracy
            save_name = 'SDIM_ResNeXt{}_{}x{}d.pth'.format(args.depth, args.cardinality, args.base_width)
            if use_cuda and args.n_gpu > 1:
                state = sdim.module.state_dict()
            else:
                state = sdim.state_dict()

            torch.save(state, os.path.join(args.save, save_name))
        print("Best accuracy: {:.4f}".format(test_accuracy))


