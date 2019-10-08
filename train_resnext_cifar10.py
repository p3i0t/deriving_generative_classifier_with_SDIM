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


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Trains ResNeXt on CIFAR',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # Positional arguments
    parser.add_argument('--data_path', type=str, default='data', help='Root for the Cifar dataset.')
    parser.add_argument('--dataset', type=str, default='cifar10', choices=['cifar10', 'cifar100'], help='Choose between Cifar10/100.')
    # Optimization options
    parser.add_argument('--epochs', '-e', type=int, default=300, help='Number of epochs to train.')
    parser.add_argument('--batch_size', '-b', type=int, default=128, help='Batch size.')
    parser.add_argument('--learning_rate', '-lr', type=float, default=0.1, help='The Learning Rate.')
    parser.add_argument('--momentum', '-m', type=float, default=0.9, help='Momentum.')
    parser.add_argument('--decay', '-d', type=float, default=0.0005, help='Weight decay (L2 penalty).')
    parser.add_argument('--test_bs', type=int, default=10)
    parser.add_argument('--schedule', type=int, nargs='+', default=[150, 225],
                        help='Decrease learning rate at these epochs.')
    parser.add_argument('--gamma', type=float, default=0.1, help='LR is multiplied by gamma on schedule.')
    # Checkpoints
    parser.add_argument('--save', '-s', type=str, default='./', help='Folder to save checkpoints.')
    parser.add_argument('--load', '-l', type=str, help='Checkpoint path to resume / test.')
    parser.add_argument('--test', '-t', action='store_true', help='Test only flag.')
    # Architecture
    parser.add_argument('--depth', type=int, default=29, help='Model depth.')
    parser.add_argument('--cardinality', type=int, default=8, help='Model cardinality (group).')
    parser.add_argument('--base_width', type=int, default=64, help='Number of channels in each group.')
    parser.add_argument('--widen_factor', type=int, default=4, help='Widen factor. 4 -> 64, 8 -> 128, ...')
    # Acceleration
    parser.add_argument('--n_gpu', type=int, default=1, help='0 = CPU.')
    parser.add_argument('--prefetch', type=int, default=2, help='Pre-fetching threads.')

    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    args = parser.parse_args()  # So error if typo

    use_cuda = not args.no_cuda and torch.cuda.is_available()

    args.device = torch.device("cuda" if use_cuda else "cpu")
    print('device: ', args.device)

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
        nlabels = 10
    else:
        train_data = dset.CIFAR100(args.data_path, train=True, transform=train_transform, download=True)
        test_data = dset.CIFAR100(args.data_path, train=False, transform=test_transform, download=True)
        nlabels = 100
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, shuffle=True,
                                               num_workers=args.prefetch, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=args.test_bs, shuffle=False,
                                              num_workers=args.prefetch, pin_memory=True)

    # Init checkpoints
    if not os.path.isdir(args.save):
        os.makedirs(args.save)

    # Init model, criterion, and optimizer
    net = CifarResNeXt(args.cardinality, args.depth, nlabels, args.base_width, args.widen_factor).to(args.device)
    print(net)

    if use_cuda and args.n_gpu > 1:
        net = torch.nn.DataParallel(net, device_ids=list(range(args.n_gpu)))

    optimizer = torch.optim.SGD(net.parameters(), lr=args.learning_rate, momentum=args.momentum,
                                weight_decay=args.decay, nesterov=True)

    #optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)
    best_train_loss = np.inf

    # train function (forward, backward, update)
    def train():
        net.train()
        loss_list = []
        for batch_idx, (x, y) in enumerate(train_loader):
            x, y = x.to(args.device), y.to(args.device)
            # forward
            output = net(x)

            # backward
            optimizer.zero_grad()
            loss = F.cross_entropy(output, y)
            loss.backward()
            optimizer.step()
            loss_list.append(loss.item())
        return np.mean(loss_list)

    # test function (forward only)
    def test():
        net.eval()
        correct = 0
        for batch_idx, (x, y) in enumerate(test_loader):
            x, y = x.to(args.device), y.to(args.device)
            # forward
            output = net(x)

            # accuracy
            pred = output.max(1)[1]
            correct += float(pred.eq(y).sum())

        test_accuracy = correct / len(test_loader.dataset)
        print('Test accuracy: {:.4f}'.format(test_accuracy))
        return test_accuracy


    for epoch in range(args.epochs):
        if epoch in args.schedule:
            args.learning_rate *= args.gamma
            for param_group in optimizer.param_groups:
                param_group['lr'] = args.learning_rate

        train_loss = train()
        print('Epoch: {}, training loss: {:.4f}.'.format(epoch + 1, train_loss))
        test_accuracy = test()
        if train_loss > best_train_loss:
            save_name = 'ResNeXt{}_{}x{}d.pth'.format(args.depth, args.cardinality, args.base_width)
            torch.save(net.state_dict(), os.path.join(args.save, save_name))
        print("Best accuracy: {:.4f}".format(test_accuracy))