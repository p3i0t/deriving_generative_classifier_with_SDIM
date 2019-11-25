import argparse
import sys
import os
import numpy as np
import pandas as pd

import torch
import torchvision.models as models
from torchvision import datasets
from torch.utils.data import DataLoader, Dataset
from torch.optim import Adam

from sdim import SDIM
from utils import cal_parameters, AverageMeter
import torchvision.transforms as transforms


def get_model(model_name='resnext50_32x4d'):
    if model_name == 'resnext101_32x8d':
        m = models.resnext101_32x8d(pretrained=True)
    elif model_name == 'resnext50_32x4d':
        m = models.resnext50_32x4d(pretrained=True)
    print('Model name: {}, # parameters: {}'.format(model_name, cal_parameters(m)))
    return m


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].float().sum() / batch_size * 100
        res.append(correct_k.item())
    return res


def train_epoch(sdim, optimizer, train_loader, hps):
    """
    One epoch training.
    """
    sdim.train()

    losses = AverageMeter('Loss', ':.4e')
    MIs = AverageMeter('MI', ':.4e')
    nlls = AverageMeter('NLL', ':.4e')
    margins = AverageMeter('Margin', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')

    for batch_id, (x, y) in enumerate(train_loader):
        x, y = x.to(hps.device), y.to(hps.device)

        # backward
        optimizer.zero_grad()

        if use_cuda and hps.n_gpu > 1:
            f_forward = sdim.module.eval_losses
        else:
            f_forward = sdim.eval_losses
        loss, mi_loss, nll_loss, ll_margin = f_forward(x, y)
        loss.backward()
        optimizer.step()

        # eval logits
        log_lik = sdim(x)
        acc1, acc5 = accuracy(log_lik, y, topk=(1, 5))
        losses.update(loss.item(), x.size(0))
        top1.update(acc1, x.size(0))
        top5.update(acc5, x.size(0))

        MIs.update(mi_loss.item(), x.size(0))
        nlls.update(nll_loss.item(), x.size(0))
        margins.append(ll_margin.item(), x.size(0))

    print('Train loss: {:.4f}, mi: {:.4f}, nll: {:.4f}, ll_margin: {:.4f}'.format(
        losses.avg, MIs.avg, nlls.avg, margins.avg
    ))
    print('Train Acc@1: {:.2f}, Acc@2: {:.2f}'.format(top1.avg, top5.avg))

    if losses.avg < hps.loss_optimal:
        hps.loss_optimal = losses.avg
        model_path = 'sdim_{}_{}_rep{}.pth'.format(hps.classifier_name, hps.problem, hps.rep_size)
        torch.save(sdim.state_dict(), os.path.join(hps.log_dir, model_path))


def test_epoch(sdim, test_loader, hps):
    """
    One epoch testing.
    """
    sdim.eval()

    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')

    with torch.no_grad():
        for batch_id, (x, y) in enumerate(test_loader):
            x = x.to(hps.device)
            y = y.to(hps.device)

            # eval logits
            log_lik = sdim(x)
            acc1, acc5 = accuracy(log_lik, y, topk=(1, 5))
            top1.update(acc1, x.size(0))
            top5.update(acc5, x.size(0))

    print('Test Acc@1: {:.2f}, Acc@2: {:.2f}'.format(top1.avg, top5.avg))


if __name__ == '__main__':
    # This enables a ctr-C without triggering errors
    import signal

    signal.signal(signal.SIGINT, lambda x, y: sys.exit(0))

    parser = argparse.ArgumentParser()
    parser.add_argument("--verbose", action='store_true', help="Verbose mode")
    parser.add_argument("--inference", action="store_true",
                        help="Used in inference mode")
    parser.add_argument("--rejection_inference", action="store_true",
                        help="Used in inference mode with rejection")
    parser.add_argument("--log_dir", type=str,
                        default='./logs', help="Location to save logs")

    # Dataset hyperparams:
    parser.add_argument("--problem", type=str, default='imagenet',
                        help="Problem (mnist/fashion/cifar10")
    parser.add_argument("--n_classes", type=int,
                        default=1000, help="number of classes of dataset.")
    parser.add_argument("--data_dir", type=str, default='data',
                        help="Location of data")

    # Optimization hyperparams:
    parser.add_argument("--n_batch_train", type=int,
                        default=128, help="Minibatch size")
    parser.add_argument("--n_batch_test", type=int,
                        default=200, help="Minibatch size")
    parser.add_argument("--optimizer", type=str,
                        default="adam", help="adam or adamax")
    parser.add_argument("--lr", type=float, default=0.001,
                        help="Base learning rate")
    parser.add_argument("--beta1", type=float, default=.9, help="Adam beta1")
    parser.add_argument("--polyak_epochs", type=float, default=1,
                        help="Nr of averaging epochs for Polyak and beta2")
    parser.add_argument("--weight_decay", type=float, default=1.,
                        help="Weight decay. Switched off by default.")
    parser.add_argument("--epochs", type=int, default=500,
                        help="Total number of training epochs")

    # Inference hyperparams:
    parser.add_argument("--percentile", type=float, default=0.01,
                        help="percentile value for inference with rejection.")

    # Model hyperparams:
    parser.add_argument("--image_size", type=int,
                        default=224, help="Image size")
    parser.add_argument("--mi_units", type=int,
                        default=256, help="output size of 1x1 conv network for mutual information estimation")
    parser.add_argument("--rep_size", type=int,
                        default=128, help="size of the global representation from encoder")
    parser.add_argument("--classifier_name", type=str, default='resnext50_32x4d',
                        help="name of discriminative classifier")
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--n_gpu', type=int, default=1, help='0 = CPU.')
    # Ablation
    parser.add_argument("--seed", type=int, default=1234, help="Random seed")
    hps = parser.parse_args()  # So error if typo

    use_cuda = not hps.no_cuda and torch.cuda.is_available()

    torch.manual_seed(hps.seed)

    hps.device = torch.device("cuda" if use_cuda else "cpu")

    # Dataloaders
    train_transform = transforms.Compose([transforms.RandomResizedCrop(224),
                                          transforms.RandomHorizontalFlip(),
                                          transforms.ToTensor(),
                                          transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    test_transform = transforms.Compose([transforms.Resize(256),
                                         transforms.CenterCrop(224),
                                         transforms.ToTensor(),
                                         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    train_set = datasets.ImageNet(root='~/data', split='train', download=True, transform=train_transform)
    train_loader = DataLoader(dataset=train_set, batch_size=hps.chunksize, shuffle=True,
                              pin_memory=True, num_workers=8)

    test_set = datasets.ImageNet(root='~/data', split='val', download=True, transform=test_transform)
    test_loader = DataLoader(dataset=test_set, batch_size=hps.chunksize, shuffle=False,
                             pin_memory=True, num_workers=8)

    # Models
    print('Classifier name: {}'.format(hps.classifier_name))
    classifier = get_model(model_name=hps.classifier_name).to(hps.device)

    train_loader = DataLoader(dataset=train_loader, batch_size=hps.n_batch_train, shuffle=True,
                              pin_memory=True, num_workers=8)

    # test_snapshot = FeatureSnapshotDataset(dir=snapshot_dir, train=False)
    test_loader = DataLoader(dataset=test_loader, batch_size=hps.n_batch_test, shuffle=False,
                             pin_memory=True, num_workers=8)

    sdim = SDIM(disc_classifier=classifier, rep_size=hps.rep_size, mi_units=hps.mi_units, n_classes=hps.n_classes).to(hps.device)

    if use_cuda and hps.n_gpu > 1:
        sdim = torch.nn.DataParallel(sdim, device_ids=list(range(hps.n_gpu)))

    optimizer = Adam(filter(lambda param: param.requires_grad is True, sdim.parameters()), lr=hps.lr)

    torch.manual_seed(hps.seed)
    np.random.seed(hps.seed)

    # Create log dir
    logdir = os.path.abspath(hps.log_dir) + "/"
    if not os.path.exists(logdir):
        os.mkdir(logdir)

    hps.loss_optimal = 1e5
    for epoch in range(1, hps.epochs + 1):
        print('===> Epoch: {}'.format(epoch))
        train_epoch(sdim, optimizer, train_loader, hps)
        test_epoch(sdim, test_loader, hps)
