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


class wrapped_model(torch.nn.Module):
    """
    Wrap a torch Module to return a list of last conv activation and logit activation.
    """
    def __init__(self, m):
        super().__init__()
        self.f_conv = torch.nn.Sequential(*list(m.children())[:-2])
        self.pooling = list(m.children())[-2]
        self.l = list(m.children())[-1]

    def forward(self, x):
        out_list = []
        out = self.f_conv(x)

        out = self.pooling(out)
        out_list.append(out)

        out = torch.squeeze(torch.squeeze(out, dim=3), dim=2)
        out = self.l(out)
        out_list.append(out)
        return out_list


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


class FeatureSnapshotDataset(Dataset):
    def __init__(self, dir='', train=True, chunk_size=200):
        key = 'train' if train else 'test'
        self.data_path = os.path.join(dir, key)
        self.chunk_size = chunk_size
        # self.features = pd.read_csv(os.path.join(data_path, 'features.csv'), header=None)
        # self.logits = pd.read_csv(os.path.join(data_path, 'logits.csv'), header=None)
        # self.targets = pd.read_csv(os.path.join(data_path, 'targets.csv'), header=None)
        # assert len(self.features) == len(self.logits) == len(self.targets)

    def __len__(self):
        chunks = os.listdir(self.data_path)
        assert torch.load(chunks[0])['feature'].size(0) == self.chunk_size
        return (len(chunks) - 1) * self.chunk_size + torch.load(chunks[-1])['feature'].size(0)

    def __getitem__(self, item):
        chunk_idx = item // self.chunk_size
        idx = item % self.chunk_size
        chunk_path = os.path.join(self.data_path, 'sample_{}.pt'.format(chunk_idx))
        chunk = torch.load(chunk_path)
        return chunk['feature'][idx], chunk['logit'][idx], chunk['target'][idx]


def train_epoch(sdim, optimizer, train_loader, hps):
    """
    One epoch training.
    """
    sdim.train()

    # batch_time = AverageMeter('Time', ':6.3f')
    # data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    MIs = AverageMeter('MI', ':.4e')
    nlls = AverageMeter('NLL', ':.4e')
    margins = AverageMeter('Margin', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')

    for batch_id, (features, logits, y) in enumerate(train_loader):
        features = features.to(hps.device)
        logits = logits.to(hps.device)
        y = y.to(hps.device)

        optimizer.zero_grad()

        if use_cuda and hps.n_gpu > 1:
            f_forward = sdim.module.eval_losses
        else:
            f_forward = sdim.eval_losses

        loss, mi_loss, nll_loss, ll_margin = f_forward(features, logits, y)

        loss.backward()
        optimizer.step()

        # eval logits
        log_lik = sdim(logits)
        acc1, acc5 = accuracy(log_lik, y, topk=(1, 5))
        losses.update(loss.item(), features.size(0))
        top1.update(acc1, features.size(0))
        top5.update(acc5, features.size(0))

        MIs.update(mi_loss.item(), features.size(0))
        nlls.update(nll_loss.item(), features.size(0))
        margins.append(ll_margin.item(), features.size(0))

    print('Train loss: {:.4f}, mi: {:.4f}, nll: {:.4f}, ll_margin: {:.4f}'.format(
        losses.avg, MIs.avg, nlls.avg, margins.avg
    ))
    print('Train Acc@1: {:.2f}, Acc@2: {:.2f}'.format(top1.avg, top5.avg))

    if losses.avg < hps.loss_optimal:
        hps.loss_optimal = losses.avg
        model_path = 'sdim_{}_{}_d{}.pth'.format(hps.classifier_name, hps.problem, hps.rep_size)
        torch.save(sdim.state_dict(), os.path.join(hps.log_dir, model_path))


def test_epoch(sdim, test_loader, hps):
    """
    One epoch testing.
    """
    sdim.eval()

    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')

    with torch.no_grad():
        for batch_id, (features, logits, y) in enumerate(test_loader):
            features = features.to(hps.device)
            logits = logits.to(hps.device)
            y = y.to(hps.device)

            # eval logits
            log_lik = sdim(logits)
            acc1, acc5 = accuracy(log_lik, y, topk=(1, 5))
            top1.update(acc1, features.size(0))
            top5.update(acc5, features.size(0))

    print('Test Acc@1: {:.2f}, Acc@2: {:.2f}'.format(top1.avg, top5.avg))


def train(sdim, optimizer, hps):
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


def extract_features(classifier, train_loader, test_loader, hps):
    classifier.eval()

    snapshot_dir = '{}_{}_data_snapshot'.format(hps.problem, hps.classifier_name)
    if not os.path.exists(snapshot_dir):
        os.mkdir(snapshot_dir)

    train_dir = os.path.join(snapshot_dir, 'train')
    if not os.path.exists(train_dir):
        os.mkdir(train_dir)

    test_dir = os.path.join(snapshot_dir, 'test')
    if not os.path.exists(test_dir):
        os.mkdir(test_dir)

    # # train data
    # train_features_file = open(os.path.join(train_dir, 'features.csv'), 'ab')
    # train_logits_file = open(os.path.join(train_dir, 'logits.csv'), 'ab')
    # train_targets_file = open(os.path.join(train_dir, 'targets.csv'), 'ab')

    print('Extracting train snapshot')
    for batch_id, (x, y) in enumerate(train_loader):
        x = x.to(hps.device)
        y = y.to(hps.device)

        with torch.no_grad():
            out_list = classifier(x)

        sample = {'feature': out_list[0], 'logit': out_list[1], 'target': y}
        torch.save(sample, os.path.join(train_dir, 'sample_{}.pt'.format(batch_id)))


        # np.savetxt(train_features_file, features, delimiter=',', fmt='%.4e')
        # np.savetxt(train_logits_file, logits, delimiter=',', fmt='%.4e')
        # np.savetxt(train_targets_file, targets, delimiter=',', fmt='%d')

    # # test data
    # test_features_file = open(os.path.join(test_dir, 'features.csv'), 'ab')
    # test_logits_file = open(os.path.join(test_dir, 'logits.csv'), 'ab')
    # test_targets_file = open(os.path.join(test_dir, 'targets.csv'), 'ab')

    print('Extracting test snapshot')
    for batch_id, (x, y) in enumerate(test_loader):
        x = x.to(hps.device)
        y = y.to(hps.device)

        with torch.no_grad():
            out_list = classifier(x)

        sample = {'feature': out_list[0], 'logit': out_list[1], 'target': y}
        torch.save(sample, os.path.join(test_dir, 'sample_{}.pt'.format(batch_id)))

        # features = out_list[0].split(split_size=1, dim=0)
        # logits = out_list[1].split(split_size=1, dim=0)
        # targets = y.split(split_size=1, dim=0)
        #
        # for idx, (feature, logit, target) in enumerate(zip(features, logits, targets)):
        #     sample = {'feature': feature, 'logit': logit, 'target': target}
        #     torch.save(sample, os.path.join(test_dir, 'sample_{}.pt'.format(global_idx)))
        #     global_idx += 1

        # features = torch.squeeze(torch.squeeze(out_list[0], dim=3), dim=2).cpu().numpy()  # squeeze to 2d tensor.
        # logits = out_list[1].cpu().numpy()
        # targets = y.cpu().numpy()
        # np.savetxt(test_features_file, features, delimiter=',', fmt='%.4e')
        # np.savetxt(test_logits_file, logits, delimiter=',', fmt='%.4e')
        # np.savetxt(test_targets_file, targets, delimiter=',', fmt='%d')


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
    parser.add_argument("--chunksize", type=int,
                        default=200, help="chunk size for storage of data snapshot.")
    parser.add_argument("--data_dir", type=str, default='data',
                        help="Location of data")

    # Optimization hyperparams:
    parser.add_argument("--n_batch_train", type=int,
                        default=256, help="Minibatch size")
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
                        default=512, help="output size of 1x1 conv network for mutual information estimation")
    parser.add_argument("--rep_size", type=int,
                        default=64, help="size of the global representation from encoder")
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

    # Extract features of classifiers as snapshots or load pre-extracted data snapshots.
    snapshot_dir = '{}_{}_data_snapshot'.format(hps.problem, hps.classifier_name)
    if not os.path.isdir(snapshot_dir):
        # ToDo: this verification is weak.
        print('Pre-extracted data snapshot not found, extracting ...')
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
        classifier = wrapped_model(classifier)
        extract_features(classifier, train_loader, test_loader, hps)

    print('Loading data snapshot ...')
    train_snapshot = FeatureSnapshotDataset(dir=snapshot_dir, train=True)
    train_loader = DataLoader(dataset=train_snapshot, batch_size=hps.n_batch_train, shuffle=True,
                              pin_memory=True, num_workers=8)

    test_snapshot = FeatureSnapshotDataset(dir=snapshot_dir, train=False)
    test_loader = DataLoader(dataset=test_snapshot, batch_size=hps.n_batch_test, shuffle=False,
                             pin_memory=True, num_workers=8)

    print(train_snapshot[0])
    print(train_snapshot[0][0])
    local_size = train_snapshot[0][0].size(1)
    sdim = SDIM(local_feature_size=local_size, rep_size=hps.rep_size, mi_units=hps.mi_units, n_classes=hps.n_classes).to(hps.device)

    if use_cuda and hps.n_gpu > 1:
        sdim = torch.nn.DataParallel(sdim, device_ids=list(range(hps.n_gpu)))

    optimizer = Adam(filter(lambda param: param.requires_grad is True, sdim.parameters()), lr=hps.lr)

    train(sdim, optimizer, hps)
