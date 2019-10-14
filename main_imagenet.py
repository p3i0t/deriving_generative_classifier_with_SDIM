import argparse
import sys
import os
import numpy as np

import torch
import torchvision.models as models
from torchvision import datasets
from torch.utils.data import DataLoader
from torch.optim import Adam

from sdim import SDIM
from utils import get_dataset, cal_parameters
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
        out_list.append(out)

        out = self.pooling(out)
        out = torch.squeeze(torch.squeeze(out, dim=3), dim=2)
        out = self.l(out)
        out_list.append(out)
        return out_list


def get_model(model_name='resnext50_32x4d'):
    if model_name == 'resnext101_32x8d':
        m = models.resnext101_32x8d(pretrained=True)
    elif model_name == 'resnext101_32x8d':
        m = models.resnext50_32x4d(pretrained=True)
    print('Model name: {}, # parameters: {}'.format(model_name, cal_parameters(m)))
    return wrapped_model(m)


def train(sdim, optimizer, hps):
    torch.manual_seed(hps.seed)
    np.random.seed(hps.seed)

    # Create log dir
    logdir = os.path.abspath(hps.log_dir) + "/"
    if not os.path.exists(logdir):
        os.mkdir(logdir)

    train_transform = transforms.Compose([transforms.RandomSizedCrop(224),
                                          transforms.RandomHorizontalFlip(),
                                          transforms.ToTensor(),
                                          transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    test_transform = transforms.Compose([transforms.Resize(256),
                                         transforms.CenterCrop(224),
                                         transforms.ToTensor(),
                                         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    train_set = datasets.ImageNet(root='~/data', split='train', download=True, transform=train_transform)
    train_loader = DataLoader(dataset=train_set, batch_size=hps.n_batch_train, shuffle=True)

    test_set = datasets.ImageNet(root='~/data', split='val', download=True, transform=test_transform)
    test_loader = DataLoader(dataset=test_set, batch_size=hps.n_batch_test, shuffle=False)

    # dataset = get_dataset(data_name=hps.problem, train=True)
    # train_loader = DataLoader(dataset=dataset, batch_size=hps.n_batch_train, shuffle=True)
    #
    # dataset = get_dataset(data_name=hps.problem, train=False)
    # test_loader = DataLoader(dataset=dataset, batch_size=hps.n_batch_test, shuffle=False)

    min_loss = 1e3
    for epoch in range(1, hps.epochs + 1):
        sdim.train()
        loss_list = []
        mi_list = []
        nll_list = []
        margin_list = []

        for batch_id, (x, y) in enumerate(train_loader):
            x = x.to(hps.device)
            y = y.to(hps.device)

            optimizer.zero_grad()

            loss, mi_loss, nll_loss, ll_margin = sdim.eval_losses(x, y)
            loss.backward()
            optimizer.step()

            loss_list.append(loss.item())
            mi_list.append(mi_loss.item())
            nll_list.append(nll_loss.item())
            margin_list.append(ll_margin.item())

        print('===> Epoch: {}'.format(epoch))
        print('loss: {:.4f}, mi: {:.4f}, nll: {:.4f}, ll_margin: {:.4f}'.format(
            np.mean(loss_list),
            np.mean(mi_list),
            np.mean(nll_list),
            np.mean(margin_list)
        ))
        if np.mean(loss_list) < min_loss:
            min_loss = np.mean(loss_list)
            model_path = 'sdim_{}_{}_d{}.pth'.format(hps.classifier_name, hps.problem, hps.rep_size)
            torch.save(sdim.state_dict(), os.path.join(hps.log_dir, model_path))

        sdim.eval()
        # Evaluate accuracy on test set.
        if epoch > 10:
            acc_list = []
            for batch_id, (x, y) in enumerate(test_loader):
                x = x.to(hps.device)
                y = y.to(hps.device)

                preds = sdim(x).argmax(dim=1)
                acc = (preds == y).float().mean()
                acc_list.append(acc.item())
            print('Test accuracy: {:.3f}'.format(np.mean(acc_list)))


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
                        default=32, help="Image size")
    parser.add_argument("--mi_units", type=int,
                        default=512, help="output size of 1x1 conv network for mutual information estimation")
    parser.add_argument("--rep_size", type=int,
                        default=64, help="size of the global representation from encoder")
    parser.add_argument("--encoder_name", type=str, default='resnet25',
                        help="encoder name: resnet#")
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')

    # Ablation
    parser.add_argument("--seed", type=int, default=1234, help="Random seed")
    hps = parser.parse_args()  # So error if typo

    use_cuda = not hps.no_cuda and torch.cuda.is_available()

    torch.manual_seed(hps.seed)

    hps.device = torch.device("cuda" if use_cuda else "cpu")

    m = get_model(model_name='resnext50_32x4d').to(hps.device)
    m = wrapped_model(m)
    sdim = SDIM(disc_classifier=m, rep_size=hps.rep_size, mi_units=hps.mi_units, n_classes=hps.n_classes).to(hps.device)

    optimizer = Adam(filter(lambda param: param.requires_grad is True, sdim.parameters()), lr=hps.lr)

    train(sdim, optimizer, hps)
