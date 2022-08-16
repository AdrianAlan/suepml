import argparse
import logging
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import tqdm
import yaml

from suep.disco import distance_corr
from suep.checkpoints import EarlyStopping
from suep.generator import CalorimeterDataset
from torch.cuda.amp import GradScaler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import Adam, lr_scheduler
from torch.utils.data import DataLoader
from tqdm import trange
from utils import IsValidFile, get_data_loader

from suepvision.smodels import (
    LeNet5,
    get_resnet18,
    get_resnet50,
    get_enet,
    get_convnext
)


SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.random.manual_seed(SEED)
torch.cuda.manual_seed(SEED)


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':1.5f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{avg' + self.fmt + '} ({name})'
        return fmtstr.format(**self.__dict__)


class Plotting():

    def __init__(self, save_dir):
        self.save_dir = save_dir
        plt.style.use('./misc/style.mplstyle')
        self.colors = ['orange', 'red', 'black']
        self.markers = ["s", "v", "o"]

    def draw_loss(self, data_train, data_val, data_acc, name, label="Loss"):
        """Plots the training and validation loss"""

        fig, ax1 = plt.subplots()
        ax1.set_xlabel("Epoch", horizontalalignment='right', x=1.0)
        ax1.set_ylabel("Loss", horizontalalignment='right', y=1.0)
        ax1.set_yscale('log')
        ax1.tick_params(axis='y', labelcolor='red')
        ax1.plot(data_train,
                 color=self.colors[0],
                 label='Training')
        ax1.plot(data_val,
                 color=self.colors[1],
                 label='Validation')
        ax2 = ax1.twinx()
        ax2.set_ylabel('Accuracy', color='black')
        ax2.tick_params(axis='y', labelcolor='black')
        ax2.plot(data_acc,
                 color=self.colors[2],
                 label='Accuracy')
        ax1.legend()
        ax2.legend()
        plt.savefig('{}/loss-{}'.format(self.save_dir, name))
        plt.close(fig)


def set_logging(name, filename, verbose):
    logger = logging.getLogger(name)
    fh = logging.FileHandler(filename)
    ch = logging.StreamHandler()

    logger.setLevel(logging.DEBUG)
    fh.setLevel(logging.DEBUG)
    if verbose:
        ch.setLevel(logging.INFO)

    f = logging.Formatter('%(asctime)s:%(name)s:%(levelname)s:%(message)s',
                          datefmt='%m/%d/%Y %I:%M')
    fh.setFormatter(f)
    ch.setFormatter(f)

    logger.addHandler(fh)
    logger.addHandler(ch)

    return logger


def execute(rank,
            world_size,
            name,
            architecture,
            dataset,
            training_pref,
            disco_mode,
            verbose=False):

    setup(rank, world_size)

    if rank == 0:
        logname = "models/{}.log".format(name)
        logger = set_logging("Train {}".format(name), logname, verbose)

    plot = Plotting("models")

    batch_size_train = training_pref['batch_size_train']
    batch_size_validation = training_pref['batch_size_validation']

    train_loader = get_data_loader(dataset['train'][rank],
                                   batch_size_train,
                                   training_pref['workers'],
                                   dataset['in_dim'],
                                   rank,
                                   boosted=dataset['boosted'],
                                   shuffle=True)

    val_loader = get_data_loader(dataset['validation'][rank],
                                 batch_size_validation,
                                 training_pref['workers'],
                                 dataset['in_dim'],
                                 rank,
                                 boosted=dataset['boosted'],
                                 shuffle=False)

    model = eval(architecture)()
    model = nn.SyncBatchNorm.convert_sync_batchnorm(model).to(rank)
    if rank == 0:
        logger.debug('Model architecture:\n{}'.format(str(model)))

    cudnn.benchmark = True
    net = DDP(model, device_ids=[rank])

    optimizer = Adam(
        net.parameters(),
        lr=training_pref['learning_rate'],
        weight_decay=training_pref['weight_decay']
    )
    scheduler = lr_scheduler.MultiStepLR(
        optimizer,
        milestones=np.arange(10, 50, 10),
        gamma=0.5
    )

    if rank == 0:
        cp_es = EarlyStopping(
            logger,
            patience=training_pref['patience'],
            save_path='models/{}'.format(name)
        )
    criterion = nn.CrossEntropyLoss().to(rank)
    scaler = GradScaler()
    verobse = verbose and rank == 0
    t_loss = torch.cuda.FloatTensor([], device=rank)
    v_loss = torch.cuda.FloatTensor([], device=rank)
    v_acc = torch.cuda.FloatTensor([], device=rank)
    acc, loss = AverageMeter('Accuracy'), AverageMeter('Loss'),
    correlation = AverageMeter('Correlation')
    for epoch in range(1, training_pref['max_epochs']+1):

        net.train()
        loss.reset()
        if verbose:
            tr = trange(len(train_loader), file=sys.stdout)

        for images, targets, tracks, spher in train_loader:

            optimizer.zero_grad()

            outputs = net(images)
            targets = torch.cuda.LongTensor(targets, device=rank)

            l = criterion(outputs, targets)

            if disco_mode:

                if disco_mode == 1:
                    value = torch.tensor(tracks).to(rank) + 0.
                elif disco_mode == 2:
                    value = torch.tensor(spher).to(rank) + 0.

                pos = targets == 0
                corr = distance_corr(
                    F.softmax(outputs, dim=1)[:, 1][pos],
                    value[pos],
                    1
                )
                if torch.isnan(corr):
                    corr = 0
                l = l + corr

            scaler.scale(l).backward()
            scaler.step(optimizer)
            scaler.update()

            loss.update(l.data)

            info = 'Epoch {}, {}'.format(epoch, loss)
            if verbose:
                tr.set_description(info)
                tr.update(1)

        if rank == 0:
            logger.debug(info)

        t_loss = torch.cat(
            (t_loss, torch.cuda.FloatTensor([loss.avg], device=rank))
        )

        if verbose:
            tr.close()

        net.eval()
        acc.reset()
        correlation.reset()
        loss.reset()

        if verbose:
            tr = trange(len(val_loader), file=sys.stdout)

        with torch.no_grad():

            for images, targets, tracks, spher in val_loader:
                outputs = net(images)
                preds = torch.argmax(outputs, dim=1)
                targets = torch.cuda.LongTensor(targets, device=rank)

                l = criterion(outputs, targets)

                if disco_mode:

                    if disco_mode == 1:
                        value = torch.tensor(tracks).to(rank) + 0.
                    elif disco_mode == 2:
                        value = torch.tensor(spher).to(rank) + 0.

                    pos = targets == 0
                    corr = distance_corr(
                        F.softmax(outputs, dim=1)[:, 1][pos],
                        value[pos],
                        1
                    )
                    if torch.isnan(corr):
                        corr = 0
                    l = l + corr
                    correlation.update(corr.data)

                l = reduce_tensor(l.data)
                loss.update(l.data)

                a = (targets == preds).sum() / batch_size_validation
                acc.update(a.data)

                info = 'Validation, {}, {}, {}'.format(loss, acc, correlation)
                if verbose:
                    tr.set_description(info)
                    tr.update(1)

            if rank == 0:
                logger.debug(info)
            v_acc = torch.cat(
                (v_acc, torch.cuda.FloatTensor([acc.avg], device=rank))
            )
            v = torch.cuda.FloatTensor([loss.avg], device=rank)
            v_loss = torch.cat((v_loss, v))

            if verbose:
                tr.close()

            plot.draw_loss(t_loss.cpu().numpy(),
                           v_loss.cpu().numpy(),
                           v_acc.cpu().numpy(),
                           name)

        flag_tensor = torch.tensor(0)
        if rank == 0 and cp_es(v.sum(0), model):
            flag_tensor += 1
        dist.all_reduce(flag_tensor, op=torch.distributed.ReduceOp.SUM)
        if flag_tensor == 1:
            break

        dist.barrier()
        scheduler.step()

    cleanup()


def reduce_tensor(loss):
    loss = loss.clone()
    dist.all_reduce(loss)
    loss /= int(os.environ['WORLD_SIZE'])
    return loss


def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '11223'
    os.environ['WORLD_SIZE'] = str(world_size)
    os.environ['RANK'] = str(rank)
    dist.init_process_group("gloo", rank=rank, world_size=world_size)


def cleanup():
    dist.destroy_process_group()


if __name__ == '__main__':

    parser = argparse.ArgumentParser('Train SUEP Classifier')
    parser.add_argument('name', type=str, help='Model name')
    parser.add_argument('disco_mode', nargs='?', type=int, help='Disco mode', default=0)
    parser.add_argument('-c', '--config',
                        action=IsValidFile,
                        type=str,
                        help='Path to config file',
                        default='config.yml')
    parser.add_argument('-v', '--verbose',
                        action='store_true',
                        help='Output verbosity')
    args = parser.parse_args()
    config = yaml.safe_load(open(args.config))

    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    world_size = torch.cuda.device_count()
    mp.spawn(execute,
             args=(world_size,
                   args.name,
                   config['architecture'],
                   config['dataset'],
                   config['training_pref'],
                   args.disco_mode,
                   args.verbose),
             nprocs=world_size,
             join=True)
