import argparse
import logging
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import torch
import torch.backends.cudnn as cudnn
import torch.cuda as tcuda
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torchvision.models as models
import tqdm
import yaml

from suep.checkpoints import EarlyStopping
from suep.generator import CalorimeterDataset
from suepvision import smodels
from torch.cuda.amp import GradScaler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import Adam, lr_scheduler
from torch.utils.data import DataLoader
from tqdm import trange
from utils import IsValidFile


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


def get_data_loader(hdf5_source_path,
                    batch_size,
                    num_workers,
                    in_dim,
                    rank=0,
                    boosted=False,
                    flip_prob=None,
                    shuffle=True):

    dataset = CalorimeterDataset(
        torch.device(rank),
        hdf5_source_path,
        in_dim,
        boosted=boosted,
        flip_prob=flip_prob
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=collate_fn,
        num_workers=num_workers,
        shuffle=shuffle
    )


def collate_fn(batch):
    transposed_data = list(zip(*batch))
    inp = torch.stack(transposed_data[0], 0)
    tgt = list(transposed_data[1])
    return inp, tgt


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


def execute(rank, world_size, name, dataset, training_pref, verbose):

    setup(rank, world_size)

    if rank == 0:
        logname = "models/{}.log".format(name)
        logger = set_logging("Train {}".format(name), logname, verbose)

    plot = Plotting("models")

    # Initialize dataset
    train_loader = get_data_loader(dataset['train'][rank],
                                   training_pref['batch_size_train'],
                                   training_pref['workers'],
                                   dataset['in_dim'],
                                   rank,
                                   boosted=training_pref['boosted'],
                                   flip_prob=0.5,
                                   shuffle=True)

    val_loader = get_data_loader(dataset['validation'][rank],
                                 training_pref['batch_size_validation'],
                                 training_pref['workers'],
                                 dataset['in_dim'],
                                 rank,
                                 boosted=training_pref['boosted'],
                                 shuffle=False)

    # Build SSD network
    model = smodels.LeNet5()
    model = nn.SyncBatchNorm.convert_sync_batchnorm(model).to(rank)
    if rank == 0:
        logger.debug('Model architecture:\n{}'.format(str(model)))

    # Data parallelization
    cudnn.benchmark = True
    net = DDP(model, device_ids=[rank])

    # Set training objective parameters
    optimizer = Adam(
        net.parameters(),
        lr=training_pref['learning_rate'],
        weight_decay=training_pref['weight_decay']
    )
    scheduler = lr_scheduler.MultiStepLR(
        optimizer,
        milestones=np.arange(20, 100, 20),
        gamma=0.5
    )

    if rank == 0:
        cp_es = EarlyStopping(
            logger,
            patience=training_pref['patience'],
            save_path='models/{}.pth'.format(name)
        )
    criterion = nn.CrossEntropyLoss().to(rank)
    scaler = GradScaler()
    verobse = verbose and rank == 0
    train_loss, val_loss = torch.tensor([]), torch.tensor([])
    v_acc = torch.tensor([])
    acc, loss = AverageMeter('Accuracy'), AverageMeter('Loss')

    for epoch in range(1, training_pref['max_epochs']+1):

        net.train()
        acc.reset()
        loss.reset()

        if verbose:
            tr = trange(len(train_loader), file=sys.stdout)

        for images, targets in train_loader:
            count = len(targets)
            targets = tcuda.LongTensor(targets, device=rank)
            outputs = net(images)
            preds = torch.argmax(outputs, dim=1)
            a = (targets == preds).sum() / count
            l = criterion(outputs, targets)

            acc.update(a)
            loss.update(l)

            scaler.scale(l).backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

            info = 'Epoch {}, {}, {}'.format(epoch, loss, acc)
            if verbose:
                tr.set_description(info)
                tr.update(1)

        if rank == 0:
            logger.debug(info)

        train_loss = torch.cat((train_loss, torch.tensor([loss.avg])))

        if verbose:
            tr.close()

        net.eval()
        acc.reset()
        loss.reset()

        # Start model validation
        if verbose:
            tr = trange(len(val_loader), file=sys.stdout)

        with torch.no_grad():

            for images, targets in val_loader:
                count = len(targets)
                targets = tcuda.LongTensor(targets, device=rank)
                outputs = net(images)
                preds = torch.argmax(outputs, dim=1)

                l = criterion(outputs, targets)
                l = reduce_tensor(l.data)
                a = (targets == preds).sum() / count
                a = reduce_tensor(a.data)

                acc.update(a)
                loss.update(l)

                info = 'Validation, {}, {}'.format(loss, acc)
                if verbose:
                    tr.set_description(info)
                    tr.update(1)

            if rank == 0:
                logger.debug(info)
            v_acc = torch.cat((v_acc, torch.tensor([acc.avg])))
            vloss = torch.tensor([loss.avg])
            val_loss = torch.cat((val_loss, vloss))

            if verbose:
                tr.close()

            plot.draw_loss(train_loss.cpu().numpy(),
                           val_loss.cpu().numpy(),
                           v_acc.cpu().numpy(),
                           name)

            if rank == 0 and cp_es(vloss.sum(0), model):
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
                   config['dataset'],
                   config['training_pref'],
                   args.verbose),
             nprocs=world_size,
             join=True)
