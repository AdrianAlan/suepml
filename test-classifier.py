import argparse
import logging
import numpy as np
import os
import sys
import torch
import torch.backends.cudnn as cudnn
import torch.cuda as tcuda
import torch.nn as nn
import torchvision.models as models
import tqdm
import yaml

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


def execute(name, architecture, dataset, evaluation_pref, verbose):

    test_loader = get_data_loader(
        dataset['test'][0],
        evaluation_pref['batch_size'],
        evaluation_pref['workers'],
        dataset['in_dim'],
        0,
        boosted=dataset['boosted'],
        shuffle=False
    )

    model = eval(architecture)()
    model.load_state_dict(torch.load("models/{}.pth".format(name)))
    cudnn.benchmark = True
    test_results = torch.tensor([])
    if verbose:
        tr = trange(len(test_loader), file=sys.stdout)
        tr.set_description('Testing')

    model.eval()
    with torch.no_grad():
        for images, targets in test_loader:
            count = len(targets)
            targets = tcuda.LongTensor(targets, device=0)
            outputs = model(images)
            preds = torch.argmax(outputs, dim=1)
            batch_results = torch.cat((targets.reshape(-1, 1), outputs), 1)
            test_results = torch.cat((test_results, batch_results), 0)
            if verbose:
                tr.update(1)
        if verbose:
            tr.close()
    np.save(
        "models/{}-results.npy".format(name),
        test_results.detach().cpu().numpy()
    )


if __name__ == '__main__':

    torch.set_default_tensor_type('torch.cuda.FloatTensor')

    parser = argparse.ArgumentParser('Test SUEP Classifier')
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

    execute(
        args.name,
        config['architecture'],
        config['dataset'],
        config['evaluation_pref'],
        args.verbose
    )
