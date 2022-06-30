import argparse
import os
import torch

from suep.generator import CalorimeterDataset
from torch.utils.data import DataLoader

class IsReadableDir(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        prospective_dir = values
        if not os.path.isdir(prospective_dir):
            raise argparse.ArgumentTypeError(
                    '{0} is not a valid path'.format(prospective_dir))
        if os.access(prospective_dir, os.R_OK):
            setattr(namespace, self.dest, prospective_dir)
        else:
            raise argparse.ArgumentTypeError(
                    '{0} is not a readable directory'.format(prospective_dir))


class IsValidFile(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        prospective_file = values
        if not os.path.exists(prospective_file):
            raise argparse.ArgumentTypeError(
                    '{0} is not a valid file'.format(prospective_file))
        else:
            setattr(namespace, self.dest, prospective_file)

def collate_fn(batch):
    transposed_data = list(zip(*batch))
    return torch.stack(transposed_data[0], 0), transposed_data[1]


def get_data_loader(hdf5_source_path,
                    batch_size,
                    num_workers,
                    in_dim,
                    rank=0,
                    boosted=False,
                    shuffle=True):

    dataset = CalorimeterDataset(
        torch.device(rank),
        hdf5_source_path,
        in_dim,
        boosted=boosted)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=collate_fn,
        num_workers=num_workers,
        shuffle=shuffle
    )
