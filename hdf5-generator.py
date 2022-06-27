import argparse
import awkward as ak
import h5py
import numpy as np
import itertools
import os
import uproot
import yaml

from utils import IsReadableDir
from tqdm import tqdm
from typing import Dict, Iterable, List, Optional, Tuple


class PhysicsConstants():

    def __init__(self):

        self.ht_threshold = 500.

        self.eta_span = (-2.5, 2.5)
        self.eta_steps = 281
        self.phi_span = (-np.pi, np.pi)
        self.phi_steps = 361
        self.set_edges()

    def set_edges(self):

        self.edges_eta = np.linspace(self.eta_span[0],
                                     self.eta_span[1],
                                     self.eta_steps)
        self.edges_phi = np.linspace(self.phi_span[0],
                                     self.phi_span[1],
                                     self.phi_steps)

    def get_edges(self) -> Tuple[List[int], List[int]]:
        return self.edges_eta, self.edges_phi


class HDF5Generator:

    def __init__(self,
                 hdf5_dataset_path: str,
                 hdf5_dataset_size: int,
                 verbose: bool = True):

        self.constants = PhysicsConstants()
        self.edges_eta, self.edges_phi = self.constants.get_edges()
        self.hdf5_dataset_path = hdf5_dataset_path
        self.hdf5_dataset_size = hdf5_dataset_size
        self.verbose = verbose

    def create_hdf5_dataset(self, suep: Iterable, qcd: Iterable):

        if self.verbose:
            progress = tqdm(total=self.hdf5_dataset_size,
                            desc=('Processing {}'.format(
                                self.hdf5_dataset_path)))

        # Create the HDF5 file.
        with h5py.File(self.hdf5_dataset_path, 'w') as hdf5_dataset:

            hdf5_eta = hdf5_dataset.create_dataset(
                name='eta',
                shape=(self.hdf5_dataset_size,),
                maxshape=(None),
                dtype=h5py.special_dtype(vlen=np.int16))

            hdf5_phi = hdf5_dataset.create_dataset(
                name='phi',
                shape=(self.hdf5_dataset_size,),
                maxshape=(None),
                dtype=h5py.special_dtype(vlen=np.int16))

            hdf5_pt = hdf5_dataset.create_dataset(
                name='pt',
                shape=(self.hdf5_dataset_size,),
                maxshape=(None),
                dtype=h5py.special_dtype(vlen=np.float32))

            hdf5_beta = hdf5_dataset.create_dataset(
                name='beta',
                shape=(self.hdf5_dataset_size,),
                maxshape=(None),
                dtype=h5py.special_dtype(vlen=np.int16))

            hdf5_bphi = hdf5_dataset.create_dataset(
                name='bphi',
                shape=(self.hdf5_dataset_size,),
                maxshape=(None),
                dtype=h5py.special_dtype(vlen=np.int16))

            hdf5_bpt = hdf5_dataset.create_dataset(
                name='bpt',
                shape=(self.hdf5_dataset_size,),
                maxshape=(None),
                dtype=h5py.special_dtype(vlen=np.float32))

            get_suep = itertools.cycle([True, False])
            labels, es, bes, ss, bss = [], [], [], [], []
            for i in range(self.hdf5_dataset_size):

                if next(get_suep):
                    event_details, label = next(suep), 1
                else:
                    event_details, label = next(qcd), 0

                pt = event_details.get('pt')
                eta = event_details.get('eta')
                phi = event_details.get('phi')
                q = event_details.get('q')
                vertex = event_details.get('vertex')
                mask = (q != 0) & (vertex == 0) & (abs(eta) < 2.4) & (pt > 0.5)
                pt, eta, phi = pt[mask], eta[mask], phi[mask]
                px_eta, px_phi, values = self.get_energy_map(eta, phi, pt)

                hdf5_eta[i] = px_eta
                hdf5_phi[i] = px_phi
                hdf5_pt[i] = values

                bpt = event_details.get('bpt')
                beta = event_details.get('beta')
                bphi = event_details.get('bphi')

                if len(pt) != len(bpt):
                    print('Ass!')

                px_beta, px_bphi, bvalues = self.get_energy_map(beta, bphi, bpt)

                hdf5_beta[i] = px_beta
                hdf5_bphi[i] = px_bphi
                hdf5_bpt[i] = bvalues

                labels.append(label)
                es.append(event_details.get('event_sphericity'))
                bes.append(event_details.get('event_bsphericity'))
                ss.append(event_details.get('suep_sphericity'))
                bss.append(event_details.get('suep_bsphericity'))

                if self.verbose:
                    progress.update(1)

            hdf5_dataset.create_dataset('label', data=labels, dtype='i1')
            hdf5_dataset.create_dataset('event_sphericity', data=es, dtype='f')
            hdf5_dataset.create_dataset('event_bsphericity', data=bes, dtype='f')
            hdf5_dataset.create_dataset('suep_sphericity', data=ss, dtype='f')
            hdf5_dataset.create_dataset('suep_bsphericity', data=bss, dtype='f')

        if self.verbose:
            progress.close()

    def get_energy_map(self,
                       etas: np.ndarray,
                       phis: np.ndarray,
                       values: np.ndarray) -> Tuple[np.ndarray,
                                                    np.ndarray,
                                                    np.ndarray]:
        """Translate eta/phi to pixel coordinates"""
        img, _, _ = np.histogram2d(etas,
                                   phis,
                                   bins=[self.edges_eta, self.edges_phi],
                                   weights=values)
        bins = np.argwhere(img)
        indices_eta = bins[:, 0]
        indices_phi = bins[:, 1]
        values = img[indices_eta, indices_phi]
        return indices_eta, indices_phi, values


class EventGenerator():

    def __init__(self, path: str):
        self.constants = PhysicsConstants()
        self.path = path
        self.root_files = self.get_files_from_dir(path)

    def __iter__(self):

        for root_file in self.root_files:

            rf = uproot.open('{}{}'.format(self.path, root_file))
            if not len(rf.keys()):
                continue
            tree = rf['mmtree/tree']

            hts = tree['ht'].array()
            n_fatjet = tree['n_fatjet'].array()

            event_sphericity = tree['event_sphericity'].array()
            event_bsphericity = tree['eventBoosted_sphericity'].array()
            suep_sphericity = tree['suepJet_sphericity'].array()
            suep_bsphericity = tree['suepJetBoosted_sphericity'].array()

            pts = tree['PFcand_pt'].array()
            phis = tree['PFcand_phi'].array()
            etas = tree['PFcand_eta'].array()
            qs = tree['PFcand_q'].array()
            vertices = tree['PFcand_vertex'].array()

            bpts = tree['bPFcand_pt'].array()
            bphis = tree['bPFcand_phi'].array()
            betas = tree['bPFcand_eta'].array()

            for i, ht in enumerate(hts):

                if ht < self.constants.ht_threshold or n_fatjet[i] < 2:
                    continue

                yield {'pt': np.array(pts[i]),
                       'eta': np.array(etas[i]),
                       'phi': np.array(phis[i]),
                       'q': np.array(qs[i]),
                       'vertex': np.array(vertices[i]),
                       'bpt': np.array(bpts[i]),
                       'beta': np.array(betas[i]),
                       'bphi': np.array(bphis[i]),
                       'event_sphericity': event_sphericity[i],
                       'event_bsphericity': event_bsphericity[i],
                       'suep_sphericity': suep_sphericity[i],
                       'suep_bsphericity': suep_bsphericity[i]}

    def get_files_from_dir(self, path: str) -> List[str]:
        return [i for i in os.listdir(path) if i.endswith(".root")]


def parse_config_for_sizes_and_names(config: str) -> List[int]:
    c = yaml.safe_load(open(config))
    return c['dataset']['size'], c['dataset']['names']


def main(path_suep: str,
         path_qcd: str,
         config: str,
         verbose: bool = True):

    dataset_sizes, dataset_paths = parse_config_for_sizes_and_names(config)

    eg_suep = iter(EventGenerator(path_suep))
    eg_qcd = iter(EventGenerator(path_qcd))

    for size, path in zip(dataset_sizes, dataset_paths):

        generator = HDF5Generator(
            hdf5_dataset_path=path,
            hdf5_dataset_size=size,
            verbose=verbose)

        generator.create_hdf5_dataset(eg_suep, eg_qcd)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        'Process SUEP and QCD ROOT files and store events and labels in to H5')

    parser.add_argument('source_dir_suep',
                        action=IsReadableDir,
                        help='SUEP files source folder',
                        type=str)

    parser.add_argument('source_dir_qcd',
                        action=IsReadableDir,
                        help='QCD files source folder',
                        type=str)

    parser.add_argument('-c', '--config',
                        default='./hdf5-config.yml',
                        dest='config',
                        help='Configuration file path',
                        type=str)

    parser.add_argument('-v', '--verbose',
                        action="store_true",
                        help='Speak')

    args = parser.parse_args()

    main(args.source_dir_suep,
         args.source_dir_qcd,
         args.config,
         args.verbose)
