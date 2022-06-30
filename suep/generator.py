import h5py
import torch


class CalorimeterDataset(torch.utils.data.Dataset):

    def __init__(self, rank, hdf5_source_path, in_dim, boosted=False):
        """Generator for calorimeter and data"""
        self.rank = rank
        self.source = hdf5_source_path
        self.in_dim = in_dim
        self.boosted = boosted

    def __getitem__(self, index):

        x_phi = torch.cuda.LongTensor(
            self.phi[index],
            device=self.rank
        ).unsqueeze(0)
        x_eta = torch.cuda.LongTensor(
            self.eta[index],
            device=self.rank
        ).unsqueeze(0)
        x_pt = torch.cuda.FloatTensor(
            self.pt[index],
            device=self.rank
        )
        calorimeter = self.process_images(x_eta, x_phi, x_pt)

        return calorimeter, self.labels[index]

    def __len__(self):

        if not hasattr(self, 'hdf5_dataset'):
            self.open_hdf5()

        return self.dataset_size

    def open_hdf5(self):
        self.hdf5_dataset = h5py.File(self.source, 'r')

        self.labels = self.hdf5_dataset['label']
        if self.boosted:
            self.eta = self.hdf5_dataset['beta']
            self.phi = self.hdf5_dataset['bphi']
            self.pt = self.hdf5_dataset['bpt']
        else:
            self.eta = self.hdf5_dataset['eta']
            self.phi = self.hdf5_dataset['phi']
            self.pt = self.hdf5_dataset['pt']

        self.dataset_size = len(self.labels)

    def process_images(self, w, h, v):
        i = torch.cat((w, h), 0)
        pixels = torch.cuda.sparse.FloatTensor(
            i, v, torch.Size(self.in_dim), device=self.rank)
        pixels = pixels.to_dense()
        pixels = self.normalize(pixels)
        return pixels.unsqueeze(0)

    def normalize(self, tensor):
        m = torch.mean(tensor)
        s = torch.std(tensor)
        if s:
            return tensor.sub(m).div(s)
        return tensor
