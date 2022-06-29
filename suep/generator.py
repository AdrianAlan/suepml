import h5py
import torch


class CalorimeterDataset(torch.utils.data.Dataset):

    def __init__(self,
                 rank,
                 hdf5_source_path,
                 in_dim,
                 boosted=False,
                 flip_prob=None):
        """Generator for calorimeter and data"""
        self.rank = rank
        self.source = hdf5_source_path
        self.in_dim = in_dim
        self.boosted = boosted
        self.flip_prob = flip_prob

    def __getitem__(self, index):

        if not hasattr(self, 'hdf5_dataset'):
            self.open_hdf5()

        labels = self.labels[index]
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

        if self.flip_prob:
            if torch.rand(1) < self.flip_prob:
                calorimeter = self.flip_image(calorimeter, vertical=True)
            if torch.rand(1) < self.flip_prob:
                calorimeter = self.flip_image(calorimeter, vertical=False)

        return calorimeter, labels

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

    def process_images(self, etas, phis, pts):
        c = torch.zeros(etas.size(1), dtype=torch.long).cuda(self.rank)
        c = c.unsqueeze(0)
        i = torch.cat((c, etas, phis), 0)
        v = pts
        pixels = torch.sparse.FloatTensor(i, v, torch.Size(self.in_dim))
        pixels = pixels.to_dense()
        pixels = self.normalize(pixels)
        return pixels

    def normalize(self, tensor):
        if tensor.sum():
            m = torch.mean(tensor)
            s = torch.std(tensor)
            return tensor.sub(m).div(s)
        return tensor

    def flip_image(self, image, vertical=True):
        if vertical:
            axis = 1
        else:
            axis = 2
        image = torch.flip(image, [axis])
        return image
