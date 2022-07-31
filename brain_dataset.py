from BrainGB.dataset.brain_dataset import BrainDataset
import torch
import numpy as np
import os.path as osp

from scipy.io import loadmat
from torch_geometric.data import InMemoryDataset, Data
from BrainGB.dataset.base_transform import BaseTransform
import sys
from torch_geometric.utils import dense_to_sparse
from torch_geometric.data.makedirs import makedirs
from BrainGB.dataset.abcd.load_abcd import load_data_abcd, load_data_abide, load_data_pnc
from torch_geometric.data.dataset import files_exist
import logging
from typing import List, Any, Sequence


def to_list(value: Any) -> Sequence:
    if isinstance(value, Sequence) and not isinstance(value, str):
        return value
    else:
        return [value]


class MyBrainDataset(torch.utils.data.Dataset):
    def __init__(self, root, name, pre_transform: BaseTransform = None, view=0):
        super(MyBrainDataset, self).__init__()
        self.view: int = view
        self.name = name.upper()
        self.raw_dir = root
        self.processed_file_name = f'{name}_data.pt'
        self.processed_dir = osp.join(root, 'processed')
        self.processed_path = osp.join(self.processed_dir, self.processed_file_name)
        self.raw_file_names = f'{self.name}.mat'
        assert self.name in ['PPMI', 'HIV', 'BP', 'ABCD', 'PNC', 'ABIDE']
        self._process()
        self.data = torch.load(self.processed_path)
        logging.info('Loaded dataset: {}'.format(self.name))

    def __getitem__(self, idx):
        return self.data[idx]

    def _download(self):
        if files_exist(self.raw_paths) or self.name in ['ABCD', 'PNC', 'ABIDE']:  # pragma: no cover
            return

        makedirs(self.raw_dir)
        self.download()

    def download(self):
        raise NotImplementedError

    @property
    def processed_paths(self) -> List[str]:
        r"""The absolute filepaths that must be present in order to skip
        processing."""
        files = to_list(self.processed_file_names)
        return [osp.join(self.processed_dir, f) for f in files]

    def process(self):
        if self.name in ['ABCD', 'PNC', 'ABIDE']:
            if self.name == 'ABCD':
                adj, y = load_data_abcd(self.raw_dir)
            elif self.name == 'PNC':
                adj, y = load_data_pnc(self.raw_dir)
            elif self.name == 'ABIDE':
                adj, y = load_data_abide(self.raw_dir)
            else:
                raise NotImplementedError
            y = torch.LongTensor(y)
            adj = torch.Tensor(adj)
            num_graphs = adj.shape[0]
            num_nodes = adj.shape[1]
        else:
            m = loadmat(osp.join(self.raw_dir, self.raw_file_names))
            if self.name == 'PPMI':
                if self.view > 2 or self.view < 0:
                    raise ValueError(f'{self.name} only has 3 views')
                raw_data = m['X']
                num_graphs = raw_data.shape[0]
                num_nodes = raw_data[0][0].shape[0]
                a = np.zeros((num_graphs, num_nodes, num_nodes))
                for i, sample in enumerate(raw_data):
                    a[i, :, :] = sample[0][:, :, self.view]
                adj = torch.Tensor(a)
            else:
                key = 'fmri' if self.view == 1 else 'dti'
                adj = torch.Tensor(m[key]).transpose(0, 2)
                num_graphs = adj.shape[0]
                num_nodes = adj.shape[1]

            y = torch.Tensor(m['label']).long().flatten()
            y[y == -1] = 0

        data_list = []
        for i in range(num_graphs):
            edge_index, edge_attr = dense_to_sparse(adj[i])
            # data = Data(num_nodes=num_nodes, y=y[i], edge_index=edge_index, edge_attr=edge_attr)
            # X, y, mask, edges
            data = (adj[i], y[i], edge_index)
            if edge_index.shape[1] != 39800:
                print("Dropped")
                continue
            data_list.append(data)

        # if self.pre_transform is not None:
        #     data_list = [self.pre_transform(data) for data in data_list]

        # data, slices = self.collate(data_list)
        torch.save(data_list, self.processed_path)

    def _process(self):
        print('Processing...', file=sys.stderr)

        if files_exist([self.processed_file_name]):  # pragma: no cover
            print('Done!', file=sys.stderr)
            return

        makedirs(self.processed_dir)
        self.process()

        print('Done!', file=sys.stderr)

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}{self.name}()'