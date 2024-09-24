import torch
from torch_geometric.data import InMemoryDataset
from torch.utils.data import Subset
from sklearn.model_selection import KFold, train_test_split
import numpy as np

class GNNDataset(InMemoryDataset):

    def __init__(self, root, train=True, transform=None, pre_transform=None, pre_filter=None):
        super().__init__(root, transform, pre_transform, pre_filter)
        if train:
            self.data, self.slices = torch.load(self.processed_paths[0])
        else:
            self.data, self.slices = torch.load(self.processed_paths[1])

    @property
    def raw_file_names(self):
        return ['data_train.csv', 'data_test.csv']

    @property
    def processed_file_names(self):
        return ['processed_data_train.pt', 'processed_data_test.pt']

    def download(self):
        # Download to `self.raw_dir`.
        pass

    def _download(self):
        pass

    def process(self):
        pass

    def split_train_val_folds(self, n_splits=3, random_state=1212):
        train_indices = list(range(len(self)))
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)

        train_val_folds = []
        for train_indices_fold, val_indices_fold in kf.split(train_indices):
            train_fold = Subset(self, train_indices_fold)
            val_fold = Subset(self, val_indices_fold)
            train_val_folds.append((train_fold, val_fold))

        return train_val_folds

    # def split_train_val_fold_with_rate(self, train_set_rate = 0.8, random_state=1212):
    #     # 获取所有索引
    #     indices = np.arange(len(self))
    #     # 划分训练集和验证集的索引
    #     train_indices, val_indices = train_test_split(indices, train_size=train_set_rate, random_state=random_state, shuffle=True)
    #
    #     # 创建训练集和验证集的子集
    #     train_fold = Subset(self, train_indices)
    #     val_fold = Subset(self, val_indices)
    #
    #     return [(train_fold, val_fold)]

if __name__ == "__main__":
    # dataset = GNNDataset('data.csv/davis')
    dataset = GNNDataset('data.csv/Kd')



