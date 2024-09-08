from torch.utils.data.dataset import Dataset


class UnifiedDataset(Dataset):
    def __init__(self, x, y, t, dataset, trsf=None):
        self.trsf = trsf
        self.dataset = dataset
        assert len(x) == len(y)
        assert len(x) == len(t)
        self._x, self._y, self._t = x, y, t

    def __getitem__(self, index):
        x, y, t = self._x[index], self._y[index], self._t[index]
        if self.trsf is not None:
            x = self.trsf(x)
        if self.dataset == 'PMNIST':
            x = x.flatten()
        return x, y, t

    def __len__(self):
        return len(self._x)
