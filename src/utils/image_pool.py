import torch
import numpy as np


class ImagePool:
    def __init__(self, size, pic_size, device, logic='old'):
        self.size = size
        self.tensor = torch.zeros(size, 3, pic_size, pic_size, device=device)
        self.logic = logic

    def query(self, batch):
        batch = batch.detach().clone()
        n = batch.size(0)
        if self.logic == 'old':
            self.tensor[n:] = self.tensor[:self.size-n].clone()
            self.tensor[:n] = batch
            idx = np.random.choice(self.size, n, replace=False)
            return self.tensor[idx]
        elif self.logic == 'new':
            random_idx = np.random.choice(n)
            self.tensor[1:] = self.tensor[:-1].clone()
            self.tensor[0] = batch[random_idx]
            idx_rhs = np.random.choice(self.size, n//2, replace=False)
            idx_lhs = np.random.choice(n, n//2, replace=False)
            batch[idx_lhs] = self.tensor[idx_rhs]
            return batch
