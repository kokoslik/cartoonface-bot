import os
from torch.utils.data import Dataset
from PIL import Image


class SingleFolderDataset(Dataset):
    def __init__(self, path, transform=None, cache=False, ext='jpg'):
        super().__init__()
        self.transform = transform
        self.image_list = [os.path.join(path, file) for file in os.listdir(path) if ext in file]
        self.len = len(self.image_list)
        self.cache = cache
        if self.cache:
            self.pics = [transform(Image.open(file).convert('RGB')) for file in self.image_list]
    def __len__(self):
        return self.len
    def __getitem__(self,i):
        if self.cache:
            return self.pics[i % self.len]
        image = Image.open(self.image_list[i % self.len]).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)
        return image
