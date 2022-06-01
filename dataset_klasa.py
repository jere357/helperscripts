import random
import torch
from torch.utils.data import Dataset
from collections import defaultdict
from random import choice
import torchvision
from torchvision.utils import make_grid

import torch
import numpy as np
import matplotlib.pyplot as plt

import torchvision.transforms.functional as F


def show(imgs):
    if not isinstance(imgs, list):
        imgs = [imgs]
    fix, axs = plt.subplots(ncols=len(imgs), squeeze=False)
    for i, img in enumerate(imgs):
        img = img.detach()
        img = F.to_pil_image(img)
        axs[0, i].imshow(np.asarray(img))
        axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
    plt.show()


class MNISTMetricDataset(Dataset):
    def __init__(self, root="/tmp/mnist/", split='train'):
        super().__init__()
        assert split in ['train', 'test', 'traineval']
        self.root = root
        self.split = split
        mnist_ds = torchvision.datasets.MNIST(self.root, train='train' in split, download=True)
        self.images, self.targets = mnist_ds.data.float() / 255., mnist_ds.targets
        self.classes = list(range(10))
        self.target2indices = defaultdict(list)
        for i in range(len(self.images)):
            self.target2indices[self.targets[i].item()] += [i]

    def _sample_negative(self, index):
        target_id = self.targets[index].item()
        while True:
            negative_id = random.randint(0, 9)
            if negative_id != target_id:
                break
        return choice(self.target2indices[negative_id])

    def _sample_positive(self, index):
        target_id = self.targets[index].item()
        return choice(self.target2indices[target_id])

    def __getitem__(self, index):
        anchor = self.images[index].unsqueeze(0)
        target_id = self.targets[index].item()
        if self.split in ['traineval', 'val', 'test']:
            return anchor, target_id
        else:
            positive_index = self._sample_positive(index)
            negative_index = self._sample_negative(index)
            positive = self.images[positive_index]
            negative = self.images[negative_index]
            return anchor, positive.unsqueeze(0), negative.unsqueeze(
                0), target_id, index, positive_index, negative_index

    def __len__(self):
        return len(self.images)


if __name__ == "__main__":
    dataset = MNISTMetricDataset()
    print("xd")
    for i in range(10):
        anchor, positive, negative, target_id, id, pid, nid = dataset[i+33]
        grid = make_grid([anchor, positive, negative])
        print(dataset.targets[id].item(), dataset.targets[pid].item(), dataset.targets[nid].item())
        show(grid)
