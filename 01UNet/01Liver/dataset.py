# -*- coding: utf-8 -*-
# @Time    : 2024/2/5 18:45
# @Author  : 楚楚
# @File    : dataset.py
# @Software: PyCharm

from torch.utils.data import Dataset
import PIL.Image as Image
import os


def make_dataset(root):
    images = list()
    n = len(os.listdir(root)) // 2

    for i in range(n):
        img = os.path.join(root, f"00{i}.png")
        mask = os.path.join(root, f"00{i}_mask.png")

        images.append((img, mask))

    return images

class LiverDataset(Dataset):
    def __init__(self,root,transform=None,target_transform=None):
        self.images=make_dataset(root)
        self.transform=transform
        self.target_transform=target_transform

    def __getitem__(self, index):
        x_path,y_path=self.images[index]
        image_x=Image.open(x_path)
        image_y=Image.open(y_path)

        if self.transform is not None:
            image_x=self.transform(image_x)
        if self.target_transform is not None:
            image_y=self.target_transform(image_y)

        return image_x,image_y

    def __len__(self):
        return len(self.images)