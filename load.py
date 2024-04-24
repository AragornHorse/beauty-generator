from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
import os
from PIL import Image
import numpy as np
import torch
import glob


def default_loader(path):
    return Image.open(path).convert("RGB")


train_transform = transforms.Compose([
    # transforms.RandomHorizontalFlip(),
    transforms.Resize(160),
    transforms.RandomCrop(size=128),
    transforms.RandomRotation(1, interpolation=transforms.InterpolationMode.NEAREST),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ColorJitter(0.03, 0.03, 0.03, (-0.01, 0.01)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

test_transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor()
])


class mydataset(Dataset):
    def __init__(self, im_list, transform=None, loader=default_loader):
        super(mydataset, self).__init__()
        imgs = []
        for im_item in im_list:
            imgs.append(im_item)
        self.imgs = imgs
        self.transform = transform
        self.loader = loader

    def __getitem__(self, index):
        im_path = self.imgs[index]

        im_data = self.loader(im_path)

        if self.transform is not None:
            im_data = self.transform(im_data)

        return im_data

    def __len__(self):
        return len(self.imgs)


im_train_list = glob.glob(
    r"C:\Users\DELL\Desktop\datasets\mask\img\*")

train_dataset = mydataset(im_train_list, transform=train_transform)


def get_loader(batch_size):
    train_data_loader = DataLoader(dataset=train_dataset, batch_size=batch_size,
                                   shuffle=True)
    return train_data_loader

# print(len(test_dataset))

if __name__ == '__main__':
    import matplotlib.pyplot as plt

    for data in get_loader(100):
        x=data
        print(x.size())
        print(torch.max(x), torch.min(x))
        plt.imshow((x[0].detach().numpy().transpose([1, 2, 0]) + 1) / 2)
        plt.show()

