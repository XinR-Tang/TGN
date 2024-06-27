from io import BytesIO

from PIL import Image
from torch.utils.data import Dataset

import cv2
import numpy as np
import torch
from skimage import io
import torchvision.datasets as datasets
from skimage.color import rgb2gray
from torch.utils.data import Dataset, DataLoader
import scipy.ndimage
from scipy import special
import os


class SatelliteDataset(Dataset):

    def __init__(self, rootdir, clip=True, seed=0, **kwargs):
        from os import listdir
        from os.path import isfile, join

        self.rootdir = rootdir
        self.img_list = [f for f in listdir(self.rootdir)]
        # self.length = len(self.img_list)
        assert (seed + 1) * len(self) - 1 <= 2 ** 32 - 1

    def __len__(self):
        return len(self.img_list)
        # return self.length

    def __getitem__(self, index):
        img_name = os.path.join(self.rootdir, self.img_list[index])
        image = io.imread(img_name)
        if image.shape[0] + image.shape[1] > 512 or image.shape[0] + image.shape[1] < 512:
            print(image.shape)
            print(img_name)
        image = image / 255.
        image *= 2.0
        image -= 1.0
        # print("inside get_item:")
        # print(np.min(image), np.max(image))
        image = image.transpose(2, 0, 1)
        return image.astype('float32')


class MultiResolutionDataset(Dataset):
    def __init__(self, path, transform, resolution=256):
        self.env = lmdb.open(
            path,
            max_readers=32,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
        )

        if not self.env:
            raise IOError('Cannot open lmdb dataset', path)

        with self.env.begin(write=False) as txn:
            self.length = int(txn.get('length'.encode('utf-8')).decode('utf-8'))

        self.resolution = resolution
        self.transform = transform

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        with self.env.begin(write=False) as txn:
            key = f'{self.resolution}-{str(index).zfill(5)}'.encode('utf-8')
            img_bytes = txn.get(key)

        buffer = BytesIO(img_bytes)
        img = Image.open(buffer)
        img = self.transform(img)

        return img


class CSatelliteDataset(Dataset):
    def __init__(self, rootdir, train=True, download=False, transform=None):
        self.rootdir = rootdir
        self.train = train
        self.transform = transform

        # 选择使用训练集还是测试集
        self.data_folder = "train" if self.train else "test"

        # 读取所有图像路径和标签
        self.img_labels = []
        for category in ['tail']:
            txt_path = os.path.join(self.rootdir, self.data_folder, category, f"{category}_classes.txt")
            with open(txt_path, 'r') as file:
                lines = file.readlines()
                for line in lines:
                    img_path, label = line.strip().split()
                    full_img_path = os.path.join(self.rootdir, self.data_folder, category, img_path)
                    self.img_labels.append((full_img_path, int(label)))

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path, label = self.img_labels[idx]
        # print(img_path, label)
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, label


class C1SatelliteDataset(Dataset):
    def __init__(self, rootdir, train=True, download=False, transform=None):
        self.rootdir = rootdir
        self.train = train
        self.transform = transform

        # 选择使用训练集还是测试集
        self.data_folder = "train" if self.train else "test"

        # 读取所有图像路径和标签
        self.img_labels = []
        category_path = os.path.join(self.rootdir, self.data_folder, "tail")
        for label in os.listdir(category_path):
            label_path = os.path.join(category_path, label)
            if os.path.isdir(label_path):
                for img_file in os.listdir(label_path):
                    full_img_path = os.path.join(label_path, img_file)
                    self.img_labels.append((full_img_path, int(label)))

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path, label = self.img_labels[idx]
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, label


class DiorDataset(Dataset):
    def __init__(self, root_dir, folder_number, transform=None):
        """
        Args:
            root_dir (string): Directory with all the images.
            folder_number (int): Specific folder number to use for the dataset.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.root_dir = root_dir
        self.folder = str(folder_number)
        self.transform = transform
        self.image_paths = [os.path.join(root_dir, self.folder, file)
                            for file in os.listdir(os.path.join(root_dir, self.folder))
                            if file.endswith(('.png', '.jpg', '.jpeg'))]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_path = self.image_paths[idx]
        image = Image.open(img_path)

        if self.transform:
            image = self.transform(image)

        # 返回图像及其文件夹名（作为标签）
        return image, int(self.folder)