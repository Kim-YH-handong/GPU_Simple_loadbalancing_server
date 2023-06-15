from torchvision import transforms
import cv2
from PIL import Image
from torch.utils.data import Dataset
from scipy.ndimage.interpolation import zoom
from scipy import ndimage
import torch
import os
import random
import h5py
import numpy as np
np.set_printoptions(threshold=np.inf, linewidth=np.inf)

class RandomGenerator_test(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        image = image.transpose(2, 0, 1)  # C H W
        label = label.transpose(2, 0, 1)  # C H W

        transform = transforms.Compose([
            transforms.Resize(self.output_size)
        ])

        transform_label = transforms.Compose([
            transforms.Resize(self.output_size)
        ])
        image = torch.from_numpy(image.astype(np.float32))
        label = torch.from_numpy(label.astype(np.float32))
        image = transform(image)
        label = transform_label(label)
        image = image / 255
        label = label / 255

        sample = {'image': image, 'label': label}
        return sample


class VFSS_dataset(Dataset):
    def __init__(self, base_dir, transform=None):
        self.transform = transform  # using transform in torch!
        self.data_dir = base_dir

        self.img_paths = []

        save_dir = os.path.join(base_dir, 'result_img')
        base_dir = os.path.join(base_dir, 'original_img')

        for p in os.listdir(base_dir):
            name = p.split('.')[0]
            self.img_paths.append(os.path.join(base_dir, name + '.jpeg'))

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):

        img = self.img_paths[idx]
        image = cv2.imread(img)
        label = np.load('/home/younghun/VFSS/Data/[Capstone]demo/clahe/mask/AhnMiza_softdiet_00000.npy')
        label = label.transpose(1, 2, 0)

        sample = {'image': image, 'label': label}

        if self.transform:
            sample = self.transform(sample)

        sample['case_name'] = self.img_paths[idx]

        return sample
