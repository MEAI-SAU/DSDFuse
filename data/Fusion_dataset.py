import glob
import os

import torch
import torchvision.transforms
import torchvision.transforms.functional as TF
from PIL import Image
from torch.utils.data.dataset import Dataset
from torchvision.transforms import Compose, RandomCrop, ToTensor

from util.util import RGB2YCrCb, randfilp, randrot


def calculate_valid_crop_size(crop_size, upscale_factor):
    return crop_size - (crop_size % upscale_factor)


def train_hr_transform(crop_size):
    return Compose([
        RandomCrop(crop_size),
    ])


def prepare_data_path(dataset_path):
    filenames = os.listdir(dataset_path)
    data_dir = dataset_path
    data = glob.glob(os.path.join(data_dir, "*.bmp"))
    data.extend(glob.glob(os.path.join(data_dir, "*.tiff")))
    data.extend(glob.glob(os.path.join(data_dir, "*.jpg")))
    data.extend(glob.glob(os.path.join(data_dir, "*.png")))
    data.sort()
    filenames.sort()
    return data, filenames


class FusionDataset(Dataset):
    def __init__(
        self,
        split,
        crop_size=256,
        min_max=(-1, 1),
        ir_path='./PathToIr/',
        vi_path='./PathToVis/',
        structure_path=None,
        function_path=None,
        is_crop=True
    ):
        super(FusionDataset, self).__init__()
        assert split in ['train', 'val', 'test'], 'split must be "train"|"val"|"test"'
        self.split = split
        self.is_crop = is_crop
        self.crop_size = crop_size
        self.crop = torchvision.transforms.RandomCrop(self.crop_size)
        self.min_max = min_max
        structure_path = structure_path or ir_path
        function_path = function_path or vi_path
        self.filepath_structure, self.filenames_structure = prepare_data_path(structure_path)
        self.filepath_function, self.filenames_function = prepare_data_path(function_path)
        self.length = min(len(self.filenames_structure), len(self.filenames_function))

    def __getitem__(self, index):
        structure_image = Image.open(self.filepath_structure[index]).convert('RGB')
        function_image = Image.open(self.filepath_function[index]).convert('RGB')

        structure_image = (ToTensor()(structure_image) * (self.min_max[1] - self.min_max[0]) + self.min_max[0]).unsqueeze(0)
        function_image = (ToTensor()(function_image) * (self.min_max[1] - self.min_max[0]) + self.min_max[0]).unsqueeze(0)
        paired_image = torch.cat([structure_image, function_image], dim=1)

        if self.split == 'train':
            paired_image = randfilp(paired_image)
            paired_image = randrot(paired_image)
            if self.is_crop:
                patch = self.crop(paired_image)
                if patch.shape[-1] <= self.crop_size or patch.shape[-2] <= self.crop_size:
                    patch = TF.resize(patch, self.crop_size)
            else:
                patch = paired_image
        else:
            patch = paired_image

        structure_image, function_image = torch.split(patch, [3, 3], dim=1)
        structure_ycrcb = RGB2YCrCb(structure_image)
        function_ycrcb = RGB2YCrCb(function_image)
        structure_y = structure_ycrcb[:, 0:1, :, :].squeeze(0)
        function_y = function_ycrcb[:, 0:1, :, :].squeeze(0)
        function_crcb = function_ycrcb[:, 1:3, :, :].squeeze(0)
        return {
            'structure_y': structure_y,
            'function_y': function_y,
            'structure_rgb': structure_image.squeeze(0),
            'function_rgb': function_image.squeeze(0),
            'function_crcb': function_crcb,
            'function_cbcr': function_crcb,
            'Index': index
        }

    def __len__(self):
        return self.length


class FusionDataset_Digtal(Dataset):
    def __init__(
        self,
        split,
        crop_size=256,
        min_max=(-1, 1),
        img1_path=None,
        img2_path=None,
        structure_path=None,
        function_path=None,
        is_crop=True
    ):
        super(FusionDataset_Digtal, self).__init__()
        assert split in ['train', 'val', 'test'], 'split must be "train"|"val"|"test"'
        self.split = split
        self.is_crop = is_crop
        self.crop_size = crop_size
        self.crop = torchvision.transforms.RandomCrop(self.crop_size)
        self.min_max = min_max
        structure_path = structure_path or img1_path
        function_path = function_path or img2_path
        self.filepath_structure, self.filenames_structure = prepare_data_path(structure_path)
        self.filepath_function, self.filenames_function = prepare_data_path(function_path)
        self.length = min(len(self.filenames_structure), len(self.filenames_function))

    def __getitem__(self, index):
        structure_image = Image.open(self.filepath_structure[index]).convert('RGB')
        function_image = Image.open(self.filepath_function[index]).convert('RGB')

        structure_image = (ToTensor()(structure_image) * (self.min_max[1] - self.min_max[0]) + self.min_max[0]).unsqueeze(0)
        function_image = (ToTensor()(function_image) * (self.min_max[1] - self.min_max[0]) + self.min_max[0]).unsqueeze(0)
        paired_image = torch.cat([structure_image, function_image], dim=1)

        if self.split == 'train':
            paired_image = randfilp(paired_image)
            paired_image = randrot(paired_image)
            if self.is_crop:
                patch = self.crop(paired_image)
                if patch.shape[-1] <= self.crop_size or patch.shape[-2] <= self.crop_size:
                    patch = TF.resize(patch, self.crop_size)
            else:
                patch = paired_image
        else:
            patch = paired_image

        structure_image, function_image = torch.split(patch, [3, 3], dim=1)
        structure_ycrcb = RGB2YCrCb(structure_image)
        function_ycrcb = RGB2YCrCb(function_image)
        structure_y = structure_ycrcb[:, 0:1, :, :].float().squeeze(0)
        function_y = function_ycrcb[:, 0:1, :, :].float().squeeze(0)
        function_crcb = function_ycrcb[:, 1:3, :, :].float().squeeze(0)
        return {
            'structure_y': structure_y,
            'function_y': function_y,
            'structure_rgb': structure_image.float().squeeze(0),
            'function_rgb': function_image.float().squeeze(0),
            'function_crcb': function_crcb,
            'function_cbcr': function_crcb,
            'Index': index
        }

    def __len__(self):
        return self.length
