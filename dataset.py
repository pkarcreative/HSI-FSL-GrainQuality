# dataset.py
# CDF data loading and episodic data loader for FSL training and evaluation.
# The load_cdf and apply_transforms functions are specific to the Specim FX17
# hyperspectral camera data format used in this project.
#
# Usage:
#   from dataset import HSIDataLoader, load_cdf, apply_transforms
#   loader = HSIDataLoader(root_dir, k_shot=5, query_size=10)
#   for support_imgs, support_lbls, query_imgs, query_lbls in loader:
#       ...

import os
import numpy as np
import torch
import cv2
from spacepy import pycdf


def load_cdf(filepath):
    # Reads a .cdf hyperspectral file and returns the data array.
    # Drops the first and last 10 spectral channels due to low camera
    # sensitivity at the spectral boundaries of the Specim FX17 sensor.
    cdf = pycdf.CDF(filepath)
    data = cdf['data'][:, :, 10:-10]
    cdf.close()
    return data


def apply_transforms(data, binning_factor=1):
    # Resize spatial dimensions to 224x224 for ResNet18 compatibility.
    data = cv2.resize(data, (224, 224), interpolation=cv2.INTER_LINEAR)

    # Spectral binning: average every binning_factor consecutive channels.
    # binning_factor=1 keeps all 204 channels (recommended, no information loss).
    # binning_factor=2 -> 102 channels, binning_factor=3 -> 68 channels.
    if binning_factor > 1:
        c = data.shape[-1]
        new_c = (c // binning_factor) * binning_factor
        data = data[:, :, :new_c].reshape(224, 224, new_c // binning_factor, binning_factor)
        data = np.mean(data, axis=-1)

    # Per-channel z-score normalisation across spatial pixels.
    reshaped = data.reshape(-1, data.shape[-1])
    mean = np.mean(reshaped, axis=0)
    std = np.std(reshaped, axis=0)
    std[std == 0] = 1e-8
    normalized = (reshaped - mean) / std
    return torch.tensor(normalized.reshape(data.shape), dtype=torch.float32)


class HSIDataLoader:
    # Episodic iterator for FSL tasks over HSI .cdf files.
    #
    # Directory structure expected:
    #   root_dir/
    #     ClassName1/  file1.cdf  file2.cdf ...
    #     ClassName2/  ...
    #
    # Each call to __next__ yields one episode:
    #   - support: k_shot samples per class
    #   - query: query_size samples per class
    # Files are iterated in sorted order so episodes are deterministic.
    #
    # Args:
    #   root_dir            : path to directory with class subfolders
    #   k_shot              : support samples per class per episode
    #   query_size          : query samples per class per episode
    #   exclude_classes     : list of class names to skip (e.g. for 6-way experiments)
    #   max_samples_per_class: cap on files used per class (default: use all)
    #   binning_factor      : spectral binning factor passed to apply_transforms

    def __init__(self, root_dir, k_shot, query_size, exclude_classes=None,
                 max_samples_per_class=None, binning_factor=1):
        self.root_dir = root_dir
        self.k_shot = k_shot
        self.query_size = query_size
        self.binning_factor = binning_factor
        self.exclude_classes = set(exclude_classes or [])

        self.classes = sorted([
            d for d in os.listdir(root_dir)
            if os.path.isdir(os.path.join(root_dir, d))
            and d not in self.exclude_classes
        ])
        if not self.classes:
            raise ValueError(f'No class directories found in {root_dir}')

        self.n_way = len(self.classes)
        self.label_to_idx = {cls: i for i, cls in enumerate(self.classes)}

        # determine effective samples per class
        counts = [
            len([f for f in os.listdir(os.path.join(root_dir, c)) if f.endswith('.cdf')])
            for c in self.classes
        ]
        min_count = min(counts)
        if max_samples_per_class is not None:
            self.max_samples = min(max_samples_per_class, min_count)
        else:
            self.max_samples = min_count

        self.current_idx = 0

    def __iter__(self):
        self.current_idx = 0
        return self

    def __len__(self):
        return self.max_samples // (self.k_shot + self.query_size)

    def __next__(self):
        if self.current_idx + self.k_shot + self.query_size > self.max_samples:
            raise StopIteration

        support_imgs, support_lbls = [], []
        query_imgs, query_lbls = [], []

        for cls in self.classes:
            cls_path = os.path.join(self.root_dir, cls)
            files = sorted([f for f in os.listdir(cls_path) if f.endswith('.cdf')])

            for fname in files[self.current_idx:self.current_idx + self.k_shot]:
                img = apply_transforms(load_cdf(os.path.join(cls_path, fname)), self.binning_factor)
                support_imgs.append(img.unsqueeze(0))
                support_lbls.append(self.label_to_idx[cls])

        self.current_idx += self.k_shot

        for cls in self.classes:
            cls_path = os.path.join(self.root_dir, cls)
            files = sorted([f for f in os.listdir(cls_path) if f.endswith('.cdf')])

            for fname in files[self.current_idx:self.current_idx + self.query_size]:
                img = apply_transforms(load_cdf(os.path.join(cls_path, fname)), self.binning_factor)
                query_imgs.append(img.unsqueeze(0))
                query_lbls.append(self.label_to_idx[cls])

        self.current_idx += self.query_size

        return (
            torch.cat(support_imgs, dim=0),
            torch.tensor(support_lbls),
            torch.cat(query_imgs, dim=0),
            torch.tensor(query_lbls)
        )
