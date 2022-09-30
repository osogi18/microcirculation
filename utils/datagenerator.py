import os
import random
import numpy as np
import glob

import albumentations as A
from .dataset import EyeDataset, PatchEyeDataset, PSEyeDataset
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split, KFold


def get_paths(cfg):
    """
    :param directories: list of directories or one directory, where data is
    :return: [[image_path, mask_path], ...]
    """
    return glob.glob(f"{cfg.data_folder}/*.png")


def data_generator(cfg):
    image_paths = get_paths(cfg)
    image_paths = np.asarray(image_paths)
    train_paths, val_paths = [], []

    if not cfg.kfold:
        train_paths, val_paths = train_test_split(image_paths, test_size=cfg.val_size, random_state=cfg.seed)
    else:
        kf = KFold(n_splits=cfg.n_splits)
        for i, (train_index, val_index) in enumerate(kf.split(image_paths)):
            if i + 1 == cfg.fold_number:
                train_paths = image_paths[train_index]
                val_paths = image_paths[val_index]

    random.shuffle(train_paths)
    random.shuffle(val_paths)
    return train_paths, val_paths


def get_transforms(cfg):
    # getting transforms from albumentations
    pre_transforms = [getattr(A, item["name"])(**item["params"]) for item in cfg.pre_transforms]
    augmentations = [getattr(A, item["name"])(**item["params"]) for item in cfg.augmentations]
    post_transforms = [getattr(A, item["name"])(**item["params"]) for item in cfg.post_transforms]

    # concatenate transforms
    train = A.Compose(pre_transforms + augmentations + post_transforms)
    test = A.Compose(pre_transforms + post_transforms)
    return train, test


def get_loaders(cfg):
    # getting transforms
    train_transforms, test_transforms = get_transforms(cfg)

    # getting train and val paths
    train_paths, val_paths = data_generator(cfg)

    # creating datasets
    if cfg.patches:
        train_ds = PatchEyeDataset(train_paths, transform=train_transforms)
        val_ds = PatchEyeDataset(val_paths, transform=test_transforms)
    elif cfg.pseudo_labeling:
        train_ds = PSEyeDataset(train_paths, transform=train_transforms)
        val_ds = PSEyeDataset(val_paths, transform=test_transforms)
    else:
        train_ds = EyeDataset(train_paths, transform=train_transforms)
        val_ds = EyeDataset(val_paths, transform=test_transforms)

    # creating data loaders
    # TODO: make drop last parameter in config
    train_dl = DataLoader(train_ds, batch_size=cfg.batch_size, drop_last=True)
    val_dl = DataLoader(val_ds, batch_size=cfg.batch_size, drop_last=True)
    return train_dl, val_dl

# if __name__ == '__main__':
#     cfg = Config()
#     cfg.data_folder = 'train_dataset_mc/'
#     cfg.batch_size = 4
#     cfg.val_size = 0.2
#     cfg.seed = 12
#     cfg.kfold = False
#     cfg.pre_transforms = [
#     ]
#
#     cfg.augmentations = [
#         #     class albumentations.augmentations.geometric.transforms.ElasticTransform
#     ]
#
#     cfg.post_transforms = []
#
#     train_dl, val_dl = get_loaders(cfg)
#     a = next(iter(train_dl))
#     print(a[0].shape, a[1].shape)
