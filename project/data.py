import PIL
import torch
import torchvision
from project.hyperparameters import *


def data_preparation(data_dir, batch_size_train, batch_size_test):
    train_transform = torchvision.transforms.Compose(
        [torchvision.transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
         torchvision.transforms.RandomHorizontalFlip(),
         torchvision.transforms.RandomRotation(15, resample=PIL.Image.BILINEAR),
         torchvision.transforms.ToTensor(),
         torchvision.transforms.RandomAffine(20),
         torchvision.transforms.Normalize(mean=IMGMEAN, std=IMGSTD)])

    test_transform = torchvision.transforms.Compose(
        [torchvision.transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
         torchvision.transforms.ToTensor(),
         torchvision.transforms.Normalize(mean=IMGMEAN, std=IMGSTD)])

    ds = torchvision.datasets.ImageFolder(data_dir)

    # val_transform = torchvision.transforms.Compose(
    #     [torchvision.transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    #      torchvision.transforms.ToTensor()])

    train_ds, test_ds = torch.utils.data.random_split(ds, DS_SPLIT)
    # test_ds, val_ds = torch.utils.data.random_split(test_ds, [2715, 2000])

    train_ds.dataset.transform = train_transform
    test_ds.dataset.transform = test_transform
    # val_ds.dataset.transform = val_transform

    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size_train,
                                               shuffle=True, pin_memory=True, drop_last=True,
                                               num_workers=N_WORKERS, prefetch_factor=PREFETCH_FACTOR)
    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=batch_size_test,
                                              shuffle=False, pin_memory=True, drop_last=True,
                                              num_workers=N_WORKERS, prefetch_factor=PREFETCH_FACTOR)
    # val_loader = torch.utils.data.DataLoader(test_ds, batch_size=batch_size_test,
    #                                           shuffle=False, pin_memory=True, drop_last=True,
    #                                           num_workers=N_WORKERS, prefetch_factor=PREFETCH_FACTOR)
    return train_loader, test_loader
