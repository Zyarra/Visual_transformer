import PIL
import torchvision
import torch


def data_preparation(data_dir, batch_size_train, batch_size_test, image_size, num_workers, prefetch_factor,
                     img_mean, img_std, ds_split):
    train_transform = torchvision.transforms.Compose(
        [torchvision.transforms.Resize((image_size, image_size)),
         torchvision.transforms.RandomHorizontalFlip(),
         torchvision.transforms.RandomRotation(15, resample=PIL.Image.BILINEAR),
         torchvision.transforms.ToTensor(),
         torchvision.transforms.RandomAffine(20),
         torchvision.transforms.Normalize(mean=img_mean, std=img_std)])

    test_transform = torchvision.transforms.Compose(
        [torchvision.transforms.Resize((image_size, image_size)),
         torchvision.transforms.ToTensor(),
         torchvision.transforms.Normalize(mean=img_mean, std=img_std)])

    ds = torchvision.datasets.ImageFolder(data_dir)

    # val_transform = torchvision.transforms.Compose(
    #     [torchvision.transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    #      torchvision.transforms.ToTensor()])

    train_ds, test_ds = torch.utils.data.random_split(ds, ds_split)

    train_ds.dataset.transform = train_transform
    test_ds.dataset.transform = test_transform
    # val_ds.dataset.transform = val_transform

    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size_train,
                                               shuffle=True, pin_memory=True, drop_last=True,
                                               num_workers=num_workers, prefetch_factor=prefetch_factor)
    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=batch_size_test,
                                              shuffle=False, pin_memory=True, drop_last=True,
                                              num_workers=num_workers, prefetch_factor=prefetch_factor)
    # val_loader = torch.utils.data.DataLoader(test_ds, batch_size=batch_size_test,
    #                                           shuffle=False, pin_memory=True, drop_last=True,
    #                                           num_workers=N_WORKERS, prefetch_factor=PREFETCH_FACTOR)
    return train_loader, test_loader
