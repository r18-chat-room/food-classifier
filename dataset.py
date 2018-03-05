import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision import transforms
from torchvision.datasets.folder import find_classes
from torchvision.datasets import ImageFolder


def get_train_valid_loader(path, scale_size, batch_size, valid_size, num_workers, shuffle):
    """
    load training and validation multi-process iterators.

    :param path: path directory to the dataset.
    :param scale_size: the scaled size of source image.
    :param batch_size: how many samples per batch to load.
    :param valid_size: percentage split of dataset used for validation. Should be a float in the range [0,1].
    :param num_workers: number of subprocess to use when loading.
    :param shuffle: whether to shuffle the indices

    :return: (tuple) training & validation dataset iterator
    """
    valid_transform = transforms.Compose([
        transforms.Resize(scale_size),
        transforms.CenterCrop(scale_size),
        transforms.ToTensor(),
    ])

    train_transform = transforms.Compose([
        transforms.Resize(scale_size),
        transforms.CenterCrop(scale_size),
        transforms.RandomHorizontalFlip(),  # augment by horizontal flip
        transforms.ToTensor(),
    ])

    train_dataset = ImageFolder(path, transform=train_transform)
    valid_dataset = ImageFolder(path, transform=valid_transform)
    indices = list(range(len(train_dataset)))
    pivot = int(np.floor(valid_size * len(train_dataset)))

    if shuffle:
        np.random.shuffle(indices)

    train_sampler, valid_sampler = SubsetRandomSampler(indices[:pivot]), SubsetRandomSampler(indices[pivot:])
    train_loader = DataLoader(train_dataset,
                              batch_size=batch_size,
                              sampler=train_sampler,
                              num_workers=num_workers,)
    valid_loader = DataLoader(valid_dataset,
                              batch_size=batch_size,
                              sampler=valid_sampler,
                              num_workers=num_workers,)
    return train_loader, valid_loader


def get_categories(path):
    """
    load categories from dataset folder and generate an dict of id2name.

    :param path: the path of folder of the dataset

    :return: a dict with id as key and name as value
    """
    categories = find_classes(path)[1]
    return dict(zip(categories.values(), categories.keys()))
