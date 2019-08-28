import glob
import numpy as np
from skimage.io import imread
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
import torch
import os
from pathlib import Path
import dataset.Morpho as Morpho
import pandas as pd

def get_mnist_dataloaders(batch_size=128, path_to_data='../data'):
    """MNIST dataloader with (32, 32) images."""
    all_transforms = transforms.Compose([
        transforms.Resize(32),
        transforms.ToTensor()
    ])
    data_loader = datasets.MNIST(path_to_data, train=True, download=True,
                                transform=all_transforms)
    test_data = datasets.MNIST(path_to_data, train=False,
                               transform=all_transforms)
##split    
    train_size = int(0.9 * len(data_loader))
    test_size = len(data_loader) - train_size
    train_loader, valid_loader = torch.utils.data.random_split(data_loader, [train_size, test_size])
###data_loader wrap
    valid_loader = DataLoader(valid_loader, batch_size=batch_size, shuffle=True)
    train_loader = DataLoader(train_loader, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True)
    return valid_loader, train_loader, test_loader


class MorphoMNISTDataset(Dataset):
    """Dataset wrapping images and target labels for Kaggle - Planet Amazon from Space competition.

    Arguments:
        A CSV file path
        Path to image folder
        Extension of images
        PIL transforms
    """
    def __init__(self, label="train", transform=None):
        self.transform = transform
        data_root = "C:/Users/Cooper/Morpho/MorphoMNIST"
        dataset_names = ["plain", "thin", "thic", "swel", "frac"]
        if label == "train":
            setname = ["train"]
        else:
            setname = ["t10k"]
        for subset in setname:
            digits_filename = f"{subset}-labels-idx1-ubyte.gz"
            images_filename = f"{subset}-images-idx3-ubyte.gz"
            metrics_filename = f"{subset}-morpho.csv"
            pert_filename = f"{subset}-pert-idx1-ubyte.gz"

            #             data_dirs = [os.path.join(data_root, dataset_names[i]) for i in pairing]
            data_dirs = [os.path.join(data_root, dataset_names[0])]
            imgs_paths = [os.path.join(data_dir, images_filename) for data_dir in data_dirs]
            digits_paths = [os.path.join(data_dir, digits_filename) for data_dir in data_dirs]
            metrics_paths = [os.path.join(data_dir, metrics_filename) for data_dir in data_dirs]
            all_digits = np.array([Morpho.load_idx(path) for path in digits_paths])
            all_images = np.array([Morpho.load_idx(path) for path in imgs_paths])
            all_images = np.squeeze(all_images)
            all_images = np.expand_dims(all_images, axis=3)
            self.all_images = all_images
            #self.all_images = torch.from_numpy(all_images/255).float()
            all_metrics = [pd.read_csv(path, index_col='index') for path in metrics_paths]
            one_hot_digits = np.zeros((all_metrics[0].values.shape[0], 10))
            one_hot_digits[np.arange(all_metrics[0].values.shape[0]), all_digits.reshape(-1, 1)[:, -1].astype(int)] += 1
            #self.labels = torch.from_numpy(np.hstack([all_metrics[0].values, all_digits.reshape(-1, 1)])).float()
            self.labels = np.hstack([all_metrics[0].values, all_digits.reshape(-1, 1)])# all_digits.reshape(-1, 1)
            self.labels = np.hstack([self.labels, one_hot_digits])
            # train = torch.utils.data.TensorDataset(all_images, labels)
            # loader = torch.utils.data.DataLoader(train, batch_size=64, shuffle=True)

    def __getitem__(self, index):
        img = self.all_images[index, :, :, :]
        if self.transform is not None:
            img = self.transform(img)

        label = torch.from_numpy(self.labels[index, :])
        return img, label

    def __len__(self):
        return self.labels.shape[0]

    def get_label_size(self):
        return self.labels.shape[1]

#home = str(Path.home())
def get_MorphoMNIST(label):
    data_root = "C:/Users/Cooper/Morpho/MorphoMNIST"
    dataset_names = ["plain", "thin", "thic", "swel", "frac"]

    if label == "train":
        setname = ["train"]
    else:
        setname = ["t10k"]
    for subset in setname:
        digits_filename = f"{subset}-labels-idx1-ubyte.gz"
        images_filename = f"{subset}-images-idx3-ubyte.gz"
        metrics_filename = f"{subset}-morpho.csv"
        pert_filename = f"{subset}-pert-idx1-ubyte.gz"

        #             data_dirs = [os.path.join(data_root, dataset_names[i]) for i in pairing]
        data_dirs = [os.path.join(data_root, dataset_names[0])]
        imgs_paths = [os.path.join(data_dir, images_filename) for data_dir in data_dirs]
        digits_paths = [os.path.join(data_dir, digits_filename) for data_dir in data_dirs]
        metrics_paths = [os.path.join(data_dir, metrics_filename) for data_dir in data_dirs]
        all_digits = np.array([Morpho.load_idx(path) for path in digits_paths])
        all_images = np.array([Morpho.load_idx(path) for path in imgs_paths])
        all_metrics = [pd.read_csv(path, index_col='index') for path in metrics_paths]
        labels = np.hstack([all_metrics[0].values, all_digits.reshape(-1,1)])
        #all_labels = np.hstack([all_metrics, all_digits.reshape((-1,1))])
        labels = torch.from_numpy(labels).float()
        all_images = torch.from_numpy(all_images/255).float()
        np.swapaxes(all_images, 0, 1)
        train = torch.utils.data.TensorDataset(all_images, labels)
        loader = torch.utils.data.DataLoader(train, batch_size=64, shuffle=True)
    return loader


class DSpritesDataset(Dataset):
    """D Sprites dataset."""
    def __init__(self, label="train", transform=None):
        """
        Parameters
        ----------
        subsample : int
            Only load every |subsample| number of images.
        """
        self.label = label
        self.dataset = np.load('C:/Users/Cooper/FYP/dsprites-dataset/dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz')
        self.transform = transform
        self.imgs = self.dataset['imgs']
        shuffle_index = np.random.permutation(len(self.imgs))
        self.imgs = self.imgs[shuffle_index]
        self.labels = self.dataset['latents_values'][shuffle_index]
        train_size = int(9*self.imgs.shape[0] / 10)
        if label == "train":
            self.imgs = self.imgs[:train_size, :, :]
            self.labels = self.labels[:train_size, 1:]
        else:
            self.imgs = self.imgs[train_size:, :, :]
            self.labels = self.labels[train_size:, 1:]
        #self.imgs = np.expand_dims(self.dataset['imgs'], dim=1)

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        # Each image in the dataset has binary values so multiply by 255 to get
        # pixel values
        sample = self.imgs[idx] * 255
        # Add extra dimension to turn shape into (H, W) -> (H, W, C)
        samples = sample.reshape(sample.shape + (1,))
        if self.transform:
            samples = self.transform(samples)
        labels = self.labels[idx]
        return samples, labels

    def get_label_size(self):
        return self.labels.shape[1]


def get_fashion_mnist_dataloaders(batch_size=128,
                                  path_to_data='../fashion_data'):
    """FashionMNIST dataloader with (32, 32) images."""
    all_transforms = transforms.Compose([
        transforms.Resize(32),
        transforms.ToTensor()
    ])
    train_data = datasets.FashionMNIST(path_to_data, train=True, download=True,
                                       transform=all_transforms)
    test_data = datasets.FashionMNIST(path_to_data, train=False,
                                      transform=all_transforms)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True)
    return train_loader, test_loader


def get_dsprites_dataloader(batch_size=128,
                            path_to_data='../data/dsprites_data.npz'):
    """DSprites dataloader."""
    dsprites_data = DSpritesDataset(path_to_data,
                                    transform=transforms.ToTensor())
    dsprites_loader = DataLoader(dsprites_data, batch_size=batch_size,
                                 shuffle=True)
    return dsprites_loader


def get_chairs_dataloader(batch_size=128,
                          path_to_data='../rendered_chairs_64'):
    """Chairs dataloader. Chairs are center cropped and resized to (64, 64)."""
    all_transforms = transforms.Compose([
        transforms.Grayscale(),
        transforms.ToTensor()
    ])
    chairs_data = datasets.ImageFolder(root=path_to_data,
                                       transform=all_transforms)
    chairs_loader = DataLoader(chairs_data, batch_size=batch_size,
                               shuffle=True)
    return chairs_loader


def get_chairs_test_dataloader(batch_size=62,
                               path_to_data='../rendered_chairs_64_test'):
    """There are 62 pictures of each chair, so get batches of data containing
    one chair per batch."""
    all_transforms = transforms.Compose([
        transforms.Grayscale(),
        transforms.ToTensor()
    ])
    chairs_data = datasets.ImageFolder(root=path_to_data,
                                       transform=all_transforms)
    chairs_loader = DataLoader(chairs_data, batch_size=batch_size,
                               shuffle=False)
    return chairs_loader


def get_celeba_dataloader(data_loader, batch_size=128):
    """CelebA dataloader with (64, 64) images."""
#    data_loader = CelebADataset(path_to_data,
#                                transform=transforms.ToTensor())
#    data_loader = DataLoader(celeba_data, batch_size=batch_size,
#                               shuffle=True)
    
##split    

    train_size = int(0.8 * len(data_loader))
    valid_size = int(0.1 * len(data_loader))
    test_size = len(data_loader) - train_size - valid_size
    print(train_size, valid_size, test_size)
    train_loader, test_loader = torch.utils.data.random_split(data_loader, [train_size+valid_size,test_size])
    train_loader, valid_loader = torch.utils.data.random_split(train_loader, [train_size,valid_size])
###data_loader wrap
    valid_loader = DataLoader(valid_loader, batch_size=batch_size, shuffle=True)
    train_loader = DataLoader(train_loader, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_loader, batch_size=batch_size, shuffle=True)
    return valid_loader, train_loader, test_loader



class CelebADataset(Dataset):
    """CelebA dataset with 64 by 64 images."""
    def __init__(self, path_to_data, subsample=1, transform=None):
        """
        Parameters
        ----------
        subsample : int
            Only load every |subsample| number of images.
        """
        self.img_paths = glob.glob(path_to_data + '/*')[::subsample]
        self.transform = transform

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        sample_path = self.img_paths[idx]
        sample = imread(sample_path)

        if self.transform:
            sample = self.transform(sample)
        # Since there are no labels, we just return 0 for the "label" here
        return sample, 0