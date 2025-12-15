import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch
import torchvision.datasets as datasets

from sklearn.datasets import load_svmlight_file
from sklearn.model_selection import train_test_split

import gdown
import sys

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

train_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10(root='./data', train=True, transform=transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, 4),
            transforms.ToTensor(),
            normalize,
        ]), download=True),
        batch_size=256, shuffle=True,
        num_workers=2, pin_memory=True)

test_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10(root='./data', train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=256, shuffle=False,
        num_workers=2, pin_memory=True)


def download_mushrooms_dataset():
    """Download mushrooms.txt dataset from Google Drive"""
    url = "https://drive.google.com/uc?id=1C9lPeL1IBgn8h4jSoFaK_dmRUOVxoRyp"

    output_file = "training/data"
    
    try:
        print(f"Downloading dataset to {output_file}...")
        
        gdown.download(url, output_file, quiet=False)
        print("Download completed successfully!")

        data = load_svmlight_file(output_file)
        X, y = data[0].toarray(), data[1]

        y = 2 * y - 3

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        return X_train, X_test, y_train, y_test
        
    except Exception as e:
        print(f"Error downloading file: {e}")
        sys.exit(1)