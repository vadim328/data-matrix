import os

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision import transforms

import pytorch_lightning as pl
from pytorch_lightning import Trainer


import dataset
from argparse import ArgumentParser

import albumentations as A
from albumentations.pytorch import ToTensorV2

from model import LitResNet


def main(hparams):

    transform = A.Compose([
         A.Resize(224, 224),
         A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
         ToTensorV2()
    ])

    train_data = dataset.DataMatrixDatasetTrain(root_dir="../part1/train/", transforms=transform)
    print("---------------\nTrain Dataset loaded \n---------------")
    train_data_loader = DataLoader(train_data, batch_size=8)
    
    test_data = dataset.DataMatrixDatasetTest(root_dir="../part1/test/", transforms=transform)
    print("---------------\nTest Dataset loaded \n---------------")
    test_data_loader = DataLoader(test_data, batch_size=8)
    
    model_x = LitResNet(num_classes=2)
    trainer = pl.Trainer(max_epochs=10, accelerator=hparams.accelerator, devices=hparams.devices)
    trainer.fit(model_x, train_data_loader)

    trainer.test(dataloaders=test_data_loader)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--accelerator", default=None)
    parser.add_argument("--devices", default=None)
    args = parser.parse_args()

    main(args)
