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

    transform_train = A.Compose([
         A.Resize(224, 224),
         A.RandomRotate90(p=0.25),
         A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15, interpolation=1, p=0.5),
         #A.cutout(max_h_size=32, max_w_size=32, p=0.05),
         A.CLAHE(clip_limit=2, p=0.1),
         A.RandomBrightnessContrast(p=0.2),
         A.GaussNoise(p=0.2),
         A.MotionBlur(blur_limit=3, p=0.2),
         A.ISONoise(p=0.2),
         A.ImageCompression(quality_lower=15, quality_upper=30, p=0.25),
         A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
         ToTensorV2()
    ])
    
    transfom_test = A.Compose([
         A.Resize(224, 224),
         A.ImageCompression(quality_lower=15, quality_upper=30, p=0.25),
         A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
         ToTensorV2()
    ])

    train_data = dataset.DataMatrixDatasetTrain(root_dir="../part1/train/", transforms=transform_train)
    print("---------------\nTrain Dataset loaded \n---------------")
    train_data_loader = DataLoader(train_data, batch_size=8)
    
    test_data = dataset.DataMatrixDatasetTest(root_dir="../part1/test/", transforms=transform_test)
    print("---------------\nTest Dataset loaded \n----------------")
    test_data_loader = DataLoader(test_data, batch_size=8)
    
    model_x = LitResNet(num_classes=54)
    trainer = pl.Trainer(max_epochs=75, accelerator=hparams.accelerator, devices=hparams.devices)
    trainer.fit(model_x, train_data_loader)

    trainer.test(dataloaders=test_data_loader)
    print(model_x.on_test_epoch_end())


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--accelerator", default=None)
    parser.add_argument("--devices", default=None)
    args = parser.parse_args()

    main(args)
