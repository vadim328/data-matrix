import json
from typing import Dict, List, Any

import cv2
import os

import pandas as pd
import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2
from sklearn.preprocessing import LabelEncoder


class DataMatrixDataset(Dataset):
    def __init__(
            self,
            root_dir: str,
    ):
        self.root_dir = root_dir

    def load_datast(self):
        image_paths, labels = [], []
        for folder_1 in os.listdir(self.root_dir):
            if os.path.isdir(os.path.join(self.root_dir, folder_1)):
                for folder_2 in os.listdir(os.path.join(self.root_dir, folder_1)):
                    for image_name in os.listdir(os.path.join(self.root_dir, folder_1, folder_2)):
                        image_path = os.path.join(self.root_dir, folder_1, folder_2, image_name)
                        if image_path.lower().endswith(("png", "jpg", "jpeg", "bmp", "gif")):
                            image_paths.append(image_path)
                            labels.append(f"{folder_1}/{folder_2}")

        self.label_encoder = LabelEncoder()
        labels = self.label_encoder.fit_transform(labels)
        labels = list(labels)
        #labels = [0 if i == 0 else 1 for i in labels]
        labels = torch.tensor(labels)
        return image_paths, labels


class DataMatrixDatasetTrain(DataMatrixDataset):
    def __init__(
        self,
        root_dir: str,
        transforms: List[Dict[str, Any]] = None,
        transform_params: Dict[str, Any] = {},
        name: str = None,
        infinite: bool = False,
    ):
        super().__init__(root_dir)
        self.image_paths, self.labels = self.load_datast()
        
        #print(self.labels[0], self.image_paths[0])
        self.transforms = transforms if transforms is not None else None

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        image = cv2.imread(img_path, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if self.transforms:
            image = self.transforms(image=image)["image"]
        return image, label

    def get_class_name(self, label):
        return self.label_encoder.inverse_transform([label])[0]


class DataMatrixDatasetTest(DataMatrixDataset):
    def __init__(
        self,
        root_dir: str,
        transforms: List[Dict[str, Any]] = None,
        transform_params: Dict[str, Any] = {},
        name: str = None,
        infinite: bool = False,
    ):
        super().__init__(root_dir)
        self.image_paths, self.labels = self.load_datast()
        
        self.transforms = transforms if transforms is not None else None

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        image = cv2.imread(img_path, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if self.transforms:
            image = self.transforms(image=image)["image"]
        return image, label

    def get_class_name(self, label):
        return self.label_encoder.inverse_transform([label])[0]
