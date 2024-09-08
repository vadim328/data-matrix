from torchvision.models import resnet50, ResNet50_Weights
from torch.nn import functional as F
from torch.utils.data import random_split
import pytorch_lightning as pl
from torch import nn
import torch
import torchmetrics


class LitResNet(pl.LightningModule):
    def __init__(self, num_classes):
        super().__init__()
        self.model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        self.model.fc = torch.nn.Linear(self.model.fc.in_features, num_classes)
        self.accuracy = torchmetrics.classification.Accuracy(task="multiclass", num_classes=num_classes)
        self.test_predictions = []
        self.acc_score = None
        self.loss = None
 
    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        #y = torch.tensor(y)
        y_hat = self(x)
        self.loss = F.cross_entropy(y_hat, y)
        self.acc_score = self.accuracy(y_hat, y)
        self.log('train_loss_step', self.loss, on_step=True, on_epoch=False)
        self.log('train_accuracy_step', self.acc_score, on_step=True, on_epoch=False)
        self.log('train_loss_epoch', self.loss, on_step=False, on_epoch=True)
        self.log('train_accuracy_epoch', self.acc_score, on_step=False, on_epoch=True)
        return self.loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        self.loss = F.cross_entropy(y_hat, y)
        self.acc_score = self.accuracy(y_hat, y)
        self.log('validation_loss', self.loss)
        self.log('validation_accuracy', self.acc_score)

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        preds = torch.argmax(y_hat, dim=1)
        self.test_predictions.append(preds)
        self.loss = F.cross_entropy(y_hat, y)
        self.acc_score = self.accuracy(y_hat, y)
        self.log('test_loss', self.loss)
        self.log('test_accuracy', self.acc_score)
    
    def on_train_epoch_end(self):
        print("epoch accuracy: ", self.acc_score.item())
        print("epoch loss: ", self.loss.item())

    def on_test_epoch_end(self):
        all_preds = torch.cat(self.test_predictions, dim=0)
        return all_preds

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)
 
