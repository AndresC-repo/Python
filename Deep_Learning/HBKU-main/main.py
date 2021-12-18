import os

import torch
import torch.nn as nn
import torchvision
from torch.utils.data import DataLoader
from dataloader import CustomDatasetFromCSV, get_transform

import pytorch_lightning as pl
from pytorch_lightning.core import LightningModule
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

import pandas as pd

from optim import get_optim
from model import initialize_model
from metrics import calc_map
from weighter import get_weigther
from loss import get_lossfunction

from argparse import ArgumentParser

from trainer_lr import get_lr_trainer
# ----------------------------------------------------------------------- #
SEED = 42
torch.manual_seed(SEED)
# ----------------------------------------------------------------------- #
def get_parser(parser):

    parser.add_argument("--loss", default="asymm", type=str, choices=["asymm", "focal", "BCE"],)  # noqa: E501

    return parser


class MyLightingModule(LightningModule):
    def __init__(self, loss, df_train, df_val, num_classes, feature_extract, im_size, IMG_DIR_TRAIN, IMG_DIR_TEST, batch_size, lr):
        super().__init__()
        # Hyperparameters loaded from flags
        self.losstype = loss
        self.df_train = df_train
        self.df_val = df_val
        self.im_size = im_size
        self.num_classes = num_classes

        self.IMG_DIR_TRAIN = IMG_DIR_TRAIN
        self.IMG_DIR_TEST = IMG_DIR_TEST
        # loss
        self.loss_train = get_lossfunction(self.losstype)
        self.loss_val = get_lossfunction(self.losstype)
        # metrics
        self.sigmoid = nn.Sigmoid()
        self.batch_size = batch_size
        self.feature_extract = feature_extract
        # Learning rate
        self.lr = lr
        # self.config
        self.model = initialize_model(
            'efficientnet', self.num_classes, self.feature_extract, use_pretrained=True)

        self.save_hyperparameters()

    # ---------------------------------------------------------------------------------------------- #

    def forward(self, x):

        return self.model(x)

    # ---------------------------------------------------------------------------------------------- #

    def training_step(self, batch, batch_idx):
        images, labels = batch
        # Model output
        model_outputs = self(images)
        # Loss
        train_loss = self.loss_train(model_outputs, labels)

        return {"loss": train_loss, 'preds': model_outputs, 'labels': labels}

    def validation_step(self, batch, batch_idx):
        images, labels = batch
        # Model output
        model_outputs = self(images)
        # Loss
        validation_loss = self.loss_val(model_outputs, labels)
        # Get accuracy
        acc = calc_map(self.sigmoid(model_outputs), labels)

        return {'val_loss': validation_loss, 'val_acc': acc}

    # ---------------------------------------------------------------------------------------------- #

    def training_epoch_end(self, outputs):
        # Loss
        train_loss_epoch = torch.stack([x["loss"] for x in outputs]).mean()  # noqa: E501

        self.log("train_loss", train_loss_epoch)

    def validation_epoch_end(self, outputs):
        # Loss
        val_loss_epoch = torch.stack([x["val_loss"] for x in outputs]).mean()  # noqa: E501
        # Accuracy
        val_acc_epoch = torch.stack([x["val_acc"] for x in outputs]).mean()  # noqa: E501

        self.log("val_loss", val_loss_epoch)
        self.log("val_acc", val_acc_epoch)

    # ---------------------------------------------------------------------------------------------- #
    # TRAINING SETUP
    # ---------------------------------------------------------------------------------------------- #

    def configure_optimizers(self):
        """
        Return optimizers and learning rate schedulers
        """
        optimizer, scheduler = get_optim(self.model, self.lr)

        return [optimizer], [scheduler]

    # ---------------------------------------------------------------------------------------------- #
    # DATA
    # ---------------------------------------------------------------------------------------------- #
    # Create a weight vector summing down for each class

    def prepare_data(self):
        transform_train = get_transform(im_size=self.im_size, split='train')
        transform_val = get_transform(im_size=self.im_size, split='val')

        self.train_data = CustomDatasetFromCSV(
            self.df_train, transform_train, self.IMG_DIR_TRAIN, split='train')
        self.val_data = CustomDatasetFromCSV(
            self.df_val, transform_val, self.IMG_DIR_TRAIN, split='val')

    # Prepare valid input shape according to the selected model

    def train_dataloader(self):
        weighted_sampler_train = get_weigther(self.df_train)
        return DataLoader(dataset=self.train_data, batch_size=self.batch_size, sampler=weighted_sampler_train,
                          num_workers=4, shuffle=False)

    def val_dataloader(self):
        return DataLoader(dataset=self.val_data, batch_size=self.batch_size,
                          num_workers=4, shuffle=False)


def main_execute(loss, amounts=None):
    """
    Main training routine specific for this project
    : param hparams:
    """    # ------------------------
    # 0. Create paths
    # ------------------------
    # paths
    CKP_DIR = "./training"
    DATA_DIR = './labels'
    IMG_DIR_TRAIN = './imgs/train/'
    IMG_DIR_TEST = './imgs/test/'
    data_csv = os.path.join(DATA_DIR, 'labels_train.csv')
    os.makedirs(CKP_DIR, exist_ok=True)
    # Cats = labels dataframe -> list with labels
    cats = pd.read_csv(os.path.join(DATA_DIR, "categories.csv"), header=None)
    cats = list(cats[0])
    # print(f' labels: {cats}')
    # ----------------------------------------------------------------------- #
    # Read features in data dataframe
    data = pd.read_csv(data_csv,
                       header=None, error_bad_lines=False)

    # print(f'Data  with name[0] and features:[1:] \n {data.head(3)}')
    # ----------------------------------------------------------------------- #
    # divide in  train set and val
    df_train = data.sample(frac=0.9, random_state=42)

    # drop all values found in train -> do this by index
    df_val = data.drop(df_train.index)
    df_train["weight"] = 1

# ------------------------
# 1. HYPERPARAMETERS
# ------------------------
    max_epochs = 250
    check_val_every_n_epoch = 5
    precision = 32
    if loss == 'BCE':
        batch_size = 256
        lr = lr = 0.001584893192461114
    elif loss == 'focal':
        batch_size = 256  # Found with trainer.tune(model)
        lr = 0.0009120108393559097   # Found with lr_finder
    elif loss == 'asymm':
        batch_size = 128
        lr = 1e-08
    num_classes = 80
    feature_extract = True
    im_size = 224

    # ------------------------
    # 2. INIT LIGHTNING MODEL
    # ------------------------
    model = MyLightingModule(
        loss, df_train, df_val, num_classes, feature_extract, im_size, IMG_DIR_TRAIN, IMG_DIR_TEST, batch_size, lr)
    model.train()
    # ------------------------
    # 3. DEFINE CALLBACKS
    # ------------------------
    early_stop_callback = EarlyStopping(
        monitor='val_acc',
        min_delta=0.00,
        patience=15,
        verbose=False,
        mode='max'
    )
    checkpoint_callback = ModelCheckpoint(
        monitor='val_acc',
        save_weights_only=True,
        filename='{epoch}-{val_loss:.2f}-{val_acc:.2f}',
        save_top_k=3,
        mode='max',
    )    # ------------------------
    # 4. DEFINE LOGGER
    # ------------------------
    logger = TensorBoardLogger(save_dir='training/')
    # ------------------------
    # 5. INIT TRAINER
    # ------------------------
    trainer = pl.Trainer(
        fast_dev_run=False,
        overfit_batches=False,
        logger=logger,
        max_epochs=max_epochs,
        gpus=1,
        distributed_backend='dp',
        precision=precision,
        check_val_every_n_epoch=check_val_every_n_epoch,
        callbacks=[checkpoint_callback, early_stop_callback],
        limit_train_batches=amounts / 100 if amounts is not None else 1.0,
        auto_lr_find=False,
        auto_scale_batch_size=True,
        profiler="simple",
    )
    # Uncomment to find BS and LR
    # trainer.tune(model)
    # get_lr_trainer(model, trainer, loss)
    # ------------------------
    # 5. START TRAINING
    # ------------------------
    trainer.fit(model)
    # ------------------------
    # 6. START TESTING
    # ------------------------
    # test loop is only called when .test() is used:
    # trainer.test(model)


# -------------------------------------------------------------------------------------------------------- #
# -------------------------------------------------------------------------------------------------------- #
if __name__ == "__main__":
    # ------------------------
    # ARGUMENTS
    # ------------------------
    print("PyTorch Version: ", torch.__version__)
    print("Torchvision Version: ", torchvision.__version__)

    parser = ArgumentParser()
    parser = get_parser(parser)
    param = parser.parse_args()
    loss = param.loss
    # tensorboard --logdir "./training/"
    main_execute(loss)
