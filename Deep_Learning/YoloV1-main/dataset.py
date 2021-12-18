import torch
import os
import pandas as pd
from PIL import Image

from albumentations.pytorch import ToTensorV2
import albumentations as A
import numpy as np


def convert_to_tensor():
    transforms = [ToTensorV2(), ]
    return transforms


def resize(input_shape):
    transforms = [A.Resize(always_apply=False, p=1.0,
                           height=input_shape, width=input_shape, interpolation=0,), ]
    return transforms


def normalize():
    transformations = [
        A.Normalize(
            mean=[0.4722914520168194], std=[0.09468680026259756],
        ),
        A.ToFloat(max_value=1),
    ]
    return transformations


class VOCDataset(torch.utils.data.Dataset):
    def __init__(
        self, csv_file, img_dir, label_dir, input_shape=448, S=7, B=2, C=20, transform=True,
    ):
        self.csv_file = pd.read_csv(csv_file, header=None)
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.transform = transform
        self.input_shape = input_shape
        self.S = S
        self.B = B
        self.C = C

    def __len__(self):
        return len(self.csv_file)

    def __getitem__(self, index):
        # Gets label path from csv file
        label_path = os.path.join(
            self.label_dir, self.csv_file.iloc[index, 1])

        # One box for each label in the text file
        BoundingBoxes = []
        with open(label_path) as f:
            for label in f.readlines():
                # numbers are input as Str so conversion to float necessary
                class_label, x, y, width, height = [
                    float(x) for x in label.replace("\n", "").split()]
                BoundingBoxes.append([class_label, x, y, width, height])

        # Get the image from csv file
        img_path = os.path.join(self.img_dir, self.csv_file.iloc[index, 0])

        image = Image.open(img_path).convert("RGB")
        image = np.array(image)

        if self.transform:
            transforms = resize(self.input_shape)
            transforms = transforms + normalize() + convert_to_tensor()
            transforms = A.Compose(transforms)
            # dict_keys(['image', 'mask'])
            transformed = transforms(image=image)
            image = transformed["image"]

        # Get cell information
        label_full = torch.zeros((self.S, self.S, self.C + 5 * self.B))
        BoundingBoxes = torch.tensor(BoundingBoxes)
        for box in BoundingBoxes:
            class_label, x, y, width, height = box.tolist()
            class_label = int(class_label)

            # i,j represents the cell row and cell column
            i, j = int(self.S * y), int(self.S * x)
            # Decimal part is the position in relation to the cell
            x_cell, y_cell = (self.S * x) % 1, (self.S * y) % 1

            width_cell, height_cell = (
                width * self.S,
                height * self.S,
            )

            # row i, column  j
            if label_full[i, j, 20] == 0:
                # Set that there exists an object
                label_full[i, j, 20] = 1
                # Pc = 1
                label_full[i, j, class_label] = 1
                # Add BoundingBox coordinates
                label_full[i, j, 21:25] = torch.tensor(
                    [x_cell, y_cell, width_cell, height_cell]
                )

        return image, label_full


if __name__ == "__main__":
    # ------------------------
    # Test shape
    # ------------------------
    csv_file = "test.csv"
    img_dir = "./images"
    label_dir = "./labels"

    dataset = VOCDataset(csv_file, img_dir, label_dir)
    image, label = dataset.__getitem__(0)
    # 5 last are not used, just for matching the predictions with double bbox
    # label = [S, S, C+(5*B)]
    # image = [C, H, W]
    assert(label.shape == (7, 7, 30))
    assert(image.shape == (3, 448, 448))
