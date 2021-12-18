# Dataloader
import numpy as np
from torchvision import transforms
from PIL import *
from torch.utils.data.dataset import Dataset
import os


def get_transform(im_size, split):
    if(split == 'train'):
        transforms_total = transforms.Compose([transforms.Resize((im_size, im_size)),
                                               transforms.ColorJitter(
                                              brightness=0.1, contrast=0.1, saturation=0.1, hue=0),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomAffine(5),
            transforms.RandomRotation(180),
            transforms.ToTensor(),
            # transforms.Lambda(lambda img: img * 2.0 - 1.0)
            transforms.Normalize([0.485, 0.456, 0.406], [
                0.229, 0.224, 0.225])
        ])

    else:
        # Transformations on inference
        transforms_total = transforms.Compose([transforms.Resize((im_size, im_size)),
                                               transforms.ToTensor(),
                                               # transforms.Lambda(lambda img: img * 2.0 - 1.0)
                                               transforms.Normalize([0.485, 0.456, 0.406], [
                                                   0.229, 0.224, 0.225])
                                               ])
    return transforms_total


class CustomDatasetFromCSV(Dataset):
    def __init__(self, df, transformations, folder, split):
        """
        Args:
            csv_path (string): path to csv file
            transformations: pytorch transforms for transforms and tensor conversion
            train: flag to determine if train or val set
        """
        # Transforms

        self.transforms = transformations
        # Second column is the photos
        self.image_arr = np.asarray(df.iloc[:, 0])

        # Second column is the labels
        if split == 'train':
            self.label_arr = np.asarray(df.iloc[:, 1:-1])
        else:
            self.label_arr = np.asarray(df.iloc[:, 1:])
        # Calculate len
        self.data_len = len(self.label_arr)

        # Init path to folder with photos
        self.folder = folder

    def __getitem__(self, index):

        # Get image name from the pandas Series
        single_image_name = self.image_arr[index]

        # Open image and convert to RGB (some dataset images are grayscale)
        img_as_img = Image.open(os.path.join(
            self.folder, single_image_name)).convert('RGB')

        # Use transforms
        if self.transforms is not None:
            img_as_tensor = self.transforms(img_as_img)

        # Get image labels from the pandas DataFrame
        single_image_label = self.label_arr[index]

        return (img_as_tensor, single_image_label)

    def __len__(self):
        return self.data_len
