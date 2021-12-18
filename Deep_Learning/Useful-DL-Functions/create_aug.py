"""
Use this if you do not want to use Data augmentation on the run but to create an augmented dataset.

Creates a certain amount of augmented images of a certain class
images created have a prefix "augmented" on their name

Images created can also be deleted using this file with the --delete_aug argument

Add the necessary augmentations for the specific project
Also update the dataset path
"""

import os
import random
import numpy as np
from pathlib import Path
from argparse import ArgumentParser
from PIL import Image
from albumentations import HorizontalFlip, VerticalFlip, Flip, Blur, Cutout, Compose


# Create Augmented images
def get_parser(parser):

    parser.add_argument("--dataset", default="dataset1", type=str,
                        choices=["dataset1", "dataset2"],)

    parser.add_argument("--aug_class", default="class1", type=str,
                        choices=["class1", "class2", "class3"],)

    parser.add_argument("--num_create_img", default=0, type=int)  # noqa: E501

    parser.add_argument('--delete_aug', default=False, action='store_true')

    return parser

# ---------------------------------------------------------------------------------------- #
#                           Augmentation                                                   #
# ---------------------------------------------------------------------------------------- #


def al_VerticalFlip():
    transformations = [
        VerticalFlip(always_apply=False, p=1.0),
    ]
    return transformations


def al_horizontal_flip():
    transformations = [
        HorizontalFlip(always_apply=False, p=1.0),
    ]
    return transformations


def al_Flip():
    transformations = [
        Flip(always_apply=False, p=1.0),
    ]
    return transformations


def al_Blur():
    transformations = [
        Blur(always_apply=False, p=1.0, blur_limit=(3, 6)),
    ]
    return transformations


def al_Cutout():
    transformations = [
        Cutout(always_apply=False, p=1.0, num_holes=100,
               max_h_size=5, max_w_size=5)
    ]
    return transformations
# ---------------------------------------------------------------------------------------- #
#                 Function
# ---------------------------------------------------------------------------------------- #


def Create_Augumented_Images(num_create_img, aug_class, dataset):
    # Normal Augmentation is done while on the run (Dataset)
    # our folder path containing some images
    folder_path = str(
        "paht/" + dataset + "/train")
    # the number of file to generate
    num_files_desired = num_create_img

    labels_dir = sorted(Path(folder_path).glob("*.txt"))

    # loop on all files of the folder and build a list of files paths
    labels = []
    for label in labels_dir:
        f = open(label, "r")
        for line in f.read().splitlines():
            if line.upper() == aug_class.upper():
                labels.append(label)

    # dictionary of the transformations functions we defined earlier
    # add here the new transformations <------- HERE update 
    available_transformations = {
        # 'VerticalFlip': al_VerticalFlip,
        'horizontal_flip': al_horizontal_flip,
        'Flip': al_Flip,
        # 'Blur': al_Blur,
        # 'Cutout': al_Cutout,
    }

    num_generated_files = 0
    while num_generated_files < num_files_desired:
        # random image from the folder
        label_path = random.choice(labels)
        # get same label as the image
        image_path = (os.path.splitext(label_path)[0] + '.jpg')
        # print(image_path)
        # read image as an two dimensional array of pixels
        image_to_transform = Image.open(image_path).convert("L")
        image_to_transform = np.array(image_to_transform)
        # read the label
        fo = open(label_path, "r")
        file_contents = fo.read()
        # random num of transformations to apply
        num_transformations_to_apply = random.randint(
            1, len(available_transformations))

        num_transformations = 0
        # transformed_image = None
        transforms = None
        # prevent performing double equal transormation
        old_key = None
        while num_transformations <= num_transformations_to_apply:
            # choose a random transformation to apply for a single image
            key = random.choice(list(available_transformations))
            if key != old_key:
                if transforms is not None:
                    transforms = transforms + available_transformations[key]()
                    trans = True
                else:
                    transforms = available_transformations[key]()
                    trans = True
            num_transformations += 1
            old_key = key
        if trans:
            transforms = Compose(transforms)
            # dict_keys(['image])
            transformed = transforms(image=image_to_transform)
            image = transformed["image"]
            # define a name for our new file: augemnted_image_number starting from 1
            new_file_path = '%s/augmented_image_%s.jpg' % (
                folder_path, num_generated_files + 1)
            new_file_label = '%s/augmented_image_%s.txt' % (
                folder_path, num_generated_files + 1)
            # write label
            with open(new_file_label, 'w') as the_file:
                the_file.write(file_contents)
            # save image in folder
            image = Image.fromarray(image)
            image.save(new_file_path)
            # marked as created
            num_generated_files += 1
    print(f"created {num_generated_files} in folder {folder_path}")

# ---------------------------------------------------------------------------------------- #


def Delete_augmented(dataset):
    # replace "path" to the one poitning to the dataset
    folder_path = str(
        "path/" + dataset + "/train")
    labels_dir = sorted(Path(folder_path).glob("*.txt"))
    labels = []
    del_count = 0
    for label in labels_dir:
        labels.append(label)
    for label in labels:
        if os.path.basename(label).startswith("augmented"):
            os.remove(Path(str(label)))
            os.remove(Path((os.path.splitext(label)[0] + '.jpg')))  # noqa: E501
            del_count = del_count + 1
    return del_count

# ---------------------------------------------------------------------------------------- #
#                               __main__                                                   #
# ---------------------------------------------------------------------------------------- #


if __name__ == "__main__":

    # Parser for JustCommand
    parser = ArgumentParser()
    parser = get_parser(parser)
    param = parser.parse_args()
    dataset = param.dataset
    delete_aug = param.delete_aug

    aug_class = param.aug_class

    num_create_img = param.num_create_img

    # Function Call
    if (delete_aug):
        del_count = Delete_augmented(dataset=dataset)
        print(
            f'{del_count} of the augmented images were deleted from dataset: {dataset}')
    else:
        Create_Augumented_Images(
            num_create_img, aug_class=aug_class, dataset=dataset)
