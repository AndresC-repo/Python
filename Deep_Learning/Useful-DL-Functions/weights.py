"""
Counts the amount of class examples there are in the Dataset and orints a pie chart and the count as list.
Prints out a weight vector containing a weight value for each class that is relative to the class presence.
In a multi-label sense, where there can be examples which class is "no_class" it does not output a weight for it.

note: Weight is multiplied by a value of 10 for multi-class and 2 for binaryclassification this is just to bring the loss value higher.


read_labels should be a function that from the dataset, when specified a class_type (class1, class2, class3 or all)
returns the amount of classes (int(class_count)) and a list of strings ["class1", "class2", "class3"]


get_ground_truth
inputs: labels_file (the .txt file with the label) and detect_class (class1, class2, class3 or all)
outputs: class of the label_file
"""

import os
import numpy as np
from pathlib import Path
from argparse import ArgumentParser

import matplotlib.pyplot as plt


# ------------------------------------------------
#       Color scheme
# ------------------------------------------------
fh_gray = (199 / 255, 202 / 255, 204 / 255)
fh_deep_green = (23 / 255, 156 / 255, 125 / 255)
fh_shade_green = (109 / 255, 191 / 255, 169 / 255)
fh_soft_green = (180 / 255, 220 / 255, 211 / 255)
turple = (0, 128 / 255, 128 / 255)
magenta = (255 / 255, 0, 255 / 255)
purple = (128 / 255, 0, 128 / 255)
red = (255 / 255, 1 / 255, 1 / 255)
blue = (1 / 255, 255 / 255, 1 / 255)
black = (1 / 255, 1 / 255, 1 / 255)
colors_scheme = [fh_deep_green, fh_soft_green,
                 fh_gray, fh_shade_green, magenta, purple, turple]

# ------------------------------------------------
# Create Augmented images
def get_parser(parser):

    parser.add_argument("--split", default="train", type=str, choices=["train", "test", "val"],)  # noqa: E501

    parser.add_argument("--dataset", default="Dataset1", type=str, choices=["Dataset1", "Dataset2"],)  # noqa: E501

    parser.add_argument("--num_of_examples", default=1, type=int)  # noqa: E501

    parser.add_argument("--class_type", default="ALL", type=str, choices=["Class1", "Class2", "Class3"],)  # noqa: E501

    return parser
# ------------------------------------------------

def get_label_count(dataset, split, class_type):

    # get all labels
    lab, class_count = read_labels(dataset, class_type)
    if class_count == 1:
        labels = []
        labels.append(lab)
    else:
        labels = lab
    # -----------------------
    root = str(
        "path" + dataset + '/' + split)
    labels_dir = sorted(Path(root).glob("*.jpg"))  # noqa: E501
    # ------------------------------------------------
    # go through all labels in path
    y = []
    for lab in labels_dir:
        label_path = (os.path.splitext(lab)[0] + '.txt')
        # get vector from label file
        label = get_ground_truth(labels_file=label_path, dataset=dataset, detect_class=class_type)  # noqa: E501
        # list with all vectors
        y.append(label)
    # ------------------------------------------------
    # array with all counted labels. Array containing damages
    count_in_dataset = np.sum(y, axis=0)
    prov = np.sum(y, axis=1)
    # account of no_class (for multilabel)
    no_damage_count = np.count_nonzero(prov == 0)
    if no_damage_count:
        # set all in one vector
        count_in_dataset = np.concatenate(
            (count_in_dataset, no_damage_count), axis=None)
        # append no_class to the labels
        labels.append('NO_class')
        class_count += 1
    # create dictionary with 'Class': number
    dict_labels = {}
    for x in range(class_count):
        dict_labels[labels[x]] = count_in_dataset[x]
    print(f"Total amount of labels in dataset is:  {dict_labels}")
    # output a pie chart
    colors = colors_scheme
    fig1, ax1 = plt.subplots()
    ax1.pie(count_in_dataset, colors=colors, autopct='%1.1f%%', textprops={'fontsize': 12})  # noqa: E501
    ax1.legend(labels=labels, loc="best", fontsize=12)
    ax1.axis('equal')
    plt.show()
    # Get weights
    inv_count_in_dataset = 1 / count_in_dataset
    weights = inv_count_in_dataset / np.sum(inv_count_in_dataset)
    # I have tested with a bigger weight in order to get higher loss values
    mult = 10
    if(class_count == 1):
        mult = 2
    if no_damage_count:
        print(f"weights: {weights[:-1] * mult}")
    else:
        print(f"weights: {weights* mult}")
    return weights

# ------------------------------------------------
# ------------------------------------------------

if __name__ == "__main__":
    # ------------------------
    # ARGUMENTS
    # ------------------------
    # Parser for JustCommand
    parser = ArgumentParser()
    parser = get_parser(parser)
    param = parser.parse_args()
    # Set Dataset
    dataset = param.dataset
    split = param.split
    class_type = param.class_type

    print(f"DATASET: {dataset} split {split} and class_type {class_type}")
    # ------------------------
    # Weight
    # ------------------------
    get_label_count(dataset=dataset, split=split, class_type=class_type)
