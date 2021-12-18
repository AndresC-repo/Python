"""
This file subsamples the dataset into a more balanced one.

"""
import os
from pathlib import Path
from argparse import ArgumentParser
import numpy as np

# Create Augmented images
def get_parser(parser):

    parser.add_argument("--split", default="train", type=str, choices=["train", "test", "val"],)  # noqa: E501

    parser.add_argument("--dataset", default="dataset1", type=str, choices=["dataset1", "dataset2"],)  # noqa: E501

    parser.add_argument("--num_of_examples", default=1, type=int)  # noqa: E501

    parser.add_argument("--class_type", default="ALL", type=str, choices=["class1", "class2", "class3", "ALL"],)  # noqa: E501

    return parser

# ---------------------------------------------------------------------------------------- #
#                           support functions                                                 #
# ---------------------------------------------------------------------------------------- #


def read_labels(dataset, detect_class='ALL'):
    """
    Input: dataset string
    Output: labels <- array with the classes
            class_count <- number of different classes in dataset
    """
    # ALL_LABELS_FILE = "./../dataset/" + dataset + "/labels.txt"
    # called with JUST Command
    if detect_class != 'ALL':
        labels = detect_class
        class_count = 1
    else:
        ALL_LABELS_FILE = "./datasets/" + dataset + "/labels.txt"
        with open(ALL_LABELS_FILE) as f:
            labels = f.read().splitlines()
        class_count = len(labels)

        if class_count == 0:
            print('No valid labels inside file {} that should contain all possible labels (= classes).'.format(
                ALL_LABELS_FILE))
            return -1
    return labels, class_count

# ---------------------------------------------------------------------------------------- #
#                           Reduce Function                                                #
# ---------------------------------------------------------------------------------------- #


def Reduce_dataset(dataset, split, num_of_examples, class_type='ALL'):
    erase_counter = 0
    # Replace path with your own pointing to the dataset
    folder_directory = str(
        "path/" + dataset + "/train")

    labels_dir = sorted(Path(folder_directory + '/').glob("*.txt"))

    if class_type == 'ALL':
        labels_tag, class_count = read_labels(dataset)
        # labels_tag [RISS, AUS...]
        labels_tag.append('NO_CLASS')
        class_count += 1
    else:
        labels_tag = np.array([class_type])
        class_count = 1
    # loop on all files of the folder and build a list of files paths
    labels = []
    for label in labels_dir:
        labels.append(label)

    # counters -> array
    counters = np.zeros(class_count)
    # number of examples
    for label in labels:
        #  Should the image be removed
        remove_flag = 0
        # read label and go through every line
        fo = open(label, "r")
        file_contents = fo.read()
        len_of_labels = len(file_contents.splitlines())
        for line in file_contents.split('\n'):
            # 1 is our default number-> do nothing
            if num_of_examples != 1:
                # line: label file  & labels_tag: array with labels
                for idx, lab in enumerate(labels_tag):
                    if line.upper() == lab:
                        # dmg counter
                        counters[idx] += 1
                        if counters[idx] >= num_of_examples:
                            remove_flag += 1

        if ((len_of_labels == 1 and remove_flag == 1) or (remove_flag == 2) or (class_count == 1 and remove_flag == 1)):
            # print(Path(str(label)))
            # print(Path((os.path.splitext(label)[0] + '.jpg')))  # noqa: E501
            os.remove(Path(str(label)))
            os.remove(Path((os.path.splitext(label)[0] + '.jpg')))  # noqa: E501
            erase_counter += 1
    print(
        f'Number of examples deleted: {erase_counter} for class_type: {class_type}')


# ---------------------------------------------------------------------------------------- #
#                               __main__                                                   #
# ---------------------------------------------------------------------------------------- #


if __name__ == "__main__":

    # Parser for JustCommand
    parser = ArgumentParser()
    parser = get_parser(parser)
    param = parser.parse_args()
    # Set Dataset
    dataset = param.dataset
    split = param.split
    # Balance Dataset
    num_of_examples = param.num_of_examples
    class_type = param.class_type
    # Function Calls
    Reduce_dataset(dataset=dataset, split=split, num_of_examples=num_of_examples, class_type=class_type)   # noqa: E501
