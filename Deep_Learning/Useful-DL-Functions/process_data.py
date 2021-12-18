"""
Arrange a folder that contains all the images and labels into folders as in "train", "val", "test"
containing a selected proportion (0.8, 0.1, 0.1 are the default values). 

Removes images or label files without the corresponding other (ex: input1.jpg without input1.txt would be removed).
Removes images and corresponding label files if they do not satisfy an specified size.


"""
import numpy as np
import os
import random
from PIL import Image
from argparse import ArgumentParser

SEED = 2334
np.random.seed(SEED)

def get_parser(parser):

    parser.add_argument("--dataset", default="dataset1", type=str,
                        choices=["dataset1", "dataset2"],)

    parser.add_argument("--data_percent_val", default=0.1, type=float,)  # noqa: E501
    parser.add_argument("--data_percent_test", default=0.1, type=float)  # noqa: E501

    return parser


def process_data(dataset, data_percent_val=0.1, data_percent_test=0.1, size=(224,224):
    # labels and images are in the same directory and the labels are txt files.
    # set directories
    # Replace path to where the Dataset is
    source_directory = str("path/" + dataset)  # noqa: E501

    train_directory = str("path/" + dataset + "/train")  # noqa: E501
    val_directory = str("path/" + dataset + "/val")  # noqa: E501
    test_directory = str("path/" + dataset + "/test")  # noqa: E501

    # Create folders
    folders = ['train', 'val', 'test']
    for folder in folders:
        if not os.path.exists(os.path.join(source_directory, folder)):
            os.mkdir(os.path.join(source_directory, folder))
            if(os.path.isdir(os.path.join(source_directory, folder))):
                print(
                    f"created folders: {os.path.join(source_directory, folder)}")

    # list all files in dir that are an image
    files = [f for f in os.listdir(source_directory) if f.endswith('.jpg')]

    for file in files:
        im = Image.open(source_directory + '/' + file)
        width, height = im.size
        if (height < size[0] & width < size[1]):
            print("removed file due to size:", file)
            os.remove(source_directory + '/' + file)
            os.remove(source_directory + '/' + (os.path.splitext(file)[0] + '.txt'))  # noqa: E501

    # -------------------- Remove txt without jpg and viceversa -------------------

    files = [f for f in os.listdir(source_directory) if f.endswith('.jpg')]

    for file in files:
        if not ((os.path.isfile(source_directory + '/' + (os.path.splitext(file)[0] + '.txt')))
                and (os.path.isfile(source_directory + '/' + file))):
            try:
                os.remove(source_directory + '/' + file)
                print("removed file due to lack of label file:", file)
            except:
                os.remove(source_directory + '/' + (os.path.splitext(file)[0] + '.txt'))  # noqa: E501
                print("removed file due to lack of image file:", file)

    # -------------------- For Val ------------------------------------------------
    files = [f for f in os.listdir(source_directory) if f.endswith('.jpg')]
    equal_split = int(len(files))
    # select a percent of the files randomly
    random_files = random.sample(files, int(equal_split * data_percent_val))

    # move the randomly selected images by renaming directory
    for random_file_name in random_files:
        os.rename(source_directory + '/' + random_file_name,
                  val_directory + '/' + random_file_name)
        continue

    # move the relevant labels for the randomly selected images
    for image_labels in random_files:
        # strip extension and add .txt to find corellating label file then rename directory.
        os.rename(source_directory + '/' + (os.path.splitext(image_labels)
                                            [0] + '.txt'), val_directory + '/' + (os.path.splitext(image_labels)[0] + '.txt'))
        continue
    print(
        f"total examples in Validation folder {int(equal_split * data_percent_val)}")
    # -------------------- For TEST ------------------------------------------------
    # data_percent_test = float(data_percent_test)
    files = [f for f in os.listdir(source_directory) if f.endswith('.jpg')]
    random_files = random.sample(files, int(equal_split * data_percent_test))
    for random_file_name in random_files:
        os.rename(source_directory + '/' + random_file_name,
                  test_directory + '/' + random_file_name)
        continue

    for image_labels in random_files:
        os.rename(source_directory + '/' + (os.path.splitext(image_labels)
                                            [0] + '.txt'), test_directory + '/' + (os.path.splitext(image_labels)[0] + '.txt'))
        continue
    print(
        f"total examples in Test folder {int(equal_split * data_percent_test)}")

    # -------------------- For Train ------------------------------------------------
    # data_percent_test = float(data_percent_test)
    files = [f for f in os.listdir(source_directory) if f.endswith('.jpg')]
    # random_files = random.sample(files, int(len(files) * data_percent_test))
    for file in files:
        os.rename(source_directory + '/' + file,
                  train_directory + '/' + file)
        continue

    for file in files:
        os.rename(source_directory + '/' + (os.path.splitext(file)
                                            [0] + '.txt'), train_directory + '/' + (os.path.splitext(file)[0] + '.txt'))
        continue
    print(f"total examples in train folder: {len(files)}")


if __name__ == "__main__":
    # ------------------------
    # ARGUMENTS
    # ------------------------
    parser = ArgumentParser()
    parser = get_parser(parser)
    param = parser.parse_args()
    dataset = param.dataset
    data_percent_val = param.data_percent_val
    data_percent_test = param.data_percent_test

    process_data(dataset=dataset, data_percent_val=data_percent_val,
                 data_percent_test=data_percent_test)
