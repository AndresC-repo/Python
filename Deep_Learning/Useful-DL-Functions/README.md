# Useful-DL-Functions
Compilation of useful tools for deep learning projects. A little bit beyond simple neural networks.

These are all functions used on a bigger Multi-label project and as such, reusing them for other projects imply to modify them as necessary.
They take on argparse arguments to work. Do not work out of the box but with some little modification.
But might be helpful for inspiration.

Normalization, weighting and process of data into folders require to manually input the Dataset path and labels.

Everything related to the Dataset is in the form "path/DatasetName/split" (where the spit is "train", "val" or "test").
It is assumed that each folders contains an image and a .txt file with the same name containing the class.

It works for Multi-label purposes as for normal multi-class projects.

# Inlcudes

### Create_aug.py
Specify one or multiple class and a number of images.
It creates that amount of images using Albumentations for each selected class (a .txt file with the same name and class is also created)
Note: Personally modify which Aumgentation should be use, as for 1-channel images it does not make sense to use channel related augmentation.

### downsample_data
Specify one or multiple class and a number of images.
Randomly remove specified class images and related labels files only leaving the selected amount.

### normalization.py
Get the normalization values 
Specify Dataset path

### process_data.py
This will arrange a folder that contains all the images and labels into folders as in "train", "val", "test" containing a selected proportion (0.8, 0.1, 0.1 are the default values).
Removes images or label files without the corresponding other (ex: input1.jpg without input1.txt would be removed).
Removes images and corresponding label files if they do not satisfy an specified size.

### weights.py
Counts the examples of each class and outputs an array of their weight as to be used for Weighted loss functions
Outputs the count and prints a Pie chart of the label distribution.

### get_w_array.py
Returns an array of the len of the train set to be containing the weights of each example.
To be used in the WeightedRandomSampler

### metrics.py
Creates a metrics class which can output: accuracy, recall, precision, f1 and  hamming distance. 
Specially useful for multi-label purposes as it assumes the non-presence of other classes as another class for which it can also output its metrics. 

## TODO:
- Making it more general and not so project specific
- Clean and properly comment

