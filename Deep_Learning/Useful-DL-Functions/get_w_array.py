"""
Get weight array for the WeightSampler function

read_labels should be a function that from the dataset, when specified a class_type (class1, class2, class3 or all)
returns the amount of classes (int(class_count)) and a list of strings ["class1", "class2", "class3"]


get_ground_truth
inputs: labels_file (the .txt file with the label) and detect_class (class1, class2, class3 or all)
outputs: class of the label_file
"""
def get_w_array(dataset, class_type):

    # get all labels
    lab, class_count = read_labels(dataset, class_type)
    if class_count == 1:
        labels = []
        labels.append(lab)
    else:
        labels = lab
    # -----------------------
    root = str(
        "path" + dataset + '/train')
    # Gets all the .jpg file directions 
    labels_dir = sorted(Path(root).glob("*.jpg"))  # noqa: E501
    # ------------------------------------------------
    # Go through all labels in path
    y = []
    for lab in labels_dir:
        label_path = (os.path.splitext(lab)[0] + '.txt')
        # get vector from label file
        label = get_ground_truth(labels_file=label_path, dataset=dataset, detect_class=class_type)  # noqa: E501
        # list with all labels of the complete dataset
        y.append(label)
    # ------------------------------------------------
    # array with all counted labels this is the damage count
    count_in_dataset = np.sum(y, axis=0)
    # account of no class presence (this is for multilabel)
    no_presence_count = np.count_nonzero(np.sum(y, axis=1) == 0)
    # if there is examples where there is no valid class
    if no_presence_count:
        # set all in one vector
        count_in_dataset = np.concatenate(
            (count_in_dataset, no_presence_count), axis=None)
        # append no_class to the labels
        labels.append('NO_CLASS')
    # Get weights
    inv_count_in_dataset = 1 / count_in_dataset
    weights = inv_count_in_dataset / np.sum(inv_count_in_dataset)
    if no_presence_count:
        w = y[:] * weights[:-1]
    else:
        w = y[:] * weights
    w = w.max(axis=1)
    w[w == 0] = weights.min(axis=None)

    return w
