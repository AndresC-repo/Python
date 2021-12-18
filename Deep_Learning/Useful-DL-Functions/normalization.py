# ------------------------------------------------
#            Normalization
# ------------------------------------------------


def get_normalization(train_dataset):
    """
    This function calculates the normalization values for a given dataset:
    channel-wise means and channel-wise stds.
    :param train_dataset: the training set of a specific dataset
    :return: channelwise means array and channelwise stds array
    """
    means = []
    stds = []
    for d, l in train_dataset:
        d = d.permute(1, 2, 0)
        means.append(d.float().mean(axis=(0, 1)))
        stds.append(d.float().std(axis=(0, 1)))

    def mean(x):
        x = [tensor.item() for tensor in x]
        return sum(x) / len(x) / 256  # Convert from [0, 255] to [0, 1]

    # [[c1, c2, c3], [c1, c2, c3] ...]
    # =>
    # [(c1, c1, c1, ...), (c2, c2, c2, ...)]

    mean_means = [mean(channel_means) for channel_means in mi.unzip(means)]
    mean_stds = [mean(channel_stds) for channel_stds in mi.unzip(stds)]

    return mean_means, mean_stds