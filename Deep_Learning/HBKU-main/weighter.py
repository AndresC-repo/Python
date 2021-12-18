# Weights
# Adds a weight column to the end of the df
# Weight col has a weight value proportional to the class count
import torch
from torch.utils.data import WeightedRandomSampler


def get_weigther(df):
    weight = df.iloc[:, 1:-1].sum()
    # long of weight: should be equal to num of classes, len(weight))

    def make_weight(x, weight):
        # Return the min weight value of the present classes
        return min(weight[x[1:-1] == 1])
    # get min weight value on the train.weight column
    df.weight = df.apply(make_weight, args=(weight,), axis=1)
    # normalize weight
    df.weight = sum(weight) / df.weight

    class_weights_train = torch.tensor(df.weight.values)

    weighted_sampler_train = WeightedRandomSampler(
        weights=class_weights_train,
        num_samples=len(class_weights_train),
        replacement=True
    )
    return weighted_sampler_train
