# ----------------------------------------------------------------------- #
# Metrics
# ----------------------------------------------------------------------- #

import numpy as np
import ml_metrics as metrics

# Function for calculating MEAN Average Precision(MAP) score


def calc_map(preds, labels):
    preds = np.around(preds.cpu().detach().numpy())
    labels = labels.cpu().detach().numpy()
    pred = []
    for i in preds:
        # for each batch
        cats = np.nonzero(list(i))[0]  # Get the index of the non zeros
        pred.append(list(cats))  # transform into list
    label = []
    for i in labels:
        cats = np.nonzero(list(i))[0]
        label.append(list(cats))
    return metrics.mapk(label, pred)  # compare idxs of every batch as lists
