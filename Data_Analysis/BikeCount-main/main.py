# main code
import os

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from training import train_model
from plotting import plot_graphs

# ------------------------------------------------------------------------------- #
# This file is the main code executer
# Gives overview about the dataframe
# Calls to plotting file
# Calls to training model file
# Created images can be found in ./'DIRNAME' folder
# ------------------------------------------------------------------------------- #
CSVFILE = 'csv_files/hour.csv'
DIRNAME = 'images'


def main(csvFile):
    # Create image dir to store plots and images
    try:
        os.mkdir(DIRNAME)
    except FileExistsError:
        pass

    df = pd.read_csv(csvFile)  # read file

    # Uncomment following lines to get a better overview of the dataframe
    # print(f'df.head() \n {df.head()}')  # print first five lines
    # print(f'df.describe() \n {df.describe()}')  # look for max&min values and std

    # overall info, look for null values
    print(f'df.info() \n {df.info(verbose=True)}')

    # Check data correlation
    corr = df.corr()
    plt.figure(figsize=(15, 10))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True,
                annot_kws={'size': 8}, square=True)
    # Save heatmap
    plt.savefig('images/Heatmap.jpg')
    # or print specific correlations in terminal: here temp and atemp are strongly correlated
    print(
        f' descending correlation: \n {corr["temp"].sort_values(ascending=False)}')
    # From correlation it can be seen which variables play a bigger factor when bike renting
    # Drop useless columns
    df = df.drop(columns=['dteday', 'yr',
                          'atemp', 'windspeed', 'mnth'])
    plot_graphs(df)
    train_model(df)


if __name__ == "__main__":
    main(CSVFILE)
