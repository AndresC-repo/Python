import pandas as pd
pd.plotting.register_matplotlib_converters()
import matplotlib.pyplot as plt

import seaborn as sns
print("Setup Complete")

# ----------------------------------------------- #
# Path of the file to read
iris_filepath = "./iris.csv"
# ----------------------------------------------- #

# Read the file into a variable iris_data
iris = pd.read_csv(iris_filepath, index_col="Id")

# Print the first 5 rows of the data
print(iris.head())
# ----------------------------------------------- #
# Simple 
# ----------------------------------------------- #
# Histogram 
sns.histplot(a=iris["PetalLengthCm"], kde=True)
plt.show()

# ----------------------------------------------- #
# Density plots
# Kernel Density Estimate plot  (smoothed histogram)
sns.kdeplot(data=iris["PetalLengthCm"], shade=True)
plt.show()

# ----------------------------------------------- #
# 2D KDE plots
sns.jointplot(x=iris["PetalLengthCm"], y=iris["SepalWidthCm"], kind="kde")
plt.show()


# ----------------------------------------------- #
# load data
# ----------------------------------------------- #

# Color-coded plots
# Paths of the files to read
iris_setosa=iris.loc[iris["Species"]=="Iris-setosa"]
iris_virginica=iris.loc[iris["Species"]=="Iris-virginica"]
iris_versicolor=iris.loc[iris["Species"]=="Iris-versicolor"]

# Print the first 5 rows of the Iris versicolor data
print(iris_setosa.head())
# ----------------------------------------------- #
# HISTO multi
# ----------------------------------------------- #
# Histograms for each species
sns.histplot(a=iris_setosa["PetalLengthCm"], label="Iris-setosa", kde=False)
sns.histplot(a=iris_virginica["PetalLengthCm"], label="Iris-versicolor", kde=False)
sns.histplot(a=iris_versicolor["PetalLengthCm"], label="Iris-virginica", kde=False)

# Add title
plt.title("Histogram of Petal Lengths, by Species")

# Force legend to appear
plt.legend()
plt.show()
# ----------------------------------------------- #
# KDE multiple 
# ----------------------------------------------- #

# KDE plots for each species
sns.kdeplot(data=iris_setosa["PetalLengthCm"], label="Iris-setosa", shade=True)
sns.kdeplot(data=iris_virginica["PetalLengthCm"], label="Iris-versicolor", shade=True)
sns.kdeplot(data=iris_versicolor["PetalLengthCm"], label="Iris-virginica", shade=True)

# Add title
plt.title("Distribution of Petal Lengths, by Species")
plt.show()