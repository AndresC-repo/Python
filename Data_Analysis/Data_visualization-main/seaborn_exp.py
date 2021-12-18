# SETUP
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# -------------------------------------------------- #
# Seaborn basic usage
# -------------------------------------------------- #
# Load the data

# Path of the file to read
spotify_filepath = "../input/spotify.csv"
spotify_data = pd.read_csv(spotify_filepath, index_col="Date", parse_dates=True
# -------------------------------------------------- #
#  Read first 5 rows // read last 5 
spotify_data.head()
spotify_data.tail()
# -------------------------------------------------- #

# Line chart
sns.lineplot(data=spotify_data)
# Set the width and height of the figure
plt.figure(figsize=(14,6))
# Add title
plt.title("Daily Global Streams of Popular Songs in 2017-2018")
plt.show()
# -------------------------------------------------- #
# Plot a subset of the data
list(spotify_data.columns)
# -------------------------------------------------- #
# Set the width and height of the figure
plt.figure(figsize=(14,6))

# Add title
plt.title("Daily Global Streams of Popular Songs in 2017-2018")

# Line chart showing daily global streams of 'Shape of You'
sns.lineplot(data=spotify_data['Shape of You'], label="Shape of You")

# Line chart showing daily global streams of 'Despacito'
sns.lineplot(data=spotify_data['Despacito'], label="Despacito")

# Add label for horizontal axis
plt.xlabel("Date")
