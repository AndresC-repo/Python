import pandas as pd
pd.plotting.register_matplotlib_converters()
import matplotlib.pyplot as plt
import seaborn as sns

# -------------------------------------------------- #
# Path of the file to read
insurance_filepath = "./insurance.csv"
# Read the file into a variable insurance_data
insurance_data = pd.read_csv(insurance_filepath)
# -------------------------------------------------- #
print(insurance_data.head())
print(insurance_data.tail())
# -------------------------------------------------- #
# Scatter plot and regression 
# -------------------------------------------------- #
# scatter plot
figure_1 = sns.scatterplot(x=insurance_data['bmi'], y=insurance_data['charges'])
plt.show()
# regression line
figure_2 = sns.regplot(x=insurance_data['bmi'], y=insurance_data['charges'])
plt.show()

# Color scatter plot
figure_3 = sns.scatterplot(x=insurance_data['bmi'], y=insurance_data['charges'], hue=insurance_data['smoker'])
plt.show()

# add two regression lines
figure_4 = sns.lmplot(x="bmi", y="charges", hue="smoker", data=insurance_data)
plt.show()

# feature a categorical variable (like "smoker")
figure_5 = sns.swarmplot(x=insurance_data['smoker'],
              y=insurance_data['charges'])
# on average, non-smokers are charged less than smoker
plt.show()


