# training and prediction
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error


def train_model(df):
    # Separate target from features
    X = df.drop(columns='cnt', axis=1)
    Y = df['cnt'].values
    # Split into train and test
    x_train, x_test, y_train, y_test = train_test_split(
        X, Y, test_size=0.20, random_state=0)
    # Build model
    LinReg = LinearRegression()
    LinReg.fit(x_train, y_train)
    y_pred = LinReg.predict(x_test)
    LinRegScore = LinReg.score(x_test, y_test)
    # print model score
    print(f'LinRegScore: {LinRegScore}')
    plot_preds(x_test, y_test, y_pred)


def plot_preds(x_test, y_test, y_pred):
    # Plots error for every data point with a black line in the error mean
    error = abs(y_test - y_pred)
    meanError = np.mean(error)
    Mad = meanError / len(error)
    fig, ax = plt.subplots()
    sns.scatterplot(x=y_test, y=error, ax=ax)
    sns.lineplot(x=y_test, y=np.mean(error), color='black', ax=ax)
    ax.set(title='Prediction_error_plot')
    ax.set_xlabel('Error')
    ax.set_ylabel('bike count')
    plt.savefig('images/Prediction_Error_plot.jpg')
    # Print error information
    print(f'MeanSqrdError: {np.sqrt(mean_squared_error(y_test, y_pred))}')
    print(f'MeanAbsError: {mean_squared_error(y_test, y_pred)}')
    print(f'boogie: {meanError}')

    print(f'MeanAbsolutionDeviation: {Mad}')

    # This means to show how on point the predictions are
    fig, ax = plt.subplots()
    sns.scatterplot(x=x_test['instant'], y=y_test,
                    ax=ax, color='#FFC20A')
    sns.scatterplot(x=x_test['instant'], y=y_pred,
                    ax=ax, marker='+', color='#0C7BDC', alpha=0.5)
    ax.set(title='Prediction_scatter')
    ax.set_xlabel('days')
    ax.set_ylabel('bike count')
    plt.savefig('images/Prediction_scatter.jpg')
