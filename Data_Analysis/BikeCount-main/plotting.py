import seaborn as sns
import matplotlib.pyplot as plt


def plot_graphs(df):
    # plot for every day of the week for total users
    fig, ax = plt.subplots(figsize=(20, 10))
    ax = sns.pointplot(data=df, x='hr', y='cnt', hue='weekday')
    ax.set(title='Count of bikes during weekdays and weekends')
    ax.set_xlabel('Hour')
    ax.set_ylabel('Bike count')
    plt.savefig('images/weekVsWeekend.jpg')

    # plot for every day of the week for total users
    fig, ax = plt.subplots(figsize=(20, 10))
    ax = sns.pointplot(data=df, x='hr', y='casual', hue='weekday')
    ax.set(title='Count of bikes during weekdays and weekends: Unregistered users')
    ax.set_xlabel('Hour')
    ax.set_ylabel('Bike count - Casual')
    plt.savefig('images/weekVsWeekend_casual.jpg')

    # plot for every day of the week for total users
    fig, ax = plt.subplots(figsize=(20, 10))
    ax = sns.pointplot(data=df, x='hr', y='registered', hue='weekday')
    ax.set(title='Count of bikes during weekdays and weekends: Registered users')
    ax.set_xlabel('Hour')
    ax.set_ylabel('Bike count - Registered')
    plt.savefig('images/weekVsWeekend_registered.jpg')

    # Weather plot
    fig, ax = plt.subplots(figsize=(20, 10))
    ax = sns.pointplot(data=df, x='hr', y='cnt', hue='weathersit')
    ax.set(title='Bike_count_weather')
    ax.set_xlabel('Hour')
    ax.set_ylabel('Bike count')
    plt.savefig('images/Bike_count_weather.jpg')

    # Seasons plot
    fig, ax = plt.subplots(figsize=(20, 10))
    ax = sns.pointplot(data=df, x='hr', y='cnt', hue='season')
    ax.set(title='Bike_count_seasons')
    ax.set_xlabel('Hour')
    ax.set_ylabel('Bike count')
    plt.savefig('images/Bike_count_seasons.jpg')

    # weekday
    fig, ax = plt.subplots(figsize=(20, 10))
    ax = sns.barplot(data=df, x='weekday', y='cnt')
    ax.set(title='Bike_count_week')
    ax.set_xlabel('Day')
    ax.set_ylabel('Bike count')
    plt.savefig('images/Bike_count_week.jpg')

    # Plot REGRESSION temp vs count
    fig, ax = plt.subplots(figsize=(20, 10))
    ax = sns.regplot(data=df, x='temp', y='cnt')
    ax.set(title="Relation_temperature_and_users")
    ax.set_xlabel('Temperature')
    ax.set_ylabel('Bike count')
    plt.savefig('images/Relation_temp_and_users.jpg')
