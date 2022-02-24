# lorenzo
import pandas as pd
import pdb
import numpy as np

# filename = "lore.csv"
def create_csv():
	filename = "input.txt"
	df = pd.read_csv(filename, sep='\t', lineterminator='\n', decimal=",")
	df['Date']= pd.to_datetime(df["Date\r"])
	df["hour"] = df["Date"].dt.hour
	df["day"] = df["Date"].dt.day
	df["month"] = df["Date"].dt.month
	df.pop("Date\r")
	df2 = df.groupby(['month', 'day', 'hour']).mean()
	df2.to_csv("output1_not_filled.csv")

def check_missing_hr():
	filename = "output1_not_filled.csv"
	df = pd.read_csv(filename)
	penis_count =0
	print(df.shape)
	df2 = df.copy()
	mth_dic = {'jan':31, 'feb':28, 'mar':31,'apr':30, 'may':31, 'jun':30, 'jul':31, 'ag':31,'sep':30,'oct':31,'nov':30, 'dic':31}
	for month in range(1, 13):
		x = list(mth_dic.values())[month-1]
		for day in range(1, x+1):
			for hour in range (24):
				if df.loc[(df['month'] == month) & (df['day'] == day) & (df['hour']==hour)].empty:
					# line = pd.DataFrame({"month": month, "day": day, "hour": hour, "Blocks":0, "Tiles":0,"Grinders":0}, index=[-1])
					line = pd.DataFrame({"month": month, "day": day, "hour": hour, "Grinders":0}, index=[-1])
					df2 = df2.append(line)
	df2 = df2.groupby(['month', 'day', 'hour']).mean()
	df2.to_csv("output2_full_year.csv")
	print(df2.shape)



if __name__ == '__main__':
	create_csv()
	check_missing_hr()
	# df['hour'].value_counts().sum()  # 8711  from 8760  -> 49