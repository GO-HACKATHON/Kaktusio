
import pandas as pd
import numpy as np

#import data from csv files
d = pd.read_csv('raw_data/demand.csv',  parse_dates=['TIME_STAMP'])
s = pd.read_csv('raw_data/supply.csv',  parse_dates=['TIME_STAMP'])


#give label for each files
d['LABEL'] = 'D'
s['LABEL'] = 'S'


#parse timestamp into week, day, hour, and minute
d['WEEK'] = d['TIME_STAMP'].dt.week
d['DAY'] = d['TIME_STAMP'].dt.dayofweek + 1
d['HOUR'] = d['TIME_STAMP'].dt.hour
d['MINUTE'] = d['TIME_STAMP'].dt.minute

s['WEEK'] = s['TIME_STAMP'].dt.week
s['DAY'] = s['TIME_STAMP'].dt.dayofweek + 1
s['HOUR'] = s['TIME_STAMP'].dt.hour
s['MINUTE'] = s['TIME_STAMP'].dt.minute

#determine minimum longitude and latitude
min_lon = min(d.LON.min(), s.LON.min())
min_lat = min(d.LAT.min(), s.LAT.min())

d['X'] = (d.LON-min_lon)/0.01
d['Y'] = (d.LAT-min_lat)/0.01

d['X'] = d['X'].astype(int)
d['Y'] = d['Y'].astype(int)

s['X'] = (s.LON-min_lon)/0.01
s['Y'] = (s.LAT-min_lat)/0.01

s['X'] = s['X'].astype(int)
s['Y'] = s['Y'].astype(int)


#combine demand and supply data
ds = pd.concat([d,s])


#create pivot table to bin data supply and demand per minute
pvt = ds.pivot_table(['TAXI_ID'], index=['WEEK', 'DAY', 'HOUR', 'MINUTE', 'X', 'Y'], columns=['LABEL'], aggfunc='count', fill_value=0)


#flatten column names
pvt.columns = list(pvt.columns)
pvt.reset_index(inplace=True)


#change column name to differentiate DEMAND and SUPPLY
pvt.rename(columns = {('TAXI_ID', 'D'):'DEMAND', ('TAXI_ID', 'S'):'SUPPLY'}, inplace=True)

#create new data based on demand and supply data
new_data = pvt[['X', 'Y', 'WEEK', 'DAY', 'HOUR', 'MINUTE', 'DEMAND', 'SUPPLY']]


# bin data into timesteps
timestep = 10 #in minutes

new_data['GAP'] = new_data['SUPPLY'] - new_data['DEMAND']
new_data['TIMESTEP'] = (new_data.HOUR*60 + new_data.MINUTE)/timestep
new_data['TIMESTEP'] = new_data.TIMESTEP.astype(int)

df = new_data.groupby(by=['WEEK', 'DAY', 'TIMESTEP','X', 'Y'], as_index=False)['DEMAND', 'SUPPLY', 'GAP'].sum()

#cut outlier/extreme data (2% tails)
df = df[df.GAP <= df.GAP.quantile(0.99)]
df = df[df.GAP >= df.GAP.quantile(0.01)]

#create dummy dataframe to fill missing area with zero demand and supply
WEEK = df['WEEK'].unique().tolist()
DAY = df['DAY'].unique().tolist()
TIMESTEP = df['TIMESTEP'].unique().tolist()
X = df['X'].unique().tolist()
X = np.sort(X)
Y = df['Y'].unique().tolist()
Y = np.sort(Y)

dummy = []

for week in WEEK:
    for day in DAY:
        for ts in TIMESTEP:
            for x in X:
                for y in Y:
                    dummy.append({'WEEK': week, 'DAY': day, 'TIMESTEP':ts, 'X':x, 'Y':y, 'DEMAND':0, 'SUPPLY':0, 'GAP':0})

df_dummy = pd.DataFrame(dummy)
df_dummy = df_dummy[['WEEK', 'DAY', 'TIMESTEP', 'X', 'Y', 'DEMAND', 'SUPPLY', 'GAP']]

#concat and regroup data
df = pd.concat([df,df_dummy])
df = df.groupby(by=['WEEK', 'DAY', 'TIMESTEP', 'X', 'Y'],  as_index=False)['DEMAND', 'SUPPLY', 'GAP'].sum()


#create forecast data
forecast_period = 3 #timesteps

df = df.sort_values(by=['X', 'Y', 'WEEK', 'DAY', 'TIMESTEP'])
df['GAP_FCST'] = df.GAP.shift(-forecast_period)
df = df.drop('GAP', axis=1)


#create 6 backward timesteps as features
df['DEMAND t-1'] = df.DEMAND.shift(periods=1)
df['DEMAND t-2'] = df.DEMAND.shift(periods=2)
df['DEMAND t-3'] = df.DEMAND.shift(periods=3)
df['DEMAND t-4'] = df.DEMAND.shift(periods=4)
df['DEMAND t-5'] = df.DEMAND.shift(periods=5)
df['DEMAND t-6'] = df.DEMAND.shift(periods=6)

df['SUPPLY t-1'] = df.DEMAND.shift(periods=1)
df['SUPPLY t-2'] = df.DEMAND.shift(periods=2)
df['SUPPLY t-3'] = df.DEMAND.shift(periods=3)
df['SUPPLY t-4'] = df.DEMAND.shift(periods=4)
df['SUPPLY t-5'] = df.DEMAND.shift(periods=5)
df['SUPPLY t-6'] = df.DEMAND.shift(periods=6)

df = df.fillna(0)
df = df.reset_index(drop=True)

#export new data to csv files
df.to_csv('processed_data/demand_supply.csv', index=False, encoding='utf-8')