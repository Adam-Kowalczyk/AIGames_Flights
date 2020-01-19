import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from sklearn.preprocessing import OneHotEncoder
import seaborn as sns
import os

from sklearn.ensemble import RandomForestRegressor

from tensorflow.keras.callbacks import ModelCheckpoint

#Evaluation:
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn import preprocessing

import ast
from geopy import distance
import datetime as dt


import utils
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

def create_model():
    model = Sequential()
    model.add(Dense(128, input_shape=(X_train.shape[1],), activation='relu'))
    model.add(Dense(4096, activation='relu'))
    #model.add(Dropout(0.4))
    model.add(Dense(2048, activation='relu'))
    #model.add(Dropout(0.3))
    model.add(Dense(1024, activation='relu'))
    #model.add(Dropout(0.2))
    model.add(Dense(512, activation='relu'))
    model.add(Dense(256, activation='relu'))
    model.add(Dense(1))
    
    model.compile(optimizer='adam', loss='mae')
    
    return model

def process_fixes(df):
    df['decoded_fixes'] = df['decoded_fixes'].apply(lambda x: ast.literal_eval(x))
    
    df.loc[:, 'lat_start'] = df['decoded_fixes'].map(lambda x: x[0][0])
    df.loc[:, 'lng_start'] = df['decoded_fixes'].map(lambda x: x[0][1])
    
    df.loc[:, 'lat_end'] = df['decoded_fixes'].map(lambda x: x[-1][0])
    df.loc[:, 'lng_end'] = df['decoded_fixes'].map(lambda x: x[-1][1])
    
    df.drop(columns=['decoded_fixes'], inplace=True)
    
def get_hour(row):
    row['departure_time'] = dt.datetime.strptime(row['departure_time'][: -4], "%Y-%m-%d %H:%M:%S")
    minutes = row['departure_time'].minute
    hour = row['departure_time'].hour
    
    if minutes >= 30:
        row['departure_time'] = (hour + 1) % 24
    else:
        row['departure_time'] = hour
        
    return row
    
if __name__ == '__main__':
        
    Y = pd.read_csv('data/flight_times_labels_train.csv')['flight_time_s'].astype(float).values.reshape(-1)
    
    df_train = pd.read_csv('data/flight_plans_train.csv')
    df_train.drop(columns=['id', 'departure_airport', 'arrival_airport', 'fixes', 'departure_time'], inplace=True)
    df_train['Distance'] = df_train.apply(lambda x: utils.dist_in_miles_from_spherical_path(ast.literal_eval(x['decoded_fixes'])), axis=1)
    process_fixes(df_train)
    df_train['DistanceStraight'] = df_train.apply(lambda x: distance.distance((x['lat_start'], x['lng_start']), (x['lat_end'], x['lng_end'])).miles, axis=1)
    df_train['DsitancesProduct'] = df_train['Distance'] * df_train['DistanceStraight']
    df_train['TimeFromDistanceStraight'] = df_train['DistanceStraight'] / df_train['requested_airspeed']
    df_train['TimeFromDistance'] = df_train['Distance'] / df_train['requested_airspeed']
    
    wheathers = pd.read_csv('weather_features.csv')
    df_train = pd.concat([df_train, wheathers], axis=1, sort=False)
    df_train = df_train.replace(-999, 0)
    df_train = df_train.fillna(0.0)
    
    '''
    df_train = df_train.apply(get_hour, axis=1)
    one_hot = pd.get_dummies(df_train['departure_time'])
    df_train.drop(columns=['departure_time'], inplace=True)
    df_train = pd.concat([df_train, one_hot], axis=1, sort=False)
    '''
    
    X = df_train.values
    
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=5)
    
    min_max_scaler = preprocessing.MinMaxScaler()
    X_train = min_max_scaler.fit_transform(X_train)
    X_test = min_max_scaler.transform(X_test)
    
    print('training')
    model = RandomForestRegressor(n_estimators=100)
    model.fit(X_test, y_test)
    
    test_data = pd.read_csv('flight_plans_test.csv')
    test_ids = test_data[['id']].squeeze()
    test_data.drop(columns=['id', 'departure_airport', 'arrival_airport', 'fixes', 'departure_time'], inplace=True)
    test_data['Distance'] = test_data.apply(lambda x: utils.dist_in_miles_from_spherical_path(ast.literal_eval(x['decoded_fixes'])), axis=1)
    process_fixes(test_data)
    test_data['DistanceStraight'] = test_data.apply(lambda x: distance.distance((x['lat_start'], x['lng_start']), (x['lat_end'], x['lng_end'])).m, axis=1)
    test_data['DsitancesProduct'] = test_data['Distance'] * test_data['DistanceStraight']
    test_data['TimeFromDistanceStraight'] = test_data['DistanceStraight'] / test_data['requested_airspeed']
    test_data['TimeFromDistance'] = test_data['Distance'] / test_data['requested_airspeed']
    
    wheathers_test = pd.read_csv('weather_features_test.csv')
    test_data = pd.concat([test_data, wheathers_test], axis=1, sort=False)
    test_data = test_data.replace(-999, 0)
    test_data = test_data.fillna(0.0)
    
    '''
    test_data = test_data.apply(get_hour, axis=1)
    one_hot = pd.get_dummies(test_data['departure_time'])
    test_data.drop(columns=['departure_time'], inplace=True)
    test_data = pd.concat([test_data, one_hot], axis=1, sort=False)
    '''
    
    X_test_data = test_data.values
    X_test_data = min_max_scaler.transform(X_test_data)
    pred = model.predict(X_test_data).flatten()
    
    test_outcome = pd.DataFrame({'id': list(test_ids), 'flight_time_s': list(pred.astype(int))})
    test_outcome.to_csv('results.csv', index=False)
    