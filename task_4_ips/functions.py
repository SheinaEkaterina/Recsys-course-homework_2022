import pandas as pd
import datetime as dtime
import numpy as np
import math



def feature_engineering(data: pd.DataFrame):
    """Немного поменяем функцию создания фичей из 1 дз"""
    data = data[::20]
    data = data.fillna(-1)
    # превратим date_time из строки в datetime тип данных:
    data['date_time'] = pd.to_datetime(data['date_time'])
    data['time'] = data['date_time'].dt.time
    data['date'] = data['date_time'].dt.date
    ######################
    morning = (data['time'] < dtime.time(10, 0, 0)) & (data['time'] >= dtime.time(6, 0, 0))
    day = (data['time'] < dtime.time(18, 0, 0)) & (data['time'] >= dtime.time(10, 0, 0))
    evening = (data['time'] <= dtime.time(23, 59, 59)) & (data['time'] >= dtime.time(18, 0, 0))
    night = (data['time'] < dtime.time(6, 0, 0)) & (data['time'] >= dtime.time(0, 0, 0))
    data.loc[(morning, 'time_of_day')] = 0
    data.loc[(day, 'time_of_day')] = 1
    data.loc[(evening, 'time_of_day')] = 2
    data.loc[(night, 'time_of_day')] = 3
    # Возьмем косинус от нормализованного времени суток, чтобы  модель знала, что после ночи идет утро
    data["time_of_day"] = math.pi * data["time_of_day"] / data["time_of_day"].max()
    data["time_of_day"] = np.cos(data["time_of_day"])
    # Отнормируем
    data["time_of_day"] = (data["time_of_day"] + 1) / 2
    ######################
    workday = (data['date_time'].dt.dayofweek >= 0) & (data['date_time'].dt.dayofweek < 5)
    weekend = (data['date_time'].dt.dayofweek >= 5) & (data['date_time'].dt.dayofweek < 7)
    data.loc[(workday, 'is_weekend')] = 0
    data.loc[(weekend, 'is_weekend')] = 1
    data['is_weekend'] = data['is_weekend'].astype('int8')
    ######################
    data['month'] = data['date_time'].dt.month
    # Отнормируем
    data['month'] = data['month'] - 9
    coefs = data[['date_time', 'g0', 'g1', 'coeff_sum0', 'coeff_sum1']]
    coefs = coefs[coefs['date_time'] >= "2021-10-02 00:00:00"].drop(['date_time'], axis=1)
    features = [ 'banner_id','banner_id0' , 'banner_id1', 'os_id', 'country_id', 'date', 'time_of_day', 'is_weekend',
                 'month', 'clicks']

    return data.loc[:, features], coefs



def train_test_split(data: pd.DataFrame):
    last_day = data['date'].max()
    test = data.loc[data['date'] == last_day]
    train = data.loc[data['date'] < last_day]
    X_train = train.drop(columns=['clicks'])
    Y_train = train.loc[:, 'clicks']
    # тестовая выборка с banner_id0
    X_test0 = test.drop(columns=['clicks'])
    # тестовая выборка с banner_id1
    X_test1 = test.drop(columns=['clicks'])
    X_test0 = X_test0.drop(columns=['banner_id1', 'banner_id'])
    X_test1 = X_test1.drop(columns=['banner_id', 'banner_id0'])
    X_test1 = X_test1.rename(columns={"banner_id1": "banner_id"})
    X_test0 = X_test0.rename(columns={"banner_id0": "banner_id"})
    Y_test = test.loc[:, 'clicks']

    return X_train, Y_train, X_test0, X_test1, Y_test
