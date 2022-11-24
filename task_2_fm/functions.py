import pandas as pd
import datetime as dtime
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.metrics import log_loss, roc_auc_score
import lightfm
from lightfm import LightFM
from lightfm.evaluation import auc_score
import scipy.sparse as sps
from sklearn.feature_extraction import DictVectorizer


def analyse_new_columns(data: pd.DataFrame):
    print("Количество записей в таблице: ", data.shape[0])

    print()
    print("Новые колонки")
    for feature in ['oaid_hash']:
        print(f"Количество уникальных значений для {feature}:  {len(data[feature].value_counts())}")
        print(f"Nan в {feature}:  {data['oaid_hash'].isnull().any()}")
        print(f"Встречаемость самого частого значения: {data['oaid_hash'].value_counts().max()}")
        print(f"Встречаемость самого редкого значения: {data['oaid_hash'].value_counts().min()}")


def feature_engineering(data: pd.DataFrame) -> pd.DataFrame:
    """
    time_of_day - время суток показа баннера
    is_weekend - выходной или нет 1/0
    month - месяц показа рекламы (месяца всего 2 в данных - октябрь и сентябрь)
    """
    # превратим date_time из строки в datetime тип данных:
    data['date_time'] = pd.to_datetime(data['date_time'])
    data['time'] = data['date_time'].dt.time
    data['date'] = data['date_time'].dt.date
    # ######################
    morning = (data['time'] < dtime.time(10, 0, 0)) & (data['time'] >= dtime.time(6, 0, 0))
    day = (data['time'] < dtime.time(18, 0, 0)) & (data['time'] >= dtime.time(10, 0, 0))
    evening = (data['time'] <= dtime.time(23, 59, 59)) & (data['time'] >= dtime.time(18, 0, 0))
    night = (data['time'] < dtime.time(6, 0, 0)) & (data['time'] >= dtime.time(0, 0, 0))
    data.loc[(morning, 'time_of_day')] = 0
    data.loc[(day, 'time_of_day')] = 1
    data.loc[(evening, 'time_of_day')] = 2
    data.loc[(night, 'time_of_day')] = 3
    # ######################
    workday = (data['date_time'].dt.dayofweek >= 0) & (data['date_time'].dt.dayofweek < 5)
    weekend = (data['date_time'].dt.dayofweek >= 5) & (data['date_time'].dt.dayofweek < 7)
    data.loc[(workday, 'is_weekend')] = 0
    data.loc[(weekend, 'is_weekend')] = 1
    # ######################
    data['month'] = data['date_time'].dt.month
    data['month'] = data['month'] - 9

    data['date_copy'] = data['date']  # ???
    return data


def one_hot(data: pd.DataFrame):
    cats_zone = data['zone_id'].value_counts()[lambda x: x > 300000].index
    cats_country = data['country_id'].value_counts()[lambda x: x > 300000].index
    other_cat_features = ['os_id', 'time_of_day']

    zone_col = pd.get_dummies(pd.Categorical(data['zone_id'], categories=cats_zone)).add_prefix('zone_')
    country_col = pd.get_dummies(pd.Categorical(data['country_id'], categories=cats_country)).add_prefix('country_')
    data = pd.get_dummies(data, columns=other_cat_features, drop_first=True)
    data = pd.concat([data, zone_col], axis=1)
    data = pd.concat([data, country_col], axis=1)
    return data


def item_feature_matrix(data):
    """
    Матрица фичей баннеров
    item features - zone_id, time_of_day, is_weekend, month"""
    # Составим списки one-hot фичей для zone_id, time_of_day
    zone_features = []
    time_features = []
    for col in data.columns:
        if col.startswith("zone_") and 'id' not in col:
            zone_features.append(col)

    for col in data.columns:
        if col.startswith("time_of_day"):
            time_features.append(col)

    data = data[["banner_id", "is_weekend", "month"] + zone_features + time_features]
    data = sps.csr_matrix(data, dtype=np.int32)
    return data


def user_feature_matrix(data):
    """
    Матрица фичей юзеров
    user features - os_id, country_id"""
    # Составим списки one-hot фичей для операционной системы и страны
    os_features = []
    country_features = []
    for col in data.columns:
        if col.startswith("country_") and 'id' not in col:
            country_features.append(col)

    for col in data.columns:
        if col.startswith("os_"):
            os_features.append(col)

    data = data[["oaid_hash"] + os_features + country_features]
    data = sps.csr_matrix(data, dtype=np.int32)
    return data


def item_user_matrix(data):
    """Матрица, где юзерам поставлены в соответствие баннеры, по которым они кликнули"""
    user_item_table = data.loc[data['clicks'] == 1]
    user_item_table = user_item_table.loc[:, ['oaid_hash', 'banner_id']]
    user_item_table = user_item_table.pivot_table(index='oaid_hash', columns='banner_id', aggfunc=len, fill_value=0)
    user_item_matrix = sps.csr_matrix(user_item_table, dtype=np.int32)
    return user_item_matrix


def prepare_data(data: pd.DataFrame):
    """Сделаем разбивку на тестовое и тренировочное множества.
    Создадим матрицы с фичами пользователя, с фичами баннера и матрицу пользователь-баннер"""
    last_day = data['date'].max()
    test = data.loc[data['date'] == last_day]
    train = data.loc[data['date'] < last_day]
    user_feat_train = user_feature_matrix(train)
    item_feat_train = item_feature_matrix(train)
    user_feat_test = user_feature_matrix(test)
    item_feat_test = item_feature_matrix(test)
    user_item_train = item_user_matrix(train)
    user_item_test = item_user_matrix(test)
    return user_item_train, user_item_test, user_feat_train, item_feat_train, user_feat_test, item_feat_test


def test_model(prediction, Y_test: pd.DataFrame):
    # возьмем только вероятность клика
    prediction = prediction[:, 1]
    auc = roc_auc_score(Y_test, prediction)
    print(f"Log loss: {log_loss(Y_test, prediction)}")
    print(f"Auc: {auc}")


def cv(X_train: pd.DataFrame, Y_train: pd.DataFrame, metric: str):
    scores = dict()
    for n in [5, 10, 20]:
        model = LightFM(no_components=n)
        model.fit(X_train, Y_train)
        score = cross_val_score(model, X_train, Y_train, n_jobs=-1, cv=3, scoring=metric)
        scores[n] = np.mean(score)
        print(f"N factors: {n}, score: {score}")

    opt_params = max(scores, key=scores.get)
    l1_ratio = opt_params
    return l1_ratio
