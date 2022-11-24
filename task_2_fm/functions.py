import pandas as pd
import datetime as dtime
import numpy as np
from tqdm import tqdm
import lightfm
from lightfm import LightFM
from lightfm.evaluation import auc_score
import scipy.sparse as sps


def analyse_new_columns(data: pd.DataFrame):
    print("Количество записей в таблице: ", data.shape[0])
    print()
    print("Новые колонки")
    for feature in ['oaid_hash']:
        print(f"Количество уникальных значений для {feature}:  {len(data[feature].value_counts())}")
        print(f"Nan в {feature}:  {data['oaid_hash'].isnull().any()}")
        print(f"Встречаемость самого частого значения: {data['oaid_hash'].value_counts().max()}")
        print(f"Встречаемость самого редкого значения: {data['oaid_hash'].value_counts().min()}")
        print(f"Минимальное значение: {data['oaid_hash'].min()}")
        print(f"Максимальное значение:  {data['oaid_hash'].min()}")


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
    # ######################
    # Объединим пользователей, которые встречаются 1 раз, в 1 пользователя с oaid_hash=0
    value_count = pd.DataFrame(data['oaid_hash'].value_counts().reset_index())
    rare_values = value_count[value_count['oaid_hash'] < 2]['index'].to_list()
    data['oaid_hash'] = np.where(data['oaid_hash'].isin(rare_values), 0, data['oaid_hash'])
    # ######################
    # Закодируем хэши пользователей
    # Сделаем словарь, где каждому хэшу поставим в соответсвие число от 0 до n_users
    users_hash = list(data['oaid_hash'].unique())
    n_users = len(users_hash)
    new_hashes = np.arange(n_users)
    mapping = dict(zip(users_hash, new_hashes))
    data['oaid_hash'] = data['oaid_hash'].map(mapping)
    # # Прологарифмируем campaign_clicks
    # data['campaign_clicks'] = data['campaign_clicks'] + 0.1
    # data['campaign_clicks'] = np.log(data['campaign_clicks'])
    # data['campaign_clicks'] = data['campaign_clicks'].astype(int)
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
    # уберем лишние колонки
    data = data.drop(columns=['date_time', 'time'])
    return data


def item_feature_matrix(data: pd.DataFrame):
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


def user_feature_matrix(data: pd.DataFrame):
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


def item_user_matrix(data: pd.DataFrame):
    """Матрица, где юзерам поставлены в соответствие баннеры, по которым они кликнули"""
    user_item_table = data.loc[data['clicks'] == 1]
    user_item_table = user_item_table.loc[:, ['oaid_hash', 'banner_id']]
    user_item_table = user_item_table.pivot_table(index='oaid_hash', columns='banner_id', aggfunc=len, fill_value=0)
    user_item_matrix = sps.csr_matrix(user_item_table, dtype=np.int32)
    return user_item_matrix


def train_test_split(data: pd.DataFrame):
    last_day = data['date'].max()
    test = data.loc[data['date'] == last_day]
    train = data.loc[data['date'] < last_day]
    return train, test


def prepare_data(train: pd.DataFrame, test: pd.DataFrame):
    """Сделаем разбивку на тестовое и тренировочное множества.
    Создадим матрицы с фичами пользователя, с фичами баннера и матрицу пользователь-баннер"""
    user_feat_train = user_feature_matrix(train)
    item_feat_train = item_feature_matrix(train)
    user_feat_test = user_feature_matrix(test)
    item_feat_test = item_feature_matrix(test)
    user_item_train = item_user_matrix(train)
    user_item_test = item_user_matrix(test)
    return user_item_train, user_item_test, user_feat_train, item_feat_train, user_feat_test, item_feat_test


def test_model(model, user_item_test, user_feat_test, item_feat_test):
    auc = auc_score(model, test_interactions=user_item_test,
                    user_features=user_feat_test,
                    item_features=item_feat_test).mean()
    print(f"Auc: {auc}")


def cv(user_item_train: sps.csr_matrix, user_feat_train: sps.csr_matrix, item_feat_train: sps.csr_matrix):
    scores = dict()
    train, test = lightfm.cross_validation.random_train_test_split(user_item_train, test_percentage=0.3,
                                                                   random_state=22)
    for n in tqdm([5, 10, 20]):
        model = LightFM(no_components=n)
        model.fit(interactions=user_item_train,
                  user_features=user_feat_train,
                  item_features=item_feat_train,
                  epochs=40)
        score = auc_score(model, train).mean()
        scores[n] = score
        print(f"N factors: {n}, score: {score}")

    opt_params = max(scores, key=scores.get)
    l1_ratio = opt_params
    return l1_ratio


def extraxt_users_items(test: pd.DataFrame):
    """Возвращает array из баннеров в тесте и array из юзеров в тесте"""
    users_array = test['oaid_hash']
    items_array = test['banner_id']
    users_array = np.array(users_array, dtype=np.int64)
    items_array = np.array(items_array, dtype=np.int32)
    return users_array, items_array
