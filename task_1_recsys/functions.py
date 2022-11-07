import pandas as pd
import datetime as dtime
import numpy as np
import sklearn.metrics as metrics
from sklearn.model_selection import GroupShuffleSplit, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss

features = ['zone_id', 'campaign_clicks', 'os_id', 'country_id', 'time_of_day',
            'is_weekend', '15_av_clicks_country', '15_av_clicks_os',
            '15_av_clicks_month', 'month']


def analysis(data: pd.DataFrame):
    print("Количество записей в таблице: ", data.shape[0])
    data_disbalance = data['clicks'].value_counts()[1] / (data['clicks'].value_counts()[0] +
                                                          data['clicks'].value_counts()[1])
    print(f"соотношние между классами:")
    print(f"1-ый класс: {100 * data_disbalance} %, 2-ой класс {100 * (1 - data_disbalance)} %")
    print()
    print("Nan в таблице:")
    print((data.isnull().any()).any())
    print()
    for feature in ['os_id', 'country_id', 'impressions', 'campaign_clicks']:
        print(f"Количество уникальных значений для {feature}:  {len(data[feature].value_counts())}")


def baseline(data: pd.DataFrame):
    # создадим бейзлайн как среднее по баннеру для страны country_id и операционной системы os_id

    average_clicks = (data.loc[:, ['banner_id', 'os_id', 'country_id', 'clicks']]
                      .groupby(['banner_id', 'os_id', 'country_id'], sort=False)
                      ['clicks']
                      .mean())

    data = data.merge(average_clicks, how='inner', on=['banner_id', 'os_id', 'country_id'])
    data = data.rename(columns={"clicks_x": "clicks", "clicks_y": "baseline"}, errors="raise")
    return data


def avg(data: pd.DataFrame, slice_col: str, feature: str = 'clicks', number_rows: int = 15):
    """Оконная функция для посдсчета среднего значения feature с размером окна number_rows
    по срезу данных по признаку slice_col """
    col_name = 'avg_' + str(number_rows) + '_' + feature
    average = (data
               .loc[:, ['banner_id', slice_col, 'date_time', feature]]
               .sort_values(['banner_id', slice_col, 'date_time'], ascending=[True, True, True])
               .groupby(['banner_id', slice_col], sort=False)
               [feature]
               .rolling(number_rows, min_periods=number_rows - 1)
               .mean()
               .reset_index()
               .rename(columns={'level_1': 'index',
                                feature: col_name}))

    return average[col_name]


def feature_engineering(data: pd.DataFrame) -> pd.DataFrame:
    """
    time_of_day - время суток показа баннера (0 - утро, 1 - день, 2 - вечер, 3 - ночь)
    is_weekend - выходной или нет 1/0
    month - месяц показа рекламы (месяца всего 2 в данных - октябрь и сентябрь)
    15_av_clicks_os - среднее кликов для баннера, сделанное с данной операционной системы, за последние 15 показов
    15_av_clicks_country - среднее кликов для баннера, сделанное из данной страны, за последние 15 показов
    15_av_clicks_month - среднее кликов для баннера, сделанное за данное время года, за последние 15 показов
    """
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
    data['time_of_day'] = data['time_of_day'].astype('int8')
    ######################
    workday = (data['date_time'].dt.dayofweek >= 0) & (data['date_time'].dt.dayofweek < 5)
    weekend = (data['date_time'].dt.dayofweek >= 5) & (data['date_time'].dt.dayofweek < 7)
    data.loc[(workday, 'is_weekend')] = 0
    data.loc[(weekend, 'is_weekend')] = 1
    data['is_weekend'] = data['is_weekend'].astype('int8')
    ######################
    data['month'] = data['date_time'].dt.month
    ######################
    data['15_av_clicks_os'] = avg(data, 'os_id', 'clicks')
    # пропуски (когда мало предыдущих данных) заполним -1
    data['15_av_clicks_os'] = data['15_av_clicks_os'].fillna(-1)
    ######################
    data['15_av_clicks_country'] = avg(data, 'country_id', 'clicks')
    data['15_av_clicks_country'] = data['15_av_clicks_country'].fillna(-1)
    ######################
    data['15_av_clicks_month'] = avg(data, 'month', 'clicks')
    data['15_av_clicks_month'] = data['15_av_clicks_month'].fillna(-1)
    return data


def train_test_split(data: pd.DataFrame):
    # Разобъем на тест и тест так, чтобы один баннер не попал и в тест и трэйн
    splitter = GroupShuffleSplit(test_size=0.33, n_splits=2, random_state=7)
    split = splitter.split(data, groups=data['banner_id'])
    train_inds, test_inds = next(split)
    train = data.iloc[train_inds]
    test = data.iloc[test_inds]
    X_train = train.loc[:, features]
    Y_train = train.loc[:, 'clicks']
    X_test = test.loc[:, features]
    Y_test = test.loc[:, 'clicks']
    return X_train, Y_train, X_test, Y_test


def create_model(X_train: pd.DataFrame, Y_train: pd.DataFrame, l1_ratio=0.6):
    model = LogisticRegression(penalty='elasticnet', solver='saga', l1_ratio=l1_ratio)
    model.fit(X_train, Y_train)
    return model


def test_model(model, X_test: pd.DataFrame, Y_test: pd.DataFrame):
    prediction = model.predict(X_test)
    fpr, tpr, thresholds = metrics.roc_curve(Y_test, prediction, pos_label=2)
    auc = metrics.auc(fpr, tpr)
    print(f"Log loss: {log_loss(Y_test, prediction)}")
    print(f"Auc: {auc}")


def cv(X_train: pd.DataFrame, Y_train: pd.DataFrame):
    scores = dict()
    for l1_ratio in np.linspace(0, 0.5, 6):
        regr = LogisticRegression(penalty='elasticnet', solver='saga', l1_ratio=l1_ratio)
        regr.fit(X_train, Y_train)
        score = cross_val_score(regr, X_train, Y_train, n_jobs=-1, cv=4, scoring='log_loss')
        scores[l1_ratio] = np.mean(score)
    opt_params = max(scores, key=scores.get)
    l1_ratio = opt_params[1]
    return l1_ratio
