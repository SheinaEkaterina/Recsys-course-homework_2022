from typing import Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn.preprocessing
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, log_loss
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import sklearn.linear_model
from sklearn.pipeline import Pipeline
import random
from scipy import sparse

random.seed(1337)
np.random.seed(1337)


class ClickPredictor:
    ohe: Optional[OneHotEncoder] = None

    def preprocess(self, X, y, remember=False, return_last_day=False):
        X['date_time'] = pd.to_datetime(X['date_time'])

        if return_last_day:
            idx = X['date_time'].dt.date == X['date_time'].dt.date.max()

        X['day_of_week'] = X['date_time'].dt.dayofweek
        X['hour'] = X['date_time'].dt.hour

        X = X.drop(columns=['date_time'])

        to_normalize = ['campaign_clicks']
        to_one_hot = ['zone_id', 'banner_id', 'os_id', 'country_id', 'day_of_week', 'hour']

        X[to_normalize] = StandardScaler().fit_transform(X[to_normalize])

        X_to = X[to_one_hot]

        if self.ohe is None:
            self.ohe = OneHotEncoder()
            X_to = self.ohe.fit_transform(X_to)
            if not remember:
                self.ohe = None
        else:
            X_to = self.ohe.transform(X_to)


        X = X.drop(columns=to_one_hot)
        X = sparse.hstack([X, X_to])
        X = X.tocsr()

        if return_last_day:
            X_last, y_last = X[idx], y[idx]
            return X, y, X_last, y_last
        else:
            return X, y

    def get_model(self, C=0.1):
        return LogisticRegression(penalty='l2',
                                  C=C,  # TODO try different
                                  solver='lbfgs',
                                  random_state=1337,
                                  max_iter=500,  # TODO more
                                  verbose=0
                                  )

    def get_data(self, nrows=None, remember=False, shuffle=True, return_last_day=False):
        df = pd.read_csv('/opt/downloads/ml/recsys/data.csv',
                         nrows=nrows,
                         usecols=[
                             'date_time',
                             'zone_id',
                             'banner_id',
                             'campaign_clicks',
                             'os_id',
                             'country_id',
                             'clicks'
                         ])

        if shuffle:
            df = df.sample(frac=1).reset_index(drop=True)

        X, y = df.drop(columns=['clicks']), df['clicks']

        return self.preprocess(X, y, remember=remember, return_last_day=return_last_day)

    def select_features(self, validate=False):
        X, y = self.get_data(1000000, shuffle=False)

        model = self.get_model()

        fs = SelectFromModel(model)
        fs.fit(X, y)

        if validate:
            X, y = self.get_data(1000000, shuffle=False)
            X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8)
            model_full, model_sel = self.get_model(), self.get_model()

            model_full.fit(X_train, y_train)
            model_sel.fit(fs.transform(X_train), y_train)
            print(model_full.score(X_test, y_test))
            print(model_sel.score(fs.transform(X_test), y_test))

        return fs

    def adjust_regularization(self):
        X, y = self.get_data(1000000, shuffle=False)

        metrics = ['neg_log_loss', 'roc_auc']

        Cs = [0.001, 0.01, 0.1, 1, 10]
        nlls = []
        print('Regularization')
        for C in Cs:
            model = self.get_model(C=C)
            cv = cross_validate(model, X, y, cv=5, n_jobs=5, return_estimator=True, scoring=metrics, verbose=0)

            print(f'C = {C}')
            for i, metric in enumerate(metrics):
                metric_values = cv[f"test_{metric}"]
                print(f'{metric}: avg: {np.average(metric_values)}, metrics: {metric_values}')
            nlls.append(np.average(cv[f"test_neg_log_loss"]))
        return Cs[np.argmax(nlls)]

    def run(self):
        X_train, y_train, X_test, y_test = self.get_data(None, remember=True, shuffle=True, return_last_day=True)

        fs = self.select_features()
        print(f'Selected {np.sum(fs.get_support())}/{len(fs.get_support())} features')
        X_train = fs.transform(X_train)
        X_test = fs.transform(X_test)

        C = self.adjust_regularization()
        print(f'Selected {C} for regularization')

        model = self.get_model(C=C)

        model.fit(X_train, y_train)

        y_test_pred = model.predict(X_test)
        auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
        ll = log_loss(y_test, model.predict_proba(X_test)[:, 1])
        print(f'Auc: {auc}, ll: {ll}')

        y_test_pred_baseline = np.full(y_test_pred.shape, np.mean(y_train))
        auc_baseline = roc_auc_score(y_test, y_test_pred_baseline)
        ll_baseline = log_loss(y_test, y_test_pred_baseline)
        print(f'Auc: {auc_baseline}, ll: {ll_baseline}')


if __name__ == '__main__':
    cp = ClickPredictor()
    cp.run()
