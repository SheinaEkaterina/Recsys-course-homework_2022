import random
from typing import Optional

import dill as dill
import numpy as np
import pandas
import pandas as pd
import scipy
import scipy.stats as st
from scipy import sparse
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, log_loss
from sklearn.preprocessing import OneHotEncoder

random.seed(1337)
np.random.seed(1337)


class CIPS:
    ohe: Optional[OneHotEncoder] = None

    def preprocess(self, X, y, remember=False):
        X['date_time'] = pd.to_datetime(X['date_time'])

        bi0 = X['banner_id'] == X['banner_id0']
        X, y = X[bi0], y[bi0]

        idx = X['date_time'].dt.date == X['date_time'].dt.date.max()
        nidx = ~idx

        X_mod = X[['banner_id0',
                   'banner_id1',
                   'rate0',
                   'rate1',
                   'g0',
                   'g1',
                   'coeff_sum0',
                   'coeff_sum1']]
        X = X.drop(columns=[
            'banner_id0',
            'banner_id1',
            'rate0',
            'rate1',
            'g0',
            'g1',
            'coeff_sum0',
            'coeff_sum1'])

        X['day_of_week'] = X['date_time'].dt.dayofweek
        X['hour'] = X['date_time'].dt.hour

        X = X.drop(columns=['date_time'])

        # to_normalize = []
        to_one_hot = ['zone_id', 'banner_id', 'os_id', 'country_id', 'day_of_week', 'hour']

        # X[to_normalize] = StandardScaler().fit_transform(X[to_normalize])

        X_to = X[to_one_hot]

        X_1 = X.copy()
        X_1['banner_id'] = X_mod['banner_id1']
        X_to1 = X_1[to_one_hot]

        if self.ohe is None:
            self.ohe = OneHotEncoder()
            X_tmp = pandas.concat([X_to, X_to1])
            self.ohe.fit(X_tmp)
            X_to = self.ohe.transform(X_to)
            X_to1 = self.ohe.transform(X_to1)
            if not remember:
                self.ohe = None
        else:
            X_to = self.ohe.transform(X_to)
            X_to1 = self.ohe.transform(X_to1)

        X = X.drop(columns=to_one_hot)
        X_1 = X_1.drop(columns=to_one_hot)
        X = sparse.hstack([X, X_to])
        X_1 = sparse.hstack([X_1, X_to1])
        X = X.tocsr()
        X_1 = X_1.tocsr()

        # print(nidx.shape, X.shape, X_last.shape)
        X, X_last = X[nidx], X[idx]
        X_1, X_1_last = X_1[nidx], X_1[idx]
        X_mod, X_mod_last = X_mod[nidx], X_mod[idx]
        y, y_last = y[nidx], y[idx]
        return X, X_last, X_1, X_1_last, X_mod, X_mod_last, y, y_last

    def get_model(self, C=0.001):
        return LogisticRegression(penalty='l2',
                                  C=C,  # TODO try different
                                  solver='lbfgs',
                                  random_state=1337,
                                  max_iter=500,  # TODO more
                                  verbose=0
                                  )

    def get_data(self, nrows=None, remember=False, shuffle=True):
        df = pd.read_csv('/opt/downloads/ml/recsys/data.csv',
                         nrows=nrows,
                         usecols=[
                             'date_time',
                             'zone_id',
                             'banner_id',
                             'os_id',
                             'country_id',
                             # 'oaid_hash',
                             'banner_id0',
                             'banner_id1',
                             'rate0',
                             'rate1',
                             'g0',
                             'g1',
                             'coeff_sum0',
                             'coeff_sum1',
                             'clicks'
                         ])

        if shuffle:
            df = df.sample(frac=1).reset_index(drop=True)

        X, y = df.drop(columns=['clicks']), df['clicks']

        return self.preprocess(X, y, remember=remember)

    def prob_x_ge_y(self, m1, d1, m2, d2):
        return 1 - st.norm.cdf(0, m1 - m2, (d1 ** 2 + d2 ** 2) ** 0.5 + 1e-6)

    def run(self, lambda_=10, saved=False):
        X, X_last, X_1, X_1_last, X_mod, X_mod_last, y, y_last = self.get_data(nrows=None)

        if not saved:
            model = self.get_model()
            model.fit(X, y)
            with open('model.ckpt', 'wb') as f:
                dill.dump(model, f)
        else:
            with open('model.ckpt', 'rb') as f:
                model = dill.load(f)

        pi0_probs = self.prob_x_ge_y(X_mod_last['coeff_sum0'],
                                     X_mod_last['g0'],
                                     X_mod_last['coeff_sum1'],
                                     X_mod_last['g1'])

        pi1_coeff_sum_new_0 = scipy.special.logit(model.predict_proba(X_last)[:, 1])
        pi1_coeff_sum_new_1 = scipy.special.logit(model.predict_proba(X_1_last)[:, 1])

        pi1_probs = self.prob_x_ge_y(pi1_coeff_sum_new_0,
                                     X_mod_last['g0'],
                                     pi1_coeff_sum_new_1,
                                     X_mod_last['g1'])

        mask = np.isfinite(pi1_probs) * np.isfinite(pi0_probs) * (np.abs(pi0_probs) > 1e-18)

        cips = np.mean(y_last[mask] * np.minimum(lambda_, (pi1_probs[mask] / pi0_probs[mask])))

        true_value = np.mean(y_last)

        auc = roc_auc_score(y_last, model.predict_proba(X_last)[:, 1])
        ll = log_loss(y_last, model.predict_proba(X_last)[:, 1])
        print(f'Auc: {auc}, ll: {ll}')

        return cips, true_value


if __name__ == '__main__':
    ans = CIPS().run()
    print(f'CIPS = {ans[0]}')
    print(f'True mean reward = {ans[1]}')
