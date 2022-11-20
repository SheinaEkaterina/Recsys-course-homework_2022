import xlearn as xl

from typing import Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn.preprocessing
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, log_loss, accuracy_score
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import sklearn.linear_model
from sklearn.pipeline import Pipeline
import random
from scipy import sparse
import os

from scipy.special import softmax

from utils import _convert_to_ffm


class FFMClicksPredictor:
    ohe: Optional[OneHotEncoder] = None
    encoder = None
    path = '/opt/downloads/ml/recsys/'
    n_rows = None

    def get_y(self, path, c):
        ys = []

        with open(path, 'r') as r:
            for i, l in enumerate(r):
                ys.append(c(l))
                if self.n_rows is not None and i >= self.n_rows:
                    break

        return np.array(ys)

    def baseline(self):
        y_last = self.get_y(self.path + "data_val.ffm", lambda l: int(l[0]))
        y_ones = np.ones_like(y_last)
        y_zeros = np.zeros_like(y_last)
        print(
            f'All 1 baseline: acc {accuracy_score(y_last, y_ones)} log loss {log_loss(y_last, y_ones)}, auc {roc_auc_score(y_last, y_ones)}')
        print(
            f'All 0 baseline: acc {accuracy_score(y_last, y_zeros)} log loss {log_loss(y_last, y_zeros)}, auc {roc_auc_score(y_last, y_zeros)}')

    def run(self, lr=0.1, l2=0.0002, k=10, quiet=False):
        train, val = "/opt/downloads/ml/recsys/data_train.ffm", "/opt/downloads/ml/recsys/data_val.ffm"
        model = xl.create_ffm()
        model.setTrain(train)
        model.setValidate(val)
        # model.setTXTModel(self.path + 'ffm/model.txt')
        if quiet:
            model.setQuiet()
        # model.setOnDisk()
        param = {'task': 'binary',
                 'lr': lr,
                 'lambda': l2,
                 'nthread': 8,
                 'epoch': 20,
                 'opt': 'adagrad',
                 'metric': 'auc',
                 'k': k,
                 'stop_window': 2,
                 'init': 0.2,
                 }
        model.fit(param, self.path + 'ffm/model.out')

        model.setTest(val)
        model.predict(self.path + 'ffm/model.out', self.path + 'ffm/output.txt')

        y_last = self.get_y(val, lambda l: int(l[0]))
        y_pred = self.get_y(self.path + "ffm/output.txt", lambda l: float(l))

        y_pred = 1 / (1 + np.exp(-y_pred))

        acc = accuracy_score(y_last, np.round(y_pred))
        ll = log_loss(y_last, y_pred)
        auc = roc_auc_score(y_last, y_pred)
        print(
            f'FFM lr {lr} l2 {l2} k {k} model: acc {acc} log loss {ll}, auc {auc}')

        return acc, ll, auc

    def k_search(self):
        self.baseline()
        reses = {}
        best_auc, best_h = None, None
        for k in [10, 8, 4, 2]:
            res = self.run(k=k, quiet=False)
            reses[k] = res
            if best_auc is None or res[2] > best_auc:
                best_auc, best_h = res[2], k

        self.run(k=best_h, quiet=False)
        return reses


if __name__ == '__main__':
    ffmcp = FFMClicksPredictor()
    # ffmcp.run()
    reses = ffmcp.k_search()
    with open('/opt/downloads/ml/recsys/ffm/reses_dict', 'w') as f:
        f.write(str(reses))
