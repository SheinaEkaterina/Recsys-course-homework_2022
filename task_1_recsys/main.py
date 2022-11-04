from tqdm import tqdm
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import log_loss
import matplotlib.pyplot as plt
from load_data import load_raw, last_day_split
from feature_engineering import feature_engineering


N_SPLITS = 5
N_C = 10

df, df_test = last_day_split(load_raw("../data"))

df = feature_engineering(df, drop_ids=False)
X = df.drop("clicks", axis=1).to_numpy()
y = df["clicks"].to_numpy()

skf = StratifiedKFold(N_SPLITS)
c_vals = np.logspace(-4, 4, N_C)
train_losses = np.zeros((N_C, N_SPLITS))
val_losses = np.zeros_like(train_losses)

for i, ci in enumerate(c_vals):
    model = LogisticRegression(solver="liblinear", C=ci)

    for j, (train_index, val_index) in tqdm(enumerate(skf.split(X, y))):
        model.fit(X[train_index], y[train_index])

        pred_train = model.predict_proba(X[train_index])[:, 1]
        train_loss = log_loss(y[train_index], pred_train)
        pred_val = model.predict_proba(X[val_index])[:, 1]
        val_loss = log_loss(y[val_index], pred_val)

        train_losses[i, j] = train_loss
        val_losses[i, j] = val_loss

    print(f"C = {ci:.2e}, train_loss = {train_losses[i].mean():.2e}, " +
          f"val_loss = {val_losses[i].mean():.2e}")


plt.figure()
plt.plot(c_vals, train_losses.mean(1), marker="o", label="train")
plt.plot(c_vals, val_losses.mean(1), marker="D", label="validation")
plt.ylabel("Loss")
plt.xlabel("$C$")
plt.legend()
plt.show()
