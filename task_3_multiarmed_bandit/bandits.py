from functools import partial
import json
import os
import time
from typing import Callable
import numpy as np
import pandas as pd
from scipy import stats
from sim_lib import simulation


def main():
    pd.options.mode.chained_assignment = None

    bandit_num = 0
    def get_fname():
        return f"bandit_{bandit_num}"

    # 1. UCB bandit
    for ni_min in (2, 10, 20, 50):
        for gamma in (0.001, 0.01, 0.1, 1.0):
            settings = {"ni_min": ni_min, "gamma": gamma}
            policy = partial(ucb, **settings)
            output = evaluate_policy(policy)
            save_results("ucb", settings, output, get_fname())
            bandit_num += 1

    # 2. Thompson sampling bandit
    for alpha in (0.1, 0.5, 1, 5):
        for beta in (0.1, 1, 10, 50):
            settings = {"alpha": alpha, "beta": beta}
            policy = partial(thompson, **settings)
            output = evaluate_policy(policy)
            save_results("thompson", settings, output, get_fname())
            bandit_num += 1


def evaluate_policy(policy: Callable, seed: int = 18475) -> dict:
    # seed for homework
    seed = 18475
    np.random.seed(seed=seed)

    start = time.time()
    output = simulation(policy, n=200000, seed=seed)
    end = time.time()

    print(f"Simulation time: {end - start:.3f} s")
    print(f"Total regret: {output['regret']}")
    print(f"Regret / rounds: {output['regret'] / output['rounds']}")
    return output


def eps_greedy(history: pd.DataFrame, eps: float):
    if stats.uniform.rvs() < eps:
        n = history.shape[0]
        return history.index[stats.randint.rvs(0, n)]

    ctr = history['clicks'] / (history['impressions'] + 10)
    n = np.argmax(ctr)
    return history.index[n]


def ucb(history: pd.DataFrame, ni_min: int = 10, gamma: float = 1.0):
    ni = np.maximum(history["impressions"], ni_min)
    theta = history["clicks"] / ni
    t = ni.sum()
    ci = (2 * np.log(t) / ni)
    i = np.argmax(theta + gamma * ci)
    return history.index[i]


def thompson(history: pd.DataFrame, alpha: float = 1., beta: float = 1.):
    k = history["clicks"]
    n = history["impressions"]
    a = alpha + k
    b = beta + n - k
    samples = [stats.beta.rvs(a_i, b_i) for a_i, b_i in zip(a, b)]
    i = np.argmax(samples)
    return history.index[i]


def save_results(bandit_name: str, bandit_settings: dict,
                 output: dict, fname: str, output_dir: str = "results"):
    dct = {k: v for k, v in output.items() if k != "history"}
    dct["bandit_name"] = bandit_name
    for setting in bandit_settings:
        dct[setting] = bandit_settings[setting]
    path = os.path.join(output_dir, fname)
    with open(path + ".json", "w") as fout:
        json.dump(dct, fout)
    pd.DataFrame.to_csv(output["history"], path + ".csv")


if __name__ == "__main__":
    main()
