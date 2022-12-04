from functools import partial
import time
from typing import Callable
import numpy as np
import pandas as pd
from scipy.stats import randint, uniform
from sim_lib import simulation


def main(policy_name: str):
    pd.options.mode.chained_assignment = None

    policy = get_policy(policy_name)
    output = evaluate_policy(policy)
    print(f"Total regret: {output['regret']}")
    print(f"Regret / rounds: {output['regret'] / output['rounds']}")


def evaluate_policy(policy: Callable, seed: int = 18475) -> dict:
    # seed for homework
    seed = 18475
    np.random.seed(seed=seed)

    start = time.time()
    output = simulation(policy, n=200000, seed=seed)
    end = time.time()

    print(f"Simulation time: {end - start:.3f} s")
    return output


def eps_greedy(history: pd.DataFrame, eps: float):
    if uniform.rvs() < eps:
        n = history.shape[0]
        return history.index[randint.rvs(0, n)]

    ctr = history['clicks'] / (history['impressions'] + 10)
    n = np.argmax(ctr)
    return history.index[n]


def get_policy(name: str) -> Callable:
    if name == "eps-greedy":
        return partial(eps_greedy, eps=0.06)
    else:
        raise ValueError(f"Unknown policy {name}")


if __name__ == "__main__":
    main("eps-greedy")
