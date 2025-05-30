import torch
import panphon
from sklearn.metrics import f1_score

from tqdm import tqdm

from sklearn.metrics import log_loss, f1_score
from math import floor
from copy import deepcopy
import numpy as np
from sklearn.neural_network import MLPClassifier

import random
from itertools import permutations
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
from panphon import sonority


def compute_online_codelength(X, y, is_weighted):
    losses = []
    portions = [0.125, 0.25, 0.5, 1]
    portions = [floor(len(X) * p) for p in portions]
    for i in tqdm(range(len(portions[:-1]))):

        indices = (portions[i], portions[i + 1])
        portion_size = indices[1] - indices[0]

        X_train_portion = X[: indices[0]]
        y_train_portion = y[: indices[0]]

        X_test = X[indices[0] : indices[1]]
        y_test = y[indices[0] : indices[1]]

        mlp = MLPClassifier(random_state=1, max_iter=300).fit(
            X_train_portion, y_train_portion
        )
        y_pred_prob = mlp.predict_proba(X_test)

        if is_weighted:
            p_label = {
                f: len(y_train_portion) / len([l for l in y_train_portion if l == f])
                for f in set(y_train_portion)
            }
            sample_weight = [p_label[_y] for _y in y_test]
            cross_entropy_loss = (
                log_loss(y_test, y_pred_prob, sample_weight=sample_weight)
                * portion_size
            )
        else:
            cross_entropy_loss = log_loss(y_test, y_pred_prob) * portion_size
        losses.append({"portion": portion_size, "loss": cross_entropy_loss})
    return losses


def get_compression_score(X, y, is_weighted, oversample=1):
    num_classes = len(set(y))
    if num_classes < 2:
        return np.nan, np.nan

    X, y = X * oversample, y * oversample
    combined = list(zip(X, y))
    random.shuffle(combined)
    X, y = zip(*combined)
    y_control = deepcopy(list(y))
    random.shuffle(y_control)
    losses = compute_online_codelength(X, y, is_weighted)
    losses_control = compute_online_codelength(X, y_control)
    train_size = len(X)

    uniform_codelength = train_size * np.log2(num_classes)
    online_codelength = losses[0]["portion"] * np.log2(num_classes) + sum(
        l["loss"] for l in losses
    )
    online_codelength_control = losses_control[0]["portion"] * np.log2(
        num_classes
    ) + sum(l["loss"] for l in losses_control)

    compression = uniform_codelength / online_codelength
    compression_control = uniform_codelength / online_codelength_control
    return compression, compression_control
