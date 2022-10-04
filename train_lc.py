from tensorflow import keras
from tensorflow.keras import layers
from util import load_params, print_params
from dataset import load_mnist_dataset, prepare_training_set_random
from models import train_basic_cnn

from sklearn.utils import resample
import pandas as pd
import os
import json
import numpy as np

if __name__ == "__main__":
    params = load_params()
    print_params(params)

    (X, y), (X_test, y_test) = load_mnist_dataset()
    print(f'train: X{X.shape} y{y.shape}')
    print(f'test:  X{X_test.shape} y{y_test.shape}')

    X_train, X_reserve, y_train, y_reserve = prepare_training_set_random(X, y, train_size=100)

    test_metrics = []
    training_logs = []
    for i in range(15):
        train_size = 100 + (i * 100)
        X_valid = X_reserve[:1000]
        y_valid = y_reserve[:1000]

        print(f"\n### Running Training Size {train_size}:")
        model, log = train_basic_cnn(X_train, y_train, X_valid, y_valid, params)

        test_scores = model.evaluate(X_test, y_test)
        print(f"\n### Train Size {train_size} Results:")
        print("Test loss:    ", test_scores[0])
        print("Test accuracy:", test_scores[1])
        print("")

        test_metrics.append({
            "train_size": train_size,
            "loss": test_scores[0],
            "accuracy": test_scores[1],
        })

        training_logs.append({
            "train_size": train_size,
            "log": log.history,
        })

        print("### Selecting next samples ...")
        max_preds = model.predict(X_valid).max(axis=1)
        sorted_preds = sorted(enumerate(max_preds), key=lambda x:x[1])
        idxs = [x[0] for x in sorted_preds[:500]]

        X_new = X_reserve[idxs]
        y_new = y_reserve[idxs]
        X_reserve = np.delete(X_reserve, idxs, 0)
        y_reserve = np.delete(y_reserve, idxs, 0)

        X_train = np.append(X_train, X_new, axis=0)
        y_train = np.append(y_train, y_new, axis=0)

    sizes = []
    accs = []
    losses = []
    for x in test_metrics:
        sizes.append(x['train_size'])
        losses.append(x['loss'])
        accs.append(x['accuracy'])
    os.makedirs("plots", exist_ok=True)

    loss_metrics = []
    for x in test_metrics:
        loss_metrics.append({
            "train_size": x['train_size'],
            "loss": x['loss'],
        })
    with open("plots/lc_cnn_loss.json", "w") as outfile:
        json.dump(loss_metrics, outfile)

    acc_metrics = []
    for x in test_metrics:
        acc_metrics.append({
            "train_size": x['train_size'],
            "acc": x['accuracy'],
        })
    with open("plots/lc_cnn_acc.json", "w") as outfile:
        json.dump(acc_metrics, outfile)