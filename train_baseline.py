from tensorflow import keras
from tensorflow.keras import layers
from util import load_params, print_params
from dataset import load_mnist_dataset, prepare_training_set_random
from models import train_basic_cnn

from sklearn.utils import resample
import pandas as pd
import os
import json

if __name__ == "__main__":
    params = load_params()
    print_params(params)

    (X, y), (X_test, y_test) = load_mnist_dataset()
    print(f'train: X{X.shape} y{y.shape}')
    print(f'test:  X{X_test.shape} y{y_test.shape}')

    test_metrics = []
    training_logs = []
    for train_size in [100, 250, 500, 750, 1000, 1500, 2000, 2500, 3000]:
        print(f"\n### Running Training Size {train_size}:")
        X_train, X_valid, y_train, y_valid = prepare_training_set_random(X, y, train_size=train_size)
        if len(X_valid) > train_size:
            X_valid, y_valid = resample(X_valid, y_valid, n_samples=train_size, stratify=y_valid)

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
    with open("plots/baseline_cnn_loss.json", "w") as outfile:
        json.dump(loss_metrics, outfile)

    acc_metrics = []
    for x in test_metrics:
        acc_metrics.append({
            "train_size": x['train_size'],
            "acc": x['accuracy'],
        })
    with open("plots/baseline_cnn_acc.json", "w") as outfile:
        json.dump(acc_metrics, outfile)