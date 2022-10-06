from util import load_params, print_params, write_plots
from dataset import load_mnist_dataset, prepare_training_set_random
from models import train_basic_cnn

import numpy as np

if __name__ == "__main__":
    params = load_params()
    print_params(params)

    (X, y), (X_test, y_test) = load_mnist_dataset()
    print(f'train: X{X.shape} y{y.shape}')
    print(f'test:  X{X_test.shape} y{y_test.shape}')

    train_size = 0
    test_metrics = []
    training_logs = []
    for stage in params['train']['stages']:
        for i in range(stage['count']):
            if train_size == 0:
                train_size = stage['size']
                X_train, X_reserve, y_train, y_reserve = prepare_training_set_random(X, y, train_size=stage['size'])
            assert(len(X_train) == train_size)

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
            idxs = [x[0] for x in sorted_preds[:stage['size']]]

            X_new = X_reserve[idxs]
            y_new = y_reserve[idxs]
            X_reserve = np.delete(X_reserve, idxs, 0)
            y_reserve = np.delete(y_reserve, idxs, 0)

            X_train = np.append(X_train, X_new, axis=0)
            y_train = np.append(y_train, y_new, axis=0)

            train_size += stage['size']

            write_plots(test_metrics, "lc_cnn")