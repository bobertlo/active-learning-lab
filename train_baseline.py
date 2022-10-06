from util import load_params, print_params
from dataset import load_mnist_dataset, prepare_training_set_random
from models import train_basic_cnn

from sklearn.utils import resample
from util import write_plots

if __name__ == "__main__":
    params = load_params()
    print_params(params)

    (X, y), (X_test, y_test) = load_mnist_dataset()
    print(f'train: X{X.shape} y{y.shape}')
    print(f'test:  X{X_test.shape} y{y_test.shape}')

    test_metrics = []
    training_logs = []
    train_size = 0
    for stage in params['train']['stages']:
        for i in range(stage['count']):
            if train_size == 0:
                train_size = stage['size']
            print(f"\n### Running Training Size {train_size}:")

            X_train, X_valid, y_train, y_valid = prepare_training_set_random(X, y, train_size=train_size)
            if len(X_valid) > train_size:
                X_valid, y_valid = resample(X_valid, y_valid, n_samples=train_size, stratify=y_valid)
            assert(len(X_train) == train_size)

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

            train_size += stage['size']
            write_plots(test_metrics, "baseline_cnn")