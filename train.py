from util import load_params, print_params, write_plots
from dataset import load_mnist_dataset
from models import train_basic_cnn

from selection import BaselineSelector, LeastConfidenceSelector, SmallestMarginSelector, MaxEntropySelector

import numpy as np

def margin(x):
    sx = sorted(x, reverse=True)
    return sx[0] - sx[1]


if __name__ == "__main__":
    params = load_params()
    print_params(params)

    (X, y), (X_test, y_test) = load_mnist_dataset()
    print("\n### Source Dataset:")
    print(f'train: X{X.shape} y{y.shape}')
    print(f'test:  X{X_test.shape} y{y_test.shape}')

    selector_name = params['train']['selector']
    selector_init_size = params['train']['init_size']

    if selector_name == 'random':
        selector = BaselineSelector(X, y, train_size=selector_init_size)
    elif selector_name == 'lc':
        selector = LeastConfidenceSelector(X, y, train_size=selector_init_size)
    elif selector_name == 'sm':
        selector = SmallestMarginSelector(X, y, train_size=selector_init_size)
    elif selector_name == 'ent':
        selector = MaxEntropySelector(X, y, train_size=selector_init_size)
    else:
        raise ValueError(f"selector name '{selector_name}' not defined")
    
    X_train, X_reserve, y_train, y_reserve = selector.get()
    print("\n### Initialed Dataset:")
    selector.print()

    first_run = True
    test_metrics = []
    training_logs = []
    for stage in params['train']['stages']:
        for i in range(stage['count']):
            if first_run:
                first_run = False
            else:
                print("### Selecting next samples ...")
                X_train, X_reserve, y_train, y_reserve = selector.select(model, size=stage['size'])
                selector.print()
            
            X_valid, y_valid = X_reserve, y_reserve
            if len(X_valid) > 5000:
                X_valid = X_valid[:5000]
                y_valid = y_valid[:5000]

            print(f"\n### Running Training Size {len(X_train)}:")
            model, log = train_basic_cnn(X_train, y_train, X_valid, y_valid, params)

            test_scores = model.evaluate(X_test, y_test)
            print(f"\n### Train Size {len(X_train)} Results:")
            print("Test loss:    ", test_scores[0])
            print("Test accuracy:", test_scores[1])
            print("")

            test_metrics.append({
                "train_size": len(X_train),
                "loss": test_scores[0],
                "accuracy": test_scores[1],
            })
            write_plots(test_metrics)
    
    model, log = train_basic_cnn(X, y, X_test, y_test, params)
    test_scores = model.evaluate(X_test, y_test)
    print(f"\n### Full Train/Test Results:")
    print("Test loss:    ", test_scores[0])
    print("Test accuracy:", test_scores[1])
    print("")

    test_metrics.append({
        "train_size": len(X),
        "loss": test_scores[0],
        "accuracy": test_scores[1],
    })
    write_plots(test_metrics)
    
