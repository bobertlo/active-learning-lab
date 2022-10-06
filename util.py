from ruamel.yaml import YAML
import os
import json

def load_params(file="params.yaml"):
    with open(file, "r") as infile:
        params = YAML().load(infile)
    return params

def print_params(params):
    for group in params:
        print(f'{group}:')
        for p in params[group]:
            print(f'  {p}: {params[group][p]}')

def write_plots(test_metrics, prefix):
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
    with open(f"plots/{prefix}_loss.json", "w") as outfile:
        json.dump(loss_metrics, outfile)

    acc_metrics = []
    for x in test_metrics:
        acc_metrics.append({
            "train_size": x['train_size'],
            "acc": x['accuracy'],
        })
    with open(f"plots/{prefix}_acc.json", "w") as outfile:
        json.dump(acc_metrics, outfile)