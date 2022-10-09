import os
import json
import pandas as pd

def run_cmd(cmd):
    print(f"### Running Command: {cmd}")
    ret = os.system(cmd)
    if ret != 0:
        raise RuntimeError(f"command {cmd} failed.")

def run_experiment(expname, selector, seed, source_branch="main"):
    try:
        run_cmd(f"git checkout {expname}-{selector}{seed}")
        print(f"experiment '{expname}-{selector}{seed}' found in cache, skipping ...")
    except RuntimeError:
        print(f"experiment f'{expname}-{selector}{seed}' not found, running ...")
        run_cmd(f"git checkout {source_branch}")
        run_cmd(f"dvc exp run -S train.selector={selector} -S prep.seed={seed}")
        run_cmd(f"git add -u")
        run_cmd(f"git commit -m 'run {expname}: {selector}{seed}'")
        run_cmd(f"git tag {expname}-{selector}{seed}")

def load_metrics(method, seed):
    sizes, accs, losses = [], [], []
    with open ("plots/accuracy.json", "r") as infile:
        acc_json = json.load(infile)
    for x in acc_json:
        accs.append(x['acc'])
        sizes.append(x['train_size'])
    with open ("plots/loss.json", "r") as infile:
        loss_json = json.load(infile)
    for x in loss_json:
        losses.append(x['loss'])

    df = pd.DataFrame({"size": sizes, "accuracy": accs, "loss": losses})
    df['method'] = method
    df['seed'] = seed
    return df

if __name__ == "__main__":
    source_branch = "main"
    exp_name = "mnist2"

    metrics = None
    for seed in range(5):
        for method in ['random', 'lc', 'sm', 'ent']:
            run_experiment(exp_name, method, seed, source_branch=source_branch)
            exp_metrics = load_metrics(method, seed)
            if metrics is None:
                metrics = exp_metrics
            else:
                metrics = metrics.append(exp_metrics)

    run_cmd(f'git checkout {source_branch}')

    print("saving metrics to metrics.csv")
    metrics.to_csv("metrics.csv")