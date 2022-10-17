import os
import json
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns


def run_cmd(cmd):
    print(f"### Running Command: {cmd}")
    ret = os.system(cmd)
    if ret != 0:
        raise RuntimeError(f"command {cmd} failed.")


def run_experiment(exp, dataset, selector, seed, source_branch="main"):
    tag = f"{exp}-{dataset}-{selector}{seed}"
    try:
        run_cmd(f"git checkout {tag}")
        print(f"experiment '{tag}' found in cache, skipping ...")
    except RuntimeError:
        print(f"experiment f'{tag}' not found, running ...")
        run_cmd(f"git checkout {source_branch}")
        run_cmd(
            f"dvc exp run -S train.selector={selector} -S train.dataset={dataset} -S train.seed={seed}"
        )
        run_cmd(f"git add -u")
        run_cmd(f"git commit -m 'run {tag}'")
        run_cmd(f"git tag {tag}")


def load_metrics(dataset, method, seed):
    sizes, accs, losses = [], [], []
    with open("plots/accuracy.json", "r") as infile:
        acc_json = json.load(infile)
    for x in acc_json:
        accs.append(x["acc"])
        sizes.append(x["train_size"])
    with open("plots/loss.json", "r") as infile:
        loss_json = json.load(infile)
    for x in loss_json:
        losses.append(x["loss"])

    df = pd.DataFrame({"size": sizes, "accuracy": accs, "loss": losses})
    df["dataset"] = dataset
    df["method"] = method
    df["seed"] = seed
    return df


if __name__ == "__main__":
    source_branch = "exp1"
    exp_name = "exp1"

    metrics = None
    for dataset in ["fashion_mnist"]:
        for seed in range(5):
            # for method in ["random", "lc", "sm", "ent", "strat_lc", "strat_sm", "strat_ent"]:
            for method in ['random', 'lc', 'sm', 'ent']:
                run_experiment(
                    exp_name, dataset, method, seed, source_branch=source_branch
                )
                exp_metrics = load_metrics(dataset, method, seed)
                if metrics is None:
                    metrics = exp_metrics
                else:
                    metrics = metrics.append(exp_metrics)

    run_cmd(f"git checkout {source_branch}")

    print("saving metrics to metrics.csv")
    metrics.to_csv("metrics.csv")

    print("generating plot ...")

    def metric_label(x):
        if x == "random":
            return "Random Baseline"
        elif x == "lc":
            return "Least Confidence"
        elif x == "sm":
            return "Smallest Margin"
        elif x == "ent":
            return "Max Entropy"
        elif x == "strat_lc":
            return "Least Confidence Stratified"
        elif x == "strat_sm":
            return "Smallest Margin Stratified"
        elif x == "strat_ent":
            return "Max Entropy Stratified"

    metrics["method"] = metrics["method"].map(metric_label)

    plt.figure(figsize=(12, 5))
    fig = sns.lineplot(
        metrics,
        x="size",
        y="accuracy",
        hue="method",
    )
    plt.ylabel("Accuracy")
    plt.xlabel("Training Samples")
    # plt.yticks([0.975, 0.98, 0.985, 0.99, 0.995])
    plt.title("Fashion MNIST")
    plt.legend(title="Selection Method")
    plt.savefig("plot.png")
