import os

def run_command(cmd):
    print(f"### Running Command: {cmd}")
    ret = os.system(cmd)
    if ret != 0:
        raise RuntimeError(f"command {cmd} failed.")

def run_experiment(expname, selector, seed):
    run_command(f"dvc exp run -S train.selector={selector} -S prep.seed={seed}")
    run_command(f"git add -u")
    run_command(f"git commit -m 'run {expname}: {selector}{seed}'")
    run_command(f"git tag {expname}-{selector}{seed}")

if __name__ == "__main__":
    for seed in range(6):
        for method in ['random', 'sm']:
            run_experiment("mnist2", method, seed)