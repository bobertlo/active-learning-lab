import os

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

if __name__ == "__main__":
    source_branch = "main"
    exp_name = "mnist2"

    for seed in [0]:
        for method in ['random', 'sm']:
            run_experiment(exp_name, method, seed, source_branch=source_branch)

    run_cmd(f'git checkout {source_branch}')