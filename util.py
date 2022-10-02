from ruamel.yaml import YAML
import os

def load_params(file="params.yaml"):
    with open(file, "r") as infile:
        params = YAML().load(infile)
    return params

def print_params(params):
    for group in params:
        print(f'{group}:')
        for p in params[group]:
            print(f'  {p}: {params[group][p]}')