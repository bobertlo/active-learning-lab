# Active Learning Lab

This experiment is a simple simulation of different active learning methods to demonstrate their performance.

## Experiment Results

![experiment results](images/exp1-mnist-fashion.png)

Starting with a small random sample of 500 labeled images, I trained a CNN model and used it to select the next samples with the following strategies:

- *Least Confidence*: Selecting samples where the highest softmax output had the lowest values.
- *Smallest Margin*: Selecting the samples with the lowest margin between the first and second highest softmax outputs.
- *Maximum Entropy*: Selecting the samples with the highest entropy in softmax outputs.

A control group was included which selects samples randomly from the remaining pool, and each group was tested 5 times.
For each iteration a new CNN is trained from scratch on the dataset with the newly added samples.

## Experiment Environment

To run the experiment reproducibly a [DVC](https://dvc.org/) pipeline was created to iteratively train a model using a selection method.
All configuration is stored in the `params.yaml` and multiple tests are run for each method using different random seeds provided in that file.
A script named `exp_run.py`, which runs each experiment and saves the results to a new commit and creates a git tag, has been included.
