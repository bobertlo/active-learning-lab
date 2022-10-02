from tensorflow import keras
import numpy as np


def load_mnist_dataset(params):
    num_classes = 10
    if params["train"]["dataset"] == "mnist":
        (X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()
    elif params["train"]["dataset"] == "fashion_mnist":
        (X_train, y_train), (X_test, y_test) = keras.datasets.fashion_mnist.load_data()
    X_train = X_train.astype("float32") / 255
    X_test = X_test.astype("float32") / 255
    X_train = np.expand_dims(X_train, -1)
    X_test = np.expand_dims(X_test, -1)
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)

    return (X_train, y_train), (X_test, y_test)
