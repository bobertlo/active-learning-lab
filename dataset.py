from tensorflow import keras
import numpy as np
from sklearn.model_selection import train_test_split

def load_mnist_dataset():
    num_classes = 10
    (X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()
    X_train = X_train.astype("float32") / 255
    X_test = X_test.astype("float32") / 255
    X_train = np.expand_dims(X_train, -1)
    X_test = np.expand_dims(X_test, -1)
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)

    return (X_train, y_train), (X_test, y_test)

def prepare_training_set_random(X, y, train_size=.2, seed=0):
    return train_test_split(
        X, y, 
        train_size=train_size,
        random_state=seed,
        stratify=y
        )
