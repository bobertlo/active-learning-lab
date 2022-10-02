from tensorflow import keras
from tensorflow.keras import layers
import tensorflow as tf
import numpy as np

import tensorflow_addons as tfa


def training_datagen(params):
    return keras.preprocessing.image.ImageDataGenerator(
        width_shift_range=params["aug"]["x_shift"],
        height_shift_range=params["aug"]["y_shift"],
        rotation_range=params["aug"]["rot"],
        zoom_range=[
            params["aug"]["zoom_min"],
            params["aug"]["zoom_max"],
        ],
    )


def basic_cnn_model():
    return keras.Sequential(
        [
            keras.Input(shape=(28, 28, 1)),
            layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
            layers.Conv2D(64, kernel_size=(5, 5), activation="relu"),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Dropout(0.25),
            layers.Flatten(),
            layers.Dense(128, activation="relu"),
            layers.Dropout(0.5),
            layers.Dense(10, activation="softmax"),
        ]
    )


def train_basic_cnn(X_train, y_train, X_valid, y_valid, params):
    model = basic_cnn_model()
    datagen = training_datagen(params)

    schedule = tfa.optimizers.CyclicalLearningRate(
        initial_learning_rate=1e-6,
        maximal_learning_rate=params["train"]["lr"],
        scale_fn=lambda x: 1 / (2.0 ** (x - 1)),
        step_size=2 * len(X_train),
    )

    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath="./checkpoint",
        save_weights_only=True,
        monitor="val_loss",
        mode="max",
        save_best_only=True,
    )

    optimizer = keras.optimizers.Adam(learning_rate=schedule)
    model.compile(
        loss="categorical_crossentropy", optimizer=optimizer, metrics=["accuracy"]
    )

    history = model.fit(
        datagen.flow(X_train, y_train, seed=params["aug"]["seed"], shuffle=True),
        validation_data=(X_valid, y_valid),
        batch_size=params["train"]["bs"],
        epochs=params["train"]["epochs"],
        callbacks=[checkpoint_callback],
    )

    return model, history
