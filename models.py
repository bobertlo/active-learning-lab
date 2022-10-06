from tensorflow import keras
from tensorflow.keras import layers
import tensorflow as tf
import numpy as np

def lr_warmup_cosine_decay(global_step,
                           warmup_steps,
                           hold = 0,
                           total_steps=0,
                           start_lr=0.0,
                           target_lr=1e-3):
    # Cosine decay
    # There is no tf.pi so we wrap np.pi as a TF constant
    learning_rate = 0.5 * target_lr * (1 + tf.cos(tf.constant(np.pi) * (global_step - warmup_steps - hold) / float(total_steps - warmup_steps - hold)))

    # Target LR * progress of warmup (=1 at the final warmup step)
    warmup_lr = target_lr * (global_step / warmup_steps)

    # Choose between `warmup_lr`, `target_lr` and `learning_rate` based on whether `global_step < warmup_steps` and we're still holding.
    # i.e. warm up if we're still warming up and use cosine decayed lr otherwise
    if hold > 0:
        learning_rate = tf.where(global_step > warmup_steps + hold,
                                 learning_rate, target_lr)
    
    learning_rate = tf.where(global_step < warmup_steps, warmup_lr, learning_rate)
    return learning_rate

class WarmUpCosineDecay(keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, start_lr, target_lr, warmup_steps, total_steps, hold):
        super().__init__()
        self.start_lr = start_lr
        self.target_lr = target_lr
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.hold = hold

    def __call__(self, step):
        lr = lr_warmup_cosine_decay(global_step=step,
                                    total_steps=self.total_steps,
                                    warmup_steps=self.warmup_steps,
                                    start_lr=self.start_lr,
                                    target_lr=self.target_lr,
                                    hold=self.hold)

        return tf.where(
            step > self.total_steps, 0.0, lr, name="learning_rate"
        )

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

    n_steps = len(X_train) / params['train']['bs'] * params['train']['epochs']
    warmup = n_steps * 0.3

    print(n_steps, warmup)
    schedule = WarmUpCosineDecay(
        start_lr = 0,
        target_lr = params['train']['lr'],
        warmup_steps = warmup,
        total_steps = n_steps,
        hold = warmup
    )

    optimizer = keras.optimizers.Adam(learning_rate=schedule)
    model.compile(loss="categorical_crossentropy", optimizer=optimizer, metrics=["accuracy"])

    history = model.fit(
        datagen.flow(
            X_train,
            y_train,
            seed=params['aug']['seed'],
            shuffle=True
        ),
        validation_data=(X_valid, y_valid),
        batch_size=params['train']['bs'],
        epochs=params['train']['epochs'],
    )

    return model, history