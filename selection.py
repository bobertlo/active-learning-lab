import numpy as np

from sklearn.model_selection import train_test_split

class BaselineSelector:
    def __init__(self, X, y, train_size=1000, seed=0):
        self.X_train, self.X_reserve, self.y_train, self.y_reserve = train_test_split(
            X, y, train_size=train_size, random_state=seed, stratify=y
        )

    def print(self):
        print(f"train:  X{self.X_train.shape} y{self.y_train.shape}")
        print(f"reserve: X{self.X_reserve.shape} y{self.y_reserve.shape}")

    def get(self):
        return self.X_train, self.X_reserve, self.y_train, self.y_reserve

    def select(self, model, size=1000):
        X_new, self.X_reserve, y_new, self.y_reserve = train_test_split(
            self.X_reserve,
            self.y_reserve,
            train_size=size,
            stratify=self.y_reserve,
        )

        self.X_train = np.append(self.X_train, X_new, axis=0)
        self.y_train = np.append(
            self.y_train,
            y_new,
            axis=0,
        )

        return self.X_train, self.X_reserve, self.y_train, self.y_reserve

class MarginSelector:
    def __init__(self, X, y, train_size=1000, seed=0):
        self.X_train, self.X_reserve, self.y_train, self.y_reserve = train_test_split(
            X, y, train_size=train_size, random_state=seed, stratify=y
        )
    
    def print(self):
        print(f"train:  X{self.X_train.shape} y{self.y_train.shape}")
        print(f"reserve: X{self.X_reserve.shape} y{self.y_reserve.shape}")
    
    def get(self):
        return self.X_train, self.X_reserve, self.y_train, self.y_reserve

    def select(self, model, size=1000):
        def margin(x):
            sx = sorted(x, reverse=True)
            return sx[0] - sx[1]

        predictions = model.predict(self.X_reserve)
        margins = np.apply_along_axis(margin, 1, predictions)

        sorted_preds = sorted(enumerate(margins), key=lambda x: x[1])
        idxs = [x[0] for x in sorted_preds[:size]]

        X_new = self.X_reserve[idxs]
        y_new = self.y_reserve[idxs]
        self.X_reserve = np.delete(self.X_reserve, idxs, 0)
        self.y_reserve = np.delete(self.y_reserve, idxs, 0)

        self.X_train = np.append(self.X_train, X_new, axis=0)
        self.y_train = np.append(self.y_train, y_new, axis=0)

        return self.X_train, self.X_reserve, self.y_train, self.y_reserve
