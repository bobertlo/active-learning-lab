import numpy as np

from sklearn.model_selection import train_test_split


class Selector:
    def __init__(self, X, y, train_size=1000, seed=0):
        self.X_train, self.X_reserve, self.y_train, self.y_reserve = train_test_split(
            X, y, train_size=train_size, random_state=seed, stratify=y
        )

    def print(self):
        print(f"train:   X{self.X_train.shape} y{self.y_train.shape}")
        print(f"reserve: X{self.X_reserve.shape} y{self.y_reserve.shape}")

    def label_samples(self, idxs):
        X_new = self.X_reserve[idxs]
        y_new = self.y_reserve[idxs]
        self.X_reserve = np.delete(self.X_reserve, idxs, 0)
        self.y_reserve = np.delete(self.y_reserve, idxs, 0)

        self.X_train = np.append(self.X_train, X_new, axis=0)
        self.y_train = np.append(self.y_train, y_new, axis=0)

    def get(self):
        return self.X_train, self.X_reserve, self.y_train, self.y_reserve

    def select(self, model, size=1000):
        raise NotImplementedError


class BaselineSelector(Selector):
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


class LeastConfidenceSelector(Selector):
    def select(self, model, size=1000):
        max_preds = model.predict(self.X_reserve).max(axis=1)
        sorted_preds = sorted(enumerate(max_preds), key=lambda x: x[1])
        idxs = [x[0] for x in sorted_preds[:size]]
        self.label_samples(idxs)

        return self.X_train, self.X_reserve, self.y_train, self.y_reserve


class StratifiedLeastConfidenceSelector(Selector):
    def select(self, model, size=1000):
        predictions = model.predict(self.X_reserve)

        pred_classes = np.apply_along_axis(np.argmax, 1, predictions)
        pred_confidence = predictions.max(axis=1)
        sorted_preds = sorted(
            enumerate(zip(pred_classes, pred_confidence)), key=lambda x: x[1][1]
        )

        samples_per_class = int(size / 10)
        idxs = []
        for i in range(10):
            idxs.extend(
                [x[0] for x in sorted_preds if x[1][0] == i][:samples_per_class]
            )

        self.label_samples(idxs)

        return self.X_train, self.X_reserve, self.y_train, self.y_reserve


class SmallestMarginSelector(Selector):
    def select(self, model, size=1000):
        def margin(x):
            sx = sorted(x, reverse=True)
            return sx[0] - sx[1]

        predictions = model.predict(self.X_reserve)
        margins = np.apply_along_axis(margin, 1, predictions)

        sorted_preds = sorted(enumerate(margins), key=lambda x: x[1])
        idxs = [x[0] for x in sorted_preds[:size]]
        self.label_samples(idxs)

        return self.X_train, self.X_reserve, self.y_train, self.y_reserve


class StratifiedSmallestMarginSelector(Selector):
    def select(self, model, size=1000):
        def margin(x):
            sx = sorted(x, reverse=True)
            return sx[0] - sx[1]

        predictions = model.predict(self.X_reserve)

        pred_classes = np.apply_along_axis(np.argmax, 1, predictions)
        pred_margins = np.apply_along_axis(margin, 1, predictions)
        sorted_preds = sorted(
            enumerate(zip(pred_classes, pred_margins)), key=lambda x: x[1][1]
        )

        samples_per_class = int(size / 10)
        idxs = []
        for i in range(10):
            idxs.extend(
                [x[0] for x in sorted_preds if x[1][0] == i][:samples_per_class]
            )

        idxs = [x[0] for x in sorted_preds[:size]]
        self.label_samples(idxs)

        return self.X_train, self.X_reserve, self.y_train, self.y_reserve


class MaxEntropySelector(Selector):
    def select(self, model, size=1000):
        def entropy(x):
            def h(x):
                if x == 0:
                    return 0
                return x * np.log2(x)

            y = np.apply_along_axis(h, 0, [x])
            return -np.sum(y)

        predictions = model.predict(self.X_reserve)
        entropies = np.apply_along_axis(entropy, 1, predictions)

        sorted_preds = sorted(enumerate(entropies), key=lambda x: x[1], reverse=True)
        idxs = [x[0] for x in sorted_preds[:size]]
        self.label_samples(idxs)

        return self.X_train, self.X_reserve, self.y_train, self.y_reserve


class StratifiedMaxEntropySelector(Selector):
    def select(self, model, size=1000):
        def entropy(x):
            def h(x):
                if x == 0:
                    return 0
                return x * np.log2(x)

            y = np.apply_along_axis(h, 0, [x])
            return -np.sum(y)

        predictions = model.predict(self.X_reserve)

        pred_classes = np.apply_along_axis(np.argmax, 1, predictions)
        pred_entropies = np.apply_along_axis(entropy, 1, predictions)
        sorted_preds = sorted(
            enumerate(zip(pred_classes, pred_entropies)),
            key=lambda x: x[1][1],
            reverse=True,
        )

        samples_per_class = int(size / 10)
        idxs = []
        for i in range(10):
            idxs.extend(
                [x[0] for x in sorted_preds if x[1][0] == i][:samples_per_class]
            )
        self.label_samples(idxs)

        return self.X_train, self.X_reserve, self.y_train, self.y_reserve
