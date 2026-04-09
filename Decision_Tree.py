import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from collections import Counter

def import_data(path):
    df = pd.read_csv(path, sep=",", header=None)
    label = (df[1] == "M").astype(int)
    df = df.drop([0, 1], axis=1)
    df.columns = [f"feature{i}" for i in range(1, df.shape[1] + 1)]
    return df, label.values

def split(x, y):
    return train_test_split(x, y, test_size=0.2, random_state=42)

class Node:
    def __init__(self, feature_idx=None, threshold=None, info_gain=None,
                 left=None, right=None, value=None, class_counts=None):
        self.feature_idx = feature_idx
        self.threshold = threshold
        self.info_gain = info_gain
        self.left = left
        self.right = right
        self.value = value
        self.class_counts = class_counts

class decision_tree:
    def __init__(self, min_samples_split=2, max_depth=4):
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth

    def fit(self, x, y):
        dataset = np.concatenate([x, y.reshape(-1, 1)], axis=1)
        self.root = self.build_tree(dataset)

    def build_tree(self, dataset, curr_depth=0):
        x, y = dataset[:, :-1], dataset[:, -1]
        n_samples, n_features = x.shape

        if n_samples >= self.min_samples_split and curr_depth <= self.max_depth:
            best = self.best_split(dataset, n_features)
            if best["info_gain"] > 0:
                left = self.build_tree(best["left_dataset"], curr_depth + 1)
                right = self.build_tree(best["right_dataset"], curr_depth + 1)
                return Node(best["feature_idx"], best["threshold"],
                            best["info_gain"], left, right)

        counts = Counter(y)
        value = counts.most_common(1)[0][0]
        return Node(value=value, class_counts=counts)

    def best_split(self, dataset, n_features):
        best = {"feature_idx": None, "threshold": None,
                "info_gain": -1, "left_dataset": None, "right_dataset": None}

        for feature_idx in range(n_features):
            values = dataset[:, feature_idx]
            thresholds = np.unique(values)

            for threshold in thresholds:
                left, right = self.split(dataset, feature_idx, threshold)

                if len(left) and len(right):
                    parent_y = dataset[:, -1]
                    left_y, right_y = left[:, -1], right[:, -1]

                    gain = self.information_gain(parent_y, left_y, right_y)

                    if gain > best["info_gain"]:
                        best.update({
                            "feature_idx": feature_idx,
                            "threshold": threshold,
                            "info_gain": gain,
                            "left_dataset": left,
                            "right_dataset": right
                        })
        return best

    def split(self, dataset, feature_idx, threshold):
        left = np.array([row for row in dataset if row[feature_idx] <= threshold])
        right = np.array([row for row in dataset if row[feature_idx] > threshold])
        return left, right

    def information_gain(self, parent_y, left_y, right_y):
        w_l = len(left_y) / len(parent_y)
        w_r = len(right_y) / len(parent_y)
        return self.entropy(parent_y) - (w_l * self.entropy(left_y) + w_r * self.entropy(right_y))

    def entropy(self, y):
        entropy = 0
        for c in np.unique(y):
            p = len(y[y == c]) / len(y)
            entropy += -p * np.log2(p)
        return entropy

    def predict_proba(self, x):
        probs = [self._predict_proba_single(row, self.root) for row in x]
        return np.array(probs)

    def _predict_proba_single(self, row, node):
        if node.value is not None:
            counts = node.class_counts
            total = sum(counts.values())
            p0 = counts.get(0, 0) / total
            p1 = counts.get(1, 0) / total
            return [p0, p1]

        if row[node.feature_idx] <= node.threshold:
            return self._predict_proba_single(row, node.left)
        else:
            return self._predict_proba_single(row, node.right)

    def predict(self, x):
        probs = self.predict_proba(x)
        return np.argmax(probs, axis=1)

def Tree(name,path, test=None):
    df, label = import_data(path)
    x_train, x_test, y_train, y_test = split(df, label)

    x_train = x_train.values.astype(float)
    x_test = x_test.values.astype(float)
    y_train = y_train.flatten()
    y_test = y_test.flatten()

    name = decision_tree(min_samples_split=2, max_depth=4)
    name.fit(x_train, y_train)

    predictions = name.predict(x_test)
    accuracy = np.mean(predictions == y_test) * 100

    if test is None:
        return accuracy
    else:
        test = np.array(test).reshape(1, -1)
        prob = name.predict_proba(test)
        pred = np.argmax(prob, axis=1)
        print("Decision Tree accuracy: ",accuracy)
        return prob, pred, accuracy