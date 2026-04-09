import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

def import_data(path):
    df = pd.read_csv(path, sep=",", header=None)
    label = (df[[1]] == "M").astype(int).values.flatten()
    df = df.drop([0, 1], axis=1)
    df.columns = [f"feature{i}" for i in range(1, df.shape[1] + 1)]
    return df, label

def split_data(df, label):
    return train_test_split(df, label, random_state=42, stratify=label, test_size=0.25)

def normalize(x_train, x_test, a=None):
    mean = np.mean(x_train, axis=0).values.reshape(1, -1)
    std  = np.std(x_train,  axis=0).values.reshape(1, -1)
    x_train = (x_train.values - mean) / std
    x_test  = (x_test.values  - mean) / std
    if a is not None:
        a = (a - mean) / std
        return x_train, x_test, a
    return x_train, x_test, None

class LogisticRegression:
    def __init__(self, x_train, y_train, epoch=100, alpha=0.25):
        self.w, self.b = self._train(x_train, y_train, epoch, alpha)

    def _train(self, x_train, y_train, epoch, alpha):
        m = y_train.shape[0]
        w = np.zeros((x_train.shape[1], 1))
        b = 0.0
        y = y_train.reshape(-1, 1)

        for _ in range(epoch):
            z = np.clip(x_train @ w + b, -500, 500)
            A = 1 / (1 + np.exp(-z))
            dz = A - y
            w -= alpha * (x_train.T @ dz) / m
            b -= alpha * np.sum(dz) / m

        return w, b

    def predict_proba(self, x):
        z = np.clip(x @ self.w + self.b, -500, 500)
        p1 = 1 / (1 + np.exp(-z))           # shape (n, 1)
        return np.hstack((1 - p1, p1))       # shape (n, 2)

    def predict(self, x):
        return (self.predict_proba(x)[:, 1] >= 0.5).astype(int)

def regression(path, new=None):
    if new is not None:
        new = new.reshape(1, -1).astype(float)

    df, label = import_data(path)
    x_train, x_test, y_train, y_test = split_data(df, label)
    x_train, x_test, new_norm = normalize(x_train, x_test, new)

    model = LogisticRegression(x_train, y_train, epoch=1000, alpha=0.5)

    y_pred = model.predict(x_test)
    acc = round(np.mean(y_pred == y_test) * 100, 2)

    if new is None:
        return acc

    proba      = model.predict_proba(new_norm)   # shape (1, 2)
    p0         = round(float(proba[0, 0]), 4)
    p1         = round(float(proba[0, 1]), 4)
    prediction = int(proba[0, 1] >= 0.5)
    label_str  = "Malignant (M)" if prediction == 1 else "Benign (B)"
    prob = np.array([p0,p1])
    print(f"Logistic Regression accuracy: {acc}")
    return p0, p1, prediction, label_str, acc
