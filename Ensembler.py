import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from Decision_Tree import import_data as dt_import, split, decision_tree
from Logistic_Regression import import_data as lr_import, split_data
from knn import load_data as knn_import, split_data as knn_split, KNN


PATH = "C:\\Users\\mdkha\\Desktop\\MY stuff\\python\\ML\\wdbc.data"


class Ensemble:
    def __init__(self, models, weights=None):
        self.models = models
        n = len(models)
        self.weights = np.array(weights) if weights else np.ones(n) / n

    def _collect_probas(self, x):
        x = np.array(x)
        if x.ndim == 1:
            x = x.reshape(1, -1)
        return np.array([m.predict_proba(x) for m in self.models])

    def soft_predict_proba(self, x):
        probas = self._collect_probas(x)
        return np.tensordot(self.weights, probas, axes=([0], [0]))

    def soft_predict(self, x):
        return np.argmax(self.soft_predict_proba(x), axis=1)

    def accuracy(self, x, y):
        preds = self.soft_predict(x)
        return round(np.mean(preds == y) * 100, 2)


class LRWrapper:
    def __init__(self, model, mean, std):
        self.model = model
        self.mean = mean
        self.std = std

    def predict_proba(self, x):
        x = np.array(x)
        if x.ndim == 1:
            x = x.reshape(1, -1)

        x_norm = (x - self.mean) / self.std
        return self.model.predict_proba(x_norm)



df_dt, lbl_dt = dt_import(PATH)
x_tr, x_te, y_tr, y_te = split(df_dt, lbl_dt)

x_tr_dt = x_tr.values.astype(float)
x_te_dt = x_te.values.astype(float)
y_tr_dt = y_tr.flatten()
y_te_dt = y_te.flatten()

dt = decision_tree(min_samples_split=2, max_depth=4)
dt.fit(x_tr_dt, y_tr_dt)


df_lr, lbl_lr = lr_import(PATH)
x_tr2, x_te2, y_tr2, y_te2 = split_data(df_lr, lbl_lr)

mean = np.mean(x_tr2, axis=0).values.reshape(1, -1)
std  = np.std(x_tr2, axis=0).values.reshape(1, -1)

x_tr_lr = (x_tr2.values - mean) / std
x_te_lr = (x_te2.values - mean) / std

from Logistic_Regression import LogisticRegression
lr = LogisticRegression(x_tr_lr, y_tr2)

lr_wrapped = LRWrapper(lr, mean, std)


# KNN
df_knn, lbl_knn = knn_import(PATH)
x_tr3, x_te3, y_tr3, y_te3 = knn_split(df_knn, lbl_knn)

x_tr_knn = x_tr3.values.astype(float)
x_te_knn = x_te3.values.astype(float)

knn = KNN(7)
knn.train(x_tr_knn, y_tr3.values)

def model_accuracy(model, x, y):
    preds = np.argmax(model.predict_proba(x), axis=1)
    return round(np.mean(preds == y) * 100, 2)


# Individual model accuracies
acc_dt  = model_accuracy(dt, x_te_dt, y_te_dt)
acc_lr  = model_accuracy(lr_wrapped, x_te_dt, y_te_dt)
acc_knn = model_accuracy(knn, x_te_dt, y_te_dt)

print("\nIndividual Model Accuracies:")
print(f"Decision Tree      : {acc_dt}%")
print(f"Logistic Regression: {acc_lr}%")
print(f"KNN                : {acc_knn}%")

# ================= ENSEMBLE =================
ens = Ensemble([dt, lr_wrapped, knn])


# ================= ACCURACY =================
print("Ensemble Accuracy:", ens.accuracy(x_te_dt, y_te_dt))


# ================= CORRELATION HEATMAP =================
probas = ens._collect_probas(x_te_dt)
prob_class1 = probas[:, :, 1]

corr_matrix = np.corrcoef(prob_class1)

plt.figure()
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm",
            xticklabels=["DT", "LR", "KNN"],
            yticklabels=["DT", "LR", "KNN"])
plt.title("Error Correlation Heatmap")
plt.show()


# ================= DIVERSITY =================
preds = np.array([
    np.argmax(dt.predict_proba(x_te_dt), axis=1),
    np.argmax(lr_wrapped.predict_proba(x_te_dt), axis=1),
    np.argmax(knn.predict_proba(x_te_dt), axis=1)
])

def disagreement(a, b):
    return np.mean(a != b)

diversity = np.mean([
    disagreement(preds[0], preds[1]),
    disagreement(preds[0], preds[2]),
    disagreement(preds[1], preds[2])
])

print("Diversity:", round(diversity, 4))


# ================= ENSEMBLE GAIN =================
def model_acc(model, x, y):
    return np.mean(np.argmax(model.predict_proba(x), axis=1) == y)

acc_dt  = model_acc(dt, x_te_dt, y_te_dt)
acc_lr  = model_acc(lr_wrapped, x_te_dt, y_te_dt)
acc_knn = model_acc(knn, x_te_dt, y_te_dt)

ensemble_acc = ens.accuracy(x_te_dt, y_te_dt) / 100
best_acc = max(acc_dt, acc_lr, acc_knn)

gain = ensemble_acc - best_acc

print("Best Individual:", round(best_acc * 100, 2), "%")
print("Ensemble Gain:", round(gain * 100, 2), "%")


# ================= CUSTOM INPUT =================
sample = np.array([[14.54,27.54,96.73,658.8,0.1139,0.1595,0.1639,0.07364,0.2303,0.07077,0.37,1.033,2.879,32.55,0.005607,0.0424,0.04741,0.0109,0.01857,0.005466,17.46,37.13,124.1,943.2,0.1678,0.6577,0.7026,0.1712,0.4218,0.1341]])

proba = ens.soft_predict_proba(sample)
pred = ens.soft_predict(sample)

print("\nCustom Input Prediction:")
print(f"P(Benign)={proba[0,0]:.4f}, P(Malignant)={proba[0,1]:.4f}")
print("Class:", "Malignant" if pred[0] == 1 else "Benign")