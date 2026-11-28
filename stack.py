import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression
from sklearn.metrics import root_mean_squared_error
import random_forest
import hist_gradient_boosting as HGB
import extra_trees
import add_features
import config

df = pd.read_csv("train.csv")
X = df.drop(columns=["project_id", "hours_spent", "log_hours", "hours_bucket"])
X = X.replace({"True": 1, "False": 0}).replace({True: 1, False: 0})
X = add_features.add_features(X)
X = pd.get_dummies(X).fillna(-1).astype(float)
y = np.log1p(df["hours_spent"]).values

builders = {
    "random_forest": random_forest.random_forest_model,
    "hist_gradient": HGB.hist_gradient,
    "extra_trees": extra_trees.extra_trees_model,
}
names = list(builders)

# Level 1: out-of-fold predictions, one column per model
P = np.zeros((len(X), len(names)))
for tr, va in KFold(config.CV_FOLDS, shuffle=True, random_state=config.RANDOM_STATE).split(X):
    for j, build in enumerate(builders.values()):
        P[va, j] = build(X.iloc[tr], y[tr]).predict(X.iloc[va])

for j, n in enumerate(names):
    print(f"{n:15s} solo: {root_mean_squared_error(y, P[:, j]):.4f}")
print(f"{'plain mean':15s}     : {root_mean_squared_error(y, P.mean(axis=1)):.4f}")

# Level 2: non-negative linear blend, scored honestly with a second CV on P
def blender():
    return LinearRegression(positive=True, fit_intercept=False)

oof2 = np.zeros(len(y))
for tr, va in KFold(5, shuffle=True, random_state=0).split(P):
    oof2[va] = blender().fit(P[tr], y[tr]).predict(P[va])
print(f"{'stacked (nested)':15s}: {root_mean_squared_error(y, oof2):.4f}")

# Final weights fit on all out-of-fold predictions
w = blender().fit(P, y).coef_
print("\nweights:", dict(zip(names, np.round(w, 3))), "sum:", round(w.sum(), 3))
print("\nPaste into config.py:")
print(f"BLEND_WEIGHTS = {[round(float(v), 4) for v in w]}  # order: {names}")