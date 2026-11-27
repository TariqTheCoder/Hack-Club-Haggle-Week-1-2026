import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import root_mean_squared_error
import random_forest
import hist_gradient_boosting as HGB

df = pd.read_csv("train.csv")
X = df.drop(columns=["project_id", "hours_spent", "log_hours", "hours_bucket"])
X = X.replace({"True": 1, "False": 0}).replace({True: 1, False: 0})
X = pd.get_dummies(X).fillna(-1).astype(float)
y = np.log1p(df["hours_spent"])

scores = []
for tr, va in KFold(5, shuffle=True, random_state=42).split(X):
    rf = random_forest.random_forest_model(X.iloc[tr], y.iloc[tr])
    hgb = HGB.hist_gradient(X.iloc[tr], y.iloc[tr])
    pred = (rf.predict(X.iloc[va]) + hgb.predict(X.iloc[va])) / 2
    scores.append(root_mean_squared_error(y.iloc[va], pred))
print(np.mean(scores), np.std(scores))