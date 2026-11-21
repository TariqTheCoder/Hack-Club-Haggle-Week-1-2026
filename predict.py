# Python script that generates model

import sklearn
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import root_mean_squared_log_error
import numpy as np
import pandas as pd

# Preprocessing
dataFrame = pd.read_csv('train.csv')
target = 'hours_spent'
X = dataFrame.drop(columns=["project_id", "hours_spent", "log_hours", "hours_bucket"])
X = X.replace({"True": 1, "False": 0}).replace({True: 1, False: 0})
X = pd.get_dummies(X).fillna(-1).astype(float)
y = np.log1p(dataFrame[target])

# Model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
model = RandomForestRegressor(n_estimators=1000, min_samples_leaf=5, n_jobs=-1, )
model.fit(X_train, y_train)
prediction = np.expm1(model.predict(X_test))
print("RMSLE: " + str(root_mean_squared_log_error(np.expm1(y_test), prediction)))

# TODO: Make a Config file and find best parameters