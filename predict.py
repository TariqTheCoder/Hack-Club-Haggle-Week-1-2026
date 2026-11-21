# Python script that generates model

import sklearn
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import root_mean_squared_log_error
import numpy as np
import pandas as pd
import ratios

# Preprocessing
dataFrame = pd.read_csv('train.csv')
# dataFrame = ratios.add_ratios(dataFrame)
target = 'hours_spent'
X = dataFrame.drop(columns=["project_id", "hours_spent", "log_hours", "hours_bucket"])
X = X.replace({"True": 1, "False": 0}).replace({True: 1, False: 0})
X = pd.get_dummies(X).fillna(-1).astype(float)
y = np.log1p(dataFrame[target])

# Model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
model0 = RandomForestRegressor(n_estimators=1000, min_samples_leaf=5, n_jobs=-1, )
model1 = HistGradientBoostingRegressor(max_depth=4, learning_rate=0.05, max_iter=300,min_samples_leaf=20)
model0.fit(X_train, y_train)
model1.fit(X_train, y_train)
model0_log = model0.predict(X_test)
model1_log = model1.predict(X_test)
final_log = np.expm1((model0_log + model1_log) / 2)
print("RMSLE: " + str(root_mean_squared_log_error(np.expm1(y_test), final_log)))

# TODO: Make a Config file and find best parameters
# TODO: Seperate models into distinct files
# TODO: Add ExtraTreesRegressor model
# TODO: Find worst features