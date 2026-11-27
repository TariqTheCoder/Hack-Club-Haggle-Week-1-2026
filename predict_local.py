# Python script that generates model

from sklearn.model_selection import train_test_split
from sklearn.metrics import root_mean_squared_log_error
import numpy as np
import pandas as pd
import random_forest
import hist_gradient_boosting as HGB


dataFrame = pd.read_csv('train.csv')
target = "hours_spent"

X = dataFrame.drop(columns=["project_id", "hours_spent", "log_hours", "hours_bucket"])
X = X.replace({"True": 1, "False": 0}).replace({True: 1, False: 0})
X = pd.get_dummies(X).fillna(-1).astype(float)
y = np.log1p(dataFrame[target])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
random_forest = random_forest.random_forest_model(X_train, y_train)
hist_gradient = HGB.hist_gradient(X_train, y_train)
random_forest_log = random_forest.predict(X_test)
hist_gradient_log = hist_gradient.predict(X_test)
final_log = np.expm1((random_forest_log + hist_gradient_log) / 2)
print("RMSLE: " + str(root_mean_squared_log_error(np.expm1(y_test), final_log)))

# TODO: Make a Config file and find best parameters
# TODO: Seperate models into distinct files
# TODO: Add ExtraTreesRegressor model
# TODO: Find worst features