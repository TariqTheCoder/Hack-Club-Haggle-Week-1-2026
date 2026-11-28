# Python script that generates model

from sklearn.model_selection import train_test_split
from sklearn.metrics import root_mean_squared_log_error
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, HistGradientBoostingRegressor
import numpy as np
import pandas as pd
import add_features
import config
from catboost import CatBoostRegressor


def random_forest_model(X, y):
    model = RandomForestRegressor(**config.RANDOM_FOREST)
    model.fit(X, y)
    return model


def hist_gradient_model(X, y):
    model = HistGradientBoostingRegressor(**config.HIST_GRADIENT)
    model.fit(X, y)
    return model


def extra_trees_model(X, y):
    model = ExtraTreesRegressor(**config.EXTRA_TREES)
    model.fit(X, y)
    return model

def cat_boost_model(X, y):
    model = CatBoostRegressor(**config.CAT_BOOST)
    model.fit(X, y)
    return model


dataFrame = pd.read_csv('train.csv')
target = "hours_spent"

X = dataFrame.drop(columns=["project_id", "hours_spent", "log_hours", "hours_bucket"])
X = X.replace({"True": 1, "False": 0}).replace({True: 1, False: 0})
X = add_features.add_features(X)
X = pd.get_dummies(X).fillna(-1).astype(float)
y = np.log1p(dataFrame[target])

testFrame = pd.read_csv('test.csv')
X_test = testFrame.drop(columns=["project_id"])
X_test = X_test.replace({"True": 1, "False": 0}).replace({True: 1, False: 0})
X_test = add_features.add_features(X_test)
X_test = pd.get_dummies(X_test).reindex(columns=X.columns, fill_value=0).fillna(-1).astype(float)

X_train, y_train = X, y
random_forest = random_forest_model(X_train, y_train)
hist_gradient = hist_gradient_model(X_train, y_train)
extra_trees = extra_trees_model(X_train, y_train)
cat_boost = cat_boost_model(X_train, y_train)
random_forest_log = random_forest.predict(X_test)
extra_trees_log = extra_trees.predict(X_test)
hist_gradient_log = hist_gradient.predict(X_test)
cat_boost_log = cat_boost.predict(X_test)
final = np.expm1((random_forest_log + hist_gradient_log + cat_boost_log) / 3)

pd.DataFrame({"project_id": testFrame["project_id"], target: final}).to_csv("predictions.csv", index=False)
