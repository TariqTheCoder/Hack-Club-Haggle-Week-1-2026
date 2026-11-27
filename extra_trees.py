from sklearn.ensemble import ExtraTreesRegressor
import config

def extra_trees_model(X, y):
    model = ExtraTreesRegressor(**config.EXTRA_TREES)
    model.fit(X, y)
    return model