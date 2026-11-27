from sklearn.ensemble import HistGradientBoostingRegressor
import config

def hist_gradient(X, y):
    model = HistGradientBoostingRegressor(**config.HIST_GRADIENT)
    model.fit(X, y)
    return model