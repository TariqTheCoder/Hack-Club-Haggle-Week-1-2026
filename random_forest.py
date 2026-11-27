from sklearn.ensemble import RandomForestRegressor
import config

def random_forest_model(X, y):
    model = RandomForestRegressor(**config.RANDOM_FOREST)
    model.fit(X, y)
    return model