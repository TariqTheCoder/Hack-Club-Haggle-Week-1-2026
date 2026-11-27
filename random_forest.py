from sklearn.ensemble import RandomForestRegressor

def random_forest_model(X,y):
    model = RandomForestRegressor(n_estimators=1000, min_samples_leaf=5, n_jobs=-1, )
    model.fit(X, y)
    return model