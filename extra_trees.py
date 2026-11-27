from sklearn.ensemble import ExtraTreesRegressor

def extra_trees_model(X, y):
    model = ExtraTreesRegressor(n_estimators=500, min_samples_leaf=3, n_jobs=-1, random_state=42)
    model.fit(X, y)
    return model