from sklearn.ensemble import HistGradientBoostingRegressor

def hist_gradient(X,y):
    model = HistGradientBoostingRegressor(learning_rate=0.03, max_iter=1000, max_depth=6,
                                      min_samples_leaf=20, l2_regularization=1.0,
                                      early_stopping=True, random_state=42)
    model.fit(X,y)
    return model