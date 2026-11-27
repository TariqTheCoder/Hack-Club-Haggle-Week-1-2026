from sklearn.ensemble import HistGradientBoostingRegressor

def hist_gradient(X,y):
    model = HistGradientBoostingRegressor(max_depth=4, learning_rate=0.05, max_iter=300,min_samples_leaf=20)
    model.fit(X,y)
    return model