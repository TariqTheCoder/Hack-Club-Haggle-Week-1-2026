RANDOM_STATE = 42

RANDOM_FOREST = dict(
    n_jobs=-1,
    n_estimators=500,
    min_samples_leaf=2,
    max_features=0.3,
    max_depth=None,
    random_state=RANDOM_STATE,
)

EXTRA_TREES = dict(
    n_jobs=-1,
    n_estimators=800,
    min_samples_leaf=1,
    max_features=1.0,
    max_depth=None,
    random_state=RANDOM_STATE,
)

HIST_GRADIENT = dict(
    early_stopping=True,
    min_samples_leaf=50,
    max_leaf_nodes=15,
    max_iter=600,
    max_depth=6,
    learning_rate=0.08,
    l2_regularization=10.0,
    random_state=RANDOM_STATE,
)

CV_FOLDS = 5
