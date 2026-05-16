from sklearn import (
    datasets,
    metrics,
    model_selection,
    neural_network,
    pipeline,
    preprocessing,
)

X, y = datasets.fetch_california_housing(return_X_y=True)

X_train, X_test, y_train, y_test = model_selection.train_test_split(
    X, y, random_state=42
)

mlp_reg = neural_network.MLPRegressor(
    loss="squared_error",
    solver="adam",
    hidden_layer_sizes=[50, 50, 50],
    early_stopping=True,
    random_state=31,
    verbose=True,
)

pipeline = pipeline.make_pipeline(preprocessing.StandardScaler(), mlp_reg)
pipeline.fit(X_train, y_train)

y_pred = pipeline.predict(X_test)
print(metrics.root_mean_squared_error(y_test, y_pred))
