import matplotlib.pyplot as plt
import numpy as np
from sklearn import preprocessing

rng = np.random.default_rng(seed=69)

m = 150
X = rng.random((m, 1))
y = 2 + 5 * X + rng.standard_normal((m, 1))

X_b = preprocessing.add_dummy_feature(X)
theta_best = np.linalg.inv(X_b.T @ X_b) @ X_b.T @ y

X_test = np.array([[0], [1]])
X_test_b = preprocessing.add_dummy_feature(X_test)
y_predict = X_test_b @ theta_best

plt.plot(X_test, y_predict, "r-", label="Predictions")
plt.plot(X, y, "b.")
plt.show()
