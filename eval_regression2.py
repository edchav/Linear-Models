# eval_regression2.py

import numpy as np
import matplotlib.pyplot as plt
from LinearRegression import LinearRegression

X_test = np.load('iris_test.npy')

X_test_model = X_test[:, [0, 1]]
y_test_model = X_test[:, 2] 

model = LinearRegression()
model.load('model2.npz')

mse = model.score(X_test_model, y_test_model)
print('Test Mean Squared Error:', mse)