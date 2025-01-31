# eval_regression4.py

import numpy as np
import matplotlib.pyplot as plt
from LinearRegression import LinearRegression

X_test = np.load('iris_test.npy')

X_test_model = X_test[:, [0, 2, 3]]
y_test_model = X_test[:, 1] 

model = LinearRegression()
model.load('model4.npz')

mse = model.score(X_test_model, y_test_model)
print('Test Mean Squared Error:', mse)