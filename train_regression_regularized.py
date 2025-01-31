# train_regression_regularized.py

import numpy as np
from LinearRegression import LinearRegression
import matplotlib.pyplot as plt

np.random.seed(42)

X_train = np.load('iris_train.npy')
X_train_model = X_train[:, [0, 1, 2]]
y_train_model = X_train[:, 3]

model_reg = LinearRegression(regularization=0.1)
model_reg.fit(X_train_model, y_train_model)

model_non_reg = LinearRegression()
model_non_reg.load("model1.npz")

model_non_reg.fit(X_train_model, y_train_model)

plt.figure()
plt.plot(model_non_reg.training_loss, label='Non-regularized')
plt.plot(model_reg.training_loss, label='Regularized')

plt.xlabel('Step')
plt.ylabel('Training Loss (MSE)')
plt.legend()
plt.show()
plt.savefig('Comparison of Regularized and Non-regularized Training Loss.png')

print("Model weights with regularization: ", model_reg.weights)
print("Model weights without regularization: ", model_non_reg.weights)