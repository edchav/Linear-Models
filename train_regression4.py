# train_regression4.py

import numpy as np
import matplotlib.pyplot as plt
from LinearRegression import LinearRegression

X_train = np.load('iris_train.npy')

# Uses sepal length, petal length, and petal width as features
# to predict sepal width
X_train_model = X_train[:, [0, 2, 3]]
y_train_model = X_train[:, 1]

model = LinearRegression()
model.fit(X_train_model, y_train_model)

model.save("model4.npz")

plt.plot(model.training_loss)
plt.xlabel('Step')
plt.ylabel('Training Loss (MSE)')
plt.savefig("model4_loss.png")
plt.show()
