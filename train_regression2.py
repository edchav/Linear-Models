# train_regression2.py

import numpy as np
import matplotlib.pyplot as plt
from LinearRegression import LinearRegression

np.random.seed(42)

X_train = np.load('iris_train.npy')

# Uses the sepal length and sepal width as features
# to predict the petal length
X_train_model = X_train[:, [0, 1]]
y_train_model = X_train[:, 2]

model = LinearRegression()
model.fit(X_train_model, y_train_model)

model.save("model2.npz")

plt.plot(model.training_loss)
plt.xlabel('Step')
plt.ylabel('Training Loss (MSE)')
plt.savefig("model2_loss.png")
plt.show()