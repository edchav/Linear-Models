# train_regression3.py

import numpy as np
import matplotlib.pyplot as plt
from LinearRegression import LinearRegression

np.random.seed(42)

X_train = np.load('iris_train.npy')

# Uses the sepal length as features
# to predict the sepal width 
X_train_model = X_train[:, [0]]
y_train_model = X_train[:, 1]

model = LinearRegression()
model.fit(X_train_model, y_train_model)

model.save("model3.npz")

plt.plot(model.training_loss)
plt.xlabel('Step')
plt.ylabel('Training Loss (MSE)')
plt.savefig("model3_loss.png")
plt.show()