# train_regression1.py

import numpy as np
import matplotlib.pyplot as plt
from LinearRegression import LinearRegression

np.random.seed(42)

X_train = np.load('iris_train.npy')

# Uses the first 3 features (sepal length, sepal width, and petal length) 
# to predict the 4th feature (petal width)
X_train_model = X_train[:, [0, 1, 2]]
y_train_model = X_train[:, 3]

model = LinearRegression()
model.fit(X_train_model, y_train_model)

model.save("model1.npz")

plt.plot(model.training_loss)
plt.xlabel("Step")
plt.ylabel("Training Loss (MSE)")
plt.savefig("model1_loss.png")
plt.show()