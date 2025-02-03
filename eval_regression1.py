# eval_regression1.py

import numpy as np
import matplotlib.pyplot as plt
from LinearRegression import LinearRegression

def main():
    # Load the test data
    X_test = np.load(r'Data\iris_x_test.npy')

    # Uses the first 3 features (sepal length, sepal width, and petal length)
    # to predict the 4th feature (petal width)
    X_test_model = X_test[:, [0, 1, 2]]
    y_test_model = X_test[:, 3] 

    # Load the model
    model = LinearRegression()
    model.load(r'Models\linear_model1.npz')

    print("Weights: ", model.weights)
    print("Bias: ", model.bias)

    # Test Mean Squared Error
    mse = model.score(X_test_model, y_test_model)
    print('Test Mean Squared Error:', mse)

if __name__ == "__main__":
    main()



