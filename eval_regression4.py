# eval_regression4.py

import numpy as np
import matplotlib.pyplot as plt
from LinearRegression import LinearRegression

def main():
    # Load the test data
    X_test = np.load(r'Data\iris_x_test.npy')

    # Uses the first, third, and fourth features (sepal length, petal length, and petal width)
    # to predict the second feature (sepal width)
    X_test_model = X_test[:, [0, 2, 3]]
    y_test_model = X_test[:, 1] 

    # Load the model
    model = LinearRegression()
    model.load(r'Models\linear_model4.npz')

    print('Weights:', model.weights)
    print('Bias:', model.bias)
    
    # Test Mean Squared Error
    mse = model.score(X_test_model, y_test_model)
    print('Test Mean Squared Error:', mse)

if __name__ == "__main__":
    main()