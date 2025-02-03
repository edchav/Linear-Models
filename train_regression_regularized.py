# train_regression_regularized.py

import numpy as np
from LinearRegression import LinearRegression
import matplotlib.pyplot as plt

def main():
    np.random.seed(42)
    
    # Load the training data
    X_train = np.load(r'Data\iris_x_train.npy')

    # Uses the sepal length, sepal width, and petal length as features
    # to predict the petal width
    X_train_model = X_train[:, [0, 1, 2]]
    y_train_model = X_train[:, 3]

    # Create a LinearRegression model with regularization
    model_reg = LinearRegression(regularization=0.1)
    model_reg.fit(X_train_model, y_train_model)

    # Load a LinearRegression model without regularization
    model_non_reg = LinearRegression()
    model_non_reg.load(r'Models\linear_model1.npz')

    plt.plot(model_non_reg.training_loss, label='Non-regularized')
    plt.plot(model_reg.training_loss, label='Regularized')

    plt.xlabel('Step')
    plt.ylabel('Training Loss (MSE)')
    plt.title('Comparison of Regularized and Non-regularized Training Loss')
    plt.legend()

    plt.savefig(r'Plots\Comparison of Regularized and Non-regularized Training Loss.png')
    plt.show()

    print('Model weights with regularization: ', model_reg.weights)
    print('Model weights without regularization: ', model_non_reg.weights)
    print('Model bias with regularization: ', model_reg.bias)
    print('Model bias without regularization: ', model_non_reg.bias)

    print('Model score with regularization: ', model_reg.score(X_train_model, y_train_model))
    print('Model score without regularization: ', model_non_reg.score(X_train_model, y_train_model))

if __name__ == '__main__':
    main()