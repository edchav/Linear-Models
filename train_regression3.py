# train_regression3.py

import numpy as np
import matplotlib.pyplot as plt
from LinearRegression import LinearRegression

def main():
    np.random.seed(42)

    # Load the training data
    X_train = np.load(r'Data\iris_x_train.npy')

    # Uses the sepal length as features
    # to predict the sepal width 
    X_train_model = X_train[:, [0]]
    y_train_model = X_train[:, 1]

    # Create a LinearRegression model
    model = LinearRegression()
    model.fit(X_train_model, y_train_model)

    # Save the model
    model.save(r'Models\linear_model3.npz')

    # Plot the training loss
    plt.plot(model.training_loss)
    plt.xlabel('Step')
    plt.ylabel('Training Loss (MSE)')
    plt.title('Training Loss of Linear Model 3')
    plt.savefig(r'Plots\linear_model3_loss.png')
    plt.show()

    print("Training Loss: ", model.training_loss[-1])
    print("Training Score: ", model.score(X_train_model, y_train_model))
    print("Weights: ", model.weights)
    print("Bias: ", model.bias)

if __name__ == '__main__':
    main()