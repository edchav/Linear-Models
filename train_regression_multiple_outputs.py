# train_regression_multiple_outputs.py

import numpy as np
import matplotlib.pyplot as plt
from LinearRegression import LinearRegression

def main():

    # Load the training data
    X_train = np.load(r'Data\iris_x_train.npy')

    # Uses sepal length and sepal width to predict petal length and petal width
    X_train_model = X_train[:, [0, 1]]
    y_train_model = X_train[:, [2, 3]]

    # Create a LinearRegression model
    model = LinearRegression()
    model.fit(X_train_model, y_train_model)

    # Save the model
    model.save(r'Models\linear_model_multiple_outputs.npz')

    # Plot the training loss
    plt.plot(model.training_loss)
    plt.xlabel('Step')
    plt.ylabel('Training Loss (MSE)')
    plt.title('Linear Model Multiple Output Loss')
    plt.savefig(r'Plots\linear_model_multiple_output_loss.png')
    plt.show()

    print("Training Loss: ", model.training_loss[-1])
    print("Score: ", model.score(X_train_model, y_train_model))
    print("Weights: ", model.weights)
    print("Bias: ", model.bias)

if __name__ == '__main__':
    main()