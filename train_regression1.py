# train_regression1.py

import numpy as np
import matplotlib.pyplot as plt
from LinearRegression import LinearRegression

def main():
    np.random.seed(42)

    # Load the training data
    X_train = np.load(r'Data\iris_x_train.npy')

    # Uses the first 3 features (sepal length, sepal width, and petal length) 
    # to predict the 4th feature (petal width)
    X_train_model = X_train[:, [0, 1, 2]]
    y_train_model = X_train[:, 3]

    # Create a LinearRegression model
    model = LinearRegression()
    model.fit(X_train_model, y_train_model)

    # Save the model
    model.save(r"Models\linear_model1.npz")

    # Plot the training loss
    plt.plot(model.training_loss)
    plt.xlabel("Step")
    plt.ylabel("Training Loss (MSE)")
    plt.title("Training Loss of Linear Model 1")
    plt.savefig(r"Plots\linear_model1_loss.png")
    plt.show()

    # Print the final training loss, weights, and bias
    print("Training Loss: ", model.training_loss[-1])
    print('score:', model.score(X_train_model, y_train_model))
    print("Weights: ", model.weights)
    print("Bias: ", model.bias)

if __name__ == '__main__':
    main()