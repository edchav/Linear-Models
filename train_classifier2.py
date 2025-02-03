# train_classifier2.py

import numpy as np
from LogisticRegression import LogisticRegression
import matplotlib.pyplot as plt

def main():
    np.random.seed(42)

    # Load data
    X_train = np.load(r'Data\iris_x_train.npy')
    y_train = np.load(r'Data\iris_y_train.npy')

    # Uses sepal length and sepal width as features
    X_train_model = X_train[:, [0, 1]]

    # Train model
    model = LogisticRegression()
    model.fit(X_train_model, y_train)

    # Save model
    model.save(r'Models\logistic_model2.npz')

    # plot decision boundary
    model.plot_decision_boundary(X_train_model, y_train, save_path=r'Plots\logistic_model2_decision_boundary.png')

    # Training Accuracy
    print("Training Accuracy:", model.score(X_train_model, y_train))
    print("Training Loss:", model.training_loss[-1])
    print('Model Weights:', model.weights)
    print('Model Bias:', model.bias)
    
    # plot loss
    plt.plot(model.training_loss)
    plt.xlabel("Step")
    plt.ylabel("Training Loss")
    plt.savefig(r'Plots\logistic_model2_loss.png')
    plt.show()

if __name__ == "__main__":
    main()