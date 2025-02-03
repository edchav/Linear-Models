# train_classifier3.py

import numpy as np
from LogisticRegression import LogisticRegression
import matplotlib.pyplot as plt

def main():
    np.random.seed(42)

    # Load data
    X_train = np.load(r'Data\iris_x_train.npy')
    y_train = np.load(r'Data\iris_y_train.npy')

    # Uses all features
    X_train_model = X_train[:, [0, 1, 2, 3]]

    # Train model
    model = LogisticRegression()
    model.fit(X_train_model, y_train)

    # Save model
    model.save(r'Models\logistic_model3.npz')

    # plot loss
    plt.plot(model.training_loss)
    plt.xlabel("Step")
    plt.ylabel("Training Loss")
    plt.savefig(r'Plots\logistic_model3_loss.png')
    plt.show()
    
    print("Training Accuracy:", model.score(X_train_model, y_train))
    print("Model weights:", model.weights)
    print("Model bias:", model.bias)

if __name__ == "__main__":
    main()