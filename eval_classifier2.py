# eval_classifier2.py

import numpy as np
from LogisticRegression import LogisticRegression

def main():

    # Load the data
    X_test = np.load(r'Data\iris_x_test.npy')
    y_test = np.load(r'Data\iris_y_test.npy')

    # Load the model
    model = LogisticRegression()
    model.load(r'Models\logistic_model2.npz')

    # Test Accuracy
    print("Test Accuracy:", model.score(X_test[:, [0, 1]], y_test))

    print("Model Weights:", model.weights)
    print("Model Bias:", model.bias)
    
    # plot decision boundary
    model.plot_decision_boundary(X_test[:, [0, 1]], y_test, save_path=r'Plots\logistic_model2_decision_boundary_test.png')

if __name__ == "__main__":
    main()