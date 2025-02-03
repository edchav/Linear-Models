# eval_classifier3.py

import numpy as np
from LogisticRegression import LogisticRegression

def main():

    # Load the data
    X_test = np.load(r'Data\iris_x_test.npy')
    y_test = np.load(r'Data\iris_y_test.npy')

    # Load the model
    model = LogisticRegression()
    model.load(r'Models\logistic_model3.npz')

    # Test Accuracy
    print("Test Accuracy:", model.score(X_test[:, [0, 1, 2, 3]], y_test))

    print("Model weights:", model.weights)
    print("Model bias:", model.bias)

if __name__ == "__main__":
    main()