from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import numpy as np

def prepare_iris():
    iris = load_iris()
    X = iris.data
    y = iris.target

    X_train, X_test, _, _ = train_test_split(X, y, test_size=0.1, stratify=y, random_state=42)

    return X_train, X_test

if __name__ == '__main__':
    X_train, X_test = prepare_iris()
    np.save('iris_train.npy', X_train)
    np.save('iris_test.npy', X_test)