from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.preprocessing import StandardScaler

def main():
    iris = load_iris()

    X = iris.data
    y = iris.target

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, stratify=y, random_state=42)

    # Scale features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    np.save(r'Data\iris_x_train.npy', X_train)
    np.save(r'Data\iris_y_train.npy', y_train)

    np.save(r'Data\iris_x_test.npy', X_test)
    np.save(r'Data\iris_y_test.npy', y_test)

if __name__ == '__main__':
    main()