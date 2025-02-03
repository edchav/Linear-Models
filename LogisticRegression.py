import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from mlxtend.plotting import plot_decision_regions
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder

class LogisticRegression:
    def __init__(self, batch_size=32, regularization=0, max_epochs=100, patience=20, learning_rate=0.1):
        """Logistic Regression using Gradient Descent.
            Parameters:
            -----------
            batch_size: int
                The number of samples per batch
            regularization: float
                The regularization parameter
            max_epochs: int
                The maximum number of epochs
            patience: int
                The number of epochs to wait before stopping if the validation loss
                does not improve.
            learning_rate: float
                The learning rate for the gradient descent
        """
        self.batch_size = batch_size
        self.regularization = regularization
        self.max_epochs = max_epochs
        self.patience = patience
        self.weights = None
        self.bias = None
        self.learning_rate = learning_rate
        self.training_loss = []
        self.encoder = OneHotEncoder(sparse_output=False)

    def _softmax(self, z):
        """
        Compute softmax values for each sets of scores in z.
        
        Parameters:
        -----------
        z: np.ndarray
            The input array
        """
        
        exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)

    def _cross_entropy_loss(self, y_true_one_hot, y_pred_proba):
        """
        Compute the cross-entropy loss.

        Parameters:
        -----------
        y_true_one_hot: np.ndarray
            The one-hot encoded target variable
        y_pred_proba: np.ndarray
            The predicted probabilities
        """

        m = y_true_one_hot.shape[0]
        loss = -np.sum(y_true_one_hot * np.log(y_pred_proba + 1e-15)) / m
        return loss

    def fit(self, X, y, batch_size=None, regularization=None, max_epochs=None, patience=None, learning_rate=None):
        """
        Fit a logistic model.

        Parameters:
        -----------
        batch_size: int
            The number of samples per batch.
        regularization: float
            The regularization parameter.
        max_epochs: int
            The maximum number of epochs.
        patience: int
            The number of epochs to wait before stopping if the validation loss
            does not improve.
        learning_rate: float
            The learning rate for the gradient descent
        """
        if batch_size is not None:
            self.batch_size = batch_size
        if regularization is not None:
            self.regularization = regularization
        if max_epochs is not None:
            self.max_epochs = max_epochs
        if patience is not None:
            self.patience = patience
        if learning_rate is not None:
            self.learning_rate = learning_rate
        
        # One-hot encode target variable
        if y.ndim == 1:
            y = self.encoder.fit_transform(y.reshape(-1, 1))

        # Initialize weights and bias
        n_features = X.shape[1]
        n_classes = y.shape[1]
        self.weights = np.random.randn(n_features, n_classes)
        self.bias = np.random.randn(n_classes)
        
        # Split into train and validation
        X_train, X_val, y_train, y_val = train_test_split(
            X, 
            y, 
            test_size=0.1, 
            random_state=42)

        best_val_loss = np.inf
        patience_counter = 0
        best_weights = None
        best_bias = None

        for epoch in range(self.max_epochs):
            # Shuffle data
            shuffled_idx = np.random.permutation(X_train.shape[0])
            X_shuffled = X_train[shuffled_idx]
            y_shuffled = y_train[shuffled_idx]

            for step in range(0, X_shuffled.shape[0], self.batch_size):
                end_idx = step + self.batch_size
                X_batch = X_shuffled[step:end_idx]
                y_batch = y_shuffled[step:end_idx]

                # Forward pass using batch data
                z = np.dot(X_batch, self.weights) + self.bias
                y_pred = self._softmax(z)
                
                # Compute loss and add regularization once
                loss = self._cross_entropy_loss(y_batch, y_pred)
                reg_term = 0.5 * self.regularization * np.sum(self.weights ** 2)
                batch_loss = loss + reg_term
                self.training_loss.append(batch_loss)

                # Compute gradients using batch data
                error = y_pred - y_batch
                grad_w = (np.dot(X_batch.T, error) / len(y_batch)) + self.regularization * self.weights
                grad_b = np.mean(error, axis=0)

                # Update parameters
                self.weights -= self.learning_rate * grad_w
                self.bias -= self.learning_rate * grad_b

            # Validation check using entire validation set
            val_z = np.dot(X_val, self.weights) + self.bias
            val_pred = self._softmax(val_z)
            val_loss = self._cross_entropy_loss(y_val, val_pred) + 0.5 * self.regularization * np.sum(self.weights ** 2)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_weights = self.weights.copy()
                best_bias = self.bias.copy()
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= self.patience:
                    print(f"Early stopping at epoch {epoch}")
                    self.weights = best_weights
                    self.bias = best_bias
                    return
                
            print(f"Epoch {epoch}, Training Loss: {batch_loss}, Validation Loss: {val_loss}")

        if best_weights is not None:
            self.weights = best_weights
            self.bias = best_bias
    
    def save(self, file_path):
        """
        Save model parameters and relevant hyperparameters.
        
        Parameters:
        -----------
        file_path: str
            The file path to save the model parameters.
        """
        np.savez(
            file_path,
            weights=self.weights,
            bias=self.bias,
            batch_size=self.batch_size,
            regularization=self.regularization,
            max_epochs=self.max_epochs,
            patience=self.patience,
            learning_rate=self.learning_rate
        )

    def load(self, file_path):
        """
        Load model parameters and hyperparameters.

        Parameters:
        -----------
        file_path: str
            The file path to load the model parameters.
        """

        data = np.load(file_path)
        self.weights = data["weights"]
        self.bias = data["bias"]
        self.batch_size = int(data["batch_size"])
        self.regularization = float(data["regularization"])
        self.max_epochs = int(data["max_epochs"])
        self.patience = int(data["patience"])
        self.learning_rate = float(data["learning_rate"])

    def predict_proba(self, X):
        """
        Predict the probabilities of each class.
        
        Parameters:
        -----------
        X: np.ndarray
            The input data
        """

        z = np.dot(X, self.weights) + self.bias
        return self._softmax(z)

    def predict(self, X):
        """
        Predict the class labels.
        
        Parameters:
        -----------
        X: np.ndarray
            The input data
        """

        probabilities = self.predict_proba(X)
        return np.argmax(probabilities, axis=1)

    def score(self, X, y):
        """
        Compute the accuracy of the model.
        
        Parameters:
        -----------
        X: np.ndarray
            The input data
        y: np.ndarray
            The target variable
        """
        if y.ndim > 1:
            y = np.argmax(y, axis=1)

        return accuracy_score(y, self.predict(X))

    def plot_decision_boundary(self, X, y, save_path=None, show=True):
        """
        Plot the decision boundary of the model.
        
        Parameters:
        -----------
        X: np.ndarray
            The input data
        y: np.ndarray
            The target variable
        save_path: str
            The file path to save the plot
        show: bool
            Whether to display the plot
        """
        
        if y.ndim > 1:
            y = np.argmax(y, axis=1)
        
        plt.figure(figsize=(8, 6))
        plot_decision_regions(X, y, clf=self, legend=2)
        plt.title("Logistic Regression Decision Boundary")

        if save_path is not None:
            plt.savefig(save_path)
        
        if show:
            plt.show()
        

