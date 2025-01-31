import numpy as np
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression as SklearnLinearRegression
from sklearn.metrics import mean_squared_error

np.random.seed(42)

class LinearRegression:
    def __init__(self, batch_size=32, regularization=0, max_epochs=100, patience=3, learning_rate=0.001):
        """Linear Regression using Gradient Descent.
        
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
        """
        self.batch_size = batch_size
        self.regularization = regularization
        self.max_epochs = max_epochs
        self.patience = patience
        self.weights = None
        self.bias = None
        self.learning_rate = learning_rate
        self.training_loss = []
    
    def fit(self, X, y, batch_size=None, regularization=None, max_epochs=None, patience=None, learning_rate=None):
        """Fit a linear model.
        
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
            
        # Split into train and validation (10% for validation) for early stopping
        X_train, X_val, y_train, y_val = train_test_split(
            X, 
            y, 
            test_size=0.1, 
            random_state=42, 
            shuffle=True
        )

        # TODO: Initialize the weights and bias based on the shape of x and y.
        n_features = X_train.shape[1]
        self.weights = np.random.rand(n_features)
        self.bias = np.random.rand()

        best_val_loss = np.inf
        patience_counter = 0
        best_weights = None
        best_bias = None

        # TODO: Implement the training loop.
        for epoch in range(self.max_epochs):
            # Shuffle the data at the beginning of each epoch to avoid local minima 
            shuffled_idx = np.random.permutation(X_train.shape[0])
            X_shuffled = X_train[shuffled_idx]
            y_shuffled = y_train[shuffled_idx]

            # Loop through the data in batches
            for step in range(0, X_shuffled.shape[0], self.batch_size):
                end_idx = step + self.batch_size
                X_batch = X_shuffled[step:end_idx]
                y_batch = y_shuffled[step:end_idx]

                # Forward pass
                # h(x;w) = w^T * x + b = w_1 * x_1 + w_2 * x_2 + ... + w_n * x_n + b
                y_pred = self.predict(X_batch)

                # Compute the loss
                errors = (y_pred - y_batch)

                # batch loss
                batch_loss = np.mean(errors ** 2)
                self.training_loss.append(batch_loss)

                # Compute the gradients for the weights and bias
                # dl/dw = X^T * (h(x;w) - y) / N + regularization * w
                # dl/db = mean(h(x;w) - y)
                grad_w = (2 * np.dot(X_batch.T, errors) / X_batch.shape[0]) + (2 * self.regularization * self.weights)
                grad_b = 2 * np.mean(errors)

                # Update the weights and bias
                self.weights = self.weights - self.learning_rate * grad_w
                self.bias = self.bias - self.learning_rate * grad_b
            
           
                val_pred = self.predict(X_val)
                val_loss = self._mse_loss(val_pred, y_val)

                # Early Stopping
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_weights = self.weights.copy()
                    best_bias = self.bias.copy()
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= self.patience:
                        print(f"Early stopping at step {step} of epoch {epoch}")
                        self.weights, self.bias = best_weights, best_bias
                        return

            print(f"Epoch {epoch+1}/{self.max_epochs} - Val Loss: {best_val_loss:.4f}")
    
        if best_weights is not None:
            self.weights = best_weights
            self.bias = best_bias

    def save(self, file_path):
        """Save model parameters and relevant hyperparameters."""
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
        """Load model parameters and hyperparameters."""
        data = np.load(file_path)
        self.weights = data["weights"]
        self.bias = data["bias"]
        self.batch_size = int(data["batch_size"])
        self.regularization = float(data["regularization"])
        self.max_epochs = int(data["max_epochs"])
        self.patience = int(data["patience"])
        self.learning_rate = float(data["learning_rate"])

    def _mse_loss(self, y_pred, y_true):
        """Compute the mean squared error loss (MSE).
        
        Parameters:
        -----------
        y_pred: numpy.ndarry
            The predicted values.
        y_true: numpy.ndarray
            The true values.
        """
        mse = np.mean((y_pred - y_true) ** 2) #np.mean((y_pred - y_true) **2)
        reg_loss = self.regularization * np.sum(self.weights ** 2)
        return mse + reg_loss

    def predict(self, X):
        """Predict using the linear model.
        
        Parameters:
        ----------
        X: numpy.ndarray
            The input data.
        """

        # TODO: Implement the prediction funciton.
        return np.dot(X, self.weights) + self.bias

    def score(self, X, y):
        """Evaluate the linear model using the mean squared error.
        
        Parameters
        ----------
        X: numpy.ndarray
            The input data.
        y: numpy.ndarray
            The target data.
        """

        # TODO: Implement the score function.
        #predictions = self.predict(X)
        #return self._mse_loss(predictions, y)
        
        predictions = self.predict(X)
        mse = np.mean((predictions - y) ** 2)
        return mse
    
# Generate synthetic data with a known linear relationship
# np.random.seed(42)
# n_samples = 200
# X = np.random.randn(n_samples, 3)  # three features

# true_weights = np.array([3.0, -2.0, 1.5])
# true_bias = 4.0

# noise = np.random.randn(n_samples) * 0.5
# y = X.dot(true_weights) + true_bias + noise


# model = LinearRegression(batch_size=32, regularization=0, max_epochs=1000, patience=5, learning_rate=0.01)
# model.fit(X, y)

# # Evaluate the model
# mse = model.score(X, y)

# print(f"Custom Model Mean Squared Error:", mse)
# print(f"Custom Model Weights:", model.weights)
# print(f"Custom Model Bias:", model.bias)




# # Compare with sklearn
# sklearn_model = SklearnLinearRegression()
# sklearn_model.fit(X, y)
# sklearn_predictions = sklearn_model.predict(X)
# sklearn_mse = mean_squared_error(y, sklearn_predictions)

# print(f"Sklearn Mean Squared Error:", sklearn_mse)
# print(f"Sklearn Weights:", sklearn_model.coef_)
# print(f"Sklearn Bias:", sklearn_model.intercept_)


# print(f"Mean Squared Error:", mse)