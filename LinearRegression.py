import numpy as np
from sklearn.model_selection import train_test_split
np.random.seed(42)

class LinearRegression:
    def __init__(self, batch_size=32, regularization=0, max_epochs=100, patience=3, learning_rate=0.01):
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
    
    def fit(self, X, y, batch_size=32, regularization=0, max_epochs=100, patience=3, learning_rate=0.01):
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

        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.regularization = regularization
        self.max_epochs = max_epochs
        self.patience = patience

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
            X_train = X_train[shuffled_idx]
            y_train = y_train[shuffled_idx]

            # Loop through the data in batches
            for start_idx in range(0, X_train.shape[0], self.batch_size):
                end_idx = start_idx + self.batch_size
                X_batch = X_train[start_idx:end_idx]
                y_batch = y_train[start_idx:end_idx]

                # Forward pass
                # h(x;w) = w^T * x + b = w_1 * x_1 + w_2 * x_2 + ... + w_n * x_n + b
                y_pred = self.predict(X_batch)

                # Compute the loss
                errors = (y_pred - y_batch)

                # Compute the gradients
                grad_w = np.dot(X_batch.T, errors) / X_batch.shape[0] \
                            + self.regularization * self.weights
                grad_b = np.mean(errors)

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
                    print(f"Early stopping at epoch {epoch+1}")
                    break

            train_loss = self._mse_loss(self.predict(X_train), y_train)
            print(f"Epoch {epoch+1}/{self.max_epochs} - Training Loss: {train_loss:.4f}")
    
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
            patience=self.patience
        )

    def load(self, file_path):
        """Load model parameters and hyperparameters."""
        data = np.load(file_path)
        self.weights = data["weights"]
        self.bias = data["bias"].item()
        self.batch_size = int(data["batch_size"].item())
        self.regularization = float(data["regularization"].item())
        self.max_epochs = int(data["max_epochs"].item())
        self.patience = int(data["patience"].item())

    def _mse_loss(self, y_pred, y_true):
        """Compute the mean squared error loss (MSE).
        
        Parameters:
        -----------
        y_pred: numpy.ndarry
            The predicted values.
        y_true: numpy.ndarray
            The true values.
        """
        mse = np.mean((y_pred - y_true) ** 2)
        reg_loss = 0.5 * self.regularization * np.sum(self.weights ** 2)
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
        predictions = self.predict(X)
        return self._mse_loss(predictions, y)

# Generate some random data, row is a sample, column is a feature
X = np.random.rand(100, 3)
y = np.random.rand(100)

model = LinearRegression()
model.fit(X, y)

# Evaluate the model
mse = model.score(X, y)

print(f"Mean Squared Error:", mse)
print(f"Weights:", model.weights)
print(f"Bias:", model.bias)