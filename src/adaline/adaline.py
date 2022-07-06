import numpy as np


class Adaline:
    """
    Implementation of an ADaptive LInear NEuron

    Attributes:
    -----------

    lr : float
        Learning rate
    random_state : int
        Random number generator seed for random weight initilization
    weights : array-like object
        Array containing the model weights
    costs : List
        A list of the cost function (SSE) in each epoch
    """

    def __init__(self, lr=0.01, random_state=42):
        """
        Initializes the object's attributes
        """
        self.lr = lr
        self.random_state = random_state
        self.weights = None
        self.costs = []

    def fit(self, X, y, epochs=50):
        """
        Fits the model according to training data

        Parameters:
        -----------
        X : array-like
            Array like object containing the training feature vectors
        y : array-like
            Array like object containing the training targets

        Returns:
        --------
        self : Adaline
        """
        # Build the vectors for training
        X_arr, y_arr = self._build_vectors(X, y)

        # Initializes the model's weights
        self.weights = self._get_initialized_model_weights(weights_size=len(X_arr[0]))
        for _ in range(epochs):
            errors = y_arr - self.net_input(X_arr)
            self.weights += (self.lr/X_arr.shape[0]) * np.dot(X_arr.T, errors)
            cost = np.sum(errors**2) / 2
            self.costs.append(cost)

    def _build_vectors(self, X=None, y=None):
        """
        Build ndarray vectors from array like objects

        Parameters:
        -----------
        X : array-like
            Array like object containing the training feature vectors
        y : array-like
            Array like object containing the training targets

        Returns:
        --------
        X_arr : ndarray if X is provided, else None
            ndarray containing feature vectors
        y_arr : ndarray if u is provided, else None
            ndarray containing target vectors
        """

        X_arr = np.hstack((
                np.ones(shape=(len(X), 1)),
                np.array(X)
        )) if X is not None else None

        y_arr = np.array(y)[..., np.newaxis] if y is not None else y

        return X_arr, y_arr
    
    def net_input(self, X):
        """Calculates net input"""
        return np.dot(X, self.weights)


    def predict(self, X):
        """
        Return the class labels for the input
        """
        X_arr, _ = self._build_vectors(X=X)

        return np.where(self.net_input(X_arr) >= 0.0, 1, -1).squeeze()

    def _get_initialized_model_weights(self, weights_size):
        """
        Initializes the model's weights

        Parameters:
        -----------

        weights_size : int:
            Length of the weights vector

        Returns:
        --------
        ndarray
            Initialized model's weights
        """
        return np.random.RandomState(self.random_state).normal(
            loc=0.0,
            scale=0.01,
            size=(weights_size, 1)
        )
