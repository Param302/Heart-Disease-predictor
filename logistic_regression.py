import numpy as np
import pandas as pd


class NotFittedError(ValueError, AttributeError):
    """
    Exception class to raise if data is not fitted.
    """


class StandardScaler:
    """
    Standard Scaler:
    Normalize the data into a specific range using Z-score normalization.

    Z-score normalization:
    Subtracts mean (µ) from value and divide it by standard deviation (σ)
        normalized_data = (X - µ) / σ
    """
    _is_fitted = False

    def fit_transform(self, X: pd.DataFrame | np.ndarray):
        """Fit and transform data using Z-score normalization
        Args:
            X: data
        Returns:
            X_scaled: normalized data
        """
        X_scaled = self._z_score_normalization(X)
        self._is_fitted = True
        return X_scaled

    def transform(self, X: pd.DataFrame | np.ndarray):
        """Transform data using Z-score normalization
        using evaluated mean (µ) and standard deviation (σ)
        by `fit_transform`.
        Args:
            X: data
        Returns:
            X_scaled: normalized data
        """
        if not self._is_fitted:
            raise NotFittedError("Data isn't fitted yet.\n"
                                 "Use `fit_transform` to fit and transform actual data and then use `transform`.")

        X_scaled = self._z_score_normalization(X)
        return X_scaled

    def _z_score_normalization(self, X: pd.DataFrame | np.ndarray):
        """
        Normalized data using Z-score normalization
        normalized_data = (X - µ) / σ
        Args:
            X: data
        Returns:
            norm_data: normalized data
            mu (μ): mean of each feature
            sigma (σ): standard deviation of each feature
        """
        if not (hasattr(self, "mu") and hasattr(self, "sigma")):
            # calculating mean of each feature
            self.mu = np.mean(X, axis=0)
            # calculating standard deviation of each feature
            self.sigma = np.std(X, axis=0)

        # normalizing data
        norm_data = (X - self.mu) / self.sigma

        return norm_data


class LogisticRegression:
    """Logistic Regression model
    ----------------------------------------------------------------------

    Logistic regression function f(x):
        f(x) = g(z)
    or more precise:
        f(x) = 1 / (1 + e^-z)
    where:
        z = x.w + b
        g(z) = 1 / (1 + e^-z)   (sigmoid)
        f(x) = 1 / (1 + e^-z)
    ----------------------------------------------------------------------

    Logistic loss function loss(f(xi), yi):
        loss(yi_hat, yi) =  -yi * log(yi_hat) - (1 - yi) log(1 - yi_hat)
    where:
        yi_hat is ith predicted value using f(x)
        yi is actual ith target value
    ----------------------------------------------------------------------

    Logistic Cost function J(w,b):
        cost = (1/m) * Σ (loss)
    where:
        w: weight(s) of parameter(s)
        b: bias
        m: no. of training data
    ----------------------------------------------------------------------
    
    Gradient descent function:
        for each jth feature:
            wj = w_init[j] - α * dj_dw
        b = b_init - α * dj_db
    where:
        for each jth feature:
            dj_dw = (1/m) * Σ (f_wb(xi) - yi)*xi
        dj_db = (1/m) * Σ (f_wb(xi) - yi)
    ----------------------------------------------------------------------

    Attributes:
        learning_rate:      learning rate of model                  (0.05)
        num_iters:          number of iteraions it takes            (1000)
        random_state:       random state of model                   (0)
        verbose:            print logged info of cost & parameters   (False)
        accuracy:           accuracy of model      
    Methods:
        fit:        Fit data to train logistic regression model
        predict:    Predict data based on trained model
        """

    def __init__(self, learning_rate: int | float = 0.05, num_iters: int = 1000, random_state: int = 0, verbose: bool = False) -> None:

        self.learning_rate = learning_rate
        self.num_iters = num_iters
        self.verbose = verbose
        self.random_state = random_state
        self.random_no_gen = np.random.default_rng(self.random_state)
        self._is_fitted = False

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        """
        Fit data into Logistic Regression model
        """
        self.X = X
        self.y = y
        self.m, self.n = X.shape

        # initalizing weight(s) and bias
        self.w = self.random_no_gen.random((self.n, ))
        self.b = self.random_no_gen.integers(min(y), max(y))

        self.w_final, self.intercept, self.j_history = \
            self.__gradient_descent(self.X, self.y,
                                    self.w, self.b,
                                    self.learning_rate,
                                    self.num_iters,
                                    self.verbose)

        train_preds, self.z = self.__logistic_regression(
            self.X, self.w_final, self.intercept)
        self.train_preds = np.array(
            [1 if i >= 0.5 else 0 for i in train_preds]
        )

        self.cost = self.__compute_cost(
            self.train_preds, self.y,
            self.w_final, self.intercept
        )

        self._is_fitted = True

    def predict(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        """
        Predicts X using logistic regression.
        Args:
            X: test data
        Returns:
            predictions:  predictions of test data
        """
        if not self._is_fitted:
            raise NotFittedError("This Logistic regression instance has not fitted yet\n"
                                 "Call 'fit' to do predictions.")

        predictions, _ = self.__logistic_regression(
            X, self.w_final, self.intercept)
        predictions = np.array(
            [1 if i >= 0.5 else 0 for i in predictions]
        )
        return predictions

    @property
    def accuracy(self) -> float:
        """
        Calculates accuracy of logistic regression model
        based on fitted data (y) and predicted data (yhat).
        Returns:
            accuracy: accuracy of model
        """
        if not self._is_fitted:
            raise NotFittedError("This Logistic regression instance has not fitted yet\n"
                                 "Call 'fit' to get accuracy.")
        n_correct = 0
        for i in range(self.m):
            if self.y[i] == self.train_preds[i]:
                n_correct += 1

        accuracy = round(n_correct / self.m, 2)
        return accuracy

    def __logistic_regression(self, X: pd.DataFrame | np.ndarray, w: np.ndarray, b: int | float) -> tuple[np.ndarray]:
        """"
        Computes the prediction using sigmoid function.
        f w,b(x) = g(z)
        z = w.x + b

        Args:
            X:  input data
            w:  weight for each feature
            b:  parameter (base value)
        Returns:
            f_wb: prediction of model
        """
        z = (X @ w) + b
        f_wb = self.__sigmoid(z)
        return f_wb, z

    def __sigmoid(self, z: pd.DataFrame | np.ndarray) -> np.ndarray:
        """"
        Computes sigmoid function.
        g(z) = 1 / 1 + e^-z

        Args:
            z: input data (w.x + b)
        Returns:
            g_z: sigmoid function
        """
        g_z = 1 / (1 + np.exp(-z))
        return g_z

    def __compute_loss(self, yi_hat: int | float, yi: int | float) -> int | float:
        """
        Computes loss from
        yi_hat (ith predicted value) and yi (ith actual value).
        using logistic loss.
        loss(yi_hat, yi) =  -yi * log(yi_hat) - (1 - yi) log(1 - yi_hat)

        Args:
            yi_hat: ith predicted value
            yi:     ith actual value
        Returns:
            loss:   logistic loss
        """
        # added epsilon to ignore "divided by zero" warning which results in "nan"
        epsilon = 1e-5
        loss = -(yi * np.log(yi_hat + epsilon) +
                 (1 - yi) * np.log(1 - yi_hat + epsilon))
        return loss

    def __compute_cost(self, yhat: np.ndarray, y: np.ndarray,
                       w: np.ndarray, b: int | float
                       ) -> int | float:
        """
        Computes ERROR cost for logistic regression
        using loss function.
        cost = (1/m) * Σ (loss)

        Args:
            yhat: predicted value(s)
            y:    actual value(s)
            w:    weight for each feature
            b:    bias
        Returns:
            cost: cost error
        """
        m = len(yhat)
        cost = 0
        for i in range(m):
            loss = self.__compute_loss(yhat[i], y[i])
            cost += loss

        cost /= m
        return cost

    def __compute_gradient(self,
                           X: pd.DataFrame | np.ndarray,
                           y: pd.Series | np.ndarray,
                           w: np.ndarray,
                           b: int | float
                           ) -> tuple[np.ndarray | int | float]:
        """
        Computes gradient
        i.e. derivatives of weights and bias.
        for each jth feature:
            dj_dw = (1/m) * Σ (f_wb(xi) - yi)*xi
        dj_db = (1/m) * Σ (f_wb(xi) - yi)

        Args:
            X:  input data
            y:  actual values
            w:  weight for each feature
            b:  bias
        Returns:
            dj_dw:  gradient of weights
            dj_db   gradient of bias
        """

        m, n = X.shape
        dj_dw = np.zeros((n, ))
        dj_db = 0
        for i in range(m):
            f_wb, _ = self.__logistic_regression(X.iloc[i], w, b)
            error = f_wb - y[i]

            for j in range(n):
                dj_dw[j] += error * X.iloc[i, j]
            dj_db += error

        dj_dw /= m
        dj_db /= m
        return dj_dw, dj_db

    def __gradient_descent(self,
                           X: pd.DataFrame | np.ndarray,
                           y: pd.Series | np.ndarray,
                           w_init: np.ndarray,
                           b_init: int | float,
                           alpha: float,
                           num_iters: int,
                           verbose: bool
                           ) -> tuple[np.ndarray | int | float]:
        """
        Performs batch gradient descent
        and computes global minima values for coefficients.
        for each jth feature:
            wj = w_init[j] - α * dj_dw
        b = b_init - α * dj_db

        Args:
            X:          input data
            y:          actual values
            w_init:     initial weight for each feature
            b_init:     initial bias
            alpha:      learning rate
            num_iters:  number of iterations to run gradient descent
        Returns:
            w:          updated weight for each feature
            b:          updated bias
            j_history:  cost function history
        """

        w = w_init.copy()
        b = b_init
        j_history = np.zeros((10, ))
        n = 0

        verbose and print(
            f"Iteration | \tCost\t| \t {'W':^60} \t |      B      |")

        for i in range(1, num_iters+1):
            # calculating gradient
            dj_dw, dj_db = self.__compute_gradient(X, y, w, b)

            # printing and saving cost at each 10%
            if i % (num_iters / 10) == 0:
                preds, _ = self.__logistic_regression(X, w, b)
                j_history[n] = self.__compute_cost(preds, y, w, b)
                n += 1

                verbose and print(
                    f"{i:^9} | {j_history[n-1]:.5e} | {w!s:^71} | {b:^12.2f} |")

            # updating parameters
            w -= alpha * dj_dw
            b -= alpha * dj_db
        return w, b, j_history


if __name__ == "__main__":
    data = pd.read_csv("./data/train.csv")
    # Training data
    m = 100
    X = data.iloc[:m, [0, 1, 2]]
    y = data.iloc[:m, -1]
    # Testing data
    X_test = data.iloc[m:m+10, [0, 1, 2]]
    y_test = data.iloc[m:m+10, -1]

    # Scaling data using StandardScaler()
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    X_test = scaler.transform(X_test)

    # Logistic Regression model
    model = LogisticRegression(learning_rate=0.3, num_iters=1000, verbose=True)
    # fitting data
    model.fit(X, y)

    print(f"Accuracy: {model.accuracy}")

    # Predicting test data
    preds = model.predict(X_test)
    print("Preds:", *preds)
    print("Actual:", *y_test)
    """
    Output:
    Accuracy: 0.71
    Preds: 1 1 0 1 0 0 0 1 1 1
    Actual: 0 1 0 1 0 1 0 1 1 1
    """
