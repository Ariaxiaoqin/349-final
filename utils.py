import numpy as np
import pandas as pd



class CustomOneHotEncoder:
    def __init__(self):
        self.categories_ = {}

    def fit(self, X):
        """Fit the encoder by finding unique values for each column."""
        if isinstance(X, pd.DataFrame):
            X_np = X.to_numpy()  # Convert to NumPy array for consistent processing
        else:
            X_np = X
        for col_idx in range(X_np.shape[1]): # Iterate over columns
            unique_vals = np.unique(X_np[:, col_idx])
            self.categories_[col_idx] = {val: idx for idx, val in enumerate(unique_vals)}


    def transform(self, X):
        """Transform the data into one-hot encoded format."""
        if isinstance(X, pd.DataFrame):
            X_np = X.to_numpy()  # Convert to NumPy array for consistent processing
        else:
            X_np = X
            
        encoded_columns = []
        for col_idx in range(X_np.shape[1]):
            unique_vals = self.categories_[col_idx]
            encoded_col = np.zeros((X_np.shape[0], len(unique_vals)))
            for row_idx, value in enumerate(X_np[:, col_idx]):
                if value in unique_vals:
                    encoded_col[row_idx, unique_vals[value]] = 1
            encoded_columns.append(encoded_col)
        return np.hstack(encoded_columns)

    def fit_transform(self, X):
        """Fit and transform the data in one step."""
        self.fit(X)
        return self.transform(X)

class StandardScaler:
    def __init__(self):
        self.means_ = None
        self.stds_ = None

    def fit(self, X):
        """Compute the mean and standard deviation for each feature."""
        self.means_ = np.mean(X, axis=0)
        self.stds_ = np.std(X, axis=0)

    def transform(self, X):
        """Standardize features by subtracting the mean and dividing by the standard deviation."""
        if self.means_ is None or self.stds_ is None:
            raise ValueError("Scaler has not been fitted yet.")
        return (X - self.means_) / self.stds_

    def fit_transform(self, X):
        """Fit to data, then transform it."""
        self.fit(X)
        return self.transform(X)

    def inverse_transform(self, X_scaled):
        """Revert the standardization (return to original scale)."""
        if self.means_ is None or self.stds_ is None:
            raise ValueError("Scaler has not been fitted yet.")
        return (X_scaled * self.stds_) + self.means_

def custom_train_test_split(X, y, test_size=0.2, random_state=None, stratify=None):
    """
    Split arrays or DataFrame into random train and test subsets with optional stratification.

    Parameters:
        X (DataFrame or array-like): Features data.
        y (Series or array-like): Target data.
        test_size (float): Proportion of the dataset to include in the test split.
        random_state (int): Random seed for reproducibility.
        stratify (array-like or Series): Column for stratified splitting.

    Returns:
        X_train, X_test, y_train, y_test: Split data, retaining original data types.
    """
    # Check if inputs are DataFrame/Series
    is_X_dataframe = isinstance(X, pd.DataFrame)
    is_y_series = isinstance(y, pd.Series)

    # Reset DataFrame indices for consistency
    if is_X_dataframe:
        X_reset = X.reset_index(drop=True)
    else:
        X_reset = np.array(X)

    if is_y_series:
        y_reset = y.reset_index(drop=True)
    else:
        y_reset = np.array(y)

    if stratify is not None and isinstance(stratify, pd.Series):
        stratify = stratify.reset_index(drop=True)

    # Set random seed for reproducibility
    if random_state is not None:
        np.random.seed(random_state)

    # Stratified splitting
    if stratify is not None:
        unique_classes, class_indices = np.unique(stratify, return_inverse=True)
        train_indices = []
        test_indices = []

        for class_label in range(len(unique_classes)):
            class_mask = (class_indices == class_label)
            class_rows = np.where(class_mask)[0]
            np.random.shuffle(class_rows)

            test_size_class = int(len(class_rows) * test_size)

            test_indices.extend(class_rows[:test_size_class])
            train_indices.extend(class_rows[test_size_class:])
    else:
        # Random split without stratification
        indices = np.arange(len(X_reset))
        np.random.shuffle(indices)

        test_size_count = int(len(X_reset) * test_size)
        test_indices = indices[:test_size_count]
        train_indices = indices[test_size_count:]

    # Use iloc to split DataFrame/Series
    if is_X_dataframe:
        X_train = X_reset.iloc[train_indices]
        X_test = X_reset.iloc[test_indices]
    else:
        X_train, X_test = X_reset[train_indices], X_reset[test_indices]

    if is_y_series:
        y_train = y_reset.iloc[train_indices]
        y_test = y_reset.iloc[test_indices]
    else:
        y_train, y_test = y_reset[train_indices], y_reset[test_indices]

    return X_train, X_test, y_train, y_test