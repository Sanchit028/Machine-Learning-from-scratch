import numpy as np
from collections import Counter

class Node:
    def __init__(self, value):
        """
        Initialize a Node object for the decision tree.

        Parameters:
        - value: The value associated with the node.
        """
        self.value = value
        self.children = {}

class DecisionTree:
    def __init__(self, max_depth=float('inf'), min_samples_split=2, min_samples_leaf=1, n_features=None):
        """
        Initialize a DecisionTree object.

        Parameters:
        - max_depth: Maximum depth of the tree.
        - min_samples_split: Minimum number of samples required to split an internal node.
        - min_samples_leaf: Minimum number of samples required to be at a leaf node.
        - n_features: Number of features to consider when looking for the best split. If None, all features will be considered.
        """
        self.root = None
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.n_features = n_features
        self.leaf_values = []

    def _proba(self, y_train, value):
        """
        Calculate the probability of a specific value in the target variable.

        Parameters:
        - y_train: The target variable values (Pandas Series).
        - value: The specific value to calculate probability for.

        Returns:
        Probability of the value in y_train.
        """
        value_counts = y_train.value_counts()
        return value_counts.get(value, 0) / y_train.shape[0]

    def _entropy(self, y_train):
        """
        Calculate entropy of the target variable.

        Parameters:
        - y_train: The target variable values (Pandas Series).

        Returns:
        Entropy value.
        """
        possible_values = y_train.unique()
        entp = np.sum([-self._proba(y_train, val) * np.log2(self._proba(y_train, val)) for val in possible_values if self._proba(y_train, val) > 0])
        return entp

    def _w_entp(self, X_train, chosen_column, y_train):
        """
        Calculate weighted entropy for a chosen feature column.

        Parameters:
        - X_train: The training data (Pandas DataFrame).
        - chosen_column: The column in X_train for which weighted entropy is calculated.
        - y_train: The target variable values (Pandas Series).

        Returns:
        Weighted entropy value.
        """
        X_train_copy = X_train.copy()
        X_train_copy[y_train.name] = y_train
        possible_values = X_train_copy[chosen_column].unique()
        we = 0
        for val in possible_values:
            df2 = X_train_copy[X_train_copy[chosen_column] == val]
            entp_i = self._entropy(df2[y_train.name])
            we += entp_i * df2.shape[0] / X_train_copy.shape[0]
        return we

    def _info_gain(self, X_train, chosen_column, y_train):
        """
        Calculate information gain for a chosen feature column.

        Parameters:
        - X_train: The training data (Pandas DataFrame).
        - chosen_column: The column in X_train for which information gain is calculated.
        - y_train: The target variable values (Pandas Series).

        Returns:
        Information gain value.
        """
        eP = self._entropy(y_train)
        e_chosen_column = self._w_entp(X_train, chosen_column, y_train)
        return eP - e_chosen_column

    def add_node(self, X_train, y_train, depth=0):
        """
        Recursively add nodes to construct the decision tree.

        Parameters:
        - X_train: The training data (Pandas DataFrame).
        - y_train: The target variable values (Pandas Series).
        - depth: Current depth in the tree.

        Returns:
        Node object.
        """
        if y_train.nunique() == 1 or depth >= self.max_depth or X_train.shape[0] < self.min_samples_split:
            leaf_value = Counter(y_train).most_common(1)[0][0]
            return Node(value=leaf_value)

        features = np.random.choice(X_train.columns, self.n_features, replace=False) if self.n_features else X_train.columns
        max_ig = float('-inf')
        max_ig_feat = None

        for feature in features:
            ig = self._info_gain(X_train, feature, y_train)
            if ig > max_ig:
                max_ig = ig
                max_ig_feat = feature

        node = Node(value=max_ig_feat)
        if self.root is None:
            self.root = node

        unique_values = X_train[max_ig_feat].unique().tolist()
        df = X_train.copy()
        df[y_train.name] = y_train

        for val in unique_values:
            df2 = df[df[max_ig_feat] == val]
            child_node = self.add_node(df2.drop(columns=[y_train.name]), df2[y_train.name], depth + 1)
            node.children[val] = child_node

        return node

    def fit(self, X_train, y_train):
        """
        Fit the decision tree model to the training data.

        Parameters:
        - X_train: The training data (Pandas DataFrame).
        - y_train: The target variable values (Pandas Series).

        Returns:
        None
        """
        self.leaf_values = y_train.unique().tolist()
        self.n_features = X_train.shape[1] if self.n_features is None else min(self.n_features, X_train.shape[1])
        self.root = self.add_node(X_train, y_train, depth=0)

    def _traverse_tree(self, x, node):
        """
        Traverse the decision tree to predict the outcome for a single instance.

        Parameters:
        - x: The input instance (Pandas Series).
        - node: Current node being processed.

        Returns:
        Predicted value.
        """
        if not node.children:
            return node.value
        value = x[node.value]
        if value in node.children:
            return self._traverse_tree(x, node.children[value])
        else:
            return Counter(self.leaf_values).most_common(1)[0][0]

    def predict(self, X_test):
        """
        Predict outcomes for multiple instances using the decision tree.

        Parameters:
        - X_test: The test data (Pandas DataFrame).

        Returns:
        Predicted values for each instance.
        """
        return X_test.apply(lambda x: self._traverse_tree(x, self.root), axis=1)
