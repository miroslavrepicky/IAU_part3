import numpy as np
import pandas as pd
from collections import Counter


class DecisionNode:
    """Represents a node in the decision tree."""

    def __init__(self, attribute=None, threshold=None, label=None, branches=None):
        self.attribute = attribute  # Feature to split on
        self.threshold = threshold  # Threshold for numerical features
        self.label = label  # Class label (for leaf nodes)
        self.branches = branches or {}  # Dictionary of branches

    def is_leaf(self):
        return self.label is not None


class ID3NumericalClassifier:
    """ID3 Decision Tree with support for numerical attributes."""

    def __init__(self, max_depth=None, min_samples_split=2, min_depth=2):
        self.root = None
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_depth = min_depth
        self.feature_types = {}  # Track if feature is numerical or categorical

    def entropy(self, y):
        """Calculate entropy of a label array."""
        if len(y) == 0:
            return 0
        counts = Counter(y)
        probs = [count / len(y) for count in counts.values()]
        return -sum(p * np.log2(p) for p in probs if p > 0)

    def find_best_threshold(self, X_col, y):
        """Find the best threshold for a numerical attribute."""
        # Sort unique values
        sorted_vals = sorted(set(X_col))

        if len(sorted_vals) <= 1:
            return None, -np.inf

        # Try midpoints between consecutive values
        best_threshold = None
        best_gain = -np.inf

        for i in range(len(sorted_vals) - 1):
            threshold = (sorted_vals[i] + sorted_vals[i + 1]) / 2

            # Split data
            left_mask = X_col <= threshold
            right_mask = X_col > threshold

            if sum(left_mask) == 0 or sum(right_mask) == 0:
                continue

            # Calculate information gain
            parent_entropy = self.entropy(y)
            n = len(y)
            n_left = sum(left_mask)
            n_right = sum(right_mask)

            left_entropy = self.entropy(y[left_mask])
            right_entropy = self.entropy(y[right_mask])

            weighted_entropy = (n_left / n) * left_entropy + (n_right / n) * right_entropy
            gain = parent_entropy - weighted_entropy

            if gain > best_gain:
                best_gain = gain
                best_threshold = threshold

        return best_threshold, best_gain

    def information_gain_categorical(self, X_col, y):
        """Calculate information gain for categorical attribute."""
        parent_entropy = self.entropy(y)
        n = len(y)

        # Calculate weighted entropy for each value
        weighted_entropy = 0
        for val in set(X_col):
            mask = X_col == val
            subset_y = y[mask]
            weighted_entropy += (len(subset_y) / n) * self.entropy(subset_y)

        return parent_entropy - weighted_entropy

    def best_attribute(self, X, y, attributes):
        """Find the best attribute to split on."""
        best_attr = None
        best_gain = -np.inf
        best_threshold = None

        for attr in attributes:
            X_col = X[:, attr]

            if self.feature_types[attr] == 'numerical':
                threshold, gain = self.find_best_threshold(X_col, y)
                if gain > best_gain:
                    best_gain = gain
                    best_attr = attr
                    best_threshold = threshold
            else:  # categorical
                gain = self.information_gain_categorical(X_col, y)
                if gain > best_gain:
                    best_gain = gain
                    best_attr = attr
                    best_threshold = None

        return best_attr, best_threshold

    def build_tree(self, X, y, attributes, depth=0):
        """Recursively build the decision tree."""

        # -------------------------------------------------------------
        # Base case 1: pure node (all labels identical)
        # BUT ONLY if we are already past min_depth
        # -------------------------------------------------------------
        if len(set(y)) == 1 and depth >= self.min_depth - 1:
            return DecisionNode(label=y[0])

        # -------------------------------------------------------------
        # Base case 2: no more attributes or not enough samples
        # BUT ONLY if depth >= min_depth-1
        # -------------------------------------------------------------
        if (len(attributes) == 0 or len(y) < self.min_samples_split) and depth >= self.min_depth - 1:
            return DecisionNode(label=Counter(y).most_common(1)[0][0])

        # -------------------------------------------------------------
        # Base case 3: reached max depth (if set)
        # BUT ONLY if depth >= min_depth-1
        # -------------------------------------------------------------
        if self.max_depth is not None and depth >= self.max_depth:
            if depth >= self.min_depth - 1:
                return DecisionNode(label=Counter(y).most_common(1)[0][0])

        # -------------------------------------------------------------
        # Otherwise: we MUST continue splitting until depth >= min_depth-1
        # -------------------------------------------------------------

        best_attr, threshold = self.best_attribute(X, y, attributes)

        if best_attr is None:
            # No attribute available for split
            return DecisionNode(label=Counter(y).most_common(1)[0][0])

        # Create node
        node = DecisionNode(attribute=best_attr, threshold=threshold)

        # Numerical split
        if self.feature_types[best_attr] == 'numerical':
            left_mask = X[:, best_attr] <= threshold
            right_mask = X[:, best_attr] > threshold

            if sum(left_mask) > 0:
                node.branches['<='] = self.build_tree(
                    X[left_mask], y[left_mask], attributes, depth + 1
                )
            if sum(right_mask) > 0:
                node.branches['>'] = self.build_tree(
                    X[right_mask], y[right_mask], attributes, depth + 1
                )

        else:
            # Categorical split
            remaining_attrs = [a for a in attributes if a != best_attr]
            for val in set(X[:, best_attr]):
                mask = X[:, best_attr] == val
                if sum(mask) > 0:
                    node.branches[val] = self.build_tree(
                        X[mask], y[mask], remaining_attrs, depth + 1
                    )

        return node

    def fit(self, X, y, feature_types=None):
        """
        Train the decision tree.

        Parameters:
        -----------
        X : array-like or DataFrame
            Training features
        y : array-like
            Target labels
        feature_types : dict, optional
            Dictionary mapping feature indices to 'numerical' or 'categorical'
            If None, all features are assumed to be numerical
        """
        if isinstance(X, pd.DataFrame):
            X = X.values
        if isinstance(y, pd.Series):
            y = y.values

        X = np.array(X)
        y = np.array(y)

        # Determine feature types
        if feature_types is None:
            self.feature_types = {i: 'numerical' for i in range(X.shape[1])}
        else:
            self.feature_types = feature_types

        attributes = list(range(X.shape[1]))
        self.root = self.build_tree(X, y, attributes)
        return self

    def predict_single(self, x, node):
        """Predict a single sample."""
        if node.is_leaf():
            return node.label

        attr_val = x[node.attribute]

        if self.feature_types[node.attribute] == 'numerical':
            if attr_val <= node.threshold:
                branch = node.branches.get('<=')
            else:
                branch = node.branches.get('>')
        else:
            branch = node.branches.get(attr_val)

        if branch is None:
            # If branch doesn't exist, return most common label in current node
            return self._most_common_label(node)

        return self.predict_single(x, branch)

    def _most_common_label(self, node):
        """Find most common label in subtree."""
        if node.is_leaf():
            return node.label

        labels = []
        for branch in node.branches.values():
            labels.append(self._most_common_label(branch))

        return Counter(labels).most_common(1)[0][0]

    def predict(self, X):
        """Predict labels for samples."""
        if isinstance(X, pd.DataFrame):
            X = X.values
        X = np.array(X)

        return np.array([self.predict_single(x, self.root) for x in X])

    def print_tree(self, node=None, depth=0, prefix="Root"):
        """Print the tree structure."""
        if node is None:
            node = self.root

        indent = "  " * depth

        if node.is_leaf():
            print(f"{indent}{prefix} -> Leaf: {node.label}")
        else:
            attr = node.attribute
            if self.feature_types[attr] == 'numerical':
                print(f"{indent}{prefix} -> Feature {attr} (threshold: {node.threshold:.3f})")
                if '<=' in node.branches:
                    self.print_tree(node.branches['<='], depth + 1, f"<= {node.threshold:.3f}")
                if '>' in node.branches:
                    self.print_tree(node.branches['>'], depth + 1, f"> {node.threshold:.3f}")
            else:
                print(f"{indent}{prefix} -> Feature {attr} (categorical)")
                for val, branch in node.branches.items():
                    self.print_tree(branch, depth + 1, f"= {val}")


# Example usage
if __name__ == "__main__":
    # Example 1: All numerical features
    print("=" * 50)
    print("Example 1: Numerical Features (Iris-like dataset)")
    print("=" * 50)

    from sklearn.datasets import load_iris
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, classification_report

    iris = load_iris()
    X_train, X_test, y_train, y_test = train_test_split(
        iris.data, iris.target, test_size=0.3, random_state=42
    )

    # Train model
    clf = ID3NumericalClassifier(max_depth=5)
    clf.fit(X_train, y_train)

    # Predict
    y_pred = clf.predict(X_test)

    print(f"\nAccuracy: {accuracy_score(y_test, y_pred):.3f}")
    print("\nTree Structure:")
    clf.print_tree()

    # Example 2: Mixed features (numerical + categorical)
    print("\n" + "=" * 50)
    print("Example 2: Mixed Features (Numerical + Categorical)")
    print("=" * 50)

    # Create sample data
    data = pd.DataFrame({
        'age': [25, 30, 45, 35, 50, 23, 40, 60, 48, 33],
        'income': [50000, 60000, 80000, 70000, 120000, 35000, 75000, 90000, 110000, 65000],
        'education': ['BS', 'MS', 'PhD', 'BS', 'PhD', 'HS', 'MS', 'PhD', 'MS', 'BS'],
        'buy': ['no', 'no', 'yes', 'yes', 'yes', 'no', 'yes', 'yes', 'yes', 'no']
    })

    # Prepare data
    X = data[['age', 'income', 'education']].values
    y = data['buy'].values

    # Specify feature types (0: numerical, 1: numerical, 2: categorical)
    feature_types = {0: 'numerical', 1: 'numerical', 2: 'categorical'}

    clf2 = ID3NumericalClassifier(max_depth=3)
    clf2.fit(X, y, feature_types=feature_types)

    print("\nTree Structure:")
    clf2.print_tree()

    # Make predictions
    test_samples = np.array([
        [28, 55000, 'BS'],
        [55, 95000, 'PhD'],
        [24, 40000, 'HS']
    ])

    predictions = clf2.predict(test_samples)
    print("\nPredictions for test samples:")
    for i, (sample, pred) in enumerate(zip(test_samples, predictions)):
        print(f"Sample {i + 1}: age={sample[0]}, income={sample[1]}, education={sample[2]} -> {pred}")