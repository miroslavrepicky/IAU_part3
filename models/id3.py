import numpy as np
import pandas as pd
from collections import Counter


class DecisionNode:

    def __init__(self, attribute=None, threshold=None, label=None, branches=None):
        self.attribute = attribute  # Feature to split on
        self.threshold = threshold  # Threshold for numerical features
        self.label = label  # Class label (for leaf nodes)
        self.branches = branches or {}  # Dictionary of branches

    def is_leaf(self):
        return self.label is not None


import numpy as np
from collections import defaultdict, Counter

class FastDecisionNode:
    def __init__(self, attribute=None, threshold=None, label=None, left=None, right=None):
        self.attribute = attribute
        self.threshold = threshold
        self.label = label
        self.left = left
        self.right = right

    def is_leaf(self):
        return self.label is not None


class ID3NumericalClassifier:
    def __init__(self, max_depth=None, min_samples_split=2, min_depth=2):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_depth = min_depth
        self.root = None

    # ---------------------------
    # Utilities: entropy + info
    # ---------------------------
    def _entropy_from_counts(self, counts):
        # counts: array-like counts per class
        total = counts.sum()
        if total == 0:
            return 0.0
        probs = counts / total
        # only non-zero probs
        nz = probs > 0
        return -np.sum(probs[nz] * np.log2(probs[nz]))

    # ---------------------------
    # Fit
    # ---------------------------
    def fit(self, X, y):
        X = np.asarray(X)
        y = np.asarray(y)

        n_samples, n_features = X.shape

        # encode labels to 0..K-1 for fast counts
        classes, y_encoded = np.unique(y, return_inverse=True)
        self.classes_ = classes
        self.n_classes_ = len(classes)
        self.y_encoded = y_encoded  # in original order
        self.n_samples_ = n_samples

        # precompute argsort for each feature and the sorted values
        self.sorted_idx = [np.argsort(X[:, j], kind='mergesort') for j in range(n_features)]
        self.sorted_vals = [X[self.sorted_idx[j], j] for j in range(n_features)]

        # store X for predict
        self.X_ = X

        # build tree starting with all indices
        all_idx = np.arange(n_samples, dtype=int)
        self.root = self._build_tree(all_idx, depth=0)
        return self

    # ---------------------------
    # Find best split for a node
    # ---------------------------
    def _best_split_for_node(self, indices):
        if indices.size == 0:
            return None, None, -np.inf

        # parent counts and entropy
        parent_counts = np.bincount(self.y_encoded[indices], minlength=self.n_classes_)
        parent_entropy = self._entropy_from_counts(parent_counts)
        n = indices.size

        best_gain = -np.inf
        best_attr = None
        best_threshold = None

        # boolean mask marking samples in this node for O(1) membership test
        node_mask = np.zeros(self.n_samples_, dtype=bool)
        node_mask[indices] = True

        # for each feature, scan the pre-sorted order and accumulate prefix counts
        for j, sorted_idx_j in enumerate(self.sorted_idx):
            sorted_vals_j = self.sorted_vals[j]

            # We'll iterate sorted indices and only process those that are in the node
            left_counts = np.zeros(self.n_classes_, dtype=int)
            left_n = 0

            last_val = None
            last_processed = False


            for pos_in_sorted, sample_idx in enumerate(sorted_idx_j):
                if not node_mask[sample_idx]:
                    continue

                lbl = self.y_encoded[sample_idx]
                left_counts[lbl] += 1
                left_n += 1

                val = sorted_vals_j[pos_in_sorted]

                k = pos_in_sorted + 1
                next_in_node_val = None
                while k < len(sorted_idx_j):
                    idx_k = sorted_idx_j[k]
                    if node_mask[idx_k]:
                        next_in_node_val = sorted_vals_j[k]
                        break
                    k += 1

                # If next_in_node_val is None -> no further in-node samples => cannot split after last sample
                if next_in_node_val is None:
                    continue

                if next_in_node_val == val:
                    # same numeric value for next in-node sample -> don't split here
                    continue

                # Now we have a valid boundary between val and next_in_node_val
                right_n = n - left_n
                if left_n < self.min_samples_split or right_n < self.min_samples_split:
                    continue

                # compute entropies from counts
                right_counts = parent_counts - left_counts
                left_entropy = self._entropy_from_counts(left_counts)
                right_entropy = self._entropy_from_counts(right_counts)
                weighted_entropy = (left_n / n) * left_entropy + (right_n / n) * right_entropy
                gain = parent_entropy - weighted_entropy

                if gain > best_gain:
                    best_gain = gain
                    best_attr = j
                    # threshold taken as midpoint between val and next_in_node_val
                    best_threshold = 0.5 * (val + next_in_node_val)

            # finished scanning this feature
        return best_attr, best_threshold, best_gain

    # ---------------------------
    # Build tree recursively
    # ---------------------------
    def _build_tree(self, indices, depth=0):
        # If pure and depth >= min_depth-1 -> leaf
        labels_here = self.y_encoded[indices]
        if labels_here.size == 0:
            # empty node, shouldn't happen often
            return FastDecisionNode(label=None)

        # majority label
        counts = np.bincount(labels_here, minlength=self.n_classes_)
        majority_label = int(np.argmax(counts))

        # stop conditions: pure & depth >= min_depth-1
        if depth >= (self.min_depth - 1) and np.all(labels_here == labels_here[0]):
            return FastDecisionNode(label=majority_label)

        # stop if not enough samples
        if depth >= (self.min_depth - 1) and indices.size < self.min_samples_split:
            return FastDecisionNode(label=majority_label)

        # stop if reached max depth (if set)
        if self.max_depth is not None and depth >= self.max_depth:
            if depth >= (self.min_depth - 1):
                return FastDecisionNode(label=majority_label)

        # find best split
        best_attr, best_threshold, best_gain = self._best_split_for_node(indices)

        if best_attr is None or best_gain <= 0:
            return FastDecisionNode(label=majority_label)

        # partition indices into left and right using threshold (no copy of X, just boolean division)
        # left: X[:, best_attr] <= threshold
        col_vals = self.X_[indices, best_attr]
        left_mask = col_vals <= best_threshold
        left_idx = indices[left_mask]
        right_idx = indices[~left_mask]

        # safety: if one side empty -> make leaf
        if left_idx.size == 0 or right_idx.size == 0:
            return FastDecisionNode(label=majority_label)

        # recursive calls
        left_node = self._build_tree(left_idx, depth + 1)
        right_node = self._build_tree(right_idx, depth + 1)

        return FastDecisionNode(attribute=best_attr, threshold=best_threshold, left=left_node, right=right_node)

    # ---------------------------
    # Predict
    # ---------------------------
    def _predict_one(self, x, node):
        while not node.is_leaf():
            if x[node.attribute] <= node.threshold:
                node = node.left
            else:
                node = node.right
        return node.label

    def predict(self, X):
        X = np.asarray(X)
        preds = np.array([self._predict_one(x, self.root) for x in X], dtype=int)
        # decode labels back to original
        return self.classes_[preds]

    # ---------------------------
    # Print tree for debug
    # ---------------------------
    def _print_node(self, node, depth=0, prefix="Root"):
        indent = "  " * depth
        if node.is_leaf():
            label = self.classes_[node.label] if node.label is not None else None
            print(f"{indent}{prefix} -> Leaf: {label}")
        else:
            print(f"{indent}{prefix} -> Feature {node.attribute} <= {node.threshold:.6f}")
            self._print_node(node.left, depth + 1, prefix=f"<= {node.threshold:.6f}")
            self._print_node(node.right, depth + 1, prefix=f"> {node.threshold:.6f}")

    def print_tree(self):
        if self.root is None:
            print("Tree is empty")
        else:
            self._print_node(self.root)



