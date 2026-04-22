# From scratch ID3 decision tree implementation 
# Includes entropy calculation, information gain, recursive tree construction

import math
from collections import Counter

# Compute entropy of a label distribution
def entropy(labels):
    total = len(labels)
    counts = Counter(labels)
    ent = 0.0

    for count in counts.values():
        p = count / total
        if p > 0:
            ent -= p * math.log2(p)

    return ent


# Compute information gain for a given feature
def information_gain(dataset, feature_index, target_index):
    entropy_before = entropy([row[target_index] for row in dataset])
    values = set(row[feature_index] for row in dataset)
    entropy_after = 0.0

    for v in values:
        subset = [row for row in dataset if row[feature_index] == v]
        weight = len(subset) / len(dataset)
        subset_entropy = entropy([row[target_index] for row in subset])
        entropy_after += weight * subset_entropy

    return entropy_before - entropy_after


# Select feature index with highest information gain
def best_split(dataset, target_index):
    features = len(dataset[0]) - 1  # Exclude target column
    best_gain = -1
    best_feature = None

    for feature_index in range(features):
        gain = information_gain(dataset, feature_index, target_index)
        if gain > best_gain:
            best_gain = gain
            best_feature = feature_index

    return best_feature


# Recursively build ID3 decision tree
def id3(dataset, target_index, features):
    labels = [row[target_index] for row in dataset]

    # If all labels are identical -> return leaf
    if labels.count(labels[0]) == len(labels):
        return labels[0]

    # If no features remain -> return majority class
    if len(features) == 0:
        return Counter(labels).most_common(1)[0][0]

    best_feature_index = best_split(dataset, target_index)
    best_feature_name = features[best_feature_index]

    tree = {best_feature_name: {}}

    # Sort values for deterministic tree structure
    values = sorted(set(row[best_feature_index] for row in dataset))

    for v in values:
        # Create subset excluding the chosen feature 
        subset = [
            row[:best_feature_index] + row[best_feature_index + 1:]
            for row in dataset if row[best_feature_index] == v
        ]

        # Remove used feature 
        new_features = features[:best_feature_index] + features[best_feature_index + 1:]

        # Adjust target index since feature column was removed 
        subtree = id3(subset, target_index - 1, new_features)
        tree[best_feature_name][v] = subtree

    return tree
