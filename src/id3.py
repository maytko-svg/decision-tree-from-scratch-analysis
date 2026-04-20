# id3.py
# Implementation of ID3 decision tree (entropy, information gain, recursive splitting)

import math
from collections import Counter

# Entropy calculation
def entropy(labels):
    total = len(labels)
    counts = Counter(labels)
    ent = 0.0

    for count in counts.values():
        p = count / total
        ent -= p * math.log2(p)

    return ent


# Information gain calculation
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


# Choose the best split function
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


# Build ID3
def id3(dataset, target_index, features):
    labels = [row[target_index] for row in dataset]

    # If all labels are the same → return leaf
    if labels.count(labels[0]) == len(labels):
        return labels[0]

    # If no features left → return majority class
    if len(features) == 0:
        return Counter(labels).most_common(1)[0][0]

    best_feature_index = best_split(dataset, target_index)
    best_feature_name = features[best_feature_index]

    tree = {best_feature_name: {}}
    values = sorted(set(row[best_feature_index] for row in dataset))

    for v in values:
        subset = [
            row[:best_feature_index] + row[best_feature_index + 1:]
            for row in dataset if row[best_feature_index] == v
        ]

        new_features = features[:best_feature_index] + features[best_feature_index + 1:]
        subtree = id3(subset, target_index - 1, new_features)
        tree[best_feature_name][v] = subtree

    return tree
