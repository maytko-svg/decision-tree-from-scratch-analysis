# From scratch ID3 decision tree implementation with pre-pruning 
# Extends the base ID3 algorithm by introducing stopping criteria to control model complexity and reduce overfitting

from src.id3 import best_split
from collections import Counter

# Build ID3 decision tree with pre-pruning parameters
def id3_pruned(dataset, target_index, features, depth=0, max_depth=None,
               min_samples_split=2, min_samples_leaf=1):
    labels = [row[target_index] for row in dataset]
  
    # If all labels are identical -> return leaf
    if labels.count(labels[0]) == len(labels):
        return labels[0]
  
    # If no features remain -> return majority class
    if len(features) == 0:
        return Counter(labels).most_common(1)[0][0]
  
    # Stop if maximum depth reached
    if max_depth is not None and depth >= max_depth:
        return Counter(labels).most_common(1)[0][0]
  
    # Stop if dataset too small to perform a split
    if len(dataset) < min_samples_split:
        return Counter(labels).most_common(1)[0][0]
  
    # Select best feature
    best_feature_index = best_split(dataset, target_index)
    best_feature_name = features[best_feature_index]
  
    tree = {best_feature_name: {}}
    values = sorted(set(row[best_feature_index] for row in dataset))
  
    # Build subtrees 
    for v in values:
      subset = [row[:best_feature_index] + row[best_feature_index+1:]
                for row in dataset if row[best_feature_index] == v]
  
      # If resulting subset is too small, assign majority label
      if len(subset) < min_samples_leaf:
          majority_label = Counter(labels).most_common(1)[0][0]
          tree[best_feature_name][v] = majority_label
      else:
          new_features = features[:best_feature_index] + features[best_feature_index+1:]
          subtree = id3_pruned(subset, -1, 
                               new_features,
                               depth=depth+1,
                               max_depth=max_depth,
                               min_samples_split=min_samples_split,
                               min_samples_leaf=min_samples_leaf)
          tree[best_feature_name][v] = subtree
    return tree
