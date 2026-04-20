from src.id3 import best_split
from collections import Counter

# Build ID3 with pruning parameters
def id3_pruned(dataset, target_index, features, depth=0, max_depth=None,
               min_samples_split=2, min_samples_leaf=1):
    labels = [row[target_index] for row in dataset]
  
    # Check if all labels are the same
    if labels.count(labels[0]) == len(labels):
        return labels[0]
  
    # Check if no features left
    if len(features) == 0:
        return Counter(labels).most_common(1)[0][0]
  
    # Check depth of tree
    if max_depth is not None and depth >= max_depth:
        return Counter(labels).most_common(1)[0][0]
  
    # Check if amount of samples too small to split
    if len(dataset) < min_samples_split:
        return Counter(labels).most_common(1)[0][0]
  
    # Pick best feature
    best_feature_index = best_split(dataset, target_index)
    best_feature_name = features[best_feature_index]
  
    # Start building tree with this feature
    tree = {best_feature_name: {}}
    values = sorted(set(row[best_feature_index] for row in dataset))
  
    # Build subtree for each value
    for v in values:
      subset = [row[:best_feature_index] + row[best_feature_index+1:]
                for row in dataset if row[best_feature_index] == v]
  
      # If subset is too small, make a leaf with majority label
      if len(subset) < min_samples_leaf:
          majority_label = Counter(labels).most_common(1)[0][0]
          tree[best_feature_name][v] = majority_label
      else:
          new_features = features[:best_feature_index] + features[best_feature_index+1:]
          subtree = id3_pruned(subset, -1, new_features,
                               depth=depth+1,
                               max_depth=max_depth,
                               min_samples_split=min_samples_split,
                               min_samples_leaf=min_samples_leaf)
          tree[best_feature_name][v] = subtree
    return tree
