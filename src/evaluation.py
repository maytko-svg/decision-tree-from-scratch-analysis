from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Predict class label by recursively traversing the decision tree based on feature values
def predict(tree, features, sample):
    # If node is a leaf, return it
    if not isinstance(tree, dict):
        return tree
    
    # Get the root feature at this node
    root = next(iter(tree))
    branches = tree[root]
    
    # Find the index of this feature in sample
    feature_index = features.index(root)
    sample_value = sample[feature_index]
    
    # Follow the branch matching this sample value
    if sample_value in branches:
        subtree = branches[sample_value]
        return predict(subtree, [f for f in features if f != root], # Recursively call function
                      [v for i, v in enumerate(sample) if i != feature_index])
      
    else:
        # If unseen value, return majority class
        return 0

# Generate true and predicted labels for a dataset using the predict() function
def get_predictions(tree, dataset, feature_names):
    y_true = []
    y_pred = []
  
    for row in dataset:
        features_only = row[:-1]
        true_label = row[-1]
    
        pred = predict(tree, feature_names[:], features_only)
        if pred is None:
            pred = 0
    
        y_true.append(true_label)
        y_pred.append(pred)
    
    return y_true, y_pred

# Evaluate decision tree on accuracy, precision, recall and F1 score
def evaluate(name, y_true, y_pred):
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='macro', zero_division=0)
    recall = recall_score(y_true, y_pred, average='macro', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)

    return {
        "Configuration": name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1": f1
    }

# Plot confusion matrix
def plot_confusion_matrix(y_true, y_pred, title, save_path=None, show=False):
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(4,4))
    sns.heatmap(cm, annot=True, square=True, fmt="d", cmap='Blues_r')
    plt.xlabel('Predicted labels')
    plt.ylabel('Actual labels')
    plt.title(title)

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
    
    if show:
        plt.show()
    
    plt.close()
