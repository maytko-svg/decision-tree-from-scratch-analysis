# ID3 Decision Tree Ablation Study
ID3 decision tree implemented from scratch with an ablation study (6 variants) and benchmarking against scikit-learn under raw vs binned/encoded feature pipelines 

# Overview 
This notebook explores how preprocessing choices (binning continuous features) and regularisation techniques (pruning) affect a custom ID3 decision tree. The goal is to understand the performance gap between a from-scratch implementation and an industry-standard library. 

# Dataset
**Wine dataset** (from `sklearn.datasets`) - 178 samples, 13 continuous features, 3 target classes. 
A 70/30 train and test split is applied with `random_state = 42` for reproducibility

# Project Structure 
decision-tree-from-scratch-analysis/
├── src/
│   ├── id3.py           # Custom ID3 implementation
│   ├── pruning.py       # Pruned ID3 variant
│   └── evaluation.py    # Metrics and confusion matrix utilities
├── outputs/
│   ├── trees/           # Rendered tree diagrams (Graphviz) of plain and best variants
│   └── confusion_matrices/ 
└── ID3_Decision_Tree_Ablation_Study.ipynb
