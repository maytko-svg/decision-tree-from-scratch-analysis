# ID3 Decision Tree Ablation Study
ID3 decision tree implemented from scratch with an ablation study (6 variants) and benchmarking against scikit-learn under raw vs binned/encoded feature pipelines 

# Overview and Dataset
This notebook explores how preprocessing choices (binning continuous features) and regularisation techniques (pruning) affect a custom ID3 decision tree. The goal is to understand the performance gap between a from-scratch implementation and an industry-standard library. 

**Wine dataset** (from `sklearn.datasets`) - 178 samples, 13 continuous features, 3 target classes. 
A 70/30 train and test split is applied with `random_state = 42` for reproducibility

# Project Structure 
```
decision-tree-from-scratch-analysis/  
├── src/  
│   ├── id3.py                 # Custom ID3 implementation  
│   ├── pruning.py             # Pruned ID3 variant  
│   └── evaluation.py          # Metrics and confusion matrix utilities  
├── outputs/  
│   ├── trees/                 # Rendered tree diagrams (Graphviz) of plain and best variants  
│   └── confusion_matrices/   
└── ID3_Decision_Tree_Ablation_Study.ipynb
```
**Note:** Clone the report before running locally or in Colab. The notebook handles this automatically when Colab is detected.

# Experiments
Eight configurations are evaluated, and split into three groups:


**1. Plain ID3 (raw continuous features)**
| Configuration       | Notes                                  |
|---------------------|----------------------------------------|
| Plain ID3           | Baseline — no preprocessing            |
| Plain ID3 + Pruning | `max_depth=2`, `min_samples_split=10`  |
**Finding:** Plain ID3 performs poorly on raw continuous data because the algorithm was designed for categorical inputs. It cannot find optimal split thresholds, which makes preprocessing essential. 

**2. Binned ID3 (discretised features)**
| Configuration       | Notes                                  |
|---------------------|----------------------------------------|
| Binned ID3           | No pruning           |
| Binned ID3 + Pruning | `max_depth=3`, `min_samples_split=15`, `min_samples_leaf=8`  |
| Binned ID3 + Pruning + Tuning | `max_depth=2`, `min_samples_split=10`, `min_samples_leaf=5` |
   (best custom model)  
| Binned ID3 + Over-Pruning     | `max_depth=1`, `min_samples_split=20`, `min_samples_leaf=10` - demonstrates underfitting |


