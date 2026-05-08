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
| Binned ID3 + Pruning + Tuning (best custom model)| `max_depth=2`, `min_samples_split=10`, `min_samples_leaf=5` |
| Binned ID3 + Over-Pruning | `max_depth=1`, `min_samples_split=20`, `min_samples_leaf=10` - demonstrates underfitting |
**Finding:** Binning significantly improves ID3 performance. Carefully tuned pruning further boosts generalisation by eliminating unnecessary splits. Over-pruning causes underfitting and degrades accuracy. 

**3. Sklearn Decision Tree (benchmark)**
| Configuration       | Notes                                  |
|---------------------|----------------------------------------|
| Sklearn DT (Raw values)           | Default `DecisionTreeClassifier` with continuous features         |
| Sklearn DT (Binned + Encoded) | Same discretisation as ID3 (`criterion="entropy"`) with matched pruning params  |
**Finding:** Sklearn on raw features achieves the highest overall performance as its ability to find optimal continuous split thresholds preserves full feature information. Applying discretisation to sklearn hurts performance because it introduces information lsos that the library's native handling avoids. 

# Key Takeaways
1. Preprocessing matters for ID3 - continuous features must be discretised to allow the algorithm to split effectively.
2. Pruning improves generalisation - Tuning `max_depth`, `min_samples_split`, and `min_samples_leaf` together is important.
3. Sklearn outperforms custom ID3 - due to optimal threshold selection on continuous variables, not just  algorithmic differences.
4. Discretisation hurts sklearn - preprocessing that helps ID3 actively harms sklearn's decision tree, which expects raw numerical inputs.

# Dependencies
```
pandas
matplotlib
seaborn
graphviz
scikit-learn
IPython
```
Install with:
```
pip install pandas matplotlib seaborn graphviz scikit-learn
```

# Running the Notebook
**Google Colab:** The notebook auto-clones the repository. Just run all cells in order. 

**Locally:**
```
git clone https://github.com/maytko-svg/decision-tree-from-scratch-analysis.git
cd decision-tree-from-scratch-analysis
jupyter notebook ID3_Decision_Tree_Ablation_Study.ipynb
```

# Outputs
- `outputs/trees/id3_plain.png` — Visualisation of the baseline tree
- `outputs/trees/id3_binned_pruned_tuned.png` — Visualisation of the best model
- `outputs/confusion_matrices/` — Confusion matrix for each of the 8 configurations
