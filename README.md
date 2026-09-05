# Task 5 — Decision Trees and Random Forests
**Elevate Labs — AI & ML Internship**

## Objective
Learn tree-based models for classification by training and comparing Decision Trees and Random Forests.

## Tools Used
- Python
- Scikit-learn
- Pandas
- Matplotlib / Seaborn

## Dataset
UCI Cleveland Heart Disease dataset — 303 patients, 13 clinical features (age, sex, chest pain type, resting blood pressure, cholesterol, etc.), binary target (0 = no disease, 1 = disease present). No missing values. Class balance: 165 disease, 138 no disease.

## Project Structure
```
heart-disease-trees/
├── data/
│   └── heart.csv
├── plots/
│   ├── 01_decision_tree_full.png
│   ├── 02_overfitting_vs_depth.png
│   ├── 03_rf_confusion_matrix.png
│   ├── 04_model_comparison.png
│   ├── 05_feature_importances.png
│   ├── 06_cross_validation_comparison.png
│   └── feature_importances.csv
├── trees.py
└── README.md
```

## How to Run
```bash
pip install pandas numpy matplotlib seaborn scikit-learn
python trees.py
```
Prints accuracy, classification report, and cross-validation results to the console, and saves all charts to `plots/`.

## Approach
1. **Train a Decision Tree** with no depth limit and visualize the top levels of the tree.
2. **Analyze overfitting** by training trees at `max_depth` 1–20 and comparing train vs. test accuracy at each depth.
3. **Train a Random Forest** (200 trees) and compare its accuracy against both the fully-grown and depth-controlled decision trees.
4. **Interpret feature importances** from the Random Forest to see which clinical measurements matter most.
5. **Evaluate with 5-fold cross-validation** for both the depth-controlled Decision Tree and the Random Forest, to get a more reliable accuracy estimate than a single train/test split.

## Results

**Overfitting analysis**
- A fully-grown Decision Tree reaches **100% training accuracy** but only **70.5% test accuracy** — a classic sign of overfitting (the tree has memorized the training data instead of learning generalizable patterns).
- Test accuracy peaks around **max_depth = 4** (78.7%), then degrades as depth increases further while train accuracy keeps climbing to 100%.

**Model comparison (test accuracy)**

| Model | Test Accuracy |
|---|---|
| Decision Tree (fully grown) | 0.705 |
| Decision Tree (max_depth=4) | 0.787 |
| Random Forest (200 trees) | 0.820 |

The Random Forest outperforms both single-tree versions — averaging many trees trained on bootstrapped samples (bagging) reduces the variance/overfitting that a single deep tree suffers from.

**Random Forest evaluation**
- Precision/recall: 0.95 precision / 0.64 recall for "no disease", and 0.76 precision / 0.97 recall for "disease" — the model is very good at catching actual disease cases (high recall on class 1), at some cost of a few more false positives.

**Feature importances (Random Forest)**
Top predictors: `cp` (chest pain type), `thal` (thalassemia test result), `thalach` (max heart rate achieved), and `oldpeak` (ST depression induced by exercise) — all clinically meaningful signals for heart disease risk. `fbs` (fasting blood sugar) and `restecg` contributed the least.

**5-fold cross-validation** (more reliable than a single split)
- Decision Tree (max_depth=4): mean accuracy **76.5%** (± 6.4%)
- Random Forest (200 trees): mean accuracy **84.1%** (± 3.3%)

The Random Forest is both more accurate and more stable (lower variance across folds) than the single decision tree, confirming the benefit of ensemble averaging over relying on one tree.

---
*Submitted as part of the Elevate Labs AI & ML Internship (MSME, Govt. of India).*
