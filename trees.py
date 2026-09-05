"""
Task 5: Decision Trees and Random Forests - Heart Disease Dataset
Elevate Labs AI & ML Internship

Run with: python trees.py
Prints evaluation metrics/cross-validation results to console and saves all plots to plots/.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay

sns.set_style("whitegrid")
plt.rcParams["figure.dpi"] = 110

# ----------------------------------------------------------------------
# 1. Load dataset (UCI Cleveland Heart Disease dataset)
# ----------------------------------------------------------------------
df = pd.read_csv("data/heart.csv")
print("Shape:", df.shape)
print("Missing values:", df.isnull().sum().sum())
print("\nClass balance (target):\n", df["target"].value_counts())

X = df.drop(columns=["target"])
y = df["target"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"\nTrain size: {X_train.shape[0]}, Test size: {X_test.shape[0]}")

# ----------------------------------------------------------------------
# 2. Train a Decision Tree Classifier (no depth limit) and visualize it
# ----------------------------------------------------------------------
dt_full = DecisionTreeClassifier(random_state=42)
dt_full.fit(X_train, y_train)

train_acc_full = accuracy_score(y_train, dt_full.predict(X_train))
test_acc_full = accuracy_score(y_test, dt_full.predict(X_test))
print("\n" + "=" * 55)
print("DECISION TREE (no max_depth - fully grown)")
print("=" * 55)
print(f"Train accuracy: {train_acc_full:.4f}")
print(f"Test accuracy : {test_acc_full:.4f}")
print(f"Tree depth reached: {dt_full.get_depth()}")

plt.figure(figsize=(20, 10))
plot_tree(
    dt_full, feature_names=X.columns, class_names=["No Disease", "Disease"],
    filled=True, rounded=True, fontsize=6, max_depth=3
)
plt.title("Decision Tree (fully grown, showing first 3 levels)")
plt.tight_layout()
plt.savefig("plots/01_decision_tree_full.png", dpi=150)
plt.close()

# ----------------------------------------------------------------------
# 3. Analyze overfitting by varying max_depth
# ----------------------------------------------------------------------
depths = range(1, 21)
train_scores, test_scores = [], []
for d in depths:
    dt = DecisionTreeClassifier(max_depth=d, random_state=42)
    dt.fit(X_train, y_train)
    train_scores.append(accuracy_score(y_train, dt.predict(X_train)))
    test_scores.append(accuracy_score(y_test, dt.predict(X_test)))

plt.figure(figsize=(8, 5))
plt.plot(depths, train_scores, marker="o", label="Train accuracy", color="steelblue")
plt.plot(depths, test_scores, marker="o", label="Test accuracy", color="darkorange")
plt.xlabel("max_depth")
plt.ylabel("Accuracy")
plt.title("Decision Tree: Overfitting Analysis (Accuracy vs Tree Depth)")
plt.legend()
plt.tight_layout()
plt.savefig("plots/02_overfitting_vs_depth.png")
plt.close()

best_depth = depths[int(np.argmax(test_scores))]
print(f"\nBest max_depth by test accuracy: {best_depth} (test acc = {max(test_scores):.4f})")

# Train the depth-controlled tree
dt_controlled = DecisionTreeClassifier(max_depth=best_depth, random_state=42)
dt_controlled.fit(X_train, y_train)
test_acc_controlled = accuracy_score(y_test, dt_controlled.predict(X_test))
print(f"Controlled-depth tree (max_depth={best_depth}) test accuracy: {test_acc_controlled:.4f}")

# ----------------------------------------------------------------------
# 4. Train a Random Forest and compare accuracy
# ----------------------------------------------------------------------
rf = RandomForestClassifier(n_estimators=200, random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)
rf_test_acc = accuracy_score(y_test, y_pred_rf)

print("\n" + "=" * 55)
print("RANDOM FOREST (200 trees)")
print("=" * 55)
print(f"Test accuracy: {rf_test_acc:.4f}")
print("\nClassification report:\n", classification_report(y_test, y_pred_rf))

print("\n" + "=" * 55)
print("MODEL COMPARISON (test accuracy)")
print("=" * 55)
print(f"Decision Tree (fully grown)      : {test_acc_full:.4f}")
print(f"Decision Tree (max_depth={best_depth})       : {test_acc_controlled:.4f}")
print(f"Random Forest (200 trees)        : {rf_test_acc:.4f}")

# Confusion matrix for random forest
cm = confusion_matrix(y_test, y_pred_rf)
fig, ax = plt.subplots(figsize=(6, 5))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["No Disease", "Disease"])
disp.plot(ax=ax, cmap="Greens", colorbar=False)
plt.title("Random Forest - Confusion Matrix")
plt.tight_layout()
plt.savefig("plots/03_rf_confusion_matrix.png")
plt.close()

# Bar chart comparing the three models
plt.figure(figsize=(7, 5))
model_names = [f"Decision Tree\n(full)", f"Decision Tree\n(depth={best_depth})", "Random Forest\n(200 trees)"]
accs = [test_acc_full, test_acc_controlled, rf_test_acc]
sns.barplot(x=model_names, y=accs, palette="Blues_d")
plt.ylim(0, 1)
plt.ylabel("Test Accuracy")
plt.title("Model Comparison: Test Accuracy")
for i, acc in enumerate(accs):
    plt.text(i, acc + 0.02, f"{acc:.3f}", ha="center")
plt.tight_layout()
plt.savefig("plots/04_model_comparison.png")
plt.close()

# ----------------------------------------------------------------------
# 5. Interpret feature importances (Random Forest)
# ----------------------------------------------------------------------
importances = pd.DataFrame({
    "Feature": X.columns,
    "Importance": rf.feature_importances_
}).sort_values("Importance", ascending=False)
print("\nRandom Forest feature importances:\n", importances.to_string(index=False))
importances.to_csv("plots/feature_importances.csv", index=False)

plt.figure(figsize=(8, 6))
sns.barplot(data=importances, x="Importance", y="Feature", palette="viridis")
plt.title("Random Forest - Feature Importances")
plt.tight_layout()
plt.savefig("plots/05_feature_importances.png")
plt.close()

# ----------------------------------------------------------------------
# 6. Evaluate using cross-validation
# ----------------------------------------------------------------------
cv_scores_dt = cross_val_score(DecisionTreeClassifier(max_depth=best_depth, random_state=42), X, y, cv=5)
cv_scores_rf = cross_val_score(RandomForestClassifier(n_estimators=200, random_state=42), X, y, cv=5)

print("\n" + "=" * 55)
print("5-FOLD CROSS-VALIDATION")
print("=" * 55)
print(f"Decision Tree (depth={best_depth}) CV scores: {np.round(cv_scores_dt, 4)}")
print(f"Decision Tree mean CV accuracy: {cv_scores_dt.mean():.4f} (+/- {cv_scores_dt.std():.4f})")
print(f"\nRandom Forest CV scores: {np.round(cv_scores_rf, 4)}")
print(f"Random Forest mean CV accuracy: {cv_scores_rf.mean():.4f} (+/- {cv_scores_rf.std():.4f})")

plt.figure(figsize=(7, 5))
plt.boxplot([cv_scores_dt, cv_scores_rf], tick_labels=["Decision Tree", "Random Forest"])
plt.ylabel("CV Accuracy")
plt.title("5-Fold Cross-Validation Accuracy Comparison")
plt.tight_layout()
plt.savefig("plots/06_cross_validation_comparison.png")
plt.close()

print("\nAll plots saved to plots/")
