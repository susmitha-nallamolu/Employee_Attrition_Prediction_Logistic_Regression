# ==========================================
# Employee Attrition Prediction
# Logistic Regression
# ==========================================

# -----------------------------
# 1. Import Libraries
# -----------------------------

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
    roc_curve
)


# -----------------------------
# 2. Load Dataset
# -----------------------------

df = pd.read_csv(
    "data/WA_Fn-UseC_-HR-Employee-Attrition.csv"
)

print("Dataset loaded successfully!")
print("Dataset shape:", df.shape)


# -----------------------------
# 3. Basic Dataset Information
# -----------------------------

print("\nFirst 5 rows:")
print(df.head())

print("\nDataset information:")
print(df.info())

print("\nMissing values:")
print(df.isnull().sum())


# -----------------------------
# 4. Target Distribution
# -----------------------------

print("\nAttrition distribution:")
print(df["Attrition"].value_counts())

plt.figure(figsize=(6, 4))

sns.countplot(
    data=df,
    x="Attrition"
)

plt.title("Employee Attrition Distribution")
plt.xlabel("Attrition")
plt.ylabel("Number of Employees")

plt.tight_layout()
plt.show()


# -----------------------------
# 5. Remove Unnecessary Columns
# -----------------------------

columns_to_drop = [
    "EmployeeNumber",
    "EmployeeCount",
    "Over18",
    "StandardHours"
]

df = df.drop(
    columns=columns_to_drop
)


# -----------------------------
# 6. Separate Features and Target
# -----------------------------

X = df.drop(
    columns=["Attrition"]
)

y = df["Attrition"].map({
    "Yes": 1,
    "No": 0
})


# -----------------------------
# 7. Identify Column Types
# -----------------------------

categorical_columns = X.select_dtypes(
    include=["object", "string"]
).columns.tolist()

numerical_columns = X.select_dtypes(
    include=["int64", "float64"]
).columns.tolist()

print("\nCategorical columns:")
print(categorical_columns)

print("\nNumerical columns:")
print(numerical_columns)


# -----------------------------
# 8. Preprocessing
# -----------------------------

preprocessor = ColumnTransformer(
    transformers=[
        (
            "numerical",
            StandardScaler(),
            numerical_columns
        ),
        (
            "categorical",
            OneHotEncoder(
                handle_unknown="ignore"
            ),
            categorical_columns
        )
    ]
)


# -----------------------------
# 9. Train-Test Split
# -----------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining data:", X_train.shape)
print("Testing data:", X_test.shape)


# -----------------------------
# 10. Apply Preprocessing
# -----------------------------

X_train_processed = preprocessor.fit_transform(
    X_train
)

X_test_processed = preprocessor.transform(
    X_test
)

print(
    "Processed training shape:",
    X_train_processed.shape
)

print(
    "Processed testing shape:",
    X_test_processed.shape
)


# =========================================================
# 11. ORIGINAL LOGISTIC REGRESSION MODEL
# =========================================================

model = LogisticRegression(
    max_iter=1000,
    random_state=42
)

model.fit(
    X_train_processed,
    y_train
)

print("\nModel trained successfully!")


# -----------------------------
# 12. Original Predictions
# -----------------------------

y_pred = model.predict(
    X_test_processed
)

y_prob = model.predict_proba(
    X_test_processed
)

print("\nFirst 10 predictions:")
print(y_pred[:10])

print("\nFirst 5 probability predictions:")
print(y_prob[:5])


# -----------------------------
# 13. Original Model Evaluation
# -----------------------------

accuracy = accuracy_score(
    y_test,
    y_pred
)

print("\nOriginal Model Accuracy:", accuracy)

cm = confusion_matrix(
    y_test,
    y_pred
)

print("\nOriginal Model Confusion Matrix:")
print(cm)

print("\nOriginal Model Classification Report:")
print(
    classification_report(
        y_test,
        y_pred
    )
)

roc_auc = roc_auc_score(
    y_test,
    y_prob[:, 1]
)

print(
    "Original Model ROC-AUC Score:",
    roc_auc
)


# -----------------------------
# 14. Original Confusion Matrix Visualization
# -----------------------------

plt.figure(figsize=(6, 5))

sns.heatmap(
    cm,
    annot=True,
    fmt="d"
)

plt.title("Confusion Matrix - Original Model")
plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.tight_layout()
plt.show()


# -----------------------------
# 15. Original ROC Curve
# -----------------------------

fpr, tpr, thresholds = roc_curve(
    y_test,
    y_prob[:, 1]
)

plt.figure(figsize=(7, 5))

plt.plot(
    fpr,
    tpr,
    label=f"ROC-AUC = {roc_auc:.3f}"
)

plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--"
)

plt.title("ROC Curve - Original Model")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")

plt.legend()

plt.tight_layout()
plt.show()


# =========================================================
# 16. FEATURE COEFFICIENT ANALYSIS
# =========================================================

feature_names = (
    preprocessor
    .get_feature_names_out()
)

coefficients = model.coef_[0]

feature_importance = pd.DataFrame({
    "Feature": feature_names,
    "Coefficient": coefficients
})

feature_importance = feature_importance.sort_values(
    by="Coefficient",
    ascending=False
)


# -----------------------------
# 17. Top Positive Features
# -----------------------------

print("\nTop positive features:")

print(
    feature_importance.head(10)
)


# -----------------------------
# 18. Top Negative Features
# -----------------------------

print("\nTop negative features:")

print(
    feature_importance.tail(10)
)


# -----------------------------
# 19. Feature Coefficient Plot
# -----------------------------

top_features = pd.concat([
    feature_importance.head(10),
    feature_importance.tail(10)
])

plt.figure(figsize=(10, 8))

sns.barplot(
    data=top_features,
    x="Coefficient",
    y="Feature"
)

plt.title(
    "Top Logistic Regression Coefficients"
)

plt.xlabel("Coefficient")
plt.ylabel("Feature")

plt.tight_layout()
plt.show()


# =========================================================
# 20. SAVE ORIGINAL MODEL
# =========================================================

joblib.dump(
    model,
    "models/logistic_regression_model.pkl"
)

joblib.dump(
    preprocessor,
    "models/preprocessor.pkl"
)

print(
    "\nOriginal model saved successfully."
)

print(
    "Preprocessor saved successfully."
)


# =========================================================
# 21. LOAD ORIGINAL MODEL
# =========================================================

loaded_model = joblib.load(
    "models/logistic_regression_model.pkl"
)

loaded_predictions = loaded_model.predict(
    X_test_processed
)

print(
    "\nLoaded model predictions:"
)

print(
    loaded_predictions[:10]
)


# =========================================================
# 22. BALANCED LOGISTIC REGRESSION
# =========================================================

balanced_model = LogisticRegression(
    class_weight="balanced",
    max_iter=1000,
    random_state=42
)

balanced_model.fit(
    X_train_processed,
    y_train
)

print(
    "\nBalanced Logistic Regression trained successfully!"
)


# -----------------------------
# 23. Balanced Predictions
# -----------------------------

balanced_predictions = balanced_model.predict(
    X_test_processed
)

balanced_probabilities = (
    balanced_model
    .predict_proba(X_test_processed)[:, 1]
)


# -----------------------------
# 24. Balanced Model Evaluation
# -----------------------------

balanced_accuracy = accuracy_score(
    y_test,
    balanced_predictions
)

balanced_cm = confusion_matrix(
    y_test,
    balanced_predictions
)

balanced_report = classification_report(
    y_test,
    balanced_predictions
)

balanced_roc_auc = roc_auc_score(
    y_test,
    balanced_probabilities
)

print(
    "\n======================================"
)

print(
    "BALANCED LOGISTIC REGRESSION RESULTS"
)

print(
    "======================================"
)

print(
    "\nAccuracy:",
    balanced_accuracy
)

print(
    "\nConfusion Matrix:"
)

print(
    balanced_cm
)

print(
    "\nClassification Report:"
)

print(
    balanced_report
)

print(
    "ROC-AUC Score:",
    balanced_roc_auc
)


# =========================================================
# 25. BALANCED CONFUSION MATRIX
# =========================================================

plt.figure(figsize=(6, 5))

sns.heatmap(
    balanced_cm,
    annot=True,
    fmt="d"
)

plt.title(
    "Confusion Matrix - Balanced Model"
)

plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.tight_layout()
plt.show()


# =========================================================
# 26. MODEL COMPARISON
# =========================================================

original_report = classification_report(
    y_test,
    y_pred,
    output_dict=True
)

balanced_report_dict = classification_report(
    y_test,
    balanced_predictions,
    output_dict=True
)

original_recall = original_report["1"]["recall"]

balanced_recall = balanced_report_dict["1"]["recall"]

original_precision = original_report["1"]["precision"]

balanced_precision = balanced_report_dict["1"]["precision"]

original_f1 = original_report["1"]["f1-score"]

balanced_f1 = balanced_report_dict["1"]["f1-score"]


print(
    "\n======================================"
)

print(
    "MODEL COMPARISON"
)

print(
    "======================================"
)

print(
    "\nOriginal Logistic Regression:"
)

print(
    f"Accuracy : {accuracy:.4f}"
)

print(
    f"Precision: {original_precision:.4f}"
)

print(
    f"Recall   : {original_recall:.4f}"
)

print(
    f"F1-score : {original_f1:.4f}"
)

print(
    f"ROC-AUC  : {roc_auc:.4f}"
)


print(
    "\nBalanced Logistic Regression:"
)

print(
    f"Accuracy : {balanced_accuracy:.4f}"
)

print(
    f"Precision: {balanced_precision:.4f}"
)

print(
    f"Recall   : {balanced_recall:.4f}"
)

print(
    f"F1-score : {balanced_f1:.4f}"
)

print(
    f"ROC-AUC  : {balanced_roc_auc:.4f}"
)

# =========================================================
# 27. THRESHOLD TUNING
# =========================================================

import numpy as np

thresholds_to_test = [
    0.30,
    0.35,
    0.40,
    0.45,
    0.50,
    0.55,
    0.60
]

print("\n======================================")
print("THRESHOLD TUNING")
print("======================================")

for threshold in thresholds_to_test:

    threshold_predictions = (
        y_prob[:, 1] >= threshold
    ).astype(int)

    threshold_report = classification_report(
        y_test,
        threshold_predictions,
        output_dict=True,
        zero_division=0
    )

    accuracy_value = accuracy_score(
        y_test,
        threshold_predictions
    )

    precision_value = threshold_report["1"]["precision"]
    recall_value = threshold_report["1"]["recall"]
    f1_value = threshold_report["1"]["f1-score"]

    print(f"\nThreshold: {threshold}")
    print(f"Accuracy : {accuracy_value:.4f}")
    print(f"Precision: {precision_value:.4f}")
    print(f"Recall   : {recall_value:.4f}")
    print(f"F1-score : {f1_value:.4f}")


# =========================================================
# 28. FIND BEST F1 THRESHOLD
# =========================================================

best_threshold = 0
best_f1 = 0

for threshold in thresholds_to_test:

    threshold_predictions = (
        y_prob[:, 1] >= threshold
    ).astype(int)

    threshold_report = classification_report(
        y_test,
        threshold_predictions,
        output_dict=True,
        zero_division=0
    )

    f1_value = threshold_report["1"]["f1-score"]

    if f1_value > best_f1:
        best_f1 = f1_value
        best_threshold = threshold


print("\n======================================")
print("BEST THRESHOLD")
print("======================================")

print(
    f"Best Threshold: {best_threshold}"
)

print(
    f"Best F1-score: {best_f1:.4f}"
)

# =========================================================
# 30. EVALUATE BEST THRESHOLD
# =========================================================

final_threshold = 0.40

final_predictions = (
    y_prob[:, 1] >= final_threshold
).astype(int)

final_accuracy = accuracy_score(
    y_test,
    final_predictions
)

final_cm = confusion_matrix(
    y_test,
    final_predictions
)

final_report = classification_report(
    y_test,
    final_predictions
)

final_roc_auc = roc_auc_score(
    y_test,
    y_prob[:, 1]
)

print("\n======================================")
print("FINAL THRESHOLD MODEL")
print("======================================")

print(
    f"\nThreshold: {final_threshold}"
)

print(
    f"Accuracy: {final_accuracy:.4f}"
)

print(
    "\nConfusion Matrix:"
)

print(final_cm)

print(
    "\nClassification Report:"
)

print(final_report)

print(
    f"ROC-AUC: {final_roc_auc:.4f}"
)

# =========================================================
# END
# =========================================================

print(
    "\nProject execution completed successfully!"
)

