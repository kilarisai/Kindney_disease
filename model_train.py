# Install dependencies if not already installed
# pip install pytorch-tabnet scikit-learn pandas numpy matplotlib joblib

print("Starting TabNet CKD training...")

import pandas as pd
import numpy as np
import torch
import joblib
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, recall_score, f1_score, confusion_matrix, roc_auc_score

from pytorch_tabnet.tab_model import TabNetClassifier


# ------------------------------
# 1. Load Dataset
# ------------------------------

print("Loading dataset...")

url = r"C:\Users\himaj\Downloads\kidney_disease.csv"

df = pd.read_csv(url)

print("Dataset Loaded Successfully")
print("Dataset Shape:", df.shape)


# ------------------------------
# 2. Robust Preprocessing
# ------------------------------

print("Cleaning dataset...")

# Drop unnamed columns
df = df.drop(columns=[col for col in df.columns if "unnamed" in col.lower()])

# Strip whitespace
df = df.applymap(lambda x: str(x).strip() if isinstance(x, str) else x)

# Replace '?' values
df.replace(['?', '\t?'], np.nan, inplace=True)


# ------------------------------
# 3. Encode Categorical Columns
# ------------------------------

print("Encoding categorical columns...")

categorical_cols = ['rbc','pc','pcc','ba','htn','dm','cad','appet','pe','ane']

for col in categorical_cols:
    df[col] = df[col].fillna(df[col].mode()[0])
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])


# ------------------------------
# 4. Handle Numeric Columns
# ------------------------------

print("Processing numeric columns...")

numeric_cols = df.columns.difference(categorical_cols + ['classification'])

for col in numeric_cols:

    df[col] = df[col].apply(lambda x: str(x).strip() if isinstance(x,str) else x)

    df[col] = pd.to_numeric(df[col], errors='coerce')

    df[col] = df[col].fillna(df[col].median())


# ------------------------------
# 5. Encode Target Column
# ------------------------------

print("Encoding target column...")

df['classification'] = df['classification'].apply(
    lambda x: 1 if str(x).lower().strip() == 'ckd' else 0
)


# ------------------------------
# 6. Split Features and Labels
# ------------------------------

print("Splitting dataset...")

X = df.drop('classification', axis=1).values
y = df['classification'].values


# ------------------------------
# 7. Feature Scaling
# ------------------------------

print("Scaling features...")

scaler = StandardScaler()
X = scaler.fit_transform(X)


# ------------------------------
# 8. Train Test Split
# ------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("Training Data Shape:", X_train.shape)
print("Testing Data Shape:", X_test.shape)


# ------------------------------
# 9. Initialize TabNet Model
# ------------------------------

print("Initializing TabNet Model...")

tabnet_params = dict(
    n_d=16,
    n_a=16,
    n_steps=5,
    gamma=1.5,
    n_independent=2,
    n_shared=2,
    optimizer_fn=torch.optim.Adam,
    optimizer_params=dict(lr=2e-2),
    mask_type='entmax',
    scheduler_params={"step_size":50, "gamma":0.9},
    scheduler_fn=torch.optim.lr_scheduler.StepLR,
    verbose=1,
    seed=42
)

clf = TabNetClassifier(**tabnet_params)


# ------------------------------
# 10. Train TabNet Model
# ------------------------------

print("Training TabNet model...")

clf.fit(
    X_train,
    y_train,
    eval_set=[(X_train, y_train), (X_test, y_test)],
    eval_name=['train','valid'],
    eval_metric=['accuracy','auc'],
    max_epochs=200,
    patience=20,
    batch_size=32,
    virtual_batch_size=16,
    num_workers=0,
    drop_last=False
)


print("Training Completed!")


# ------------------------------
# 11. Evaluate Model
# ------------------------------

print("Evaluating model...")

y_pred = clf.predict(X_test)
y_pred_proba = clf.predict_proba(X_test)[:,1]

accuracy = accuracy_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred_proba)
cm = confusion_matrix(y_test, y_pred)

print("\n----- Model Evaluation -----")
print("Accuracy:", accuracy)
print("Recall:", recall)
print("F1 Score:", f1)
print("ROC AUC:", roc_auc)
print("Confusion Matrix:\n", cm)


# ------------------------------
# 12. Feature Importance
# ------------------------------

print("Generating Feature Importance Plot...")

feature_importances = clf.feature_importances_

plt.figure(figsize=(12,6))

plt.bar(range(X.shape[1]), feature_importances)

plt.xticks(
    range(X.shape[1]),
    df.drop('classification', axis=1).columns,
    rotation=45,
    ha='right'
)

plt.title("Feature Importance from TabNet")

plt.tight_layout()

plt.show()


# ------------------------------
# 13. Save Model and Scaler
# ------------------------------

print("Saving model and scaler...")

clf.save_model("tabnet_kidney_model")

joblib.dump(scaler,"scaler.pkl")

print("Model Saved Successfully!")


print("Training Script Finished.")