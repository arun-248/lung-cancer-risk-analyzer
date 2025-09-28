# train_model.py

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, f1_score, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import joblib
import json
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay

DATA_PATH = Path("cancer patient data sets.csv")
MODEL_PATH = Path("model.pkl")
META_PATH = Path("model_meta.json")

df = pd.read_csv(DATA_PATH)
print("Loaded data:", df.shape)

for col in ["index", "Patient Id"]:
    if col in df.columns:
        df = df.drop(columns=[col])

TARGET = "Level"
X = df.drop(columns=[TARGET])
y = df[TARGET].astype(str)

numeric_cols = ["Age"]
cat_like_cols = [c for c in X.columns if c not in numeric_cols]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

numeric_tf = Pipeline([("scaler", StandardScaler())])
categorical_tf = Pipeline([("onehot", OneHotEncoder(handle_unknown="ignore"))])
preprocess = ColumnTransformer(
    transformers=[
        ("num", numeric_tf, numeric_cols),
        ("cat", categorical_tf, cat_like_cols),
    ]
)

logreg = Pipeline([
    ("prep", preprocess),
    ("clf", LogisticRegression(max_iter=200))
])
rf = Pipeline([
    ("prep", preprocess),
    ("clf", RandomForestClassifier(random_state=42))
])

param_grid_lr = {"clf__C": [0.1, 1.0, 3.0]}
param_grid_rf = {"clf__n_estimators": [200, 400], "clf__max_depth": [None, 8, 12]}

models = [
    ("LogReg", logreg, param_grid_lr),
    ("RandomForest", rf, param_grid_rf),
]

best_model, best_name, best_score = None, None, -np.inf

for name, pipe, grid in models:
    print(f"\nTraining {name}...")
    gs = GridSearchCV(pipe, param_grid=grid, scoring="f1_macro", cv=5, n_jobs=-1)
    gs.fit(X_train, y_train)
    preds = gs.predict(X_test)
    f1 = f1_score(y_test, preds, average="macro")
    print(f"{name} F1_macro: {f1:.4f}")
    print(classification_report(y_test, preds))

    if f1 > best_score:
        best_score, best_model, best_name = f1, gs.best_estimator_, name

joblib.dump(best_model, MODEL_PATH)
print(f"Saved best model: {best_name} → {MODEL_PATH}")

meta = {"best_model": best_name, "f1_macro": float(best_score), "classes": sorted(y.unique())}
META_PATH.write_text(json.dumps(meta, indent=2))

cm = confusion_matrix(y_test, best_model.predict(X_test), labels=meta["classes"])
fig, ax = plt.subplots()
ConfusionMatrixDisplay(cm, display_labels=meta["classes"]).plot(ax=ax)
plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=160)
