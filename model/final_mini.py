import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import xgboost as xgb
import joblib

from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier

# ==========================================================
# Load Dataset
# ==========================================================
np.random.seed(0)
df = pd.read_csv("Android_Permission.csv")
print("Loaded:", df.shape)

print("Nulls in original df:\n", df.isnull().sum())

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

# Columns to scale
columns = ['Rating', 'Number of ratings', 
           'Dangerous permissions count', 'Safe permissions count']

# ----------------------------------------------------------
# 1. FIX missing values in these columns
# ----------------------------------------------------------
imputer = SimpleImputer(strategy='mean')
df[columns] = imputer.fit_transform(df[columns])

# ----------------------------------------------------------
# 2. SCALE these columns
# ----------------------------------------------------------
scaler = StandardScaler()
df[columns] = scaler.fit_transform(df[columns])

# Save scaler and imputer for Streamlit use later
joblib.dump(scaler, "model/scaler.joblib")
joblib.dump(imputer, "model/imputer.joblib")

print("Scaler + Imputer saved in /model/")


# ==========================================================
# Fix Missing Values
# ==========================================================
def replace_mean(x):
    if x.dtype != 'object':
        return x.fillna(x.mean())
    return x

new_df = df.apply(lambda x: replace_mean(x), axis=1)

# ==========================================================
# Permission-only dataframe
# ==========================================================
df_permission = pd.concat([new_df['App'], new_df.iloc[:, 10:]], axis=1)

# Impute numeric missing values
missing_values_column = df_permission.columns[df_permission.isna().any()]
for c in missing_values_column:
    if df_permission[c].dtype in ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']:
        df_permission[c].replace(np.nan, np.nanmean(df_permission[c].unique()), inplace=True)

# ==========================================================
# Fix index alignment (IMPORTANT FIX)
# ==========================================================
df = df.reset_index(drop=True)                # ensures unique index
df_permission.index = df.index               # align permissions to df

# Target variable
df_y = df['Class']
df_y.index = df.index

# ==========================================================
# Features + Labels
# ==========================================================
x = df_permission.iloc[:, :-1]
y = df['Class']

# ==========================================================
# PCA Feature Extraction (NUMERIC ONLY)
# ==========================================================

# Keep only numeric permission columns
df_permission_numeric = df_permission.select_dtypes(include=[np.number])

print("PCA input shape:", df_permission_numeric.shape)

pca = PCA(n_components=100)
principalComponents = pca.fit_transform(df_permission_numeric)

# Reconstruct + binarize
temp_df = pd.DataFrame(pca.inverse_transform(principalComponents))
temp_df[temp_df <= 0] = 0
temp_df[temp_df != 0] = 1
temp_df = temp_df.astype(int)

# Keep unique index alignment
temp_df.index = df.index


# ==========================================================
# Final Feature Matrix – take first 10 PCA-derived binary features
# ==========================================================
take_cols = temp_df.columns[:10]
df_feature = pd.concat([temp_df[take_cols], df_y.rename('Class')], axis=1)

X_feature = df_feature.iloc[:, :-1]
y_feature = df_feature.iloc[:, -1]

X_feature_train, X_feature_test, y_feature_train, y_feature_test = train_test_split(
    X_feature, y_feature, test_size=0.2, random_state=42
)

print("Feature shapes:", X_feature_train.shape, X_feature_test.shape)

# ==========================================================
# XGBoost Model
# ==========================================================
def xgboost_model(X_train, X_test, y_train, y_test):
    reg = GridSearchCV(
        estimator=xgb.XGBClassifier(
            scale_pos_weight=0.5,
            n_jobs=-1,
            eval_metric='logloss'
        ),
        param_grid={},   # no tuning, identical to your original
        cv=10
    )

    reg = reg.fit(X_train, y_train)

    y_pred = reg.predict(X_test)
    y_train_pred = reg.predict(X_train)

    print("\n==== XGBoost Results ====")
    print("Precision (train):", precision_score(y_train, y_train_pred))
    print("Precision (test):", precision_score(y_test, y_pred))
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Recall:", recall_score(y_test, y_pred))
    print("ROC AUC:", roc_auc_score(y_test, y_pred))

    joblib.dump(reg, "xgboost_model.joblib")
    print("Model saved -> xgboost_model.joblib")

    return reg

# Train using PCA features
xgb_model = xgboost_model(X_feature_train, X_feature_test, y_feature_train, y_feature_test)

# ==========================================================
# MLP Model
# ==========================================================
def mlp_model(X_train, X_test, y_train, y_test):
    reg = MLPClassifier(random_state=42, max_iter=300).fit(X_train, y_train)

    y_pred = reg.predict(X_test)
    y_train_pred = reg.predict(X_train)

    print("\n==== MLP Results ====")
    print("Precision (train):", precision_score(y_train, y_train_pred))
    print("Precision (test):", precision_score(y_test, y_pred))
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Recall:", recall_score(y_test, y_pred))
    print("ROC AUC:", roc_auc_score(y_test, y_pred))

    return reg

mlp_model(X_feature_train, X_feature_test, y_feature_train, y_feature_test)

# ==========================================================
# Load Saved XGBoost Model & Predict
# ==========================================================
loaded_model = joblib.load("xgboost_model.joblib")
preds = loaded_model.predict(X_feature_test)
print("\nLoaded model prediction sample:", preds[:10])

joblib.dump(pca, "model/pca_model.joblib")
joblib.dump(scaler, "model/scaler.joblib")



