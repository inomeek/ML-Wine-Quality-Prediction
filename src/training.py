#%% 
# LIBRARIES

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from rich import print
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.svm import SVC
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression
import pickle
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, StackingClassifier
import xgboost as xgb


# %%
# IMPORT DATASET

df = pd.read_csv(r'wine_data.csv')
print(df.head(3))


# HANDLE MISSING VALUES

# MISSING VALUES
print("[bold]Missing values before filling:[/bold]")
print(df.isnull().sum())
nullsum = df.isnull().sum().sum()
print("\n[bold]Total null values:[/bold]",nullsum)
print('\n\n')

# FILLING MISSING VALUES
# Fill missing values with median for all numeric columns
for col in df.select_dtypes(include='number').columns:
    if df[col].isnull().any():
        median_val = df[col].median()
        df[col] = df[col].fillna(median_val)

print("[bold]Missing values after filling:[/bold]")
print(df.isnull().sum())

nullsum = df.isnull().sum().sum()
print("\n[bold]Total null values:[/bold]", nullsum)
print('\n')


# %%
# DUPLICATE VALUES

# DUPLICATES

df_duplicates = df[df.duplicated(keep="first")]
total_dup = df.duplicated().sum()
print(f"\n[bold]Total numer of duplicate rows:[/bold] {total_dup}")

# REMOVING DUPLICATES
df = df.drop_duplicates(keep="first")
print(f"[bold]Data shape after removing duplicates:[/bold] {df.shape}")
print("\n\n")
df['quality'] = df['quality'].astype(int)

# SAVING NEW CLEANED DATASET
cleaned_file_path = r'wine_cleaned.csv'
df.to_csv(cleaned_file_path, index=False)

# %%

# IMPORT CLEANED WINES DATASET
df = pd.read_csv(r'wine_cleaned.csv')

# BINNING 'QUALITY' COLUMN

# 0–3: Low Quality, 4–7: Medium Quality, 8–10: High Quality
def bin_quality(val):
    if val <= 3:
        return 'Low Quality'
    elif val <= 7:
        return 'Medium Quality'
    else:
        return 'High Quality'

# Creating binned quality column
df['quality_binned'] = df['quality'].apply(bin_quality)

# Encode type and quality binned
type_map = {'red': 0, 'white': 1}
quality_map = {'Low Quality': 0, 'Medium Quality': 1, 'High Quality': 2}

df['type'] = df['type'].map(type_map)
df['quality_binned'] = df['quality_binned'].map(quality_map)

# Reverse maps for decoding later
type_map_rev = {v: k.title() for k, v in type_map.items()}  # {0: 'Red', 1: 'White'}
quality_map_rev = {v: k for k, v in quality_map.items()}


# %%
# TRAIN TEST SPLIT (SCALED AND SMOTE)
X = df.drop(['quality', 'quality_binned'], axis=1)
y = df['quality_binned']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y, shuffle=True)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("[bold]Shape of scaled training data[/bold]")
print(X_train_scaled.shape)
print("[bold]Shape of scaled testing data[/bold]")
print(X_test_scaled.shape)
print('\n\n')

# Apply SMOTE

print("[bold]Before SMOTE:[/bold]")
print(pd.Series(y_train.map(quality_map_rev)).value_counts().sort_index())
print('\n\n')

smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train_scaled, y_train)

print("[bold]After SMOTE:[/bold]")
print(pd.Series(y_train_smote.map(quality_map_rev)).value_counts().sort_index())

# Map test and train labels
y_train_smote_named = y_train_smote.map(quality_map_rev)
y_test_named = y_test.map(quality_map_rev)

# %%
# KNN CLASSIFICATION 

# KNN classifier
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train_smote, y_train_smote)

# Predict on test set
y_pred = knn.predict(X_test_scaled)

# Classification Report
report = classification_report(y_test, y_pred, output_dict=True)

# Convert report dict to DataFrame
report_df = pd.DataFrame(report).transpose()

# Convert metrics to percentage and round
metrics = ['precision', 'recall', 'f1-score']
for metric in metrics:
    report_df.loc[report_df.index != 'support', metric] = report_df.loc[report_df.index != 'support', metric].apply(lambda x: round(x * 100, 2))

# Convert support to int
report_df['support'] = report_df['support'].astype(int)

# Rename index using quality_map_rev for class labels
rename_map = {str(k): v for k, v in quality_map_rev.items()}
report_df.rename(index=rename_map, inplace=True)

print("[bold]KNN Classification Report (%):[/bold]\n")
print(report_df)
print("\n[bold]Overall Accuracy:[/bold]", f"{report_df.loc['accuracy', 'precision']:.2f}%\n\n")

# Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
            xticklabels=[quality_map_rev[i] for i in sorted(y_test.unique())],
            yticklabels=[quality_map_rev[i] for i in sorted(y_test.unique())])
plt.title("KNN Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.show()

# K-Fold Cross Validation (on SMOTE data)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(knn, X_train_smote, y_train_smote, cv=cv, scoring='accuracy')

# Convert to percentage and round
cv_scores_percent = [round(score * 100, 2) for score in cv_scores]

print('\n\n')
print("[bold]K-Fold Cross Validation Scores (% Accuracy):[/bold]")
for i, score in enumerate(cv_scores_percent, 1):
    print(f"Fold {i}: {score}%")
print(f"\n[bold]Average CV Accuracy:[/bold] {round(np.mean(cv_scores_percent), 2)}%")\


# %%
# SVM CLASSIFICATION

# Initialize SVM
svm = SVC(kernel='linear', random_state=42)
svm.fit(X_train_smote, y_train_smote)

# Predict on test set
y_pred = svm.predict(X_test_scaled)

# Classification report
report = classification_report(y_test, y_pred, output_dict=True)
report_df = pd.DataFrame(report).transpose()

# Format precision, recall, f1-score as %
metrics = ['precision', 'recall', 'f1-score']
for metric in metrics:
    report_df.loc[report_df.index != 'support', metric] = report_df.loc[report_df.index != 'support', metric].apply(lambda x: round(x * 100, 2))

report_df['support'] = report_df['support'].astype(int)

# Decode class labels
report_df.rename(index={str(k): v for k, v in quality_map_rev.items()}, inplace=True)

# Reorder for display
order = list(quality_map_rev.values()) + ['macro avg', 'weighted avg', 'accuracy']
report_df = report_df.loc[order]

print("[bold]SVM Classification Report (%):[/bold]\n")
print(report_df)

print("\n[bold]Overall Accuracy:[/bold]", f"{report_df.loc['accuracy', 'precision']:.2f}%")

# Confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
            xticklabels=[quality_map_rev[i] for i in sorted(y_test.unique())],
            yticklabels=[quality_map_rev[i] for i in sorted(y_test.unique())])
plt.title("SVM Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.show()

# K-Fold CV on SMOTE data
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(svm, X_train_smote, y_train_smote, cv=cv, scoring='accuracy')
cv_scores_percent = [round(score * 100, 2) for score in cv_scores]

print("\n[bold]SVM K-Fold Cross-Validation Scores (% Accuracy):[/bold]")
for i, score in enumerate(cv_scores_percent, 1):
    print(f"Fold {i}: {score}%")
print(f"\n[bold]Average CV Accuracy:[/bold] {round(np.mean(cv_scores_percent), 2)}%")


# %%
# LOGISTIC REGRESSION 

# Train Logistic Regression
logreg = LogisticRegression(max_iter=1000, random_state=42)
logreg.fit(X_train_smote, y_train_smote)

# Predict on test data
y_pred = logreg.predict(X_test_scaled)

# Classification Report
report = classification_report(y_test, y_pred, output_dict=True)
report_df = pd.DataFrame(report).transpose()

# Format precision, recall, f1-score as %
metrics = ['precision', 'recall', 'f1-score']
for metric in metrics:
    report_df.loc[report_df.index != 'support', metric] = report_df.loc[report_df.index != 'support', metric].apply(lambda x: round(x * 100, 2))

# Convert support to int
report_df['support'] = report_df['support'].astype(int)

# Rename index using quality_map_rev (e.g., 0 → Low Quality)
report_df.rename(index={str(k): v for k, v in quality_map_rev.items()}, inplace=True)

print("[bold]Logistic Regression Classification Report (%):[/bold]\n")
print(report_df)

print(f"\n[bold]Overall Accuracy:[/bold] {report_df.loc['accuracy', 'precision']:.2f}%")

# Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
            xticklabels=[quality_map_rev[i] for i in sorted(y_test.unique())],
            yticklabels=[quality_map_rev[i] for i in sorted(y_test.unique())])
plt.title("Logistic Regression Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.show()

# K-Fold Cross-Validation on SMOTE-applied training data
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(logreg, X_train_smote, y_train_smote, cv=cv, scoring='accuracy')
cv_scores_percent = [round(score * 100, 2) for score in cv_scores]

print("\n[bold]K-Fold Cross Validation Scores (% Accuracy):[/bold]")
for i, score in enumerate(cv_scores_percent, 1):
    print(f"Fold {i}: {score}%")
print(f"\n[bold]Average CV Accuracy:[/bold] {round(np.mean(cv_scores_percent), 2)}%")


# %%
# TRAIN TEST (UNSCALED)

# Features (drop target columns)
X_unscaled = df.drop(['quality', 'quality_binned'], axis=1)
y = df['quality_binned']

# Train-test split (unscaled)
X_train_unscaled, X_test_unscaled, y_train_unscaled, y_test_unscaled = train_test_split(
    X_unscaled, y, test_size=0.2, random_state=42, stratify=y, shuffle=True
)


# %%
# GRADIENT BOOSTING

# Train Gradient Boosting
gbc = GradientBoostingClassifier(random_state=42)
gbc.fit(X_train_unscaled, y_train_unscaled)

# Predict on test set
y_pred = gbc.predict(X_test_unscaled)

# Classification report
report = classification_report(y_test_unscaled, y_pred, output_dict=True)
report_df = pd.DataFrame(report).transpose()

# Format as %
metrics = ['precision', 'recall', 'f1-score']
for metric in metrics:
    report_df.loc[report_df.index != 'support', metric] = report_df.loc[report_df.index != 'support', metric].apply(lambda x: round(x * 100, 2))

report_df['support'] = report_df['support'].astype(int)

# Rename index for readability
report_df.rename(index={str(k): v for k, v in quality_map_rev.items()}, inplace=True)

print("[bold]Gradient Boosting Classification Report (%):[/bold]\n")
print(report_df)
print("\n[bold]Overall Accuracy:[/bold]", f"{report_df.loc['accuracy', 'precision']:.2f}%\n\n")


# Confusion Matrix
conf_matrix = confusion_matrix(y_test_unscaled, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
            xticklabels=[quality_map_rev[i] for i in sorted(y_test_unscaled.unique())],
            yticklabels=[quality_map_rev[i] for i in sorted(y_test_unscaled.unique())])
plt.title("Gradient Boosting Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.show()

# K-Fold Cross Validation (Stratified, 5 folds)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(gbc, X_unscaled, y, cv=cv, scoring='accuracy')

cv_scores_percent = [round(score * 100, 2) for score in cv_scores]

# Print K-Fold results
print("\n[bold]K-Fold Cross Validation Scores (% Accuracy):[/bold]")
for i, score in enumerate(cv_scores_percent, 1):
    print(f"Fold {i}: {score}%")

# Print average accuracy
print(f"\n[bold]Average CV Accuracy:[/bold] {round(np.mean(cv_scores_percent), 2)}%")


# %%
# RANDOM FOREST CLASSIFICATION 


# Initialize Random Forest
rf_clf = RandomForestClassifier(n_estimators=100, random_state=42)

# Fit the model
rf_clf.fit(X_train_unscaled, y_train_unscaled)

# Predict on test set
y_pred = rf_clf.predict(X_test_unscaled)

# Classification report
report = classification_report(y_test_unscaled, y_pred, output_dict=True)
report_df = pd.DataFrame(report).transpose()

# Format as %
metrics = ['precision', 'recall', 'f1-score']
for metric in metrics:
    report_df.loc[report_df.index != 'support', metric] = report_df.loc[report_df.index != 'support', metric].apply(lambda x: round(x * 100, 2))

report_df['support'] = report_df['support'].astype(int)

# Rename index for readability
report_df.rename(index={str(k): v for k, v in quality_map_rev.items()}, inplace=True)

print("[bold]Random Forest Classification Report (%):[/bold]\n")
print(report_df)
print("\n[bold]Overall Accuracy:[/bold]", f"{report_df.loc['accuracy', 'precision']:.2f}%\n\n")


# Confusion Matrix
cm = confusion_matrix(y_test_unscaled, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=[quality_map_rev[i] for i in sorted(y_test_unscaled.unique())],
            yticklabels=[quality_map_rev[i] for i in sorted(y_test_unscaled.unique())])
plt.title("Random Forest Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.show()


# K-Fold Cross Validation (Stratified, 5 folds)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(rf_clf, X_unscaled, y, cv=cv, scoring='accuracy')

# Print K-Fold results
print("\n[bold]K-Fold Cross Validation Scores (% Accuracy):[/bold]")
for i, score in enumerate(cv_scores, 1):
    print(f"Fold {i}: {round(score * 100, 2)}%")

# Print average accuracy
print(f"\n[bold]Average CV Accuracy:[/bold] {round(np.mean(cv_scores) * 100, 2)}%")

# %%
# XG BOOST


# Initialize the XGBoost classifier
xgb_clf = xgb.XGBClassifier(
    objective='multi:softmax',  # For multi-class classification
    num_class=3,                # Number of classes
    eval_metric='mlogloss',
    random_state=42
)

# Fit the model
xgb_clf.fit(X_train_unscaled, y_train_unscaled)

# Predict on test set
y_pred = xgb_clf.predict(X_test_unscaled)

# Classification report
report_dict = classification_report(
    y_test_unscaled,
    y_pred,
    target_names=[quality_map_rev[i] for i in sorted(quality_map_rev.keys())],
    output_dict=True
)

report_df = pd.DataFrame(report_dict).transpose()

# Format as % for precision, recall, f1-score (except 'support')
metrics = ['precision', 'recall', 'f1-score']
for metric in metrics:
    report_df.loc[report_df.index != 'support', metric] = report_df.loc[report_df.index != 'support', metric].apply(lambda x: round(x * 100, 2))

# Convert support to integer
report_df['support'] = report_df['support'].astype(int)

print("[bold]XGBoost Classification Report:[/bold]\n")
print(report_df)
print("\n[bold]Overall Accuracy:[/bold]", f"{report_df.loc['accuracy', 'precision']:.2f}%\n\n")


# Confusion matrix plot
conf_matrix = confusion_matrix(y_test_unscaled, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Oranges',
            xticklabels=[quality_map_rev[i] for i in sorted(quality_map_rev.keys())],
            yticklabels=[quality_map_rev[i] for i in sorted(quality_map_rev.keys())])
plt.title("XGBoost Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.show()

# K-Fold Cross Validation (Stratified, 5 folds)
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(xgb_clf, X_unscaled, y, cv=kf, scoring='accuracy')

# Convert K-Fold cross-validation scores to percentages
cv_scores_percent = [round(score * 100, 2) for score in cv_scores]

# Print K-Fold results in desired format
print("\n[bold]K-Fold Cross Validation Scores (% Accuracy):[/bold]")
for i, score in enumerate(cv_scores_percent, 1):
    print(f"Fold {i}: {score}%")

# Print average accuracy
print(f"\n[bold]Average CV Accuracy:[/bold] {round(np.mean(cv_scores_percent), 2)}%")
print('\n\n')

# Get feature importances (gain, weight, cover - default is 'weight')
xgb_importances = xgb_clf.feature_importances_

# DataFrame and plot as above
feat_importance_df = pd.DataFrame({
    'Feature': X_train.columns,
    'Importance': xgb_importances
}).sort_values(by='Importance', ascending=False)

# Round importance values to 2 decimal places
feat_importance_df['Importance'] = feat_importance_df['Importance'].round(2)

print("[bold]Feature Importance Ranking[\bold]")
print(feat_importance_df)

# Plotting Feature importance
plt.figure(figsize=(9,6))
sns.barplot(data=feat_importance_df, x='Importance', y='Feature',hue=None)
plt.title('Feature Importance Ranking - XGBoost')
plt.xlabel('Importance Score')
plt.ylabel('Feature')
plt.tight_layout()
plt.show()


# %%
# STACKING

# Define base models
estimators = [
    ('rf', RandomForestClassifier(random_state=42)),
    ('gb', GradientBoostingClassifier(random_state=42)),
    ('xgb', xgb.XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42))
]

# Meta-model (final estimator)
final_estimator = LogisticRegression(max_iter=1000, random_state=42)

# Define stacking classifier
stacking_clf = StackingClassifier(
    estimators=estimators,
    final_estimator=final_estimator,
    cv=5,
    n_jobs=-1,
    passthrough=False  
)

# Fit on training data (use unscaled)
stacking_clf.fit(X_train_unscaled, y_train_unscaled)

# Predict on test set
y_pred = stacking_clf.predict(X_test_unscaled)

# Classification Report
report = classification_report(
    y_test_unscaled,
    y_pred,
    target_names=[quality_map_rev[i] for i in sorted(quality_map_rev.keys())],
    output_dict=True
)

report_df = pd.DataFrame(report).transpose()

# Format precision, recall, f1-score as percentages
metrics = ['precision', 'recall', 'f1-score']
for metric in metrics:
    report_df.loc[report_df.index != 'support', metric] = report_df.loc[report_df.index != 'support', metric].apply(lambda x: round(x * 100, 2))

# Convert support to integer
report_df['support'] = report_df['support'].astype(int)

# Print final report
print("\nStacking Classifier Classification Report (%):\n")
print(report_df)

print("\n[bold]Overall Accuracy:[/bold]", f"{report_df.loc['accuracy', 'precision']:.2f}%\n\n")

# Confusion Matrix
conf_matrix = confusion_matrix(y_test_unscaled, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Purples',
            xticklabels=[quality_map_rev[i] for i in sorted(quality_map_rev.keys())],
            yticklabels=[quality_map_rev[i] for i in sorted(quality_map_rev.keys())])
plt.title("Stacking Classifier Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.show()

# %%
# SAVING TRAINED STACKED MODEL (ONLY ENSEMBLE TECHNIQUES)

# Save the trained stacked model
with open(r'stacked_model.pkl', 'wb') as f:
    pickle.dump(stacking_clf, f)


# %%
