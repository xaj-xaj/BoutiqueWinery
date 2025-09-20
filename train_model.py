import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
import joblib
import os

# Load dataset
df = pd.read_csv("winequality-red-selected-missing.csv", sep=',')
print(df.shape)
print(df.columns)
print(df.head())

# Create binary target
df['target'] = (df['quality'] >= 7).astype(int)
print("target\n", df['target'].value_counts(normalize=True).rename("proportion"))

# Check missing values
print("Missing per column:\n", df.isnull().sum())

X = df.drop(columns=['quality', 'target'])
y = df['target']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Pipeline: impute + model
pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("model", RandomForestClassifier(random_state=42))
])

# Cross-validation
cv_scores = cross_val_score(pipeline, X_train, y_train, cv=5, scoring="accuracy")
print(f"Cross-validation accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

# Train final model
pipeline.fit(X_train, y_train)

# Evaluate on test set
y_pred = pipeline.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"Test Accuracy: {acc:.4f}")
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Not Good", "Good"], yticklabels=["Not Good", "Good"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

# Save model + imputer
os.makedirs("models", exist_ok=True)
joblib.dump(pipeline.named_steps['model'], "models/rf_model.joblib")
joblib.dump(pipeline.named_steps['imputer'].statistics_, "models/median_values.joblib")
print("Model and imputer saved.")
