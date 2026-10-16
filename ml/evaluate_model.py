# AutoMLOps Suite
# Member 1: Khaliyah Paen
# Heart Disease Model Evaluation


from pathlib import Path
import json
import joblib
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    ConfusionMatrixDisplay
)

# Step 1: Define project paths
project_dir = Path(__file__).resolve().parent.parent

data_path = project_dir / "data" / "heart.csv"
model_path = project_dir / "models" / "heart_disease_model.joblib"
assets_dir = project_dir / "assets"

assets_dir.mkdir(exist_ok=True)

# Step 2: Load the Heart Disease Dataset
df = pd.read_csv(data_path)

# Remove duplicate records
df = df.drop_duplicates()

# Separate features and target
X = df.drop("target", axis=1)
y = df["target"]

# Step 3: Recreate the training and testing split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Step 4: Load the trained model
model = joblib.load(model_path)

# Step 5: Make predictions
train_predictions = model.predict(X_train)
test_predictions = model.predict(X_test)

# Step 6: Calculate evaluation metrics
def calculate_metrics(actual, predicted):
    return {
        "accuracy": accuracy_score(actual, predicted),
        "precision": precision_score(
            actual, predicted, zero_division=0
        ),
        "recall": recall_score(
            actual, predicted, zero_division=0
        ),
        "f1_score": f1_score(
            actual, predicted, zero_division=0
        )
    }

train_metrics = calculate_metrics(
    y_train, train_predictions
)

test_metrics = calculate_metrics(
    y_test, test_predictions
)

# Step 7: Display evaluation results
print("AUTOMLOPS SUITE - MODEL EVALUATION")
print("----------------------------------")

print("\nTraining Set Results:")
for metric, value in train_metrics.items():
    print(f"{metric}: {value:.4f}")

print("\nTesting Set Results:")
for metric, value in test_metrics.items():
    print(f"{metric}: {value:.4f}")

print("\nClassification Report:")
print(classification_report(
    y_test,
    test_predictions,
    zero_division=0
))

# Step 8: Generate confusion matrix
ConfusionMatrixDisplay.from_predictions(
    y_test,
    test_predictions,
    display_labels=["No Heart Disease", "Heart Disease"],
    cmap="Blues"
)

plt.title("Heart Disease Prediction - Confusion Matrix")
plt.tight_layout()
plt.savefig(
    assets_dir / "confusion_matrix.png",
    dpi=300
)
plt.close()

# Step 9: Generate feature importance graph
classifier = model.named_steps["classifier"]

importance = pd.Series(
    classifier.feature_importances_,
    index=X.columns
).sort_values(ascending=False)

plt.figure(figsize=(10, 6))
importance.head(10).sort_values().plot(
    kind="barh",
    color="steelblue"
)

plt.title("Top 10 Important Features")
plt.xlabel("Feature Importance")
plt.tight_layout()
plt.savefig(
    assets_dir / "feature_importance.png",
    dpi=300
)
plt.close()

# Step 10: Save evaluation results
results = {
    "model": "Random Forest Classifier",
    "dataset": "Kaggle Heart Disease Dataset",
    "training": train_metrics,
    "testing": test_metrics
}

with open(assets_dir / "evaluation_results.json", "w") as file:
    json.dump(results, file, indent=4)

print("\nEvaluation completed successfully!")
print("Confusion matrix saved!")
print("Feature importance graph saved!")
print("Evaluation results saved!")
