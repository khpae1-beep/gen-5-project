
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split

# Project directories
BASE_DIR = Path(__file__).resolve().parent.parent
from pathlib import Path
DATA_PATH = Path(__file__).resolve().parent / "data" / "heart.csv"
MODEL_PATH = BASE_DIR / "models" / "heart_disease_model.joblib"
ASSETS_DIR = BASE_DIR / "assets"

ASSETS_DIR.mkdir(parents=True, exist_ok=True)

# Load dataset and remove duplicates
df = pd.read_csv(DATA_PATH)
df = df.drop_duplicates()

X = df.drop(columns=["target"])
y = df["target"]

# Use the same split as model training
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Load trained model
model = joblib.load(MODEL_PATH)

# Make predictions
y_pred = model.predict(X_test)

# -----------------------------------
# Graph 1: Confusion Matrix
# -----------------------------------
ConfusionMatrixDisplay.from_predictions(
    y_test,
    y_pred,
    display_labels=["No Disease", "Disease"],
    cmap="Blues",
    values_format="d"
)

plt.title("Heart Disease Prediction - Confusion Matrix")
plt.tight_layout()
plt.savefig(
    ASSETS_DIR / "confusion_matrix.png",
    dpi=300,
    bbox_inches="tight"
)
plt.close()

print("Confusion Matrix saved successfully!")

# -----------------------------------
# Graph 2: Feature Importance
# -----------------------------------
classifier = model.named_steps["classifier"]

importance = pd.Series(
    classifier.feature_importances_,
    index=X.columns
).sort_values(ascending=True)

plt.figure(figsize=(10, 6))
importance.plot(kind="barh", color="steelblue")

plt.title("Heart Disease Prediction - Feature Importance")
plt.xlabel("Feature Importance Score")
plt.ylabel("Features")
plt.tight_layout()

plt.savefig(
    ASSETS_DIR / "feature_importance.png",
    dpi=300,
    bbox_inches="tight"
)
plt.close()

print("Feature Importance graph saved successfully!")
print("Both graphs are located in the assets folder.")
