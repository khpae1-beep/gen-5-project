
# AutoMLOps Suite
# Member 1: Khaliyah Paen
# Heart Disease Model Training

from pathlib import Path
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

# Load dataset
from pathlib import Path

DATA_PATH = Path(__file__).resolve().parent / "data" / "heart.csv"

df = pd.read_csv(DATA_PATH)

# Remove duplicate records
df = df.drop_duplicates()

# Separate features and target
X = df.drop("target", axis=1)
y = df["target"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Create machine learning pipeline
model = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
    ("classifier", RandomForestClassifier(
        n_estimators=100,
        random_state=42
    ))
])

# Train the model
model.fit(X_train, y_train)

# Save trained model
output_dir = Path(__file__).resolve().parent.parent / "models"
output_dir.mkdir(exist_ok=True)

joblib.dump(model, output_dir / "heart_disease_model.joblib")

print("AUTOMLOPS SUITE - MODEL TRAINING")
print("--------------------------------")
print("Algorithm: Random Forest Classifier")
print("Training Samples:", len(X_train))
print("Testing Samples:", len(X_test))
print("Model training completed successfully!")
print("Trained model saved successfully!")
