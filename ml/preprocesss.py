# AutoMLOps Suite
# Member 1: Khaliyah Paen
# Heart Disease Prediction - Data Preprocessing
# Dataset Source: Kaggle
# https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset

import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

# Step 1: Load the Heart Disease Dataset
data_path = Path(__file__).resolve().parent.parent / "data" / "heart.csv"

df = pd.read_csv(data_path)

print("AUTOMLOPS SUITE - HEART DISEASE DATA PREPROCESSING")
print("------------------------------------------------")

# Step 2: Display Dataset Information
print("\nDataset Shape:")
print(df.shape)

print("\nFirst Five Rows:")
print(df.head())

print("\nDataset Information:")
df.info()

# Step 3: Check for Missing Values
print("\nMissing Values:")
print(df.isnull().sum())

# Step 4: Check and Remove Duplicate Records
print("\nDuplicate Records:")
print(df.duplicated().sum())

df = df.drop_duplicates()

print("\nDataset Shape After Removing Duplicates:")
print(df.shape)

# Step 5: Separate Features and Target
X = df.drop("target", axis=1)
y = df["target"]

print("\nFeature Columns:")
print(X.columns.tolist())

print("\nTarget Distribution:")
print(y.value_counts())

# Step 6: Split Dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Step 7: Handle Missing Values
imputer = SimpleImputer(strategy="median")

X_train = imputer.fit_transform(X_train)
X_test = imputer.transform(X_test)

# Step 8: Feature Scaling
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Step 9: Display Final Results
print("\nTraining Samples:", len(X_train))
print("Testing Samples:", len(X_test))

print("\nTraining Features Shape:", X_train.shape)
print("Testing Features Shape:", X_test.shape)


print("\nData Preprocessing completed successfully!!!!!!")



