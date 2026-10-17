"""The heart disease training pipeline from Khaliyah's ml/train_model.py, with tunable parameters."""
import time

from django.conf import settings

LABEL_COLUMN = 'target'
RUNS_DIR = settings.BASE_DIR / 'models' / 'runs'
# Checked in order; the first one that exists is used.
DATA_PATHS = [
    settings.BASE_DIR / 'ml' / 'data' / 'heart.csv',
    settings.BASE_DIR / 'data' / 'heart.csv',
    settings.BASE_DIR / 'heart.csv',
]


def find_data_path():
    return next((path for path in DATA_PATHS if path.exists()), None)


def load_data(path):
    """Read the CSV and drop duplicate rows; returns (features, labels)."""
    import pandas as pd

    data = pd.read_csv(path).drop_duplicates()
    if LABEL_COLUMN not in data.columns:
        raise ValueError(f'{path} has no {LABEL_COLUMN!r} column.')
    return data.drop(columns=[LABEL_COLUMN]), data[LABEL_COLUMN]


def build_pipeline(n_estimators=100, max_depth=None, min_samples_split=2, random_state=42):
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    return Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
        ('classifier', RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            random_state=random_state,
        )),
    ])


def classification_metrics(prefix, actual, predicted):
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

    return {
        f'{prefix}_accuracy': float(accuracy_score(actual, predicted)),
        f'{prefix}_precision': float(precision_score(actual, predicted, zero_division=0)),
        f'{prefix}_recall': float(recall_score(actual, predicted, zero_division=0)),
        f'{prefix}_f1': float(f1_score(actual, predicted, zero_division=0)),
    }


def split_data(features, labels, test_size=0.2, random_state=42):
    """The stratified train/test split; returns (X_train, X_test, y_train, y_test).

    publish_predictions uses this too, to send only the rows a run was not trained on.
    """
    from sklearn.model_selection import train_test_split

    return train_test_split(features, labels, test_size=test_size, random_state=random_state, stratify=labels)


def train_and_evaluate(features, labels, *, test_size=0.2, random_state=42, **model_params):
    """Stratified split, fit, and score on both sides.

    Returns (fitted pipeline, metrics, train size, test size, duration in seconds).
    """
    start = time.perf_counter()
    X_train, X_test, y_train, y_test = split_data(features, labels, test_size, random_state)
    pipeline = build_pipeline(random_state=random_state, **model_params)
    pipeline.fit(X_train, y_train)
    metrics = {
        **classification_metrics('train', y_train, pipeline.predict(X_train)),
        **classification_metrics('test', y_test, pipeline.predict(X_test)),
    }
    return pipeline, metrics, len(X_train), len(X_test), time.perf_counter() - start
