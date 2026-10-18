"""Scoring a trial: k-fold cross-validation accuracy on the training split only.

The held-out test split is the same one train_experiment and publish_predictions use. Its
accuracy is recorded for reference, but never used to choose params.
"""
import time
from dataclasses import dataclass

from experiments.training import build_pipeline, classification_metrics, split_data


@dataclass
class DataSplit:
    X_train: object
    X_test: object
    y_train: object
    y_test: object


@dataclass
class TrialScore:
    cv_score: float
    cv_std: float
    test_accuracy: float
    metrics: dict
    pipeline: object
    duration_seconds: float


def make_split(features, labels, test_size=0.2, seed=42):
    """train_experiment's stratified split, made once per tuning job."""
    return DataSplit(*split_data(features, labels, test_size, seed))


def score_params(split, params, *, cv_folds=5, seed=42):
    """Score params by stratified k-fold CV accuracy on split's training rows.

    Then fit on all training rows and record train and test metrics. The returned metrics hold
    cv_accuracy and cv_std plus train_* and test_* accuracy, precision, recall and F1.
    """
    from sklearn.model_selection import StratifiedKFold, cross_val_score

    start = time.perf_counter()
    pipeline = build_pipeline(random_state=seed, **params)
    folds = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=seed)
    scores = cross_val_score(pipeline, split.X_train, split.y_train, cv=folds, scoring='accuracy')

    pipeline.fit(split.X_train, split.y_train)
    metrics = {
        'cv_accuracy': float(scores.mean()),
        'cv_std': float(scores.std()),
        **classification_metrics('train', split.y_train, pipeline.predict(split.X_train)),
        **classification_metrics('test', split.y_test, pipeline.predict(split.X_test)),
    }
    return TrialScore(
        cv_score=metrics['cv_accuracy'],
        cv_std=metrics['cv_std'],
        test_accuracy=metrics['test_accuracy'],
        metrics=metrics,
        pipeline=pipeline,
        duration_seconds=time.perf_counter() - start,
    )
