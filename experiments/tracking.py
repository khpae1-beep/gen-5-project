"""Experiment tracking: record a training run as the next version of an experiment.

Example:

    from experiments.tracking import log_run

    run = log_run(
        'heart-disease-rf',
        params={'n_estimators': 300, 'max_depth': 8},
        metrics={'test_accuracy': 0.787},
        checkpoint_path='models/runs/heart-disease-rf-v3.joblib',
        data_path='ml/data/heart.csv',
    )
    print(run.version, run.changes)
"""
import hashlib
import subprocess
from pathlib import Path

from django.conf import settings
from django.db import transaction

from .models import Experiment, Run


def log_run(experiment_name, params, metrics, checkpoint_path='', data_path=None, *, duration_seconds=None):
    """Save a run as the next version of experiment_name and return it.

    - Creates the experiment if it does not exist yet.
    - Numbers runs 1, 2, 3, ... within the experiment and links each run to the previous
      version as its parent.
    - Stores in run.changes which params and metrics differ from the parent, as
      {'params': {name: {'old': ..., 'new': ...}}, 'metrics': {...}}. A name missing on one
      side has None there. Version 1 has no changes.
    - Stores the SHA-256 of data_path (if given) and the current git commit (blank if git
      is not available).

    params and metrics are dicts of JSON-serialisable values (convert numpy numbers with
    float() or int() first). checkpoint_path is stored as given.
    """
    params = dict(params)
    metrics = dict(metrics)
    data_hash = file_sha256(data_path) if data_path else ''
    git_commit = current_git_commit()

    with transaction.atomic():
        experiment, _ = Experiment.objects.get_or_create(name=experiment_name)
        # Lock the experiment so two runs logged at once cannot get the same version.
        Experiment.objects.select_for_update().filter(pk=experiment.pk).first()
        parent = experiment.runs.order_by('-version').first()
        return Run.objects.create(
            experiment=experiment,
            version=parent.version + 1 if parent else 1,
            parent=parent,
            params=params,
            metrics=metrics,
            changes=compute_changes(parent, params, metrics),
            checkpoint_path=str(checkpoint_path),
            data_hash=data_hash,
            git_commit=git_commit,
            duration_seconds=duration_seconds,
        )


def compute_changes(parent, params, metrics):
    """Which params and metrics differ from the parent run; {} when there is no parent."""
    if parent is None:
        return {}
    return {
        'params': diff_dicts(parent.params, params),
        'metrics': diff_dicts(parent.metrics, metrics),
    }


def diff_dicts(old, new):
    """{key: {'old': ..., 'new': ...}} for every key whose value differs, in a stable order."""
    keys = list(old) + [key for key in new if key not in old]
    return {
        key: {'old': old.get(key), 'new': new.get(key)}
        for key in keys
        if old.get(key) != new.get(key) or (key in old) != (key in new)
    }


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def current_git_commit(cwd=None):
    """The current commit hash, or '' if git is missing or this is not a repository."""
    try:
        result = subprocess.run(
            ['git', 'rev-parse', 'HEAD'],
            cwd=cwd or settings.BASE_DIR,
            capture_output=True,
            text=True,
            timeout=5,
            check=True,
        )
    except (OSError, subprocess.SubprocessError):
        return ''
    return result.stdout.strip()


def resolve_path(path):
    """Paths stored relative to the project root are resolved against it."""
    path = Path(path)
    return path if path.is_absolute() else settings.BASE_DIR / path
