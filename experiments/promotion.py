"""Promoting ("checking out") a run as the model publish_predictions serves."""
import json
import os
import shutil
import tempfile

from django.conf import settings
from django.utils import timezone

from .tracking import resolve_path

# The model publish_predictions serves, and which run it came from.
PROMOTED_MODEL_PATH = settings.BASE_DIR / 'models' / 'heart_disease_model.joblib'
CURRENT_MODEL_INFO_PATH = settings.BASE_DIR / 'models' / 'current_model.json'


class PromotionError(Exception):
    """The run cannot be promoted, e.g. its checkpoint is missing."""


def replace_file(path, write):
    """Write to a temporary file next to path, then swap it in, so readers never see half a file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f'.{path.name}.')
    os.close(fd)
    try:
        write(tmp)
        os.chmod(tmp, 0o644)  # mkstemp creates owner-only files.
        os.replace(tmp, path)
    except BaseException:
        os.unlink(tmp)
        raise


def write_json(path, data):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
        f.write('\n')


def checkpoint_problem(run):
    """Why run's checkpoint cannot be promoted, or None if it can."""
    if not run.checkpoint_path:
        return f'{run} has no checkpoint_path.'
    if not resolve_path(run.checkpoint_path).exists():
        return f'Checkpoint {run.checkpoint_path} for {run} does not exist.'
    return None


def promote(run):
    """Copy run's checkpoint to PROMOTED_MODEL_PATH and record it in CURRENT_MODEL_INFO_PATH.

    Raises PromotionError if the checkpoint is missing. Returns the info written to the JSON file.
    """
    problem = checkpoint_problem(run)
    if problem:
        raise PromotionError(problem)

    info = {
        'experiment': run.experiment.name,
        'version': run.version,
        'run_id': run.pk,
        'checkpoint_path': run.checkpoint_path,
        'version_note': run.version_note,
        'metrics': run.metrics,
        'promoted_at': timezone.now().isoformat(),
    }
    checkpoint = resolve_path(run.checkpoint_path)
    replace_file(PROMOTED_MODEL_PATH, lambda tmp: shutil.copyfile(checkpoint, tmp))
    replace_file(CURRENT_MODEL_INFO_PATH, lambda tmp: write_json(tmp, info))
    return info


def current_promotion():
    """(experiment name, version) of the promoted run, or None if nothing usable is promoted."""
    try:
        with open(CURRENT_MODEL_INFO_PATH, encoding='utf-8') as f:
            info = json.load(f)
    except (OSError, ValueError):
        return None
    if not isinstance(info, dict) or not isinstance(info.get('version'), int):
        return None
    return info.get('experiment'), info['version']
