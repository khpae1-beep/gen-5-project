from pathlib import Path

import joblib
from django.conf import settings
from django.core.management.base import BaseCommand, CommandError
from django.db import transaction
from django.utils.text import slugify

from experiments import training
from experiments.tracking import log_run
from experiments.versioning import write_version_note

DEFAULT_EXPERIMENT = 'heart-disease-rf'


def project_relative(path):
    """Store paths inside the project relative to it, so they survive moving the checkout."""
    path = Path(path).resolve()
    try:
        return str(path.relative_to(settings.BASE_DIR.resolve()))
    except ValueError:
        return str(path)


class Command(BaseCommand):
    help = (
        'Train the heart disease Random Forest pipeline (as in ml/train_model.py), save the model to '
        'models/runs/<experiment>-v<N>.joblib and log it as the next run of the experiment with a version note.'
    )

    def add_arguments(self, parser):
        parser.add_argument('--experiment', default=DEFAULT_EXPERIMENT, help=f'Default {DEFAULT_EXPERIMENT!r}.')
        parser.add_argument(
            '--data',
            help='CSV with a "target" column. Default: ml/data/heart.csv, data/heart.csv or heart.csv, '
                 'whichever exists first.',
        )
        parser.add_argument('--n-estimators', type=int, default=100, help='Default 100.')
        parser.add_argument('--max-depth', type=int, default=None, help='Default None (unlimited).')
        parser.add_argument('--min-samples-split', type=int, default=2, help='Default 2.')
        parser.add_argument('--test-size', type=float, default=0.2, help='Default 0.2.')
        parser.add_argument('--random-state', type=int, default=42, help='Used for the split and the forest. Default 42.')
        parser.add_argument('--no-llm', action='store_true', help='Write the version note without the LLM.')

    def handle(self, *args, **options):
        self.validate(options)
        data_path = Path(options['data']) if options['data'] else training.find_data_path()
        if data_path is None or not data_path.exists():
            raise CommandError(
                f'Data file {options["data"]} not found.' if options['data']
                else 'heart.csv not found in ml/data/, data/ or the project root. Pass --data.'
            )

        try:
            features, labels = training.load_data(data_path)
        except ValueError as exc:
            raise CommandError(str(exc)) from exc

        params = {
            'n_estimators': options['n_estimators'],
            'max_depth': options['max_depth'],
            'min_samples_split': options['min_samples_split'],
            'test_size': options['test_size'],
            'random_state': options['random_state'],
        }
        self.stdout.write(f'Training on {project_relative(data_path)} ({len(features)} rows after dropping duplicates).')
        pipeline, metrics, n_train, n_test, duration = training.train_and_evaluate(features, labels, **params)

        experiment = options['experiment']
        with transaction.atomic():
            run = log_run(experiment, params, metrics, data_path=data_path, duration_seconds=round(duration, 3))
            checkpoint = training.RUNS_DIR / f'{slugify(experiment)}-v{run.version}.joblib'
            checkpoint.parent.mkdir(parents=True, exist_ok=True)
            joblib.dump(pipeline, checkpoint)
            run.checkpoint_path = project_relative(checkpoint)
            run.save(update_fields=['checkpoint_path'])

        note, _ = write_version_note(run, use_llm=not options['no_llm'])

        self.stdout.write(f'{n_train} training rows, {n_test} test rows, {duration:.1f} s.')
        for split in ('train', 'test'):
            self.stdout.write(f'  {split}: ' + ', '.join(
                f'{name} {metrics[f"{split}_{name}"]:.3f}' for name in ('accuracy', 'precision', 'recall', 'f1')
            ))
        self.stdout.write(f'Saved {run.checkpoint_path}')
        self.stdout.write(self.style.SUCCESS(f'Logged {run}: {note}'))

    def validate(self, options):
        if options['n_estimators'] < 1:
            raise CommandError('--n-estimators must be at least 1.')
        if options['max_depth'] is not None and options['max_depth'] < 1:
            raise CommandError('--max-depth must be at least 1.')
        if options['min_samples_split'] < 2:
            raise CommandError('--min-samples-split must be at least 2.')
        if not 0 < options['test_size'] < 1:
            raise CommandError('--test-size must be between 0 and 1.')
