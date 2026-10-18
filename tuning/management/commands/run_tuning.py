import json
from pathlib import Path

from django.core.management.base import BaseCommand, CommandError

from experiments import training
from experiments.management.commands.train_experiment import project_relative
from tuning.engine import experiment_name_for, run_job
from tuning.models import Trial, TuningJob
from tuning.scoring import make_split
from tuning.search_space import HEART_DISEASE_RF_SPACE

TEST_SIZE = 0.2


class Command(BaseCommand):
    help = (
        'Tune the heart disease RandomForest: propose params with an LLM or at random, score each trial by '
        'k-fold CV accuracy on the training split, and log every trial to the experiment tracker under '
        'heart-disease-rf-tuning-<strategy>. Test accuracy is shown for reference only.'
    )

    def add_arguments(self, parser):
        parser.add_argument('--strategy', choices=['llm', 'random'], default='llm', help='Default llm.')
        parser.add_argument('--trials', type=int, default=15, help='Number of trials (default 15).')
        parser.add_argument('--cv', type=int, default=5, help='Cross-validation folds (default 5).')
        parser.add_argument('--seed', type=int, default=42, help='Seed for the split, folds, forest and random proposals (default 42).')
        parser.add_argument(
            '--data',
            help='CSV with a "target" column. Default: ml/data/heart.csv, data/heart.csv or heart.csv, '
                 'whichever exists first.',
        )

    def handle(self, *args, **options):
        if options['trials'] < 1:
            raise CommandError('--trials must be at least 1.')
        if options['cv'] < 2:
            raise CommandError('--cv must be at least 2.')
        data_path = Path(options['data']) if options['data'] else training.find_data_path()
        if data_path is None or not data_path.exists():
            raise CommandError(
                f'Data file {options["data"]} not found.' if options['data']
                else 'heart.csv not found in ml/data/, data/ or the project root. Pass --data.'
            )
        try:
            features, labels = training.load_data(data_path)
            split = make_split(features, labels, TEST_SIZE, options['seed'])
        except ValueError as exc:
            raise CommandError(str(exc)) from exc

        job = TuningJob.objects.create(
            strategy=options['strategy'],
            n_trials=options['trials'],
            cv_folds=options['cv'],
            search_space=HEART_DISEASE_RF_SPACE,
            seed=options['seed'],
            experiment_name=experiment_name_for(options['strategy']),
        )
        self.stdout.write(
            f'Tuning job {job.pk}: {job.get_strategy_display()}, {job.n_trials} trials, {job.cv_folds}-fold CV, '
            f'seed {job.seed}, on {project_relative(data_path)} ({len(split.X_train)} training rows, '
            f'{len(split.X_test)} held-out test rows). Logging to experiment {job.experiment_name}.'
        )

        try:
            run_job(job, split, data_path, test_size=TEST_SIZE, on_trial=self.print_trial)
        except Exception as exc:
            raise CommandError(f'Tuning job {job.pk} failed: {exc}') from exc

        best = job.best_trial
        fallbacks = job.trials.filter(proposed_by=Trial.PROPOSED_BY_FALLBACK).count()
        self.stdout.write('')
        self.stdout.write(self.style.SUCCESS(
            f'Best: trial {best.number} of {job.n_trials} (found at trial {best.number}), CV accuracy '
            f'{best.cv_score:.4f} ± {best.cv_std:.4f}, test accuracy {best.test_accuracy:.4f} (reference only).'
        ))
        self.stdout.write(f'  params: {json.dumps(best.params)}')
        self.stdout.write(
            f'  logged as {best.run}. Promote it with: python manage.py promote_run {job.experiment_name} '
            f'{best.run.version}'
        )
        if job.strategy == TuningJob.STRATEGY_LLM:
            self.stdout.write(f'  {job.n_trials - fallbacks} trials proposed by the LLM, {fallbacks} random fallbacks.')

    def print_trial(self, trial):
        job = trial.job
        best = ' (best so far)' if job.best_trial_id == trial.pk else ''
        self.stdout.write(
            f'Trial {trial.number}/{job.n_trials} [{trial.proposed_by}] {json.dumps(trial.params)}\n'
            f'  CV accuracy {trial.cv_score:.4f} ± {trial.cv_std:.4f}{best}; '
            f'test accuracy {trial.test_accuracy:.4f} (reference only)\n'
            f'  {trial.reasoning}'
        )
