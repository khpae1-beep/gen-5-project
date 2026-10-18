"""Running a tuning job: propose, score by CV, log to the experiment tracker, repeat."""
import random

import joblib
from django.db import transaction
from django.utils.text import slugify

from experiments import training
from experiments.management.commands.train_experiment import project_relative
from experiments.tracking import log_run

from .llm_proposer import Proposal, propose_next
from .models import Trial, TuningJob
from .scoring import score_params
from .search_space import params_key, sample_unique_params

RANDOM_REASON = 'Random search: sampled uniformly from the search space.'


def experiment_name_for(strategy):
    return f'heart-disease-rf-tuning-{strategy}'


def next_proposal(job, history, rng, llm=None):
    if job.strategy == TuningJob.STRATEGY_RANDOM:
        seen = {params_key(t['params']) for t in history}
        return Proposal(sample_unique_params(job.search_space, rng, seen), RANDOM_REASON, Trial.PROPOSED_BY_RANDOM)
    return propose_next(job.search_space, history, rng, llm=llm, cv_folds=job.cv_folds)


def run_job(job, split, data_path=None, *, test_size=0.2, llm=None, on_trial=None):
    """Run job's trials on split (a scoring.DataSplit), logging each one as a run of job.experiment_name.

    The best trial is the one with the highest mean CV accuracy (the earliest wins ties); test
    accuracy is only recorded. on_trial(trial) is called after each trial. If anything fails,
    the job is marked failed with the error and the exception is re-raised.
    """
    rng = random.Random(job.seed)
    history = []
    try:
        for number in range(1, job.n_trials + 1):
            proposal = next_proposal(job, history, rng, llm)
            score = score_params(split, proposal.params, cv_folds=job.cv_folds, seed=job.seed)
            trial = log_trial(job, number, proposal, score, data_path, test_size)

            if job.best_cv_score is None or trial.cv_score > job.best_cv_score:
                job.best_trial, job.best_cv_score = trial, trial.cv_score
                job.save(update_fields=['best_trial', 'best_cv_score'])
            history.append({
                'number': number, 'params': proposal.params, 'cv_score': score.cv_score, 'cv_std': score.cv_std,
            })
            if on_trial:
                on_trial(trial)
    except Exception as exc:
        job.status, job.error = TuningJob.STATUS_FAILED, f'{type(exc).__name__}: {exc}'
        job.save(update_fields=['status', 'error'])
        raise
    job.status = TuningJob.STATUS_COMPLETED
    job.save(update_fields=['status'])
    return job


def log_trial(job, number, proposal, score, data_path, test_size):
    """Save the trial, log it as the next run of the job's experiment and save its checkpoint."""
    run_params = {
        **proposal.params,
        # So publish_predictions reproduces this run's held-out rows after it is promoted.
        'test_size': test_size,
        'random_state': job.seed,
        'cv_folds': job.cv_folds,
    }
    with transaction.atomic():
        run = log_run(
            job.experiment_name, run_params, score.metrics, data_path=data_path,
            duration_seconds=round(score.duration_seconds, 3),
        )
        checkpoint = training.RUNS_DIR / f'{slugify(job.experiment_name)}-v{run.version}.joblib'
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(score.pipeline, checkpoint)
        run.checkpoint_path = project_relative(checkpoint)
        run.version_note = (
            f'Tuning job {job.pk}, trial {number} ({proposal.proposed_by}): CV accuracy '
            f'{score.cv_score:.3f} ± {score.cv_std:.3f}. {proposal.reasoning}'
        )
        run.save(update_fields=['checkpoint_path', 'version_note'])
        return Trial.objects.create(
            job=job,
            number=number,
            params=proposal.params,
            cv_score=score.cv_score,
            cv_std=score.cv_std,
            test_accuracy=score.test_accuracy,
            reasoning=proposal.reasoning,
            proposed_by=proposal.proposed_by,
            run=run,
        )
