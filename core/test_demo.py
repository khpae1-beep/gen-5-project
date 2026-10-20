"""The read-only demo: every page must return 200 to a logged-out visitor, with DEBUG off, when
Ollama, RabbitMQ and Terraform are unavailable and the files under models/ are missing."""
import os
import tempfile
from pathlib import Path
from unittest import mock

import pika
from django.test import TestCase, override_settings
from django.urls import reverse
from django.utils import timezone

from experiments import promotion
from experiments.models import Experiment, Run
from monitoring.models import MetricSnapshot, PredictionLog
from tuning.models import Trial, TuningJob


def no_rabbitmq(*args, **kwargs):
    raise pika.exceptions.AMQPConnectionError('RabbitMQ is not running')


@override_settings(DEBUG=False)
class DemoPagesTests(TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        missing = Path(tmp.name) / 'models'  # Never created: no promoted model or checkpoints.
        unavailable = [
            # Ollama: a port where nothing listens, so any LLM call fails to connect.
            mock.patch.dict(os.environ, {'LLM_PROVIDER': 'ollama', 'OLLAMA_BASE_URL': 'http://127.0.0.1:9'}),
            mock.patch('pika.BlockingConnection', side_effect=no_rabbitmq),
            mock.patch('shutil.which', return_value=None),  # Terraform is not installed.
            mock.patch.object(promotion, 'PROMOTED_MODEL_PATH', missing / 'heart_disease_model.joblib'),
            mock.patch.object(promotion, 'CURRENT_MODEL_INFO_PATH', missing / 'current_model.json'),
        ]
        for patcher in unavailable:
            patcher.start()
            self.addCleanup(patcher.stop)

    def make_data(self):
        """A small version of the demo data: runs, tuning jobs with suggestions, predictions and snapshots."""
        experiment = Experiment.objects.create(name='heart-disease-rf')
        runs = []
        for version, (n, test) in enumerate([(100, 0.754), (300, 0.77)], start=1):
            runs.append(Run.objects.create(
                experiment=experiment, version=version, parent=runs[-1] if runs else None,
                params={'n_estimators': n, 'max_depth': None},
                metrics={'train_accuracy': 1.0, 'test_accuracy': test, 'test_f1': test},
                changes={'params': {'n_estimators': {'old': 100, 'new': n}}} if runs else {},
                checkpoint_path=f'models/runs/heart-disease-rf-v{version}.joblib',  # Not on disk.
                version_note=f'v{version}: note',
            ))
        jobs = []
        for strategy in (TuningJob.STRATEGY_LLM, TuningJob.STRATEGY_RANDOM):
            job = TuningJob.objects.create(
                strategy=strategy, n_trials=2, cv_folds=5, search_space={}, seed=42,
                status=TuningJob.STATUS_COMPLETED, experiment_name='heart-disease-rf', finished_at=timezone.now(),
                suggestions={'created_at': timezone.now().isoformat(), 'rejected': [], 'error': '', 'configs': [
                    {'params': {'n_estimators': 200}, 'reasoning': 'Try more trees.', 'reason_check': 'passed',
                     'llm_reason': 'Try more trees.', 'source': 'llm', 'label': 'LLM', 'tried_trial': None},
                ]},
            )
            for number, run in enumerate(runs, start=1):
                Trial.objects.create(
                    job=job, number=number, params=run.params, cv_score=0.84 + number / 100, cv_std=0.05,
                    test_accuracy=run.metrics['test_accuracy'], reasoning='Reason.', run=run,
                    proposed_by=Trial.PROPOSED_BY_LLM if strategy == 'llm' else Trial.PROPOSED_BY_RANDOM,
                )
            job.best_trial = job.trials.last()
            job.save()
            jobs.append(job)
        now = timezone.now()
        PredictionLog.objects.bulk_create([
            PredictionLog(model_name='heart_disease', model_version='v2', prediction=i % 2, actual=1,
                          latency_ms=5.0, timestamp=now)
            for i in range(20)
        ])
        MetricSnapshot.objects.create(
            model_name='heart_disease', window_start=now, window_end=now, accuracy=0.5, avg_latency_ms=5.0,
            count=10, is_anomaly=True, accuracy_anomaly=True, summary='Window 1 is the latest: accuracy 0.50.',
        )
        return experiment, runs, jobs

    def page_urls(self, experiment=None, runs=(), jobs=()):
        urls = [
            reverse('home'),
            reverse('monitoring:dashboard'),
            reverse('experiments:list'),
            reverse('tuning:list'),
            reverse('tuning:compare'),
            reverse('deployment:index'),
            reverse('admin:login'),
        ]
        if experiment:
            urls += [
                reverse('experiments:detail', args=[experiment.name]),
                reverse('experiments:compare', args=[experiment.name]),
                *[reverse('experiments:run', args=[experiment.name, run.version]) for run in runs],
                *[reverse('tuning:detail', args=[job.pk]) for job in jobs],
            ]
        return urls

    def assert_all_ok(self, urls):
        for url in urls:
            with self.subTest(url=url):
                response = self.client.get(url)
                self.assertEqual(response.status_code, 200)

    def test_every_page_with_demo_data(self):
        experiment, runs, jobs = self.make_data()
        self.assert_all_ok(self.page_urls(experiment, runs, jobs))

    def test_every_page_with_an_empty_database(self):
        self.assert_all_ok(self.page_urls())

    def test_fallback_messages(self):
        experiment, runs, jobs = self.make_data()

        deployments = self.client.get(reverse('deployment:index'))
        self.assertContains(deployments, 'The Terraform CLI is not installed or not on the PATH')

        # The checkpoint is not in models/; the page still shows the run, and only staff could promote.
        run = self.client.get(reverse('experiments:run', args=[experiment.name, 2]))
        self.assertContains(run, 'models/runs/heart-disease-rf-v2.joblib')
        self.assertContains(run, 'Only staff can promote')

        self.assertContains(self.client.get(reverse('tuning:detail', args=[jobs[0].pk])), 'Only staff can ask for suggestions')

    def test_explain_falls_back_when_the_llm_is_unavailable(self):
        experiment, runs, jobs = self.make_data()
        response = self.client.post(reverse('experiments:explain', args=[experiment.name]), {'a': 1, 'b': 2})
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, 'Written from the numbers, because the LLM was unavailable')

    def test_unknown_pages_are_404_not_500(self):
        for url in ('/experiments/nope/', '/tuning/9999/', '/no-such-page/'):
            with self.subTest(url=url):
                self.assertEqual(self.client.get(url).status_code, 404)
