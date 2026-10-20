import hashlib
import io
import json
import tempfile
from datetime import datetime, timedelta, timezone as dt_timezone
from pathlib import Path
from unittest import mock

from django.contrib.auth import get_user_model
from django.core.management import call_command
from django.test import TestCase
from django.urls import reverse

from experiments.models import Experiment, Run
from monitoring.messages import handle_prediction_message
from monitoring.models import MetricSnapshot, PredictionLog


class LifecycleIntegrationTests(TestCase):
    def setUp(self):
        temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(temp_dir.cleanup)
        self.temp_path = Path(temp_dir.name)
        self.data_path = self.temp_path / 'heart.csv'
        self.data_path.write_text('feature,target\n1,0\n2,1\n3,0\n4,1\n', encoding='utf-8')

        runs_dir = mock.patch('experiments.training.RUNS_DIR', self.temp_path / 'runs')
        runs_dir.start()
        self.addCleanup(runs_dir.stop)
        git_commit = mock.patch('experiments.tracking.current_git_commit', return_value='')
        git_commit.start()
        self.addCleanup(git_commit.stop)

    def test_training_registers_runs_and_comparison_page_reads_them(self):
        metrics_v1 = {
            'train_accuracy': 0.9,
            'train_precision': 0.8,
            'train_recall': 0.75,
            'train_f1': 0.77,
            'test_accuracy': 0.75,
            'test_precision': 0.7,
            'test_recall': 0.65,
            'test_f1': 0.67,
        }
        metrics_v2 = {
            **metrics_v1,
            'train_accuracy': 0.95,
            'test_accuracy': 0.8,
            'test_precision': 0.76,
            'test_recall': 0.7,
            'test_f1': 0.72,
        }
        mocked_training = mock.patch(
            'experiments.training.train_and_evaluate',
            side_effect=[
                ({'pipeline': 1}, metrics_v1, 3, 1, 1.25),
                ({'pipeline': 2}, metrics_v2, 3, 1, 1.5),
            ],
        )
        with mocked_training as train_and_evaluate:
            for args in ([], ['--max-depth', '4']):
                call_command(
                    'train_experiment', '--experiment', 'lifecycle-rf', '--data', str(self.data_path),
                    '--n-estimators', '10' if not args else '20', *args, '--no-llm', stdout=io.StringIO(),
                )

        train_and_evaluate.assert_has_calls([
            mock.call(mock.ANY, mock.ANY, n_estimators=10, max_depth=None, min_samples_split=2,
                      test_size=0.2, random_state=42),
            mock.call(mock.ANY, mock.ANY, n_estimators=20, max_depth=4, min_samples_split=2,
                      test_size=0.2, random_state=42),
        ])
        experiment = Experiment.objects.get(name='lifecycle-rf')
        first, second = Run.objects.filter(experiment=experiment).order_by('version')
        self.assertEqual((first.version, first.parent_id), (1, None))
        self.assertEqual((second.version, second.parent), (2, first))
        self.assertEqual(first.params['n_estimators'], 10)
        self.assertEqual(second.params, {
            'n_estimators': 20,
            'max_depth': 4,
            'min_samples_split': 2,
            'test_size': 0.2,
            'random_state': 42,
        })
        self.assertEqual(first.metrics, metrics_v1)
        self.assertEqual(second.metrics, metrics_v2)
        self.assertEqual(second.changes['metrics']['test_accuracy'], {'old': 0.75, 'new': 0.8})
        self.assertEqual(second.changes['params']['max_depth'], {'old': None, 'new': 4})
        self.assertEqual(first.data_hash, hashlib.sha256(self.data_path.read_bytes()).hexdigest())
        self.assertTrue(Path(first.checkpoint_path).is_file())
        self.assertTrue(Path(second.checkpoint_path).is_file())

        response = self.client.get(
            reverse('experiments:compare', args=[experiment.name]), {'a': first.version, 'b': second.version},
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.context['run_a'], first)
        self.assertEqual(response.context['run_b'], second)
        accuracy = next(row for row in response.context['comparison']['metrics'] if row['key'] == 'test_accuracy')
        self.assertAlmostEqual(accuracy['delta'], 0.05)
        self.assertEqual(accuracy['direction'], 'up')

    def test_prediction_messages_persist_and_compute_dashboard_metrics(self):
        experiment = Experiment.objects.create(name='lifecycle-monitoring')
        Run.objects.create(
            experiment=experiment,
            version=1,
            params={'n_estimators': 10},
            metrics={'test_accuracy': 0.8},
        )
        start = datetime(2026, 10, 19, tzinfo=dt_timezone.utc)
        observations = [(1, 1, 10), (0, 1, 20), (1, 1, 30), (0, 0, 40)]
        for index, (prediction, actual, latency) in enumerate(observations):
            handle_prediction_message(json.dumps({
                'model_name': experiment.name,
                'model_version': 'v1',
                'prediction': prediction,
                'actual': actual,
                'latency_ms': latency,
                'timestamp': (start + timedelta(seconds=index)).isoformat(),
            }))

        self.assertEqual(PredictionLog.objects.count(), 4)
        call_command('compute_metrics', '--window-size', '2', '--no-llm', stdout=io.StringIO())

        snapshots = list(MetricSnapshot.objects.filter(model_name=experiment.name).order_by('window_start'))
        self.assertEqual(len(snapshots), 2)
        self.assertEqual([(row.count, row.accuracy, row.avg_latency_ms) for row in snapshots], [
            (2, 0.5, 15.0),
            (2, 1.0, 35.0),
        ])
        self.assertEqual(
            [(row.accuracy_anomaly, row.latency_anomaly, row.is_anomaly) for row in snapshots],
            [(True, False, True), (False, True, True)],
        )

        response = self.client.get(reverse('monitoring:dashboard'))
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.context['model_name'], experiment.name)
        self.assertEqual(response.context['total_predictions'], 4)
        self.assertEqual(response.context['chart_data']['accuracy'], [0.5, 1.0])
        self.assertEqual(response.context['anomaly_count'], 2)

    def test_deployment_generation_result_is_saved_in_session_and_rendered(self):
        staff = get_user_model().objects.create_user('lifecycle-staff', is_staff=True)
        self.client.force_login(staff)
        result = {
            'provider': 'aws',
            'description': 'A test bucket',
            'finished_at': '2026-10-19T00:00:00+00:00',
            'attempts': 1,
            'status': 'success',
            'message': 'The configuration passed terraform validate.',
            'error': '',
            'code': 'resource "aws_s3_bucket" "test" {}',
            'path': 'generated/test/main.tf',
            'seconds': 0.1,
        }
        with mock.patch('deployment.views.run_generation', return_value=result) as generate, \
                mock.patch('deployment.views.terraform_installed', return_value=True):
            response = self.client.post(reverse('deployment:generate'), {
                'provider': 'aws',
                'description': 'A test bucket',
            })

        self.assertRedirects(response, reverse('deployment:index') + '#result', fetch_redirect_response=False)
        generate.assert_called_once_with('aws', 'A test bucket')
        self.assertEqual(self.client.session['deployment_result'], result)

        page = self.client.get(reverse('deployment:index'))
        self.assertEqual(page.status_code, 200)
        self.assertContains(page, 'The configuration passed terraform validate.')
        self.assertContains(page, 'resource &quot;aws_s3_bucket&quot; &quot;test&quot; {}')