import json
import tempfile
from datetime import datetime, timedelta, timezone as dt_timezone
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from django.core.management import CommandError, call_command
from django.test import TestCase, override_settings
from django.urls import reverse

from monitoring.management.commands.consume_predictions import Command as ConsumeCommand
from monitoring.management.commands import publish_predictions
from monitoring.management.commands.publish_predictions import build_message, find_data_path
from monitoring.messages import InvalidMessage, handle_prediction_message
from monitoring.metrics import compute_snapshots, flag_anomalies
from monitoring.models import MetricSnapshot, PredictionLog
from monitoring.summary import summarize_model

T0 = datetime(2026, 10, 16, 12, 0, tzinfo=dt_timezone.utc)


def make_body(**overrides):
    data = {
        'model_name': 'heart_disease',
        'model_version': 'v1',
        'prediction': 1,
        'actual': 0,
        'latency_ms': 12.5,
        'timestamp': '2026-10-16T12:00:00+00:00',
    }
    data.update(overrides)
    return json.dumps(data).encode()


class HandlePredictionMessageTests(TestCase):
    def test_saves_prediction_log(self):
        log = handle_prediction_message(make_body())

        self.assertEqual(PredictionLog.objects.count(), 1)
        log.refresh_from_db()
        self.assertEqual(log.model_name, 'heart_disease')
        self.assertEqual(log.model_version, 'v1')
        self.assertEqual(log.prediction, 1)
        self.assertEqual(log.actual, 0)
        self.assertEqual(log.latency_ms, 12.5)
        self.assertEqual(log.timestamp, datetime(2026, 10, 16, 12, 0, tzinfo=dt_timezone.utc))
        self.assertFalse(log.is_correct)

    def test_accepts_str_body(self):
        handle_prediction_message(make_body().decode())
        self.assertEqual(PredictionLog.objects.count(), 1)

    def test_actual_is_optional(self):
        body = json.loads(make_body())
        del body['actual']
        log = handle_prediction_message(json.dumps(body))
        self.assertIsNone(log.actual)
        self.assertIsNone(log.is_correct)

    def test_missing_timestamp_defaults_to_now(self):
        body = json.loads(make_body())
        del body['timestamp']
        with mock.patch('monitoring.messages.timezone.now') as now:
            now.return_value = datetime(2026, 1, 1, tzinfo=dt_timezone.utc)
            log = handle_prediction_message(json.dumps(body))
        self.assertEqual(log.timestamp, datetime(2026, 1, 1, tzinfo=dt_timezone.utc))

    def test_naive_timestamp_is_treated_as_utc(self):
        log = handle_prediction_message(make_body(timestamp='2026-10-16T12:00:00'))
        self.assertEqual(log.timestamp, datetime(2026, 10, 16, 12, 0, tzinfo=dt_timezone.utc))

    def test_accepts_message_from_publisher(self):
        log = handle_prediction_message(json.dumps(build_message('heart_disease', 'simulated', 1, 1, 20.1234)))
        self.assertEqual(log.model_version, 'simulated')
        self.assertTrue(log.is_correct)

    def test_rejects_invalid_messages(self):
        bad_bodies = {
            'not json': b'not json',
            'not an object': b'[1, 2]',
            'missing field': json.dumps({'model_name': 'm', 'prediction': 1}),
            'non-numeric prediction': make_body(prediction='yes'),
            'non-numeric latency': make_body(latency_ms=None),
            'bad timestamp': make_body(timestamp='yesterday'),
            'impossible timestamp': make_body(timestamp='2026-13-40T00:00:00'),
        }
        for case, body in bad_bodies.items():
            with self.subTest(case), self.assertRaises(InvalidMessage):
                handle_prediction_message(body)
        self.assertEqual(PredictionLog.objects.count(), 0)


class ConsumeOnMessageTests(TestCase):
    def setUp(self):
        self.command = ConsumeCommand(stdout=mock.Mock(), stderr=mock.Mock())
        self.channel = mock.Mock()
        self.method = mock.Mock(delivery_tag=7)

    def test_acks_valid_message(self):
        self.command.on_message(self.channel, self.method, None, make_body())

        self.channel.basic_ack.assert_called_once_with(delivery_tag=7)
        self.channel.basic_nack.assert_not_called()
        self.assertEqual(PredictionLog.objects.count(), 1)

    def test_rejects_invalid_message_without_requeue(self):
        self.command.on_message(self.channel, self.method, None, b'not json')

        self.channel.basic_nack.assert_called_once_with(delivery_tag=7, requeue=False)
        self.channel.basic_ack.assert_not_called()
        self.assertEqual(PredictionLog.objects.count(), 0)


class FindDataPathTests(TestCase):
    def test_uses_first_existing_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            paths = [Path(tmp) / name for name in ('first.csv', 'second.csv', 'third.csv')]
            with mock.patch.object(publish_predictions, 'DATA_PATHS', paths):
                self.assertIsNone(find_data_path())
                paths[2].touch()
                self.assertEqual(find_data_path(), paths[2])
                paths[1].touch()
                self.assertEqual(find_data_path(), paths[1])
                paths[0].touch()
                self.assertEqual(find_data_path(), paths[0])


def log_predictions(results, model_name='heart_disease', latency_ms=10.0, timestamp=T0):
    """Create one PredictionLog per (prediction, actual) pair, all sharing a timestamp by default."""
    return PredictionLog.objects.bulk_create([
        PredictionLog(
            model_name=model_name,
            model_version='v1',
            prediction=prediction,
            actual=actual,
            latency_ms=latency_ms,
            timestamp=timestamp,
        )
        for prediction, actual in results
    ])


def window_snapshots(*windows):
    """Unsaved snapshots from (accuracy, avg_latency_ms) pairs."""
    return [
        MetricSnapshot(
            model_name='heart_disease', window_start=T0, window_end=T0,
            accuracy=accuracy, avg_latency_ms=latency, count=10,
        )
        for accuracy, latency in windows
    ]


class WindowingTests(TestCase):
    def test_splits_into_full_windows_ordered_by_id_when_timestamps_tie(self):
        # 25 predictions sharing one timestamp: the first 10 are right, the next 10 wrong.
        log_predictions([(1, 1)] * 10 + [(0, 1)] * 10 + [(1, 1)] * 5)

        snapshots, leftover = compute_snapshots('heart_disease', window_size=10)

        self.assertEqual([s.accuracy for s in snapshots], [1.0, 0.0])
        self.assertEqual([s.count for s in snapshots], [10, 10])
        self.assertEqual(leftover, 5)
        self.assertEqual(MetricSnapshot.objects.count(), 2)

    def test_orders_by_timestamp_before_id(self):
        log_predictions([(1, 1)] * 2, timestamp=T0 + timedelta(minutes=1))
        log_predictions([(0, 1)] * 2, timestamp=T0)  # Created later but happened earlier.

        snapshots, _ = compute_snapshots('heart_disease', window_size=2)

        self.assertEqual([s.accuracy for s in snapshots], [0.0, 1.0])
        self.assertEqual(snapshots[0].window_start, T0)

    def test_accuracy_ignores_rows_without_actual(self):
        log_predictions([(1, 1)] * 4 + [(0, 1)] + [(1, None)] * 5 + [(1, None)] * 10)

        snapshots, _ = compute_snapshots('heart_disease', window_size=10)

        self.assertAlmostEqual(snapshots[0].accuracy, 0.8)
        self.assertIsNone(snapshots[1].accuracy)
        self.assertFalse(snapshots[1].is_anomaly)

    def test_average_latency(self):
        PredictionLog.objects.bulk_create([
            PredictionLog(model_name='m', model_version='v1', prediction=1, actual=1, latency_ms=ms, timestamp=T0)
            for ms in (10, 20, 30, 40)
        ])
        snapshots, _ = compute_snapshots('m', window_size=2)
        self.assertEqual([s.avg_latency_ms for s in snapshots], [15, 35])

    def test_replaces_only_that_models_old_snapshots(self):
        log_predictions([(1, 1)] * 10)
        log_predictions([(1, 1)] * 10, model_name='other')
        compute_snapshots('heart_disease', window_size=5)
        compute_snapshots('other', window_size=5)

        compute_snapshots('heart_disease', window_size=10)

        self.assertEqual(MetricSnapshot.objects.filter(model_name='heart_disease').count(), 1)
        self.assertEqual(MetricSnapshot.objects.filter(model_name='other').count(), 2)

    def test_command_computes_each_model_without_llm(self):
        log_predictions([(1, 1)] * 10)
        log_predictions([(1, 1)] * 6, model_name='other')

        with mock.patch('monitoring.summary.get_llm') as get_llm:
            call_command('compute_metrics', '--window-size', '3', '--no-llm', stdout=mock.Mock())

        get_llm.assert_not_called()
        self.assertEqual(MetricSnapshot.objects.filter(model_name='heart_disease').count(), 3)
        self.assertEqual(MetricSnapshot.objects.filter(model_name='other').count(), 2)

    def test_command_rejects_window_size_below_one(self):
        with self.assertRaises(CommandError):
            call_command('compute_metrics', '--window-size', '0', '--no-llm')


class AnomalyTests(TestCase):
    def test_flags_accuracy_more_than_ten_points_below_baseline(self):
        snapshots = flag_anomalies(window_snapshots((0.70, 10), (0.65, 10), (0.60, 10), (None, 10)), baseline=0.75)

        self.assertEqual([s.is_anomaly for s in snapshots], [False, False, True, False])
        self.assertIn('0.15 below the 0.75 baseline', snapshots[2].summary)

    @override_settings(MONITORING_BASELINE_ACCURACY=0.9)
    def test_baseline_comes_from_settings(self):
        snapshots = flag_anomalies(window_snapshots((0.85, 10), (0.75, 10)))
        self.assertEqual([s.is_anomaly for s in snapshots], [False, True])

    def test_default_baseline_is_075(self):
        snapshots = flag_anomalies(window_snapshots((0.66, 10), (0.64, 10)))
        self.assertEqual([s.is_anomaly for s in snapshots], [False, True])

    def test_flags_latency_more_than_double_earlier_median(self):
        # Window 3: earlier median 11 ms, so 22 ms is exactly double and not flagged.
        # Window 4: earlier median 12 ms, so 30 ms is over the 24 ms limit.
        snapshots = flag_anomalies(window_snapshots((1, 10), (1, 12), (1, 22), (1, 30)), baseline=0.75)

        self.assertEqual([s.is_anomaly for s in snapshots], [False, False, False, True])
        self.assertIn('2.5 times the earlier median of 12.0 ms', snapshots[3].summary)

    def test_latency_median_uses_only_earlier_windows(self):
        # The first window has nothing to compare with. Window 3's earlier median is 55 ms,
        # so 30 ms is fine even though it is triple window 1.
        snapshots = flag_anomalies(window_snapshots((1, 10), (1, 100), (1, 30)), baseline=0.75)
        self.assertEqual([s.is_anomaly for s in snapshots], [False, True, False])

    def test_compute_snapshots_saves_flags(self):
        log_predictions([(1, 1)] * 10, latency_ms=10)
        log_predictions([(0, 1)] * 10, latency_ms=10)
        log_predictions([(1, 1)] * 10, latency_ms=50)

        compute_snapshots('heart_disease', window_size=10)

        flags = list(MetricSnapshot.objects.order_by('window_start', 'id').values_list('is_anomaly', flat=True))
        self.assertEqual(flags, [False, True, True])


class FakeLLM:
    def __init__(self, reply=None, error=None):
        self.reply = reply
        self.error = error
        self.prompts = []

    def invoke(self, prompt):
        self.prompts.append(prompt)
        if self.error:
            raise self.error
        return SimpleNamespace(content=self.reply)


class SummaryTests(TestCase):
    def setUp(self):
        log_predictions([(1, 1)] * 10 + [(0, 1)] * 10)
        compute_snapshots('heart_disease', window_size=10)

    def latest(self):
        return MetricSnapshot.objects.order_by('window_start', 'id').last()

    def test_saves_llm_reply_on_latest_snapshot(self):
        llm = FakeLLM(reply='  Accuracy fell sharply in window 2.  ')

        snapshot, used_llm = summarize_model('heart_disease', llm=llm)

        self.assertTrue(used_llm)
        self.assertEqual(self.latest().summary, 'Accuracy fell sharply in window 2.')
        self.assertEqual(snapshot.pk, self.latest().pk)
        self.assertIn('Window 2 (10 predictions): accuracy 0.00', llm.prompts[0])
        self.assertIn('ANOMALY: accuracy 0.00 is 0.75 below the 0.75 baseline', llm.prompts[0])

    def test_falls_back_to_rules_when_llm_fails(self):
        _, used_llm = summarize_model('heart_disease', llm=FakeLLM(error=ConnectionError('refused')))

        self.assertFalse(used_llm)
        summary = self.latest().summary
        self.assertTrue(summary.startswith('The LLM was unavailable (ConnectionError)'))
        self.assertIn('20 predictions', summary)
        self.assertIn('window 2: accuracy 0.00 is 0.75 below the 0.75 baseline', summary)

    def test_falls_back_when_llm_reply_is_empty(self):
        _, used_llm = summarize_model('heart_disease', llm=FakeLLM(reply='   '))
        self.assertFalse(used_llm)
        self.assertIn('LLM was unavailable', self.latest().summary)

    def test_falls_back_when_get_llm_fails(self):
        with mock.patch('monitoring.summary.get_llm', side_effect=ValueError('bad provider')):
            _, used_llm = summarize_model('heart_disease')
        self.assertFalse(used_llm)
        self.assertTrue(self.latest().summary.startswith('The LLM was unavailable (ValueError)'))

    def test_rule_based_summary_without_anomalies(self):
        MetricSnapshot.objects.all().delete()
        log_predictions([(1, 1)] * 10, model_name='steady')
        compute_snapshots('steady', window_size=10)

        summarize_model('steady', llm=FakeLLM(error=RuntimeError()))

        self.assertIn('No anomalies were detected.', MetricSnapshot.objects.get().summary)

    def test_only_latest_snapshot_changes(self):
        first = MetricSnapshot.objects.order_by('window_start', 'id').first()
        summarize_model('heart_disease', llm=FakeLLM(reply='Summary.'))
        first.refresh_from_db()
        self.assertEqual(first.summary, '')

    def test_returns_none_without_snapshots(self):
        self.assertEqual(summarize_model('unknown', llm=FakeLLM(reply='x')), (None, False))

    def test_compute_metrics_summarises_by_default(self):
        llm = FakeLLM(reply='All good.')
        with mock.patch('monitoring.summary.get_llm', return_value=llm):
            call_command('compute_metrics', stdout=mock.Mock())
        self.assertEqual(len(llm.prompts), 1)
        self.assertEqual(self.latest().summary, 'All good.')


@mock.patch('monitoring.summary.get_llm', side_effect=AssertionError('The dashboard must not call the LLM.'))
class DashboardTests(TestCase):
    url = reverse('monitoring:dashboard')

    def test_loads_without_data(self, get_llm):
        response = self.client.get(self.url)

        self.assertEqual(response.status_code, 200)
        self.assertTemplateUsed(response, 'monitoring/dashboard.html')
        self.assertTemplateUsed(response, 'base.html')
        self.assertContains(response, 'No predictions logged yet')

    def test_shows_metrics_charts_summary_and_recent_predictions(self, get_llm):
        log_predictions([(1, 1)] * 10 + [(0, 1)] * 10 + [(1, None)] * 5, latency_ms=20)
        compute_snapshots('heart_disease', window_size=10)
        MetricSnapshot.objects.filter(pk=MetricSnapshot.objects.order_by('id').last().pk).update(summary='Saved summary.')

        with self.assertNumQueries(4):
            response = self.client.get(self.url)

        self.assertEqual(response.status_code, 200)
        context = response.context
        self.assertEqual(context['total_predictions'], 25)
        self.assertAlmostEqual(context['overall_accuracy'], 0.5)
        self.assertAlmostEqual(context['avg_latency_ms'], 20)
        self.assertEqual(context['anomaly_count'], 1)
        self.assertEqual(len(context['recent_predictions']), 20)
        self.assertEqual(context['chart_data']['accuracy'], [1.0, 0.0])
        self.assertEqual(context['chart_data']['anomaly'], [False, True])
        self.assertEqual(context['chart_data']['baseline'], 0.75)
        self.assertContains(response, 'id="accuracy-chart"')
        self.assertContains(response, 'id="latency-chart"')
        self.assertContains(response, 'Saved summary.')
        get_llm.assert_not_called()

    def test_model_can_be_chosen(self, get_llm):
        log_predictions([(1, 1)] * 3)
        log_predictions([(0, 1)] * 2, model_name='zeta')

        response = self.client.get(self.url, {'model': 'zeta'})

        self.assertEqual(response.context['model_name'], 'zeta')
        self.assertEqual(response.context['total_predictions'], 2)
        self.assertContains(response, '<select id="model"')

    def test_unknown_model_falls_back_to_first(self, get_llm):
        log_predictions([(1, 1)] * 3)
        response = self.client.get(self.url, {'model': 'nope'})
        self.assertEqual(response.context['model_name'], 'heart_disease')
