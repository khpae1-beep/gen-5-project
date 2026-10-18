import io
import json
import tempfile
from datetime import datetime, timedelta, timezone as dt_timezone
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from django.core.management import CommandError, call_command
from django.test import TestCase, override_settings
from django.urls import reverse

from experiments.tracking import log_run
from experiments.training import load_data, split_data
from monitoring.management.commands.consume_predictions import Command as ConsumeCommand
from monitoring.management.commands import publish_predictions
from monitoring.management.commands.publish_predictions import build_message, find_data_path, promoted_version
from monitoring.messages import InvalidMessage, handle_prediction_message
from monitoring.metrics import anomaly_reasons, compute_snapshots, flag_anomalies
from monitoring.models import MetricSnapshot, PredictionLog
from monitoring.summary import FactCheckError, build_prompt, fact_check, summarize_model, window_numbers

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


class PublisherFilesMixin:
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.tmp = Path(tmp.name)
        self.info = self.tmp / 'current_model.json'
        patcher = mock.patch.object(publish_predictions, 'CURRENT_MODEL_INFO_PATH', self.info)
        patcher.start()
        self.addCleanup(patcher.stop)

    def make_model_and_data(self, name='heart.csv', rows=20, duplicates=0):
        """A trivial model that always predicts 1, and a CSV of rows unique rows (two thirds labelled 1)
        followed by copies of its first duplicates rows."""
        import joblib
        import pandas as pd
        from sklearn.dummy import DummyClassifier

        data = pd.DataFrame({'age': range(50, 50 + rows), 'target': [int(i % 3 != 0) for i in range(rows)]})
        data_path = self.tmp / name
        pd.concat([data, data.head(duplicates)]).to_csv(data_path, index=False)
        model_path = self.tmp / 'model.joblib'
        joblib.dump(DummyClassifier(strategy='most_frequent').fit(data[['age']], data['target']), model_path)
        return model_path, data_path


class PromotedVersionTests(PublisherFilesMixin, TestCase):
    def test_reads_version_from_current_model_json(self):
        self.assertIsNone(promoted_version())
        self.info.write_text(json.dumps({'experiment': 'heart-disease-rf', 'version': 3}))
        self.assertEqual(promoted_version(), 'v3')

    def test_ignores_unusable_files(self):
        for content in ('not json', '[]', '{}', '{"version": "3"}'):
            with self.subTest(content):
                self.info.write_text(content)
                self.assertIsNone(promoted_version())

    def test_real_model_is_published_with_promoted_version(self):
        model_path, data_path = self.make_model_and_data()
        command = publish_predictions.Command(stdout=mock.Mock())

        with mock.patch.object(publish_predictions, 'MODEL_PATH', model_path), \
                mock.patch.object(publish_predictions, 'DATA_PATHS', [data_path]):
            self.assertEqual(command.get_predictor().version, 'v1')
            self.info.write_text(json.dumps({'experiment': 'heart-disease-rf', 'version': 4}))
            predictor = command.get_predictor()

        self.assertEqual(predictor.version, 'v4')
        self.assertEqual(predictor.predict()[0], 1)


class PublishDataOptionTests(PublisherFilesMixin, TestCase):
    def test_uses_given_csv_instead_of_default_locations(self):
        model_path, default_csv = self.make_model_and_data()
        _, chosen_csv = self.make_model_and_data('chosen.csv', rows=30)
        command = publish_predictions.Command(stdout=io.StringIO())

        with mock.patch.object(publish_predictions, 'MODEL_PATH', model_path), \
                mock.patch.object(publish_predictions, 'DATA_PATHS', [default_csv]):
            predictor = command.get_predictor(str(chosen_csv), all_rows=True)

        self.assertEqual(len(predictor.features), 30)
        self.assertIn(f'on rows from {chosen_csv}', command.stdout.getvalue())

    def test_errors_instead_of_simulating(self):
        _, data_path = self.make_model_and_data()
        cases = {
            'missing csv': ([str(self.tmp / 'missing.csv')], self.tmp / 'model.joblib'),
            'missing model': ([str(data_path)], self.tmp / 'no-model.joblib'),
        }
        for case, (args, model_path) in cases.items():
            with self.subTest(case), mock.patch.object(publish_predictions, 'MODEL_PATH', model_path), \
                    mock.patch('monitoring.management.commands.publish_predictions.open_channel') as open_channel:
                with self.assertRaises(CommandError):
                    call_command('publish_predictions', '--data', *args, stdout=io.StringIO())
                open_channel.assert_not_called()

    def test_without_data_still_falls_back_to_simulated(self):
        command = publish_predictions.Command(stdout=io.StringIO())
        with mock.patch.object(publish_predictions, 'MODEL_PATH', self.tmp / 'no-model.joblib'), \
                mock.patch.object(publish_predictions, 'DATA_PATHS', []):
            self.assertIsInstance(command.get_predictor(), publish_predictions.SimulatedPredictor)


class HeldOutRowsTests(PublisherFilesMixin, TestCase):
    def setUp(self):
        super().setUp()
        # 50 unique rows plus 5 duplicates, as in the real heart.csv.
        self.model_path, self.data_path = self.make_model_and_data(rows=50, duplicates=5)
        for name, value in (('MODEL_PATH', self.model_path), ('DATA_PATHS', [self.data_path])):
            patcher = mock.patch.object(publish_predictions, name, value)
            patcher.start()
            self.addCleanup(patcher.stop)

    def get_predictor(self, **kwargs):
        command = publish_predictions.Command(stdout=io.StringIO())
        return command.get_predictor(**kwargs), command.stdout.getvalue()

    def promote(self, test_size, random_state):
        log_run('heart-disease-rf', {'test_size': 0.2, 'random_state': 42}, {})
        log_run('heart-disease-rf', {'test_size': test_size, 'random_state': random_state}, {})
        self.info.write_text(json.dumps({'experiment': 'heart-disease-rf', 'version': 2}))

    def test_samples_the_promoted_runs_held_out_test_rows(self):
        self.promote(test_size=0.3, random_state=7)

        predictor, output = self.get_predictor()

        # The same rows train_experiment held out for that run, and none it trained on.
        features, labels = load_data(self.data_path)
        X_train, X_test, _, y_test = split_data(features, labels, test_size=0.3, random_state=7)
        self.assertEqual(list(predictor.features.index), list(X_test.index))
        self.assertEqual(list(predictor.labels), list(y_test))
        self.assertFalse(set(predictor.features.index) & set(X_train.index))
        self.assertEqual(len(predictor.features), 15)
        self.assertEqual(predictor.version, 'v2')
        self.assertIn(
            'Mode: held-out test rows, sampling 15 rows not used for training '
            '(test_size 0.3, random_state 7 from heart-disease-rf v2)',
            output,
        )

    def test_defaults_to_train_experiment_settings_without_a_promoted_run(self):
        expected = split_data(*load_data(self.data_path), test_size=0.2, random_state=42)[1]
        for info in (None, {'experiment': 'heart-disease-rf', 'version': 9}):
            with self.subTest(info):
                if info:
                    self.info.write_text(json.dumps(info))
                predictor, output = self.get_predictor()
                self.assertEqual(list(predictor.features.index), list(expected.index))
                self.assertIn('(test_size 0.2, random_state 42 from defaults)', output)

    def test_all_rows_uses_every_row(self):
        self.promote(test_size=0.3, random_state=7)

        predictor, output = self.get_predictor(all_rows=True)

        self.assertEqual(len(predictor.features), 55)  # Duplicates included, as before.
        self.assertIn('Mode: all rows (--all-rows), sampling 55 rows', output)

    def test_command_prints_mode(self):
        for args, expected in (([], 'Mode: held-out test rows'), (['--all-rows'], 'Mode: all rows')):
            with self.subTest(args), mock.patch.object(publish_predictions, 'open_channel') as open_channel:
                open_channel.return_value = (mock.Mock(), mock.Mock())
                out = io.StringIO()
                call_command('publish_predictions', '--count', '3', *args, stdout=out)
                self.assertIn(expected, out.getvalue())
                self.assertEqual(open_channel.return_value[1].basic_publish.call_count, 3)

    def test_too_small_to_split_suggests_all_rows(self):
        _, tiny = self.make_model_and_data('tiny.csv', rows=3)
        with self.assertRaisesMessage(CommandError, 'Use --all-rows'):
            self.get_predictor(data=str(tiny))


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

    def test_records_which_rule_flagged_each_window(self):
        snapshots = flag_anomalies(
            window_snapshots((0.9, 10), (0.5, 10), (0.9, 40), (0.5, 40)), baseline=0.75
        )

        self.assertEqual(
            [(s.is_anomaly, s.accuracy_anomaly, s.latency_anomaly) for s in snapshots],
            [(False, False, False), (True, True, False), (True, False, True), (True, True, True)],
        )
        self.assertIn('accuracy 0.50', snapshots[3].summary)
        self.assertIn('average latency 40.0 ms', snapshots[3].summary)

    def test_compute_snapshots_saves_flags(self):
        log_predictions([(1, 1)] * 10, latency_ms=10)
        log_predictions([(0, 1)] * 10, latency_ms=10)
        log_predictions([(1, 1)] * 10, latency_ms=50)

        compute_snapshots('heart_disease', window_size=10)

        flags = list(
            MetricSnapshot.objects.order_by('window_start', 'id')
            .values_list('is_anomaly', 'accuracy_anomaly', 'latency_anomaly')
        )
        self.assertEqual(flags, [(False, False, False), (True, True, False), (True, False, True)])


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

    OPENING = 'Window 2 is the latest: accuracy 0.00, average latency 10.0 ms, flagged as an anomaly (accuracy 0.00 is 0.75 below the 0.75 baseline).'

    def test_saves_opening_sentence_and_llm_reply_on_latest_snapshot(self):
        llm = FakeLLM(reply='  Window 2 is an anomaly: accuracy fell from 1.00 to 0.00.  ')

        snapshot, used_llm = summarize_model('heart_disease', llm=llm)

        self.assertTrue(used_llm)
        self.assertEqual(
            self.latest().summary, f'{self.OPENING} Window 2 is an anomaly: accuracy fell from 1.00 to 0.00.'
        )
        self.assertEqual(snapshot.pk, self.latest().pk)
        self.assertIn('Window 2 (LATEST) (10 predictions): accuracy 0.00', llm.prompts[0])
        self.assertIn('Window 1 (10 predictions)', llm.prompts[0])
        self.assertIn('Window 2 is the latest (current) window', llm.prompts[0])
        self.assertIn('ANOMALY: accuracy 0.00 is 0.75 below the 0.75 baseline', llm.prompts[0])

    def test_prompt_lists_recent_windows_in_order_with_latest_marked(self):
        log_predictions([(1, 1)] * 30)
        compute_snapshots('heart_disease', window_size=10)  # Windows 1 to 5.
        llm = FakeLLM(reply='Fine.')

        summarize_model('heart_disease', llm=llm, recent=3)

        prompt = llm.prompts[0]
        self.assertIn('These are windows 3 to 5, listed in time order', prompt)
        self.assertIn('use window 5 only', prompt)
        positions = [prompt.index(f'Window {n}') for n in ('3 (', '4 (', '5 (LATEST)')]
        self.assertEqual(positions, sorted(positions))
        self.assertNotIn('Window 2 (', prompt)
        self.assertEqual(prompt.count('(LATEST)'), 1)

    def test_falls_back_to_rules_when_llm_fails(self):
        _, used_llm = summarize_model('heart_disease', llm=FakeLLM(error=ConnectionError('refused')))

        self.assertFalse(used_llm)
        summary = self.latest().summary
        self.assertTrue(summary.startswith(f'{self.OPENING} The LLM was unavailable (ConnectionError)'))
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
        self.assertTrue(self.latest().summary.startswith(f'{self.OPENING} The LLM was unavailable (ValueError)'))

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
        llm = FakeLLM(reply='Window 2 is an anomaly.')
        with mock.patch('monitoring.summary.get_llm', return_value=llm) as get_llm:
            call_command('compute_metrics', stdout=mock.Mock())
        get_llm.assert_called_once_with(temperature=0)
        self.assertEqual(len(llm.prompts), 1)
        self.assertEqual(self.latest().summary, f'{self.OPENING} Window 2 is an anomaly.')

    def test_falls_back_when_llm_summary_fails_fact_check(self):
        replies = {
            'calls an unflagged window an anomaly': (
                'Windows 1 and 2 are anomalies.', 'it calls window 1 an anomaly, but it was not flagged',
            ),
            'misses a flagged window': ('The model is doing fine.', 'it does not mention flagged window 2'),
            'wrong accuracy': (
                'Window 2 is an anomaly with accuracy 0.55.', 'it mentions numbers that do not match the data: 0.55',
            ),
        }
        for case, (reply, reason) in replies.items():
            with self.subTest(case):
                _, used_llm = summarize_model('heart_disease', llm=FakeLLM(reply=reply))

                self.assertFalse(used_llm)
                summary = self.latest().summary
                self.assertTrue(summary.startswith(f'{self.OPENING} The LLM summary failed the fact check ({reason}'))
                self.assertIn('so this is a rule-based summary.', summary)
                self.assertIn('window 2: accuracy 0.00 is 0.75 below the 0.75 baseline', summary)
                self.assertNotIn(reply, summary)

    def test_opening_sentence_for_unflagged_latest_window(self):
        log_predictions([(1, 1)] * 10, latency_ms=12.34)
        compute_snapshots('heart_disease', window_size=10)

        summarize_model('heart_disease', llm=FakeLLM(reply='Window 2 was an anomaly; window 3 recovered.'))

        self.assertEqual(
            self.latest().summary,
            'Window 3 is the latest: accuracy 1.00, average latency 12.3 ms, not flagged as an anomaly. '
            'Window 2 was an anomaly; window 3 recovered.',
        )


def summary_windows(*windows, baseline=0.75):
    """(number, snapshot, reasons) for (accuracy, latency) pairs, as summarize_model builds them."""
    snapshots = window_snapshots(*windows)
    return [
        (i + 1, s, anomaly_reasons(s, [e.avg_latency_ms for e in snapshots[:i]], baseline))
        for i, s in enumerate(snapshots)
    ]


class SummaryPromptTests(TestCase):
    def test_lists_flagged_windows_as_the_only_anomalies(self):
        windows = summary_windows((0.80, 10), (0.70, 12), (0.40, 11), (0.79, 30))

        prompt = build_prompt('heart_disease', windows, 0.75)

        self.assertIn(
            'Flagged windows (the ONLY anomalies): window 3 (accuracy 0.40 is 0.35 below the 0.75 baseline); '
            'window 4 (average latency 30.0 ms is 2.7 times the earlier median of 11.0 ms).',
            prompt,
        )
        self.assertIn('Only the flagged windows listed above are anomalies.', prompt)
        self.assertIn('Do not call any other window an anomaly, anomalous, flagged or a problem.', prompt)
        self.assertIn('slightly below the baseline but is not flagged is normal variation', prompt)
        self.assertIn('Mention every flagged window by its number', prompt)
        self.assertNotIn('No windows are flagged.', prompt)

    def test_says_when_no_windows_are_flagged(self):
        # Window 2 is below the 0.75 baseline but not by more than 0.10, so it is not flagged.
        prompt = build_prompt('heart_disease', summary_windows((0.80, 10), (0.70, 12)), 0.75)

        self.assertIn('No windows are flagged.', prompt)
        self.assertIn('Do not call any window an anomaly, anomalous, flagged or a problem.', prompt)
        self.assertIn('slightly below the baseline but is not flagged is normal variation', prompt)
        self.assertIn('say that there are no anomalies', prompt)
        self.assertNotIn('ANOMALY', prompt)


class FactCheckTests(TestCase):
    def setUp(self):
        # Window 3 is an accuracy anomaly, window 4 a latency anomaly (median of 10, 12, 11 is 11 ms).
        self.windows = summary_windows((0.80, 10), (0.78, 12), (0.40, 11), (0.79, 30))
        self.prompt = build_prompt('heart_disease', self.windows, 0.75)

    def check(self, text):
        fact_check(text, self.windows, self.prompt, 0.75)

    def assertRejected(self, text, reason):
        with self.assertRaisesMessage(FactCheckError, reason):
            self.check(text)

    def test_accepts_correct_summaries(self):
        accepted = [
            'Windows 3 and 4 are anomalies. Accuracy fell to 0.40 in window 3, 0.35 below the 0.75 baseline.',
            'Window 3 is an anomaly (accuracy 40%), and window 4 was flagged for latency of 30.0 ms, 2.7 times '
            'the 11.0 ms median.',
            'Windows 3 to 4 were anomalous, but windows 1 and 2 were not.',
            'Window 1 had accuracy 0.8 and was not an anomaly; window 3 was an anomaly, as was window 4.',
            'Window 3 is an anomaly, while window 4 is also flagged as an anomaly.',
        ]
        for text in accepted:
            with self.subTest(text):
                self.check(text)

    def test_rejects_unflagged_window_called_anomaly(self):
        self.assertRejected('Windows 2, 3 and 4 are anomalies.', 'it calls window 2 an anomaly, but it was not flagged')
        self.assertRejected('Windows 3 to 5 are anomalies.', 'it calls window 5 an anomaly')

    def test_back_references_apply_to_windows_named_before(self):
        self.check('Accuracy fell to 0.40 in window 3. This is an anomaly. Window 4 was slow; it is an anomaly too.')
        # The bug that prompted the check: unflagged windows called anomalies through "These".
        self.assertRejected(
            'Window 1 has an accuracy of 0.80 and window 2 has 0.78. These are marked as ANOMALY. '
            'Windows 3 and 4 are anomalies.',
            'it calls windows 1 and 2 anomalies, but they were not flagged',
        )

    def test_allows_gap_from_baseline(self):
        self.check('Windows 3 and 4 are anomalies; window 3 is 0.35 below the baseline and window 2 is 0.03 above it.')
        self.assertRejected('Windows 3 and 4 are anomalies; window 3 is 0.30 below the baseline.', '0.30')

    def test_rejects_flagged_window_called_normal(self):
        self.assertRejected(
            'Window 3 is an anomaly. Window 4 was not an anomaly.',
            'it says flagged window 4 is not an anomaly',
        )

    def test_rejects_missing_flagged_window(self):
        self.assertRejected('Window 3 is an anomaly.', 'it does not mention flagged window 4 as anomalous')
        self.assertRejected('There are no anomalies.', 'it does not mention flagged windows 3 and 4 as anomalous')

    def test_rejects_numbers_that_do_not_match(self):
        prefix = 'Windows 3 and 4 are anomalies. '
        cases = {
            'Accuracy dropped to 0.45.': 'do not match the data: 0.45',
            'Accuracy dropped to 45%.': 'do not match the data: 45%',
            'Latency rose to 25 ms.': 'do not match the data: 25',
            'Window 3 had accuracy 0.40 and latency 11.5 ms.': 'do not match the data: 11.5',
        }
        for text, reason in cases.items():
            with self.subTest(text):
                self.assertRejected(prefix + text, reason)

    def test_no_anomalies(self):
        windows = summary_windows((0.80, 10), (0.78, 12))
        prompt = build_prompt('heart_disease', windows, 0.75)
        fact_check('There are no anomalies; accuracy stayed near 0.78.', windows, prompt, 0.75)
        with self.assertRaisesMessage(FactCheckError, 'it calls window 2 an anomaly'):
            fact_check('Window 2 is an anomaly.', windows, prompt, 0.75)

    def test_window_numbers(self):
        self.assertEqual(window_numbers('windows 4 and 5, Window 7'), {4, 5, 7})
        self.assertEqual(window_numbers('windows 2 to 4 and window 9'), {2, 3, 4, 9})
        self.assertEqual(window_numbers('windows 3-5'), {3, 4, 5})
        self.assertEqual(window_numbers('accuracy 0.40 over 10 predictions'), set())


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
        self.assertEqual(context['chart_data']['accuracy_anomaly'], [False, True])
        self.assertEqual(context['chart_data']['latency_anomaly'], [False, False])
        self.assertEqual(context['chart_data']['baseline'], 0.75)
        self.assertContains(response, 'id="accuracy-chart"')
        self.assertContains(response, 'id="latency-chart"')
        self.assertContains(response, 'Saved summary.')
        get_llm.assert_not_called()

    def test_charts_mark_only_their_own_rule(self, get_llm):
        log_predictions([(1, 1)] * 10, latency_ms=10)
        log_predictions([(1, 1)] * 10, latency_ms=50)  # Latency spike, accuracy fine.
        compute_snapshots('heart_disease', window_size=10)

        chart_data = self.client.get(self.url).context['chart_data']

        self.assertEqual(chart_data['accuracy_anomaly'], [False, False])
        self.assertEqual(chart_data['latency_anomaly'], [False, True])

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
