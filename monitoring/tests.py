import json
from datetime import datetime, timezone as dt_timezone
from unittest import mock

from django.test import TestCase

from monitoring.management.commands.consume_predictions import Command as ConsumeCommand
from monitoring.management.commands.publish_predictions import build_message
from monitoring.messages import InvalidMessage, handle_prediction_message
from monitoring.models import PredictionLog


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
