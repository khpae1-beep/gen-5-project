import json
from datetime import timezone as dt_timezone

from django.utils import timezone
from django.utils.dateparse import parse_datetime

from .models import PredictionLog

REQUIRED_FIELDS = ('model_name', 'model_version', 'prediction', 'latency_ms')


class InvalidMessage(ValueError):
    """A queue message that cannot be turned into a PredictionLog."""


def handle_prediction_message(body):
    """Parse one JSON prediction message (str or bytes) and save it as a PredictionLog.

    'actual' and 'timestamp' are optional; a missing timestamp means now.
    Raises InvalidMessage if the body is not valid JSON or a field is missing or malformed.
    """
    try:
        data = json.loads(body)
    except (TypeError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise InvalidMessage(f'Message is not valid JSON: {exc}') from exc
    if not isinstance(data, dict):
        raise InvalidMessage('Message must be a JSON object.')

    missing = [field for field in REQUIRED_FIELDS if data.get(field) in (None, '')]
    if missing:
        raise InvalidMessage(f'Message is missing {", ".join(missing)}.')

    try:
        prediction = int(data['prediction'])
        actual = None if data.get('actual') is None else int(data['actual'])
        latency_ms = float(data['latency_ms'])
    except (TypeError, ValueError) as exc:
        raise InvalidMessage(f'Message has a non-numeric field: {exc}') from exc

    return PredictionLog.objects.create(
        model_name=str(data['model_name']),
        model_version=str(data['model_version']),
        prediction=prediction,
        actual=actual,
        latency_ms=latency_ms,
        timestamp=_parse_timestamp(data.get('timestamp')),
    )


def _parse_timestamp(value):
    if value is None:
        return timezone.now()
    try:
        parsed = parse_datetime(value)
    except (TypeError, ValueError) as exc:
        raise InvalidMessage(f'Invalid timestamp {value!r}.') from exc
    if parsed is None:
        raise InvalidMessage(f'Invalid timestamp {value!r}.')
    if timezone.is_naive(parsed):
        parsed = timezone.make_aware(parsed, dt_timezone.utc)
    return parsed
