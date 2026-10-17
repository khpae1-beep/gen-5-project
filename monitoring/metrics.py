from statistics import median

from django.conf import settings
from django.db import transaction

from .models import MetricSnapshot, PredictionLog

DEFAULT_WINDOW_SIZE = 10
# The heart disease model's test accuracy.
DEFAULT_BASELINE_ACCURACY = 0.75
# A window is anomalous when accuracy is more than this far below the baseline...
MAX_ACCURACY_DROP = 0.10
# ...or average latency is more than this multiple of the earlier windows' median.
MAX_LATENCY_RATIO = 2.0
# Keeps float error (e.g. 0.75 - 0.65) from deciding the boundary case.
EPSILON = 1e-9


def baseline_accuracy():
    return getattr(settings, 'MONITORING_BASELINE_ACCURACY', DEFAULT_BASELINE_ACCURACY)


def split_into_windows(rows, window_size):
    """Split rows into consecutive windows of window_size; an incomplete last window is left out."""
    full = len(rows) - len(rows) % window_size
    return [rows[i:i + window_size] for i in range(0, full, window_size)]


def window_stats(window):
    """Metrics for one window of (prediction, actual, latency_ms, timestamp) rows.

    Accuracy only counts rows with a known actual, and is None when there are none.
    """
    labelled = [(prediction, actual) for prediction, actual, _, _ in window if actual is not None]
    correct = sum(prediction == actual for prediction, actual in labelled)
    return {
        'window_start': window[0][3],
        'window_end': window[-1][3],
        'accuracy': correct / len(labelled) if labelled else None,
        'avg_latency_ms': sum(row[2] for row in window) / len(window),
        'count': len(window),
    }


def anomaly_reasons(snapshot, earlier_latencies, baseline):
    """Why a window is anomalous (empty if it is not), given the earlier windows' average latencies."""
    reasons = []
    if snapshot.accuracy is not None and baseline - snapshot.accuracy > MAX_ACCURACY_DROP + EPSILON:
        reasons.append(
            f'accuracy {snapshot.accuracy:.2f} is {baseline - snapshot.accuracy:.2f} below the {baseline:.2f} baseline'
        )
    if earlier_latencies:
        typical = median(earlier_latencies)
        if snapshot.avg_latency_ms > MAX_LATENCY_RATIO * typical + EPSILON:
            reasons.append(
                f'average latency {snapshot.avg_latency_ms:.1f} ms is {snapshot.avg_latency_ms / typical:.1f} times '
                f'the earlier median of {typical:.1f} ms'
            )
    return reasons


def flag_anomalies(snapshots, baseline=None):
    """Set is_anomaly, and a summary of the reasons, on each snapshot in window order."""
    if baseline is None:
        baseline = baseline_accuracy()
    for i, snapshot in enumerate(snapshots):
        reasons = anomaly_reasons(snapshot, [s.avg_latency_ms for s in snapshots[:i]], baseline)
        snapshot.is_anomaly = bool(reasons)
        snapshot.summary = f'Anomaly: {"; ".join(reasons)}.' if reasons else ''
    return snapshots


def compute_snapshots(model_name, window_size=DEFAULT_WINDOW_SIZE):
    """Replace the model's MetricSnapshots with one per full window of its predictions.

    Returns (snapshots in window order, number of predictions left over for the next window).
    """
    # Published predictions often share a timestamp, so id breaks ties.
    rows = list(
        PredictionLog.objects.filter(model_name=model_name)
        .order_by('timestamp', 'id')
        .values_list('prediction', 'actual', 'latency_ms', 'timestamp')
    )
    windows = split_into_windows(rows, window_size)
    snapshots = flag_anomalies([MetricSnapshot(model_name=model_name, **window_stats(window)) for window in windows])

    with transaction.atomic():
        MetricSnapshot.objects.filter(model_name=model_name).delete()
        MetricSnapshot.objects.bulk_create(snapshots)
    return snapshots, len(rows) % window_size
