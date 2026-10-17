from django.db import transaction

from .models import MetricSnapshot, PredictionLog

DEFAULT_WINDOW_SIZE = 10


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
    snapshots = [MetricSnapshot(model_name=model_name, **window_stats(window)) for window in windows]

    with transaction.atomic():
        MetricSnapshot.objects.filter(model_name=model_name).delete()
        MetricSnapshot.objects.bulk_create(snapshots)
    return snapshots, len(rows) % window_size
