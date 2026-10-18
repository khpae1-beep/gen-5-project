from django.db.models import Avg, Count, F, Q
from django.shortcuts import render

from .metrics import baseline_accuracy
from .models import MetricSnapshot, PredictionLog

RECENT_PREDICTIONS = 20


def dashboard(request):
    """Metrics, charts and the latest saved summary for one model. Never calls the LLM."""
    model_names = list(
        PredictionLog.objects.order_by('model_name').values_list('model_name', flat=True).distinct()
    )
    model_name = request.GET.get('model')
    if model_name not in model_names:
        model_name = model_names[0] if model_names else None

    logs = PredictionLog.objects.filter(model_name=model_name)
    totals = logs.aggregate(
        total=Count('id'),
        labelled=Count('id', filter=Q(actual__isnull=False)),
        correct=Count('id', filter=Q(prediction=F('actual'))),
        avg_latency_ms=Avg('latency_ms'),
    )
    snapshots = list(MetricSnapshot.objects.filter(model_name=model_name).order_by('window_start', 'id'))
    recent_predictions = logs.order_by('-timestamp', '-id')[:RECENT_PREDICTIONS]
    baseline = baseline_accuracy()

    context = {
        'model_names': model_names,
        'model_name': model_name,
        'total_predictions': totals['total'],
        'overall_accuracy': totals['correct'] / totals['labelled'] if totals['labelled'] else None,
        'avg_latency_ms': totals['avg_latency_ms'],
        'anomaly_count': sum(s.is_anomaly for s in snapshots),
        'window_count': len(snapshots),
        'latest_snapshot': snapshots[-1] if snapshots else None,
        'recent_predictions': recent_predictions,
        'chart_data': {
            'labels': [f'Window {i}' for i in range(1, len(snapshots) + 1)],
            'accuracy': [s.accuracy for s in snapshots],
            'latency': [round(s.avg_latency_ms, 2) for s in snapshots],
            'accuracy_anomaly': [s.accuracy_anomaly for s in snapshots],
            'latency_anomaly': [s.latency_anomaly for s in snapshots],
            'baseline': baseline,
        },
        'baseline': baseline,
    }
    return render(request, 'monitoring/dashboard.html', context)
