from django.contrib import admin

from .models import MetricSnapshot, PredictionLog


@admin.register(PredictionLog)
class PredictionLogAdmin(admin.ModelAdmin):
    list_display = ('timestamp', 'model_name', 'model_version', 'prediction', 'actual', 'latency_ms')
    list_filter = ('model_name', 'model_version')
    date_hierarchy = 'timestamp'


@admin.register(MetricSnapshot)
class MetricSnapshotAdmin(admin.ModelAdmin):
    list_display = ('model_name', 'window_start', 'window_end', 'accuracy', 'avg_latency_ms', 'count', 'is_anomaly')
    list_filter = ('model_name', 'is_anomaly')
    date_hierarchy = 'window_end'
