from django.db import models
from django.utils import timezone


class PredictionLog(models.Model):
    """One prediction made by a deployed model, as received from the queue."""

    model_name = models.CharField(max_length=100)
    model_version = models.CharField(max_length=50)
    prediction = models.IntegerField()
    # Ground truth is not always known when the prediction is made.
    actual = models.IntegerField(null=True, blank=True)
    latency_ms = models.FloatField()
    timestamp = models.DateTimeField(default=timezone.now, db_index=True)

    class Meta:
        ordering = ['-timestamp']
        indexes = [models.Index(fields=['model_name', 'timestamp'])]

    def __str__(self):
        return f'{self.model_name}@{self.model_version}: {self.prediction} ({self.timestamp:%Y-%m-%d %H:%M:%S})'

    @property
    def is_correct(self):
        if self.actual is None:
            return None
        return self.prediction == self.actual


class MetricSnapshot(models.Model):
    """Aggregated metrics for one model over a time window."""

    model_name = models.CharField(max_length=100)
    window_start = models.DateTimeField()
    window_end = models.DateTimeField()
    # Null when no prediction in the window had a known actual value.
    accuracy = models.FloatField(null=True, blank=True)
    avg_latency_ms = models.FloatField()
    count = models.PositiveIntegerField()
    is_anomaly = models.BooleanField(default=False)
    summary = models.TextField(blank=True)

    class Meta:
        ordering = ['-window_end']
        indexes = [models.Index(fields=['model_name', 'window_end'])]

    def __str__(self):
        return f'{self.model_name} {self.window_start:%Y-%m-%d %H:%M} to {self.window_end:%Y-%m-%d %H:%M}'
