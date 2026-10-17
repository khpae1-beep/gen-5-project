from django.core.management.base import BaseCommand, CommandError

from monitoring.metrics import DEFAULT_WINDOW_SIZE, compute_snapshots
from monitoring.models import PredictionLog


class Command(BaseCommand):
    help = (
        'Group each model\'s PredictionLog rows into windows of --window-size predictions and save '
        'one MetricSnapshot per window, replacing that model\'s old snapshots. Windows are flagged '
        'as anomalies when accuracy falls more than 0.10 below MONITORING_BASELINE_ACCURACY or average '
        'latency is more than double the median of the earlier windows.'
    )

    def add_arguments(self, parser):
        parser.add_argument(
            '--window-size',
            type=int,
            default=DEFAULT_WINDOW_SIZE,
            help=f'Predictions per window (default {DEFAULT_WINDOW_SIZE}).',
        )

    def handle(self, *args, **options):
        window_size = options['window_size']
        if window_size < 1:
            raise CommandError('--window-size must be at least 1.')

        model_names = PredictionLog.objects.order_by('model_name').values_list('model_name', flat=True).distinct()
        if not model_names:
            self.stdout.write('No predictions logged yet.')
            return

        for model_name in model_names:
            snapshots, leftover = compute_snapshots(model_name, window_size)
            anomalies = sum(s.is_anomaly for s in snapshots)
            self.stdout.write(self.style.SUCCESS(
                f'{model_name}: saved {len(snapshots)} snapshots of {window_size} predictions, '
                f'{anomalies} anomalous ({leftover} waiting for the next window).'
            ))
