import json
import math
import platform
import socket
import statistics
import sys
import time
from contextlib import ExitStack
from datetime import timedelta
from unittest import mock

from django.contrib.auth import get_user_model
from django.db import connection
from django.template.base import Template
from django.test import TestCase
from django.test.utils import CaptureQueriesContext
from django.urls import reverse
from django.utils import timezone

from experiments.models import Experiment, Run
from monitoring.models import MetricSnapshot, PredictionLog
from tuning.models import Trial, TuningJob


WARMUP_REQUESTS = 2
MEASURED_REQUESTS = 40


def percentile_95(samples):
    ordered = sorted(samples)
    return ordered[math.ceil(0.95 * len(ordered)) - 1]


class RoutePerformanceTests(TestCase):
    @classmethod
    def setUpTestData(cls):
        cls.staff = get_user_model().objects.create_user('benchmark-staff', is_staff=True)
        cls.experiment = Experiment.objects.create(name='benchmark-experiment')
        parent = None
        for version in range(1, 6):
            run = Run.objects.create(
                experiment=cls.experiment,
                version=version,
                parent=parent,
                params={'n_estimators': 50 + version, 'max_depth': version + 2},
                metrics={
                    'train_accuracy': 0.9 + version / 100,
                    'test_accuracy': 0.8 + version / 100,
                    'test_f1': 0.75 + version / 100,
                },
                changes={} if parent is None else {
                    'params': {'n_estimators': {'old': 49 + version, 'new': 50 + version}},
                    'metrics': {'test_accuracy': {'old': 0.79 + version / 100, 'new': 0.8 + version / 100}},
                },
                version_note=f'Benchmark run {version}',
            )
            parent = run
        cls.latest_run = parent

        cls.jobs = {}
        for strategy in (TuningJob.STRATEGY_LLM, TuningJob.STRATEGY_RANDOM):
            tuning_experiment = Experiment.objects.create(name=f'benchmark-tuning-{strategy}')
            job = TuningJob.objects.create(
                strategy=strategy,
                n_trials=5,
                cv_folds=3,
                search_space={},
                seed=42,
                status=TuningJob.STATUS_COMPLETED,
                experiment_name=tuning_experiment.name,
                best_cv_score=0.84,
                finished_at=timezone.now(),
            )
            best_trial = None
            parent_run = None
            for number in range(1, 6):
                run = Run.objects.create(
                    experiment=tuning_experiment,
                    version=number,
                    parent=parent_run,
                    params={'n_estimators': 50 + number},
                    metrics={'train_accuracy': 0.9, 'test_accuracy': 0.8},
                )
                parent_run = run
                trial = Trial.objects.create(
                    job=job,
                    number=number,
                    params={'n_estimators': 50 + number},
                    cv_score=0.80 + number / 100,
                    cv_std=0.02,
                    test_accuracy=0.78 + number / 100,
                    reasoning='Deterministic benchmark fixture.',
                    proposed_by=(Trial.PROPOSED_BY_LLM if strategy == TuningJob.STRATEGY_LLM
                                 else Trial.PROPOSED_BY_RANDOM),
                    reason_check=Trial.REASON_PASSED if strategy == TuningJob.STRATEGY_LLM else '',
                    run=run,
                )
                best_trial = trial
            job.best_trial = best_trial
            job.save(update_fields=['best_trial'])
            cls.jobs[strategy] = job

        start = timezone.now() - timedelta(minutes=10)
        logs = []
        snapshots = []
        for model_index, model_name in enumerate(('benchmark-model-a', 'benchmark-model-b')):
            for number in range(30):
                logs.append(PredictionLog(
                    model_name=model_name,
                    model_version='v1',
                    prediction=number % 2,
                    actual=number % 2 if number % 5 else (number + 1) % 2,
                    latency_ms=10 + number,
                    timestamp=start + timedelta(seconds=model_index * 60 + number),
                ))
            for window in range(5):
                snapshots.append(MetricSnapshot(
                    model_name=model_name,
                    window_start=start + timedelta(minutes=window),
                    window_end=start + timedelta(minutes=window + 1),
                    accuracy=0.8 - window / 100,
                    avg_latency_ms=12 + window,
                    count=10,
                    is_anomaly=window == 4,
                    accuracy_anomaly=window == 4,
                    latency_anomaly=False,
                    summary=f'Benchmark summary {window}.',
                ))
        PredictionLog.objects.bulk_create(logs)
        MetricSnapshot.objects.bulk_create(snapshots)

    def setUp(self):
        self.client.force_login(self.staff)

    def benchmark_routes(self):
        random_job = self.jobs[TuningJob.STRATEGY_RANDOM]
        return [
            ('home', reverse('home'), lambda: self.client.get(reverse('home')), 200),
            ('experiments list', reverse('experiments:list'),
             lambda: self.client.get(reverse('experiments:list')), 200),
            ('experiment detail', reverse('experiments:detail', args=[self.experiment.name]),
             lambda: self.client.get(reverse('experiments:detail', args=[self.experiment.name])), 200),
            ('experiment comparison', reverse('experiments:compare', args=[self.experiment.name]),
             lambda: self.client.get(reverse('experiments:compare', args=[self.experiment.name]),
                                     {'a': 1, 'b': self.latest_run.version}), 200),
            ('run detail', reverse('experiments:run', args=[self.experiment.name, self.latest_run.version]),
             lambda: self.client.get(reverse('experiments:run', args=[self.experiment.name, self.latest_run.version])), 200),
            ('monitoring dashboard', reverse('monitoring:dashboard'),
             lambda: self.client.get(reverse('monitoring:dashboard')), 200),
            ('tuning list', reverse('tuning:list'), lambda: self.client.get(reverse('tuning:list')), 200),
            ('tuning comparison', reverse('tuning:compare'), lambda: self.client.get(reverse('tuning:compare')), 200),
            ('tuning job detail', reverse('tuning:detail', args=[random_job.pk]),
             lambda: self.client.get(reverse('tuning:detail', args=[random_job.pk])), 200),
            ('deployment index', reverse('deployment:index'),
             lambda: self.client.get(reverse('deployment:index')), 200),
            ('deployment generation POST', reverse('deployment:generate'),
             lambda: self.client.post(reverse('deployment:generate'), {
                 'provider': 'aws', 'description': 'Benchmark-only generated configuration',
             }), 302),
        ]

    def test_representative_routes_report_latency_and_query_counts(self):
        fake_result = {
            'provider': 'aws',
            'description': 'Benchmark-only generated configuration',
            'finished_at': timezone.now().isoformat(),
            'attempts': 1,
            'status': 'success',
            'message': 'Mocked benchmark result.',
            'error': '',
            'code': 'resource "aws_s3_bucket" "benchmark" {}',
            'path': 'generated/benchmark/main.tf',
            'seconds': 0.0,
        }
        external_call = AssertionError('An external integration was reached by a route benchmark.')
        guards = [
            mock.patch('experiments.comparison.get_llm', side_effect=external_call),
            mock.patch('monitoring.summary.get_llm', side_effect=external_call),
            mock.patch('tuning.suggestions.get_llm', side_effect=external_call),
            mock.patch('deployment.generator.get_llm', side_effect=external_call),
            mock.patch('monitoring.rabbitmq.open_channel', side_effect=external_call),
            mock.patch('deployment.generator.run_terraform', side_effect=external_call),
            mock.patch('deployment.views.run_generation', return_value=fake_result),
            mock.patch('deployment.views.terraform_installed', return_value=True),
            mock.patch('experiments.training.train_and_evaluate', side_effect=external_call),
            mock.patch('socket.create_connection', side_effect=external_call),
        ]
        original_template_render = Template.render
        template_render_ms = []
        render_depth = 0

        def measure_template_render(template, context):
            nonlocal render_depth
            outermost = render_depth == 0
            started = time.perf_counter()
            render_depth += 1
            try:
                return original_template_render(template, context)
            finally:
                render_depth -= 1
                if outermost:
                    template_render_ms.append((time.perf_counter() - started) * 1000)

        with ExitStack() as stack:
            entered = [stack.enter_context(guard) for guard in guards]
            stack.enter_context(mock.patch.object(Template, 'render', new=measure_template_render))
            benchmark_output = []
            dashboard_query_profile = []
            for name, path, request, expected_status in self.benchmark_routes():
                for _ in range(WARMUP_REQUESTS):
                    response = request()
                    self.assertEqual(response.status_code, expected_status, f'{name} warm-up: {path}')

                latencies_ms = []
                query_counts = []
                database_times_ms = []
                template_times_ms = []
                for _ in range(MEASURED_REQUESTS):
                    render_start = len(template_render_ms)

                    def time_query(execute, sql, params, many, context):
                        started = time.perf_counter()
                        try:
                            return execute(sql, params, many, context)
                        finally:
                            query_durations.append((time.perf_counter() - started) * 1000)

                    query_durations = []
                    with CaptureQueriesContext(connection) as queries, connection.execute_wrapper(time_query):
                        started = time.perf_counter()
                        response = request()
                        latencies_ms.append((time.perf_counter() - started) * 1000)
                    self.assertEqual(response.status_code, expected_status, f'{name}: {path}')
                    query_counts.append(len(queries))
                    self.assertEqual(len(query_durations), len(queries.captured_queries))
                    database_times_ms.append(sum(query_durations))
                    template_times_ms.append(sum(template_render_ms[render_start:]))
                    if name == 'monitoring dashboard' and not dashboard_query_profile:
                        self.assertEqual(len(queries), 4)
                        self.assertEqual(response.context['model_names'], ['benchmark-model-a', 'benchmark-model-b'])
                        self.assertEqual(response.context['model_name'], 'benchmark-model-a')
                        self.assertEqual(response.context['total_predictions'], 30)
                        self.assertEqual(response.context['window_count'], 5)
                        self.assertEqual(response.context['anomaly_count'], 1)
                        self.assertContains(
                            response,
                            '<option value="benchmark-model-a" selected>benchmark-model-a</option>',
                            html=True,
                        )
                        dashboard_query_profile = [
                            {
                                'sql': ' '.join(query['sql'].split()),
                                'database_execute_ms': round(duration, 3),
                            }
                            for query, duration in zip(queries.captured_queries, query_durations)
                        ]

                benchmark_output.append({
                    'route': f'{name} {path}',
                    'status': expected_status,
                    'median_ms': round(statistics.median(latencies_ms), 3),
                    'p95_ms': round(percentile_95(latencies_ms), 3),
                    'template_median_ms': round(statistics.median(template_times_ms), 3),
                    'database_execute_median_ms': round(statistics.median(database_times_ms), 6),
                    'database_execute_max_ms': round(max(database_times_ms), 6),
                    'median_queries': statistics.median(query_counts),
                    'query_min': min(query_counts),
                    'query_max': max(query_counts),
                })

            for guard in entered[:6] + entered[8:]:
                guard.assert_not_called()
            self.assertEqual(entered[6].call_count, WARMUP_REQUESTS + MEASURED_REQUESTS)

        print('ROUTE_PERFORMANCE_BASELINE=' + json.dumps({
            'python': sys.version.split()[0],
            'django': __import__('django').get_version(),
            'platform': platform.platform(),
            'database': connection.vendor,
            'client': 'in-process Django test client; authenticated staff',
            'warmups_per_route': WARMUP_REQUESTS,
            'measured_requests_per_route': MEASURED_REQUESTS,
            'p95_method': 'nearest rank',
            'dashboard_query_profile': dashboard_query_profile,
            'results': benchmark_output,
        }, sort_keys=True))