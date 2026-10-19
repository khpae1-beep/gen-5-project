import io
import json
import tempfile
from pathlib import Path
import subprocess
from unittest import mock

from django.core.management import call_command, CommandError
from django.test import SimpleTestCase

from core.management.commands.benchmark import RESULT_MARKER


class BenchmarkCommandTests(SimpleTestCase):
    def test_writes_labeled_current_code_report(self):
        benchmark = {
            'python': '3.12.3',
            'django': '4.2.30',
            'platform': 'test platform',
            'database': 'sqlite',
            'client': 'test client',
            'warmups_per_route': 2,
            'measured_requests_per_route': 40,
            'p95_method': 'nearest rank',
            'results': [{
                'route': 'monitoring dashboard /monitoring/',
                'status': 200,
                'median_ms': 5.0,
                'p95_ms': 8.0,
                'template_median_ms': 3.0,
                'database_execute_median_ms': 0.1,
                'median_queries': 4,
                'query_min': 4,
                'query_max': 4,
            }],
            'dashboard_query_profile': [],
        }
        completed = subprocess.CompletedProcess(
            args=[], returncode=0,
            stdout=f'test summary\n{RESULT_MARKER}{json.dumps(benchmark)}\n', stderr='',
        )

        with tempfile.TemporaryDirectory() as directory:
            output_path = Path(directory) / 'before.txt'
            stdout = io.StringIO()
            with mock.patch('core.management.commands.benchmark.subprocess.run', return_value=completed) as run:
                call_command('benchmark', '--output', str(output_path), stdout=stdout)

            run.assert_called_once()
            report = output_path.read_text(encoding='utf-8')
            self.assertIn('current code snapshot; not a historical before/after result', report)
            self.assertIn('monitoring dashboard /monitoring/ | 200 | 5.000 | 8.000', report)
            self.assertIn('Wrote current-code benchmark result', stdout.getvalue())

    def test_propagates_failed_benchmark_test(self):
        failed = subprocess.CompletedProcess(args=[], returncode=1, stdout='FAILED', stderr='')
        with mock.patch('core.management.commands.benchmark.subprocess.run', return_value=failed):
            with self.assertRaisesMessage(CommandError, 'Benchmark test failed with exit code 1.'):
                call_command('benchmark', stdout=io.StringIO(), stderr=io.StringIO())