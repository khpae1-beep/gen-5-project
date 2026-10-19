import json
from datetime import datetime, timezone
from pathlib import Path
import subprocess
import sys

from django.core.management.base import BaseCommand, CommandError


RESULT_MARKER = 'ROUTE_PERFORMANCE_BASELINE='


def format_report(report):
    benchmark = report['benchmark']
    lines = [
        'Django route benchmark',
        'Measurement: current code snapshot; not a historical before/after result.',
        f"Recorded at (UTC): {report['recorded_at_utc']}",
        f"Python: {benchmark['python']}",
        f"Django: {benchmark['django']}",
        f"Platform: {benchmark['platform']}",
        f"Database: {benchmark['database']}",
        f"Client: {benchmark['client']}",
        f"Warm-ups per route: {benchmark['warmups_per_route']}",
        f"Measured requests per route: {benchmark['measured_requests_per_route']}",
        f"p95 method: {benchmark['p95_method']}",
        '',
        'Routes:',
        'Route | Status | Median ms | p95 ms | Template median ms | SQL median ms | Query count (min-max)',
    ]
    for result in benchmark['results']:
        lines.append(
            f"{result['route']} | {result['status']} | {result['median_ms']:.3f} | "
            f"{result['p95_ms']:.3f} | {result['template_median_ms']:.3f} | "
            f"{result['database_execute_median_ms']:.6f} | {result['median_queries']} "
            f"({result['query_min']}-{result['query_max']})"
        )

    lines.extend(['', 'Monitoring dashboard SQL profile:'])
    for index, query in enumerate(benchmark['dashboard_query_profile'], start=1):
        lines.append(f"{index}. {query['database_execute_ms']:.3f} ms: {query['sql']}")
    lines.extend([
        '',
        'This report measures the current working-tree code with deterministic test-database fixtures.',
        'It is an in-process Django test-client benchmark, not production load testing.',
    ])
    return '\n'.join(lines) + '\n'


class Command(BaseCommand):
    help = 'Run the deterministic Django route benchmark.'

    def add_arguments(self, parser):
        parser.add_argument(
            '--output',
            type=Path,
            help='Write a plain-text current-code benchmark report to this path.',
        )

    def handle(self, *args, **options):
        manage_py = Path(sys.argv[0]).resolve()
        result = subprocess.run(
            [
                sys.executable,
                str(manage_py),
                'test',
                'core.test_performance',
                '--verbosity',
                str(options['verbosity']),
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.stdout:
            self.stdout.write(result.stdout, ending='')
        if result.stderr:
            self.stderr.write(result.stderr, ending='')
        if result.returncode:
            raise CommandError(f'Benchmark test failed with exit code {result.returncode}.')

        output_path = options.get('output')
        if output_path:
            record_line = next(
                (line for line in result.stdout.splitlines() if line.startswith(RESULT_MARKER)),
                None,
            )
            if record_line is None:
                raise CommandError('Benchmark completed without emitting its result record.')
            report = {
                'label': 'current-code measurement; not a historical before/after result',
                'recorded_at_utc': datetime.now(timezone.utc).isoformat(),
                'benchmark': json.loads(record_line[len(RESULT_MARKER):]),
            }
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(format_report(report), encoding='utf-8')
            self.stdout.write(f'Wrote current-code benchmark result to {output_path}')