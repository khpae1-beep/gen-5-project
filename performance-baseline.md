# Route Performance Baseline

Measured 2026-10-19 with `python manage.py test core.test_performance -v 2`.
The test prints a fresh `ROUTE_PERFORMANCE_BASELINE` JSON record on each run.

## Method

- In-process Django test client with authenticated staff user; this is not a browser or production server benchmark.
- Django `TestCase` test database using SQLite in-memory (`connection.vendor == "sqlite"`).
- Two warm-up requests, then 40 measured requests per route.
- Request wall time is measured with `time.perf_counter()` while `CaptureQueriesContext` records SQL statement counts.
- p95 uses the nearest-rank method. Query values below are median (minimum-maximum) counts per response.
- Fixtures: three experiments, five linked runs in the benchmark experiment, two tuning jobs with five linked trials each, two monitored models with 30 prediction rows and five metric snapshots each.
- External integrations are guarded: LLM entry points, RabbitMQ, Terraform subprocesses, socket connections, and model training cannot run. Deployment POST calls a mocked generator and measures only Django request handling, form validation, session persistence, and redirect work.

## Environment

| Property | Value |
| --- | --- |
| Python | 3.12.3 |
| Django | 4.2.30 |
| Platform | Linux-6.8.0-1064-azure-x86_64-with-glibc2.39 |
| Database | SQLite in-memory test database |
| Requests per route | 2 warm-ups + 40 measured |
| Request type | In-process test client; no concurrent load |

## Results

| Route | Status | Median (ms) | p95 (ms) | SQL queries, median (min-max) |
| --- | ---: | ---: | ---: | ---: |
| Home `/` | 200 | 0.549 | 0.721 | 0 (0-0) |
| Experiments list `/experiments/` | 200 | 3.268 | 4.678 | 1 (1-1) |
| Experiment detail `/experiments/benchmark-experiment/` | 200 | 4.051 | 4.899 | 2 (2-2) |
| Experiment comparison `/experiments/benchmark-experiment/compare/` | 200 | 3.222 | 4.972 | 3 (3-3) |
| Run detail `/experiments/benchmark-experiment/v5/` | 200 | 3.113 | 4.027 | 3 (3-3) |
| Monitoring dashboard `/monitoring/` | 200 | 7.337 | 13.511 | 4 (4-4) |
| Tuning list `/tuning/` | 200 | 2.614 | 7.348 | 1 (1-1) |
| Tuning comparison `/tuning/compare/` | 200 | 2.657 | 3.663 | 2 (2-2) |
| Tuning job detail `/tuning/2/` | 200 | 4.690 | 6.148 | 4 (4-4) |
| Deployment index `/deployments/` | 200 | 2.654 | 3.467 | 2 (2-2) |
| Deployment generation POST `/deployments/generate/` | 302 | 1.883 | 2.093 | 5 (5-5) |

## Observations

The monitoring dashboard had the highest measured median and p95 in this run. Its four SQL statements cover model discovery, prediction aggregation, metric snapshots, and recent predictions, making it the clearest candidate for follow-up profiling with representative production-sized data. The tuning job detail also used four SQL statements, while the deployment POST used five, including authenticated session/request work.

These are small-fixture, single-process measurements with SQLite and query-capture instrumentation. They do not establish production latency or load capacity. The deployment POST result excludes LLM generation, Terraform execution, provider calls, and any deployment operation; those systems are intentionally not run by the benchmark. No pass/fail latency threshold is set because the repository documents no performance service-level requirement.

## Day 4: Monitoring Dashboard Investigation

The dashboard route was profiled with the same benchmark fixtures, two warm-ups, and 40 measured requests per run. The benchmark now also times the outermost Django `Template.render()` call and wraps database execution with `connection.execute_wrapper()`; the latter is used because SQLite's `CaptureQueriesContext` duration field rounds these short statements to zero. `CaptureQueriesContext` still records and asserts the query count. The test asserts the selected model, prediction total, metric-window count, anomaly count and selected model option in the rendered response.

The four dashboard queries are:

1. Distinct model names, used to populate the model selector.
2. Aggregate prediction count, labelled/correct counts and mean latency for the selected model.
3. Ordered metric snapshots, needed for chart series and anomaly totals.
4. The latest 20 prediction rows, needed for the recent-predictions table.

Each query serves distinct displayed data. The fixture and query profile show no repeated query or per-row query pattern to remove while preserving the existing output. Query count remained exactly four in every measured request.

| Profile run | Dashboard median (ms) | p95 (ms) | Template median (ms) | SQL execute median (ms) | SQL queries |
| --- | ---: | ---: | ---: | ---: | ---: |
| Instrumented 1 | 7.402 | 10.399 | 4.505 | 0.149 | 4 |
| Instrumented 2 | 6.616 | 11.352 | 4.222 | 0.134 | 4 |
| Instrumented 3 | 5.568 | 12.167 | 3.389 | 0.125 | 4 |
| Instrumented 4 | 6.028 | 9.881 | 3.542 | 0.143 | 4 |
| Instrumented 5 | 5.923 | 11.470 | 3.601 | 0.128 | 4 |

The original, uninstrumented single-run baseline was 7.337 ms median / 13.511 ms p95. Three further uninstrumented runs measured 5.541-5.590 ms median and 7.928-12.444 ms p95. The instrumented request medians span 5.568-7.402 ms and p95 spans 9.881-12.167 ms. These variations, especially at p95, make a timing-only before/after claim unreliable. The instrumented template call is roughly 3.4-4.5 ms median, whereas measured SQLite execution is about 0.13 ms median across the four queries. This small-fixture result suggests rendering and request overhead dominate local request time, but does not demonstrate a production bottleneck.

**Conclusion:** no meaningful, reproducible database or rendering inefficiency was demonstrated, so the production dashboard view, models and template remain unchanged. No before/after optimization claim is made. Follow-up profiling should use production-like database/data volume and a production server before considering query consolidation, indexing changes or template restructuring.