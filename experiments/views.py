from django.db.models import Count, FloatField, Max, OuterRef, Subquery
from django.db.models.fields.json import KT
from django.db.models.functions import Cast
import re

from django.contrib import messages
from django.shortcuts import get_object_or_404, redirect, render
from django.views.decorators.http import require_POST

from .comparison import compare_runs
from .models import Experiment, Run
from .promotion import PromotionError, checkpoint_problem, current_promotion, promote
from .versioning import format_value, metric_label

SPLIT_METRIC = re.compile(r'^(train|test)_(.+)$')


def test_accuracy_desc(runs):
    """runs with a known test accuracy, best first (earliest version wins ties)."""
    return (
        runs.annotate(test_acc=Cast(KT('metrics__test_accuracy'), FloatField()))
        .filter(test_acc__isnull=False)
        .order_by('-test_acc', 'version')
    )


def experiment_list(request):
    best = test_accuracy_desc(Run.objects.filter(experiment=OuterRef('pk')))
    experiments = list(
        Experiment.objects.annotate(
            run_count=Count('runs'),
            last_run_at=Max('runs__created_at'),
            best_test_accuracy=Subquery(best.values('test_acc')[:1]),
            best_version=Subquery(best.values('version')[:1]),
        ).order_by('name')
    )
    promoted = current_promotion()
    for experiment in experiments:
        experiment.promoted_version = promoted[1] if promoted and promoted[0] == experiment.name else None
    return render(request, 'experiments/experiment_list.html', {'experiments': experiments})


def promoted_version_of(experiment):
    promoted = current_promotion()
    return promoted[1] if promoted and promoted[0] == experiment.name else None


def key_param_names(runs):
    """The params that differ between runs, in first-seen order; all params if none differ."""
    names = list(dict.fromkeys(name for run in runs for name in run.params))
    varying = [name for name in names if len({repr(run.params.get(name)) for run in runs}) > 1]
    return varying or names


def experiment_detail(request, name):
    experiment = get_object_or_404(Experiment, name=name)
    runs = list(experiment.runs.order_by('version'))
    promoted_version = promoted_version_of(experiment)
    param_names = key_param_names(runs)

    rows = []
    for run in runs:
        train, test = run.metrics.get('train_accuracy'), run.metrics.get('test_accuracy')
        rows.append({
            'run': run,
            'params': [format_value(run.params[name]) if name in run.params else '–' for name in param_names],
            'train_accuracy': train,
            'test_accuracy': test,
            'test_f1': run.metrics.get('test_f1'),
            'gap': train - test if train is not None and test is not None else None,
            'promoted': run.version == promoted_version,
        })

    return render(request, 'experiments/experiment_detail.html', {
        'experiment': experiment,
        'rows': rows,
        'param_names': param_names,
        'promoted_version': promoted_version,
        'chart_data': {
            'labels': [f'v{run.version}' for run in runs],
            'train': [row['train_accuracy'] for row in rows],
            'test': [row['test_accuracy'] for row in rows],
            'promoted': [row['promoted'] for row in rows],
        },
    })


def is_number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def change_rows(changes, label=str):
    """Rows for a {name: {'old', 'new'}} dict, with a numeric delta where both sides are numbers."""
    rows = []
    for name, change in changes.items():
        old, new = change['old'], change['new']
        delta = new - old if is_number(old) and is_number(new) else None
        rows.append({'name': label(name), 'old': format_value(old), 'new': format_value(new), 'delta': delta})
    return rows


def metric_table(metrics):
    """Train/test metrics as rows of (name, train, test, gap); other metrics separately."""
    by_name, other = {}, []
    for name, value in metrics.items():
        match = SPLIT_METRIC.match(name)
        if match:
            by_name.setdefault(match.group(2), {})[match.group(1)] = value
        else:
            other.append({'name': metric_label(name), 'value': value})
    rows = []
    for name, values in by_name.items():
        train, test = values.get('train'), values.get('test')
        gap = train - test if is_number(train) and is_number(test) else None
        rows.append({'name': metric_label(name), 'train': train, 'test': test, 'gap': gap})
    return rows, other


def get_run(name, version):
    return get_object_or_404(
        Run.objects.select_related('experiment', 'parent'), experiment__name=name, version=version,
    )


def run_detail(request, name, version):
    run = get_run(name, version)
    changed_params = run.changes.get('params', {})
    metric_rows, other_metrics = metric_table(run.metrics)
    promoted = promoted_version_of(run.experiment) == run.version
    return render(request, 'experiments/run_detail.html', {
        'run': run,
        'params': [
            {'name': name, 'value': format_value(value), 'changed': name in changed_params}
            for name, value in run.params.items()
        ],
        'metric_rows': metric_rows,
        'other_metrics': other_metrics,
        'param_changes': change_rows(changed_params),
        'metric_changes': change_rows(run.changes.get('metrics', {}), label=metric_label),
        'promoted': promoted,
        'checkpoint_problem': checkpoint_problem(run),
    })


@require_POST
def promote_run(request, name, version):
    """Same as `python manage.py promote_run <name> <version>`."""
    run = get_run(name, version)
    try:
        promote(run)
    except PromotionError as exc:
        messages.error(request, f'Could not promote {run}: {exc}')
    else:
        messages.success(
            request,
            f'Promoted {run}. Restart publish_predictions to publish predictions from this version.',
        )
    return redirect('experiments:run', name=run.experiment.name, version=run.version)


def parse_version(value, versions, default):
    try:
        version = int(value)
    except (TypeError, ValueError):
        return default
    return version if version in versions else default


def comparison_context(request_data, name):
    """The compare page's context for versions a and b given in request_data (GET or POST)."""
    experiment = get_object_or_404(Experiment, name=name)
    versions = list(experiment.runs.order_by('version').values_list('version', flat=True))
    context = {'experiment': experiment, 'versions': versions}
    if len(versions) < 2:
        return context

    a = parse_version(request_data.get('a'), versions, versions[-2])
    b = parse_version(request_data.get('b'), versions, versions[-1])
    runs = {run.version: run for run in experiment.runs.filter(version__in={a, b})}
    run_a, run_b = runs[a], runs[b]
    run_a.experiment = run_b.experiment = experiment  # Saves a query per run in the template.
    context.update({'run_a': run_a, 'run_b': run_b, 'comparison': compare_runs(run_a, run_b)})
    return context


def compare(request, name):
    return render(request, 'experiments/compare.html', comparison_context(request.GET, name))
