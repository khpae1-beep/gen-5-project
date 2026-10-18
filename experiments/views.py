from django.db.models import Count, FloatField, Max, OuterRef, Subquery
from django.db.models.fields.json import KT
from django.db.models.functions import Cast
from django.shortcuts import get_object_or_404, render

from .models import Experiment, Run
from .promotion import current_promotion
from .versioning import format_value


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
