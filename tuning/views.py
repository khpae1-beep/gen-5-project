from django.db.models import Count
from django.shortcuts import get_object_or_404, render

from .models import Trial, TuningJob


def job_list(request):
    jobs = TuningJob.objects.select_related('best_trial').annotate(trial_count=Count('trials')).order_by('-created_at', '-id')
    return render(request, 'tuning/job_list.html', {'jobs': jobs})


def format_params(params):
    return ', '.join(f'{name}={"None" if value is None else value}' for name, value in params.items())


def best_so_far(trials):
    """The running maximum of CV accuracy, trial by trial."""
    best, series = None, []
    for trial in trials:
        best = trial.cv_score if best is None else max(best, trial.cv_score)
        series.append(best)
    return series


def time_taken(job):
    return (job.finished_at - job.created_at).total_seconds() if job.finished_at else None


def job_context(job):
    """Everything the job page shows; used by the page and by the POST actions that re-render it."""
    trials = list(job.trials.select_related('run__experiment').order_by('number'))
    return {
        'job': job,
        'trials': [{'trial': t, 'params': format_params(t.params), 'best': t.pk == job.best_trial_id} for t in trials],
        'time_taken': time_taken(job),
        'chart_data': {
            'labels': [t.number for t in trials],
            'cv': [t.cv_score for t in trials],
            'std': [t.cv_std for t in trials],
            'best_so_far': best_so_far(trials),
            'proposed_by': [t.get_proposed_by_display() for t in trials],
            'best': [t.pk == job.best_trial_id for t in trials],
        },
    }


def job_detail(request, pk):
    job = get_object_or_404(TuningJob.objects.select_related('best_trial'), pk=pk)
    return render(request, 'tuning/job_detail.html', job_context(job))
