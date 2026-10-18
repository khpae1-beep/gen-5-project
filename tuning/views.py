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


def pick_job(jobs, value):
    """The job with id value from jobs (newest first), or the newest."""
    try:
        return next(job for job in jobs if job.pk == int(value))
    except (TypeError, ValueError, StopIteration):
        return jobs[0] if jobs else None


def job_summary(job, trials):
    best = max(trials, key=lambda t: (t.cv_score, -t.number)) if trials else None
    llm_reasons = [t for t in trials if t.proposed_by in (Trial.PROPOSED_BY_LLM, Trial.PROPOSED_BY_SUGGESTION)]
    return {
        'job': job,
        'trial_count': len(trials),
        'best': best,
        'time_taken': time_taken(job),
        'fallbacks': sum(t.proposed_by == Trial.PROPOSED_BY_FALLBACK for t in trials),
        'llm_reasons': len(llm_reasons),
        'reasons_passed': sum(t.reason_check == Trial.REASON_PASSED for t in llm_reasons),
        'reasons_not_recorded': sum(t.reason_check == '' for t in llm_reasons),
    }


def compare(request):
    jobs = list(TuningJob.objects.order_by('-created_at', '-id'))
    llm_jobs = [job for job in jobs if job.strategy == TuningJob.STRATEGY_LLM]
    random_jobs = [job for job in jobs if job.strategy == TuningJob.STRATEGY_RANDOM]
    llm_job, random_job = pick_job(llm_jobs, request.GET.get('llm')), pick_job(random_jobs, request.GET.get('random'))
    context = {'llm_jobs': llm_jobs, 'random_jobs': random_jobs, 'llm_job': llm_job, 'random_job': random_job}
    if llm_job and random_job:
        trials = {llm_job.pk: [], random_job.pk: []}
        for trial in Trial.objects.filter(job__in=[llm_job, random_job]).order_by('number'):
            trials[trial.job_id].append(trial)
        llm_trials, random_trials = trials[llm_job.pk], trials[random_job.pk]
        context.update({
            'summaries': [job_summary(llm_job, llm_trials), job_summary(random_job, random_trials)],
            'chart_data': {
                'labels': list(range(1, max(len(llm_trials), len(random_trials)) + 1)),
                'llm': best_so_far(llm_trials),
                'random': best_so_far(random_trials),
                'llm_label': f'LLM job {llm_job.pk}',
                'random_label': f'Random job {random_job.pk}',
            },
        })
    return render(request, 'tuning/compare.html', context)
