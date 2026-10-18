from django.contrib import messages
from django.contrib.admin.views.decorators import staff_member_required
from django.db.models import Count
from django.shortcuts import get_object_or_404, redirect, render
from django.urls import reverse
from django.utils.dateparse import parse_datetime
from django.views.decorators.http import require_POST

from .engine import TrialError, add_trial
from .llm_proposer import Proposal
from .models import Trial, TuningJob
from .search_space import InvalidParams, validate_params
from .suggestions import recheck_reason, suggest_next


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
    suggestions = job.suggestions or None
    if suggestions:
        suggestions = {
            **suggestions,
            'created_at': parse_datetime(suggestions['created_at']),
            'configs': [{**config, 'params_text': format_params(config['params'])} for config in suggestions['configs']],
        }
    return {
        'job': job,
        'suggestions': suggestions,
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


@require_POST
@staff_member_required
def suggest(request, pk):
    """Ask the LLM for the next configurations, save them on the job and go back to its page.

    Only this POST calls the LLM; the redirect means refreshing the page does not ask again.
    """
    job = get_object_or_404(TuningJob, pk=pk)
    job.suggestions = suggest_next(job, list(job.trials.order_by('number'))).as_json()
    job.save(update_fields=['suggestions'])
    return redirect(reverse('tuning:detail', args=[job.pk]) + '#suggestions')


@require_POST
@staff_member_required
def try_suggestion(request, pk):
    """Run one trial with saved suggestion number `index` and add it to the job."""
    job = get_object_or_404(TuningJob, pk=pk)
    configs = (job.suggestions or {}).get('configs', [])
    try:
        index = int(request.POST.get('index', ''))
        config = configs[index] if index >= 0 else None
    except (ValueError, IndexError):
        config = None
    if config is None:
        messages.error(request, 'Could not try that suggestion: it is no longer saved on this job.')
        return redirect('tuning:detail', pk=job.pk)
    try:
        params = validate_params(config['params'], job.search_space)
    except InvalidParams as exc:
        messages.error(request, f'Could not try that suggestion: invalid params ({exc}).')
        return redirect('tuning:detail', pk=job.pk)

    trials = list(job.trials.order_by('number'))
    if not trials:
        messages.error(request, 'Could not try that suggestion: the job has no trials.')
        return redirect('tuning:detail', pk=job.pk)
    if config.get('source') == 'random':  # A slot filled at random: no LLM reason to re-check.
        proposal = Proposal(params, config['reasoning'], Trial.PROPOSED_BY_FALLBACK)
    else:
        reasoning, status = recheck_reason(job, trials, params, config.get('llm_reason', ''))
        proposal = Proposal(params, reasoning, Trial.PROPOSED_BY_SUGGESTION, status)
    try:
        trial, new_best = add_trial(job, proposal)
    except TrialError as exc:
        messages.error(request, f'Could not try that suggestion: {exc}')
        return redirect('tuning:detail', pk=job.pk)

    job.refresh_from_db()
    job.suggestions['configs'][index]['tried_trial'] = trial.number
    job.save(update_fields=['suggestions'])
    outcome = 'the new best trial.' if new_best else f'the best is still trial {job.best_trial.number}.'
    messages.success(request, f'Trial {trial.number}: CV accuracy {trial.cv_score:.4f} ± {trial.cv_std:.4f}; {outcome}')
    return redirect(reverse('tuning:detail', args=[job.pk]) + f'#trial-{trial.number}')
