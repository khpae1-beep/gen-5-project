from django.db.models import Count
from django.shortcuts import render

from .models import TuningJob


def job_list(request):
    jobs = TuningJob.objects.select_related('best_trial').annotate(trial_count=Count('trials')).order_by('-created_at', '-id')
    return render(request, 'tuning/job_list.html', {'jobs': jobs})
