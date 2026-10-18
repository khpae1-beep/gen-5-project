from django.db import models


class TuningJob(models.Model):
    """One hyperparameter search: n_trials proposals, each scored by cross-validation."""

    STRATEGY_LLM = 'llm'
    STRATEGY_RANDOM = 'random'
    STRATEGIES = [(STRATEGY_LLM, 'LLM proposer'), (STRATEGY_RANDOM, 'Random search')]

    STATUS_RUNNING = 'running'
    STATUS_COMPLETED = 'completed'
    STATUS_FAILED = 'failed'
    STATUSES = [(STATUS_RUNNING, 'Running'), (STATUS_COMPLETED, 'Completed'), (STATUS_FAILED, 'Failed')]

    strategy = models.CharField(max_length=10, choices=STRATEGIES)
    n_trials = models.PositiveIntegerField()
    cv_folds = models.PositiveIntegerField()
    search_space = models.JSONField()
    seed = models.IntegerField()
    status = models.CharField(max_length=10, choices=STATUSES, default=STATUS_RUNNING)
    # The experiment the trials are logged under with experiments.tracking.log_run().
    experiment_name = models.CharField(max_length=100)
    best_cv_score = models.FloatField(null=True, blank=True)
    best_trial = models.ForeignKey('Trial', on_delete=models.SET_NULL, null=True, blank=True, related_name='+')
    error = models.TextField(blank=True)
    # The data and split the trials were scored on, so more trials can be added later.
    data_path = models.CharField(max_length=500, blank=True)
    test_size = models.FloatField(default=0.2)
    # The latest "Suggest next experiments" result: {'created_at', 'configs': [{'params', 'reasoning',
    # 'reason_check', 'llm_reason', 'tried_trial'}], 'rejected': [...], 'error': ''}; {} before any.
    suggestions = models.JSONField(default=dict, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    # Blank for jobs that are still running or were run before this was recorded.
    finished_at = models.DateTimeField(null=True, blank=True)

    class Meta:
        ordering = ['-created_at']

    def __str__(self):
        return f'{self.get_strategy_display()} job {self.pk} ({self.n_trials} trials)'


class Trial(models.Model):
    """One proposed set of params and its cross-validation score."""

    PROPOSED_BY_LLM = 'llm'
    PROPOSED_BY_RANDOM = 'random'
    # The LLM's proposal was invalid or the LLM was unavailable, so a random one was used.
    PROPOSED_BY_FALLBACK = 'random-fallback'
    # Added from the job page's "Suggest next experiments".
    PROPOSED_BY_SUGGESTION = 'llm-suggestion'
    PROPOSERS = [
        (PROPOSED_BY_LLM, 'LLM'), (PROPOSED_BY_RANDOM, 'Random'), (PROPOSED_BY_FALLBACK, 'Random (LLM fallback)'),
        (PROPOSED_BY_SUGGESTION, 'LLM suggestion'),
    ]

    # Whether the LLM's reason passed reason_check; blank when there is no LLM reason or it was
    # not recorded (trials from before it was).
    REASON_PASSED = 'passed'
    REASON_FAILED = 'failed'
    REASON_MISSING = 'missing'
    REASON_CHECKS = [(REASON_PASSED, 'Passed'), (REASON_FAILED, 'Failed'), (REASON_MISSING, 'No reason given')]

    job = models.ForeignKey(TuningJob, on_delete=models.CASCADE, related_name='trials')
    number = models.PositiveIntegerField()
    params = models.JSONField()
    # Mean and standard deviation of k-fold CV accuracy on the training split; used to pick the best trial.
    cv_score = models.FloatField()
    cv_std = models.FloatField()
    # Accuracy on the held-out test split, for reference only: never used to choose params.
    test_accuracy = models.FloatField(null=True, blank=True)
    reasoning = models.TextField(blank=True)
    proposed_by = models.CharField(max_length=20, choices=PROPOSERS)
    reason_check = models.CharField(max_length=10, choices=REASON_CHECKS, blank=True)
    run = models.ForeignKey(
        'experiments.Run', on_delete=models.SET_NULL, null=True, blank=True, related_name='tuning_trials',
    )
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ['job', 'number']
        constraints = [models.UniqueConstraint(fields=['job', 'number'], name='unique_trial_number_per_job')]

    def __str__(self):
        return f'Trial {self.number} of job {self.job_id}: CV {self.cv_score:.3f}'
