from django.db import models


class Experiment(models.Model):
    """A named series of training runs, e.g. one model family on one dataset."""

    name = models.CharField(max_length=100, unique=True)
    description = models.TextField(blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ['name']

    def __str__(self):
        return self.name


class Run(models.Model):
    """One versioned training run of an experiment."""

    experiment = models.ForeignKey(Experiment, on_delete=models.CASCADE, related_name='runs')
    version = models.PositiveIntegerField()
    # The previous version of the same experiment; null for version 1.
    parent = models.ForeignKey('self', on_delete=models.SET_NULL, null=True, blank=True, related_name='children')
    params = models.JSONField(default=dict, blank=True)
    metrics = models.JSONField(default=dict, blank=True)
    # {'params': {name: {'old': ..., 'new': ...}}, 'metrics': {...}} compared with the parent.
    changes = models.JSONField(default=dict, blank=True)
    checkpoint_path = models.CharField(max_length=500, blank=True)
    # SHA-256 of the training data file; blank when no file was given.
    data_hash = models.CharField(max_length=64, blank=True)
    # Blank when git is not available.
    git_commit = models.CharField(max_length=40, blank=True)
    version_note = models.TextField(blank=True)
    duration_seconds = models.FloatField(null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ['experiment', 'version']
        constraints = [
            models.UniqueConstraint(fields=['experiment', 'version'], name='unique_run_version_per_experiment'),
        ]

    def __str__(self):
        return f'{self.experiment.name} v{self.version}'
