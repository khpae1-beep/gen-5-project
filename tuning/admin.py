from django.contrib import admin

from .models import Trial, TuningJob


class TrialInline(admin.TabularInline):
    model = Trial
    fields = ('number', 'params', 'cv_score', 'cv_std', 'test_accuracy', 'proposed_by', 'reasoning', 'run')
    readonly_fields = fields
    extra = 0
    can_delete = False
    show_change_link = True


@admin.register(TuningJob)
class TuningJobAdmin(admin.ModelAdmin):
    list_display = ('id', 'strategy', 'n_trials', 'cv_folds', 'seed', 'status', 'best_cv_score', 'experiment_name', 'created_at')
    list_filter = ('strategy', 'status')
    list_select_related = ('best_trial',)
    raw_id_fields = ('best_trial',)
    inlines = [TrialInline]


@admin.register(Trial)
class TrialAdmin(admin.ModelAdmin):
    list_display = ('job', 'number', 'cv_score', 'cv_std', 'test_accuracy', 'proposed_by', 'created_at')
    list_filter = ('job__strategy', 'proposed_by')
    list_select_related = ('job',)
    raw_id_fields = ('job', 'run')
