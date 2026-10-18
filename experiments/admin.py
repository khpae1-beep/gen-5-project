from django.contrib import admin

from .models import Experiment, Run


class RunInline(admin.TabularInline):
    model = Run
    fk_name = 'experiment'
    fields = ('version', 'version_note', 'created_at')
    readonly_fields = fields
    extra = 0
    can_delete = False
    show_change_link = True


@admin.register(Experiment)
class ExperimentAdmin(admin.ModelAdmin):
    list_display = ('name', 'description', 'created_at')
    search_fields = ('name', 'description')
    inlines = [RunInline]


@admin.register(Run)
class RunAdmin(admin.ModelAdmin):
    list_display = ('experiment', 'version', 'parent', 'version_note', 'git_commit', 'created_at')
    list_filter = ('experiment',)
    list_select_related = ('experiment', 'parent__experiment')
    search_fields = ('version_note',)
    readonly_fields = ('created_at',)
    raw_id_fields = ('parent',)
