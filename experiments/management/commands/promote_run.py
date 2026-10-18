from django.core.management.base import BaseCommand, CommandError

from experiments.management.commands.train_experiment import project_relative
from experiments.models import Run
from experiments.promotion import PROMOTED_MODEL_PATH, PromotionError, promote


class Command(BaseCommand):
    help = (
        'Check out a run: copy its checkpoint to models/heart_disease_model.joblib and record the '
        'experiment and version in models/current_model.json. Example: promote_run heart-disease-rf 3'
    )

    def add_arguments(self, parser):
        parser.add_argument('experiment')
        parser.add_argument('version', type=int)

    def handle(self, *args, **options):
        try:
            run = Run.objects.select_related('experiment').get(
                experiment__name=options['experiment'], version=options['version'],
            )
        except Run.DoesNotExist:
            raise CommandError(f'No run {options["experiment"]} v{options["version"]}.')
        try:
            promote(run)
        except PromotionError as exc:
            raise CommandError(str(exc)) from exc

        self.stdout.write(self.style.SUCCESS(
            f'Promoted {run} to {project_relative(PROMOTED_MODEL_PATH)}'
            + (f' ({run.version_note})' if run.version_note else '')
        ))
