import json
import os
import shutil
import tempfile

from django.conf import settings
from django.core.management.base import BaseCommand, CommandError
from django.utils import timezone

from experiments.management.commands.train_experiment import project_relative
from experiments.models import Run
from experiments.tracking import resolve_path

# The model publish_predictions serves, and which run it came from.
PROMOTED_MODEL_PATH = settings.BASE_DIR / 'models' / 'heart_disease_model.joblib'
CURRENT_MODEL_INFO_PATH = settings.BASE_DIR / 'models' / 'current_model.json'


def replace_file(path, write):
    """Write to a temporary file next to path, then swap it in, so readers never see half a file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f'.{path.name}.')
    os.close(fd)
    try:
        write(tmp)
        os.chmod(tmp, 0o644)  # mkstemp creates owner-only files.
        os.replace(tmp, path)
    except BaseException:
        os.unlink(tmp)
        raise


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
        if not run.checkpoint_path:
            raise CommandError(f'{run} has no checkpoint_path.')
        checkpoint = resolve_path(run.checkpoint_path)
        if not checkpoint.exists():
            raise CommandError(f'Checkpoint {run.checkpoint_path} for {run} does not exist.')

        info = {
            'experiment': run.experiment.name,
            'version': run.version,
            'run_id': run.pk,
            'checkpoint_path': run.checkpoint_path,
            'version_note': run.version_note,
            'metrics': run.metrics,
            'promoted_at': timezone.now().isoformat(),
        }
        replace_file(PROMOTED_MODEL_PATH, lambda tmp: shutil.copyfile(checkpoint, tmp))
        replace_file(CURRENT_MODEL_INFO_PATH, lambda tmp: self.write_json(tmp, info))

        self.stdout.write(self.style.SUCCESS(
            f'Promoted {run} to {project_relative(PROMOTED_MODEL_PATH)}'
            + (f' ({run.version_note})' if run.version_note else '')
        ))

    @staticmethod
    def write_json(path, data):
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
            f.write('\n')
