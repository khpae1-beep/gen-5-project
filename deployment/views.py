import shutil
import time
import uuid

from django import forms
from django.conf import settings
from django.contrib.admin.views.decorators import staff_member_required
from django.shortcuts import redirect, render
from django.urls import reverse
from django.utils import timezone
from django.views.decorators.http import require_POST

from .generator import SUPPORTED_PROVIDERS, generate_terraform

# Each web generation writes to its own folder here (generated/ is git-ignored).
WEB_OUTPUT_DIR = settings.BASE_DIR / 'generated' / 'web'
SAMPLES = [
    ('AWS', settings.BASE_DIR / 'generated' / 'aws' / 'main.tf'),
    ('GCP', settings.BASE_DIR / 'generated' / 'gcp' / 'main.tf'),
]
# The last result is kept in the session, so refreshing the page shows it without generating again.
SESSION_KEY = 'deployment_result'
TERRAFORM_MISSING = (
    'The Terraform CLI is not installed or not on the PATH, so the generated configuration cannot be '
    'validated. Install it (see the README setup steps) and try again.'
)
MAX_ERROR_LENGTH = 4000


class DeploymentForm(forms.Form):
    provider = forms.ChoiceField(choices=[(p, p.upper()) for p in SUPPORTED_PROVIDERS])
    description = forms.CharField(
        max_length=2000,
        widget=forms.Textarea(attrs={'rows': 5, 'placeholder': 'For example: a private S3 bucket for model artifacts with versioning enabled.'}),
        help_text='What infrastructure should be created.',
    )


def terraform_installed():
    return shutil.which('terraform') is not None


def project_relative(path):
    try:
        return str(path.relative_to(settings.BASE_DIR))
    except ValueError:
        return str(path)


def read_samples():
    return [
        {'name': name, 'path': project_relative(path), 'code': path.read_text(encoding='utf-8').strip('\n')}
        for name, path in SAMPLES
        if path.exists()
    ]


def last_error_from(message):
    """The "Last error:" part of generate_terraform's failure message."""
    if 'Last error:' not in message:
        return message
    error = message.split('Last error:', 1)[1]
    return error.split('Last generated file:', 1)[0].strip()


def run_generation(provider, description):
    """Run generate_terraform and describe the outcome as a JSON-serialisable dict for the page."""
    result = {
        'provider': provider,
        'description': description,
        'finished_at': None,
        'attempts': None,
        'status': 'error',
        'message': '',
        'error': '',
        'code': '',
        'path': '',
    }
    if not terraform_installed():
        result['message'] = TERRAFORM_MISSING
        result['finished_at'] = timezone.now().isoformat()
        return result

    output_dir = WEB_OUTPUT_DIR / f'{timezone.now():%Y%m%d-%H%M%S}-{uuid.uuid4().hex[:6]}'
    main_tf = output_dir / provider / 'main.tf'
    attempts = []
    start = time.monotonic()
    try:
        path = generate_terraform(provider, description, output_dir=str(output_dir), on_attempt=attempts.append)
    except RuntimeError as exc:
        message = str(exc)
        if 'Terraform CLI is not installed' in message:
            result['message'] = TERRAFORM_MISSING
        else:
            result['status'] = 'failed'
            result['message'] = f'Terraform validation still failed after {len(attempts) or "all"} attempts.'
            result['error'] = last_error_from(message)[:MAX_ERROR_LENGTH]
    except ValueError as exc:  # e.g. the LLM returned empty or non-text content.
        result['message'] = f'The generator could not produce Terraform: {exc}'
    except Exception as exc:  # The LLM provider could not be reached or returned an error.
        result['message'] = (
            f'The LLM is unavailable or returned an error ({type(exc).__name__}: {str(exc)[:300]}). '
            'Check that Ollama is running, or the LLM_PROVIDER settings in .env.'
        )
    else:
        result['status'] = 'success'
        result['message'] = 'The configuration passed terraform validate.'
        main_tf = path

    result['attempts'] = len(attempts) or None
    result['seconds'] = round(time.monotonic() - start, 1)
    result['finished_at'] = timezone.now().isoformat()
    if main_tf.exists():
        result['code'] = main_tf.read_text(encoding='utf-8').strip('\n')
        result['path'] = project_relative(main_tf)
    return result


def index(request, form=None):
    return render(request, 'deployment/index.html', {
        'form': form or DeploymentForm(),
        'result': request.session.get(SESSION_KEY),
        'samples': read_samples(),
        'terraform_installed': terraform_installed(),
    })


@require_POST
@staff_member_required
def generate(request):
    """Generate Terraform from the form. Staff only; may take a few minutes."""
    form = DeploymentForm(request.POST)
    if not form.is_valid():
        return index(request, form)
    request.session[SESSION_KEY] = run_generation(form.cleaned_data['provider'], form.cleaned_data['description'])
    return redirect(reverse('deployment:index') + '#result')
