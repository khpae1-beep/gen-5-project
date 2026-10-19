import tempfile
from pathlib import Path
from unittest import mock

from django.contrib.auth import get_user_model
from django.test import Client, TestCase
from django.urls import reverse

from deployment import views

HCL = 'terraform {\n  required_version = ">= 1.5.0"\n}\n\nprovider "aws" {\n  region = var.region\n}\n'
FAILURE = (
    'Terraform generation failed after 4 attempts.\n'
    'Last error:\nError: Reference to undeclared input variable\n\n  on main.tf line 6\n'
    'Last generated file: {path}'
)


class DeploymentPageTestCase(TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.tmp = Path(tmp.name)
        samples = []
        for name in ('aws', 'gcp'):
            path = self.tmp / 'samples' / name / 'main.tf'
            path.parent.mkdir(parents=True)
            path.write_text(f'# {name} sample\nprovider "{name}" {{}}\n', encoding='utf-8')
            samples.append((name.upper(), path))
        for patcher in (
            mock.patch.object(views, 'WEB_OUTPUT_DIR', self.tmp / 'web'),
            mock.patch.object(views, 'SAMPLES', samples),
            mock.patch('deployment.views.shutil.which', return_value='/usr/local/bin/terraform'),
        ):
            patcher.start()
            self.addCleanup(patcher.stop)
        self.index_url = reverse('deployment:index')
        self.generate_url = reverse('deployment:generate')

    def login_staff(self):
        self.client.force_login(get_user_model().objects.create_user('staff', password='pw', is_staff=True))

    def generate(self, provider='aws', description='A private S3 bucket.', **kwargs):
        return self.client.post(self.generate_url, {'provider': provider, 'description': description}, **kwargs)


def fake_generator(attempts=2, fail=False):
    """A stand-in for generate_terraform that writes main.tf like the real one."""

    def generate_terraform(provider, description, output_dir, on_attempt):
        path = Path(output_dir) / provider / 'main.tf'
        path.parent.mkdir(parents=True, exist_ok=True)
        for attempt in range(1, attempts + 1):
            on_attempt(attempt)
            path.write_text(HCL if not fail else f'# attempt {attempt}\n{HCL}', encoding='utf-8')
        if fail:
            raise RuntimeError(FAILURE.format(path=path))
        return path

    return generate_terraform


class DeploymentPageTests(DeploymentPageTestCase):
    def test_non_staff_see_samples_and_a_disabled_button(self):
        self.client.force_login(get_user_model().objects.create_user('viewer', password='pw'))
        for client in (Client(), self.client):
            response = client.get(self.index_url)
            self.assertEqual(response.status_code, 200)
            self.assertTemplateUsed(response, 'base.html')
            self.assertContains(response, '<button type="button" disabled>Generate Terraform</button>', html=True)
            self.assertContains(response, '/admin/login/?next=/deployments/')
            self.assertContains(response, 'Generation can take a few minutes')
            self.assertNotContains(response, self.generate_url)
            self.assertContains(response, '# aws sample')
            self.assertContains(response, '# gcp sample')
            self.assertContains(response, 'provider &quot;gcp&quot; {}')

    def test_staff_see_the_form(self):
        self.login_staff()
        response = self.client.get(self.index_url)
        self.assertContains(response, f'action="{self.generate_url}"')
        self.assertContains(response, 'csrfmiddlewaretoken')
        self.assertContains(response, '<option value="aws">AWS</option>', html=True)
        self.assertContains(response, '<option value="gcp">GCP</option>', html=True)
        self.assertContains(response, 'Generation can take a few minutes')

    def test_warns_when_terraform_is_missing(self):
        self.login_staff()
        with mock.patch('deployment.views.shutil.which', return_value=None):
            response = self.client.get(self.index_url)
        self.assertContains(response, 'The Terraform CLI is not installed or not on the PATH')
        self.assertContains(response, '<button type="submit" disabled>Generate Terraform</button>', html=True)


class GenerateTests(DeploymentPageTestCase):
    def test_success_shows_attempts_and_main_tf(self):
        self.login_staff()
        with mock.patch('deployment.views.generate_terraform', side_effect=fake_generator(attempts=2)) as generate:
            response = self.generate('gcp', 'A Cloud Storage bucket.')

        self.assertRedirects(response, self.index_url + '#result', fetch_redirect_response=False)
        args, kwargs = generate.call_args
        self.assertEqual(args, ('gcp', 'A Cloud Storage bucket.'))
        self.assertEqual(Path(kwargs['output_dir']).parent, self.tmp / 'web')

        page = self.client.get(self.index_url)
        result = page.context['result']
        self.assertEqual((result['status'], result['attempts'], result['code']), ('success', 2, HCL.strip()))
        self.assertTrue(result['path'].endswith('/gcp/main.tf'))
        self.assertContains(page, 'Succeeded')
        self.assertContains(page, '2 attempts')
        self.assertContains(page, 'The configuration passed terraform validate.')
        self.assertContains(page, 'Generated main.tf')
        self.assertContains(page, 'required_version = &quot;&gt;= 1.5.0&quot;')

        # Refreshing shows the saved result without generating again.
        with mock.patch('deployment.views.generate_terraform') as generate:
            self.assertContains(self.client.get(self.index_url), 'Succeeded')
        generate.assert_not_called()

    def test_each_generation_gets_its_own_folder(self):
        self.login_staff()
        with mock.patch('deployment.views.generate_terraform', side_effect=fake_generator(attempts=1)) as generate:
            self.generate()
            self.generate()
        folders = [call.kwargs['output_dir'] for call in generate.call_args_list]
        self.assertNotEqual(folders[0], folders[1])
        self.assertEqual(self.client.get(self.index_url).context['result']['attempts'], 1)

    def test_failure_shows_the_last_terraform_error(self):
        self.login_staff()
        with mock.patch('deployment.views.generate_terraform', side_effect=fake_generator(attempts=4, fail=True)):
            page = self.generate(follow=True)

        result = page.context['result']
        self.assertEqual((result['status'], result['attempts']), ('failed', 4))
        self.assertEqual(result['error'], 'Error: Reference to undeclared input variable\n\n  on main.tf line 6')
        self.assertContains(page, 'Failed')
        self.assertContains(page, 'Terraform validation still failed after 4 attempts.')
        self.assertContains(page, 'Last Terraform error')
        self.assertContains(page, 'Reference to undeclared input variable')
        self.assertContains(page, 'Last generated main.tf')
        self.assertContains(page, '# attempt 4')

    def test_terraform_missing_does_not_call_the_generator(self):
        self.login_staff()
        with mock.patch('deployment.views.shutil.which', return_value=None), \
                mock.patch('deployment.views.generate_terraform') as generate:
            page = self.generate(follow=True)
        generate.assert_not_called()
        self.assertEqual(page.context['result']['status'], 'error')
        self.assertContains(page, 'The Terraform CLI is not installed or not on the PATH, so the generated configuration')

    def test_generator_reporting_terraform_missing(self):
        self.login_staff()
        error = RuntimeError('Terraform CLI is not installed.')
        with mock.patch('deployment.views.generate_terraform', side_effect=error):
            page = self.generate(follow=True)
        self.assertContains(page, 'Could not generate')
        self.assertContains(page, 'The Terraform CLI is not installed or not on the PATH, so the generated configuration')

    def test_llm_unavailable(self):
        self.login_staff()
        error = ConnectionError('Connection refused')
        with mock.patch('deployment.views.generate_terraform', side_effect=error):
            page = self.generate(follow=True)
        self.assertContains(page, 'Could not generate')
        self.assertContains(page, 'The LLM is unavailable or returned an error (ConnectionError: Connection refused).')
        self.assertContains(page, 'Check that Ollama is running')

    def test_generator_value_error(self):
        self.login_staff()
        with mock.patch('deployment.views.generate_terraform', side_effect=ValueError('LLM returned empty Terraform code.')):
            page = self.generate(follow=True)
        self.assertContains(page, 'The generator could not produce Terraform: LLM returned empty Terraform code.')

    def test_invalid_form(self):
        self.login_staff()
        with mock.patch('deployment.views.generate_terraform') as generate:
            for provider, description in (('aws', ''), ('azure', 'A bucket.')):
                with self.subTest(provider=provider, description=description):
                    response = self.generate(provider, description)
                    self.assertEqual(response.status_code, 200)
                    self.assertContains(response, 'errorlist')
        generate.assert_not_called()
        self.assertNotIn(views.SESSION_KEY, self.client.session)

    def test_staff_only_post_only_and_csrf(self):
        with mock.patch('deployment.views.generate_terraform') as generate:
            non_staff = Client()
            non_staff.force_login(get_user_model().objects.create_user('viewer', password='pw'))
            for client in (Client(), non_staff):
                response = client.post(self.generate_url, {'provider': 'aws', 'description': 'A bucket.'})
                self.assertRedirects(response, f'/admin/login/?next={self.generate_url}', fetch_redirect_response=False)
            staff = Client(enforce_csrf_checks=True)
            staff.force_login(get_user_model().objects.create_user('staff', password='pw', is_staff=True))
            self.assertEqual(staff.post(self.generate_url, {'provider': 'aws', 'description': 'A bucket.'}).status_code, 403)
            self.assertEqual(staff.get(self.generate_url).status_code, 405)
        generate.assert_not_called()
