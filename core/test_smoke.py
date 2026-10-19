from types import SimpleNamespace
from unittest import mock

from django.contrib.auth import get_user_model
from django.test import TestCase
from django.urls import reverse

from experiments.models import Experiment, Run
from tuning.models import Trial, TuningJob


class SmokePageTests(TestCase):
    def setUp(self):
        self.staff_user = get_user_model().objects.create_user(
            username='smoke-staff', password='test-password', is_staff=True,
        )
        self.experiment = Experiment.objects.create(name='smoke-experiment')
        self.run = Run.objects.create(
            experiment=self.experiment,
            version=1,
            params={'n_estimators': 100},
            metrics={'train_accuracy': 0.9, 'test_accuracy': 0.8},
        )
        self.job = TuningJob.objects.create(
            strategy=TuningJob.STRATEGY_RANDOM,
            n_trials=1,
            cv_folds=3,
            search_space={},
            seed=42,
            status=TuningJob.STATUS_COMPLETED,
            experiment_name=self.experiment.name,
        )
        self.trial = Trial.objects.create(
            job=self.job,
            number=1,
            params={'n_estimators': 100},
            cv_score=0.8,
            cv_std=0.02,
            test_accuracy=0.75,
            proposed_by=Trial.PROPOSED_BY_RANDOM,
        )
        self.job.best_trial = self.trial
        self.job.save(update_fields=['best_trial'])

    def page_urls(self):
        return [
            reverse('home'),
            reverse('monitoring:dashboard'),
            reverse('experiments:list'),
            reverse('tuning:list'),
            reverse('tuning:compare'),
            reverse('deployment:index'),
            reverse('experiments:detail', args=[self.experiment.name]),
            reverse('experiments:run', args=[self.experiment.name, self.run.version]),
            reverse('tuning:detail', args=[self.job.pk]),
        ]

    def test_requested_pages_are_available_logged_out_and_to_staff(self):
        for url in self.page_urls():
            with self.subTest(user='logged out', url=url):
                self.assertEqual(self.client.get(url).status_code, 200)

        self.client.force_login(self.staff_user)
        for url in self.page_urls():
            with self.subTest(user='staff', url=url):
                self.assertEqual(self.client.get(url).status_code, 200)

    def test_staff_controls_are_visible_only_to_staff(self):
        run_url = reverse('experiments:run', args=[self.experiment.name, self.run.version])
        job_url = reverse('tuning:detail', args=[self.job.pk])
        deployment_url = reverse('deployment:index')

        run_response = self.client.get(run_url)
        self.assertContains(run_response, 'Only staff can promote')
        self.assertContains(run_response, '<button type="button" disabled>Promote this version</button>', html=True)

        job_response = self.client.get(job_url)
        self.assertContains(job_response, 'Only staff can ask for suggestions')
        self.assertContains(job_response, '<button type="button" disabled>Suggest next experiments</button>', html=True)

        deployment_response = self.client.get(deployment_url)
        self.assertContains(deployment_response, 'Only staff can generate configurations')
        self.assertContains(deployment_response, '<button type="button" disabled>Generate Terraform</button>', html=True)

        self.client.force_login(self.staff_user)
        staff_run_response = self.client.get(run_url)
        self.assertContains(staff_run_response, 'action="{}"'.format(
            reverse('experiments:promote', args=[self.experiment.name, self.run.version]),
        ))
        self.assertNotContains(staff_run_response, 'Only staff can promote')

        staff_job_response = self.client.get(job_url)
        self.assertContains(staff_job_response, 'action="{}#suggestions"'.format(
            reverse('tuning:suggest', args=[self.job.pk]),
        ))
        self.assertNotContains(staff_job_response, 'Only staff can ask for suggestions')

        staff_deployment_response = self.client.get(deployment_url)
        self.assertContains(staff_deployment_response, 'action="{}"'.format(reverse('deployment:generate')))
        self.assertNotContains(staff_deployment_response, 'Only staff can generate configurations')

    def test_staff_only_actions_redirect_anonymous_and_mock_integrations_for_staff(self):
        actions = [
            reverse('experiments:promote', args=[self.experiment.name, self.run.version]),
            reverse('tuning:suggest', args=[self.job.pk]),
            reverse('deployment:generate'),
        ]
        for url in actions:
            with self.subTest(url=url):
                response = self.client.post(url)
                self.assertEqual(response.status_code, 302)
                self.assertTrue(response.url.startswith('/admin/login/'))

        self.client.force_login(self.staff_user)
        with mock.patch('experiments.views.promote') as promote:
            response = self.client.post(actions[0])
        self.assertEqual(response.status_code, 302)
        promote.assert_called_once_with(self.run)

        suggestion = SimpleNamespace(as_json=lambda: {'configs': []})
        with mock.patch('tuning.views.suggest_next', return_value=suggestion) as suggest_next:
            response = self.client.post(actions[1])
        self.assertEqual(response.status_code, 302)
        suggest_next.assert_called_once()

        with mock.patch('deployment.views.run_generation', return_value={'status': 'success'}) as run_generation:
            response = self.client.post(actions[2], {'provider': 'aws', 'description': 'A test bucket'})
        self.assertEqual(response.status_code, 302)
        run_generation.assert_called_once_with('aws', 'A test bucket')