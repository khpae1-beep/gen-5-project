import json
import tempfile
from pathlib import Path
from unittest import mock

from django.contrib.auth import get_user_model
from django.test import Client, TestCase
from django.urls import reverse
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.runnables import RunnableLambda

from experiments import promotion
from experiments.comparison import (
    ExplanationCheckError, check_explanation, compare_runs, describe_comparison, explain_difference,
    fallback_explanation,
)
from experiments.models import Experiment, Run
from experiments.tracking import log_run

RUNS = [
    ({'n_estimators': 100, 'max_depth': None, 'test_size': 0.2},
     {'train_accuracy': 1.0, 'test_accuracy': 0.754, 'train_f1': 1.0, 'test_f1': 0.776}),
    ({'n_estimators': 300, 'max_depth': 8, 'test_size': 0.2},
     {'train_accuracy': 1.0, 'test_accuracy': 0.77, 'train_f1': 1.0, 'test_f1': 0.788}),
    ({'n_estimators': 200, 'max_depth': 5, 'test_size': 0.2},
     {'train_accuracy': 0.95, 'test_accuracy': 0.77, 'train_f1': 0.956, 'test_f1': 0.788}),
]


def no_llm(*args, **kwargs):
    raise AssertionError('The LLM must not be called on a normal page load.')


class ExperimentPagesTestCase(TestCase):
    """Three runs of heart-disease-rf with checkpoints, and promotion files in a temp dir."""

    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.tmp = Path(tmp.name)
        self.model_path = self.tmp / 'models' / 'heart_disease_model.joblib'
        self.info_path = self.tmp / 'models' / 'current_model.json'
        for name, value in (('PROMOTED_MODEL_PATH', self.model_path), ('CURRENT_MODEL_INFO_PATH', self.info_path)):
            patcher = mock.patch.object(promotion, name, value)
            patcher.start()
            self.addCleanup(patcher.stop)
        patcher = mock.patch('experiments.tracking.current_git_commit', return_value='abc123def456')
        patcher.start()
        self.addCleanup(patcher.stop)

        self.runs = []
        for version, (params, metrics) in enumerate(RUNS, start=1):
            checkpoint = self.tmp / f'heart-disease-rf-v{version}.joblib'
            checkpoint.write_bytes(f'model v{version}'.encode())
            run = log_run('heart-disease-rf', params, metrics, checkpoint_path=str(checkpoint), duration_seconds=1.5)
            run.version_note = f'v{version}: note {version}'
            run.save()
            self.runs.append(run)

    def make_user(self, username='staff', is_staff=True):
        return get_user_model().objects.create_user(username, password='pw', is_staff=is_staff)

    def mark_promoted(self, version, experiment='heart-disease-rf'):
        self.info_path.parent.mkdir(parents=True, exist_ok=True)
        self.info_path.write_text(json.dumps({'experiment': experiment, 'version': version}))


@mock.patch('experiments.comparison.get_llm', side_effect=no_llm)
class ExperimentListTests(ExperimentPagesTestCase):
    url = reverse('experiments:list')

    def test_lists_runs_best_accuracy_and_promoted_version(self, get_llm):
        self.mark_promoted(1)
        log_run('no-test-metrics', {}, {'train_accuracy': 0.9})

        with self.assertNumQueries(1):
            response = self.client.get(self.url)

        self.assertEqual(response.status_code, 200)
        self.assertTemplateUsed(response, 'base.html')
        rows = {e.name: e for e in response.context['experiments']}
        heart = rows['heart-disease-rf']
        self.assertEqual((heart.run_count, heart.best_test_accuracy, heart.best_version), (3, 0.77, 2))
        self.assertEqual(heart.promoted_version, 1)
        self.assertIsNone(rows['no-test-metrics'].best_test_accuracy)
        self.assertIsNone(rows['no-test-metrics'].promoted_version)
        self.assertContains(response, '0.770 <a href="/experiments/heart-disease-rf/v2/">v2</a>', html=False)
        self.assertContains(response, 'v1 promoted')
        self.assertContains(response, 'href="/experiments/heart-disease-rf/"')

    def test_empty(self, get_llm):
        Experiment.objects.all().delete()
        self.assertContains(self.client.get(self.url), 'No experiments yet')


@mock.patch('experiments.comparison.get_llm', side_effect=no_llm)
class ExperimentDetailTests(ExperimentPagesTestCase):
    url = reverse('experiments:detail', args=['heart-disease-rf'])

    def test_shows_history_and_chart(self, get_llm):
        self.mark_promoted(2)

        with self.assertNumQueries(2):
            response = self.client.get(self.url)

        self.assertEqual(response.status_code, 200)
        # Only the params that differ between runs.
        self.assertEqual(response.context['param_names'], ['n_estimators', 'max_depth'])
        rows = response.context['rows']
        self.assertEqual([row['params'] for row in rows], [['100', 'None'], ['300', '8'], ['200', '5']])
        self.assertEqual([round(row['gap'], 3) for row in rows], [0.246, 0.23, 0.18])
        self.assertEqual([row['promoted'] for row in rows], [False, True, False])
        self.assertEqual(response.context['chart_data'], {
            'labels': ['v1', 'v2', 'v3'],
            'train': [1.0, 1.0, 0.95],
            'test': [0.754, 0.77, 0.77],
            'promoted': [False, True, False],
        })
        self.assertContains(response, 'id="accuracy-chart"')
        self.assertContains(response, 'v3: note 3')
        self.assertContains(response, '<span class="badge">promoted</span>', html=True)
        self.assertContains(response, 'Compare versions')

    def test_experiment_without_runs_and_unknown_experiment(self, get_llm):
        Experiment.objects.create(name='empty')
        self.assertContains(self.client.get(reverse('experiments:detail', args=['empty'])), 'No runs yet')
        self.assertEqual(self.client.get(reverse('experiments:detail', args=['nope'])).status_code, 404)


@mock.patch('experiments.comparison.get_llm', side_effect=no_llm)
class RunDetailTests(ExperimentPagesTestCase):
    def url(self, version):
        return reverse('experiments:run', args=['heart-disease-rf', version])

    def test_shows_everything_about_the_run(self, get_llm):
        with self.assertNumQueries(1):
            response = self.client.get(self.url(3))

        self.assertEqual(response.status_code, 200)
        context = response.context
        self.assertEqual(
            [(p['name'], p['value'], p['changed']) for p in context['params']],
            [('n_estimators', '200', True), ('max_depth', '5', True), ('test_size', '0.2', False)],
        )
        accuracy = next(row for row in context['metric_rows'] if row['name'] == 'accuracy')
        self.assertEqual((accuracy['train'], accuracy['test'], round(accuracy['gap'], 3)), (0.95, 0.77, 0.18))
        self.assertEqual(
            [(c['name'], c['old'], c['new'], c['delta']) for c in context['param_changes']],
            [('n_estimators', '300', '200', -100), ('max_depth', '8', '5', -3)],
        )
        self.assertEqual([c['name'] for c in context['metric_changes']], ['train accuracy', 'train f1'])
        run = self.runs[2]
        for text in (run.checkpoint_path, run.data_hash or '–', 'abc123def456', '1.5 s', 'v3: note 3',
                     'Compare v2 and v3 side by side'):
            self.assertContains(response, text)

    def test_non_staff_see_a_disabled_button_and_login_note(self, get_llm):
        self.client.force_login(self.make_user('viewer', is_staff=False))
        for client in (Client(), self.client):
            with self.subTest(anonymous=client is not self.client):
                response = client.get(self.url(3))
                self.assertContains(response, '<button type="button" disabled>Promote this version</button>', html=True)
                self.assertContains(
                    response, '<a href="/admin/login/?next=/experiments/heart-disease-rf/v3/">'
                    'Log in as a staff user at /admin/</a>', html=True,
                )
                self.assertNotContains(response, 'csrfmiddlewaretoken')
                self.assertNotContains(response, reverse('experiments:promote', args=['heart-disease-rf', 3]))

    def test_staff_see_the_promote_form(self, get_llm):
        self.client.force_login(self.make_user())
        response = self.client.get(self.url(3))
        self.assertContains(response, '<button type="submit">Promote this version</button>', html=True)
        self.assertNotContains(response, 'Only staff can promote')

    def test_first_run_has_nothing_to_compare(self, get_llm):
        response = self.client.get(self.url(1))
        self.assertContains(response, 'This is the first run')
        self.assertNotContains(response, 'side by side')

    def test_promote_button_disabled_when_promoted_or_checkpoint_missing(self, get_llm):
        self.client.force_login(self.make_user())
        self.mark_promoted(3)
        disabled = '<button type="submit" disabled>Promote this version</button>'
        response = self.client.get(self.url(3))
        self.assertContains(response, disabled, html=True)
        self.assertContains(response, 'This version is the promoted model.')

        Path(self.runs[0].checkpoint_path).unlink()
        response = self.client.get(self.url(1))
        self.assertContains(response, disabled, html=True)
        self.assertContains(response, 'does not exist')

    def test_unknown_run(self, get_llm):
        self.assertEqual(self.client.get(self.url(9)).status_code, 404)


@mock.patch('experiments.comparison.get_llm', side_effect=no_llm)
class PromoteViewTests(ExperimentPagesTestCase):
    def setUp(self):
        super().setUp()
        self.staff = self.make_user()
        self.client.force_login(self.staff)

    def url(self, version):
        return reverse('experiments:promote', args=['heart-disease-rf', version])

    def test_promotes_like_the_command(self, get_llm):
        response = self.client.post(self.url(2), follow=True)

        self.assertRedirects(response, reverse('experiments:run', args=['heart-disease-rf', 2]))
        self.assertEqual(self.model_path.read_bytes(), b'model v2')
        info = json.loads(self.info_path.read_text())
        self.assertEqual((info['experiment'], info['version'], info['run_id']), ('heart-disease-rf', 2, self.runs[1].pk))
        self.assertContains(response, 'Promoted heart-disease-rf v2.')
        self.assertContains(response, 'This version is the promoted model.')

    def test_missing_checkpoint_shows_error_and_changes_nothing(self, get_llm):
        Path(self.runs[0].checkpoint_path).unlink()

        response = self.client.post(self.url(1), follow=True)

        self.assertContains(response, 'Could not promote heart-disease-rf v1')
        self.assertFalse(self.model_path.exists())
        self.assertFalse(self.info_path.exists())

    def test_requires_post_and_csrf(self, get_llm):
        self.assertEqual(self.client.get(self.url(2)).status_code, 405)
        csrf_client = Client(enforce_csrf_checks=True)
        csrf_client.force_login(self.staff)
        self.assertEqual(csrf_client.post(self.url(2)).status_code, 403)
        self.assertEqual(self.client.post(self.url(9)).status_code, 404)
        self.assertFalse(self.model_path.exists())

    def test_requires_staff(self, get_llm):
        non_staff = Client()
        non_staff.force_login(self.make_user('viewer', is_staff=False))
        for name, client in (('anonymous', Client()), ('non-staff', non_staff)):
            with self.subTest(name):
                response = client.post(self.url(2))
                self.assertRedirects(
                    response, f'/admin/login/?next={self.url(2)}', fetch_redirect_response=False,
                )
        self.assertFalse(self.model_path.exists())
        self.assertFalse(self.info_path.exists())
        self.assertEqual(Client().get(self.url(2)).status_code, 405)

    def test_run_page_form_posts_with_csrf_token(self, get_llm):
        response = self.client.get(reverse('experiments:run', args=['heart-disease-rf', 2]))
        self.assertContains(response, f'action="{self.url(2)}"')
        self.assertContains(response, 'csrfmiddlewaretoken')


@mock.patch('experiments.comparison.get_llm', side_effect=no_llm)
class CompareTests(ExperimentPagesTestCase):
    url = reverse('experiments:compare', args=['heart-disease-rf'])

    def test_defaults_to_the_two_latest_versions(self, get_llm):
        with self.assertNumQueries(3):
            response = self.client.get(self.url)

        self.assertEqual((response.context['run_a'].version, response.context['run_b'].version), (2, 3))
        self.assertContains(response, 'Explain the difference')
        self.assertNotIn('explanation', response.context)
        get_llm.assert_not_called()

    def test_side_by_side_with_differences_and_deltas(self, get_llm):
        response = self.client.get(self.url, {'a': 1, 'b': 3})

        params = {row['key']: row for row in response.context['comparison']['params']}
        self.assertEqual((params['n_estimators']['a'], params['n_estimators']['b']), ('100', '200'))
        self.assertEqual(params['n_estimators']['delta'], 100)
        self.assertTrue(params['max_depth']['differs'])
        self.assertIsNone(params['max_depth']['delta'])  # None → 5 has no numeric delta.
        self.assertFalse(params['test_size']['differs'])

        metrics = {row['key']: row for row in response.context['comparison']['metrics']}
        self.assertAlmostEqual(metrics['test_accuracy']['delta'], 0.016)
        self.assertEqual(metrics['test_accuracy']['direction'], 'up')
        self.assertEqual(metrics['train_accuracy']['direction'], '')  # Train metrics are not coloured.
        self.assertContains(response, '+0.016')
        self.assertContains(response, '-0.050')
        self.assertContains(response, '+100')
        self.assertContains(response, 'class="changed"')

    def test_invalid_versions_fall_back_to_defaults(self, get_llm):
        response = self.client.get(self.url, {'a': 'x', 'b': 99})
        self.assertEqual((response.context['run_a'].version, response.context['run_b'].version), (2, 3))

    def test_needs_two_runs(self, get_llm):
        log_run('single', {}, {})
        response = self.client.get(reverse('experiments:compare', args=['single']))
        self.assertContains(response, 'Comparing needs at least two runs')
        self.assertEqual(self.client.get(reverse('experiments:compare', args=['nope'])).status_code, 404)


class ExplainViewTests(ExperimentPagesTestCase):
    url = reverse('experiments:explain', args=['heart-disease-rf'])
    GOOD = (
        'v3 has fewer, shallower trees than v1. Performance on the training data fell, with lower accuracy, '
        'while test accuracy rose from 0.754 to 0.77. The gap between train and test accuracy shrank, so v3 '
        'overfits less.'
    )

    def test_shows_checked_llm_explanation(self):
        llm = FakeListChatModel(responses=[self.GOOD])
        with mock.patch('experiments.comparison.get_llm', return_value=llm) as get_llm:
            response = self.client.post(self.url, {'a': 1, 'b': 3})

        get_llm.assert_called_once_with(temperature=0)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.context['explanation'], {'text': self.GOOD, 'used_llm': True, 'reason': ''})
        self.assertContains(response, 'Written by the LLM and checked against the numbers.')

    def test_falls_back_when_llm_fails_or_is_wrong(self):
        cases = {
            'unavailable': (mock.Mock(side_effect=ConnectionError('refused')), 'the LLM was unavailable (ConnectionError)'),
            'wrong': (
                mock.Mock(return_value=FakeListChatModel(responses=['Test accuracy fell from 0.754 to 0.77.'])),
                'the LLM explanation failed the fact check: it says test accuracy went down',
            ),
        }
        for case, (get_llm, reason) in cases.items():
            with self.subTest(case), mock.patch('experiments.comparison.get_llm', get_llm):
                response = self.client.post(self.url, {'a': 1, 'b': 3})
                explanation = response.context['explanation']
                self.assertFalse(explanation['used_llm'])
                self.assertTrue(explanation['reason'].startswith(reason))
                self.assertTrue(explanation['text'].startswith('v3 changes n_estimators from 100 to 200'))
                self.assertContains(response, 'Written from the numbers, because')

    def test_requires_post_and_csrf(self):
        with mock.patch('experiments.comparison.get_llm', side_effect=no_llm):
            self.assertEqual(self.client.get(self.url, {'a': 1, 'b': 3}).status_code, 405)
            self.assertEqual(Client(enforce_csrf_checks=True).post(self.url, {'a': 1, 'b': 3}).status_code, 403)
            response = self.client.get(reverse('experiments:compare', args=['heart-disease-rf']))
        self.assertContains(response, f'action="{self.url}"')
        self.assertContains(response, 'csrfmiddlewaretoken')


class ExplanationTests(ExperimentPagesTestCase):
    def setUp(self):
        super().setUp()
        self.v1, self.v2, self.v3 = self.runs
        self.comparison = compare_runs(self.v1, self.v3)
        self.details = describe_comparison(self.v1, self.v3, self.comparison)

    def check(self, text, a=None, b=None):
        a, b = a or self.v1, b or self.v3
        comparison = compare_runs(a, b)
        check_explanation(text, describe_comparison(a, b, comparison), comparison, a, b)

    def test_details_given_to_the_llm(self):
        self.assertIn('- n_estimators: 100 → 200 (+100)', self.details)
        self.assertIn('- test accuracy: 0.754 → 0.77 (+0.016, higher)', self.details)
        self.assertIn('- test f1: 0.776 → 0.788 (+0.012, higher)', self.details)
        self.assertIn('Train minus test accuracy gap: 0.246 → 0.18 (-0.066, smaller, less overfitting).', self.details)

    def test_fallback_explanation(self):
        self.assertEqual(
            fallback_explanation(self.v1, self.v3, self.comparison),
            'v3 changes n_estimators from 100 to 200 and max_depth from None to 5 compared with v1. '
            'Test accuracy rose from 0.754 to 0.77 (+0.016); test f1 rose from 0.776 to 0.788 (+0.012). '
            'The gap between train and test accuracy narrowed from 0.246 to 0.18, so v3 overfits less.',
        )
        self.assertEqual(
            fallback_explanation(self.v2, self.v3, compare_runs(self.v2, self.v3)).split('. ')[1],
            'Test accuracy stayed at 0.77; test f1 stayed at 0.788',
        )

    def test_accepts_correct_explanations(self):
        accepted = [
            ExplainViewTests.GOOD,
            'Going from v1 to v3, n_estimators went 100→200 and test accuracy improved by 1.6 percentage points.',
            'The model now overfits less: the gap between train and test accuracy decreased from 0.246 to 0.18.',
            'Training accuracy dropped to 0.95, but testing accuracy increased.',
        ]
        for text in accepted:
            with self.subTest(text):
                self.check(text)
        # v2 → v3: test metrics unchanged, train metrics lower.
        self.check('Test accuracy stayed the same at 0.77, while train accuracy decreased.', self.v2, self.v3)

    def test_rejects_wrong_explanations(self):
        rejected = {
            'Test accuracy rose to 0.80.': 'numbers that are not in the data: 0.80',
            'Test accuracy rose by 7%.': 'numbers that are not in the data: 7%',
            'n_estimators went from 200 to 100.': 'a change 200→100 that is not in the data',
            'Test accuracy decreased.': 'it says test accuracy went down',
            'Train accuracy increased.': 'it says train accuracy went up',
            'Test F1 stayed the same.': 'it says test f1 stayed the same, but it changed',
            'The model overfits more now.': 'overfitting increased, but the train-test gap did not grow',
        }
        for text, reason in rejected.items():
            with self.subTest(text), self.assertRaisesMessage(ExplanationCheckError, reason):
                self.check(text)

    def test_same_version_skips_llm(self):
        llm = RunnableLambda(no_llm)
        self.assertEqual(
            explain_difference(self.v2, self.v2, llm=llm),
            ('v2 and v2 are the same run.', False, 'both sides are the same run'),
        )
