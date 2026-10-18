import json
import tempfile
from pathlib import Path
from unittest import mock

from django.contrib.auth import get_user_model
from django.test import Client, TestCase
from django.urls import reverse
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.runnables import RunnableLambda

from experiments import training
from experiments.models import Run
from experiments.tests import write_heart_csv
from tuning.engine import run_job
from tuning.models import Trial, TuningJob
from tuning.scoring import make_split
from tuning.search_space import HEART_DISEASE_RF_SPACE as SPACE


def no_llm(*args, **kwargs):
    raise AssertionError('The LLM must not be called on a normal page load.')


def suggestion(params, reason):
    return {'params': params, 'reason': reason}


class TuningPagesTestCase(TestCase):
    """A real random job (3 trials) on a small generated CSV, plus helpers."""

    @classmethod
    def setUpClass(cls):
        cls.tmp_dir = tempfile.TemporaryDirectory()
        cls.tmp = Path(cls.tmp_dir.name)
        cls.data = cls.tmp / 'heart.csv'
        write_heart_csv(cls.data)
        cls.runs_patch = mock.patch.object(training, 'RUNS_DIR', cls.tmp / 'runs')
        cls.runs_patch.start()
        super().setUpClass()

    @classmethod
    def tearDownClass(cls):
        super().tearDownClass()
        cls.runs_patch.stop()
        cls.tmp_dir.cleanup()

    @classmethod
    def setUpTestData(cls):
        cls.job = cls.make_job('random', seed=42)
        cls.trials = list(cls.job.trials.order_by('number'))
        cls.best = cls.job.best_trial

    @classmethod
    def make_job(cls, strategy, seed=42, n_trials=3, llm=None):
        job = TuningJob.objects.create(
            strategy=strategy, n_trials=n_trials, cv_folds=3, search_space=SPACE, seed=seed,
            experiment_name=f'heart-disease-rf-tuning-{strategy}', data_path=str(cls.data), test_size=0.2,
        )
        split = make_split(*training.load_data(cls.data), 0.2, seed)
        return run_job(job, split, cls.data, llm=llm)

    def make_user(self, username='staff', is_staff=True):
        return get_user_model().objects.create_user(username, password='pw', is_staff=is_staff)

    def login_staff(self):
        self.client.force_login(self.make_user())

    def untried(self, **changes):
        """Params near the best trial's that no trial has used."""
        tried = [t.params for t in Trial.objects.filter(job=self.job)]
        for n_estimators in range(50, 501):
            params = {**self.best.params, **changes, 'n_estimators': n_estimators}
            if params not in tried:
                return params
        raise AssertionError('No untried params found.')


@mock.patch('tuning.suggestions.get_llm', side_effect=no_llm)
class JobListTests(TuningPagesTestCase):
    url = reverse('tuning:list')

    def test_lists_jobs(self, get_llm):
        with self.assertNumQueries(1):
            response = self.client.get(self.url)

        self.assertEqual(response.status_code, 200)
        self.assertTemplateUsed(response, 'base.html')
        job = response.context['jobs'][0]
        self.assertEqual((job.pk, job.trial_count), (self.job.pk, 3))
        best = self.best
        for text in (f'Job {self.job.pk}', 'Random search', f'{best.cv_score:.4f} ± {best.cv_std:.4f}',
                     f'{best.test_accuracy:.4f}', 'Completed', reverse('tuning:detail', args=[self.job.pk])):
            self.assertContains(response, text)
        self.assertContains(response, f'<td class="num">{best.number}</td>', html=True)

    def test_empty(self, get_llm):
        TuningJob.objects.all().delete()
        self.assertContains(self.client.get(self.url), 'No tuning jobs yet')


@mock.patch('tuning.suggestions.get_llm', side_effect=no_llm)
class JobDetailTests(TuningPagesTestCase):
    def url(self, pk=None):
        return reverse('tuning:detail', args=[pk or self.job.pk])

    def test_shows_chart_and_trials(self, get_llm):
        with self.assertNumQueries(2):
            response = self.client.get(self.url())

        self.assertEqual(response.status_code, 200)
        data = response.context['chart_data']
        scores = [t.cv_score for t in self.trials]
        self.assertEqual(data['labels'], [1, 2, 3])
        self.assertEqual(data['cv'], scores)
        self.assertEqual(data['best_so_far'], [max(scores[:i + 1]) for i in range(3)])
        self.assertEqual(data['best'], [t.pk == self.best.pk for t in self.trials])
        self.assertContains(response, 'id="cv-chart"')
        self.assertContains(response, f'<tr class="best" id="trial-{self.best.number}">', html=False)
        for trial in self.trials:
            self.assertContains(response, reverse('experiments:run', args=[trial.run.experiment.name, trial.run.version]))
        self.assertContains(response, 'Random search: sampled uniformly from the search space.')
        self.assertContains(response, reverse('experiments:detail', args=['heart-disease-rf-tuning-random']))
        self.assertIsNotNone(response.context['time_taken'])

    def test_non_staff_see_disabled_suggest_button(self, get_llm):
        for client in (Client(), self.client):
            if client is self.client:
                client.force_login(self.make_user('viewer', is_staff=False))
            response = client.get(self.url())
            self.assertContains(response, '<button type="button" disabled>Suggest next experiments</button>', html=True)
            self.assertContains(response, f'/admin/login/?next={self.url()}')
            self.assertNotContains(response, reverse('tuning:suggest', args=[self.job.pk]))

    def test_staff_see_suggest_form(self, get_llm):
        self.login_staff()
        response = self.client.get(self.url())
        self.assertContains(response, f'action="{reverse("tuning:suggest", args=[self.job.pk])}#suggestions"')
        self.assertContains(response, 'csrfmiddlewaretoken')

    def test_unknown_job(self, get_llm):
        self.assertEqual(self.client.get(self.url(9999)).status_code, 404)


@mock.patch('tuning.suggestions.get_llm', side_effect=no_llm)
class CompareTests(TuningPagesTestCase):
    url = reverse('tuning:compare')

    def make_llm_job(self, seed=42):
        replies = [json.dumps(suggestion(self.untried(max_depth=3), 'Use a shallower max_depth.'))]

        def llm(prompt):
            return replies[0] if 'Trial 1' not in prompt.to_string() else 'not json'

        return self.make_job('llm', seed=seed, llm=RunnableLambda(llm))

    def test_compares_latest_llm_and_random_jobs(self, get_llm):
        old_llm = self.make_llm_job(seed=1)
        llm_job = self.make_llm_job(seed=2)
        # An older job with no recorded times or fact checks, as before they were recorded.
        TuningJob.objects.filter(pk=old_llm.pk).update(finished_at=None)
        Trial.objects.filter(job=old_llm).update(reason_check='')

        with self.assertNumQueries(2):
            response = self.client.get(self.url)

        self.assertEqual((response.context['llm_job'], response.context['random_job']), (llm_job, self.job))
        llm_summary, random_summary = response.context['summaries']
        # Trial 1 came from the LLM (its "shallower" reason has no previous trial, so it failed the
        # check); trials 2 and 3 were fallbacks because the LLM replied "not json".
        self.assertEqual(llm_summary['fallbacks'], 2)
        self.assertEqual((llm_summary['llm_reasons'], llm_summary['reasons_passed']), (1, 0))
        self.assertEqual(random_summary['best'], self.best)
        self.assertEqual(random_summary['best'].number, self.best.number)
        self.assertIsNotNone(llm_summary['time_taken'])
        data = response.context['chart_data']
        self.assertEqual(data['labels'], [1, 2, 3])
        self.assertEqual(data['random'], [max(t.cv_score for t in self.trials[:i + 1]) for i in range(3)])
        self.assertContains(response, 'id="compare-chart"')
        self.assertContains(response, '0 of 1')

        response = self.client.get(self.url, {'llm': old_llm.pk, 'random': self.job.pk})
        self.assertEqual(response.context['llm_job'], old_llm)
        self.assertContains(response, 'not recorded')
        self.assertContains(response, '(1 not recorded)')

    def test_needs_both_kinds_of_job(self, get_llm):
        self.assertContains(self.client.get(self.url), 'Comparing needs at least one LLM job and one random job')


class SuggestTests(TuningPagesTestCase):
    def url(self):
        return reverse('tuning:suggest', args=[self.job.pk])

    def detail_url(self):
        return reverse('tuning:detail', args=[self.job.pk])

    def reply(self):
        best = self.best.params
        good = self.untried(min_samples_leaf=best['min_samples_leaf'])
        good['n_estimators'] = 500 if best['n_estimators'] < 500 else 50
        direction = 'more' if good['n_estimators'] > best['n_estimators'] else 'fewer'
        wrong = self.untried(max_features='log2' if best['max_features'] != 'log2' else 'sqrt')
        wrong_direction = 'larger' if wrong['n_estimators'] < best['n_estimators'] else 'smaller'
        self.good, self.wrong = good, wrong
        self.good_reason = f'Use {direction} trees than trial {self.best.number}.'
        return json.dumps({'suggestions': [
            suggestion(good, self.good_reason),
            suggestion({**good, 'n_estimators': 9999}, 'Far more trees.'),
            suggestion(self.trials[0].params, 'Repeat trial 1.'),
            suggestion(good, 'The same again.'),
            suggestion(wrong, f'A {wrong_direction} n_estimators.'),
        ]})

    def test_saves_validated_checked_suggestions_and_redirects(self):
        self.login_staff()
        llm = FakeListChatModel(responses=[self.reply()])
        with mock.patch('tuning.suggestions.get_llm', return_value=llm) as get_llm:
            response = self.client.post(self.url())

        get_llm.assert_called_once_with(temperature=0)
        self.assertRedirects(response, self.detail_url() + '#suggestions', fetch_redirect_response=False)
        self.job.refresh_from_db()
        saved = self.job.suggestions
        configs = saved['configs']
        self.assertEqual([c['params'] for c in configs], [self.good, self.wrong])
        self.assertEqual([c['reason_check'] for c in configs], [Trial.REASON_PASSED, Trial.REASON_FAILED])
        self.assertEqual([c['tried_trial'] for c in configs], [None, None])
        self.assertEqual((configs[0]['reasoning'], configs[0]['llm_reason']), (self.good_reason, self.good_reason))
        self.assertIn(f"compared with the best trial (trial {self.best.number}). The LLM's reason failed the fact check",
                      configs[1]['reasoning'])
        self.assertEqual(saved['rejected'], [
            'Suggestion 2 was dropped: n_estimators=9999 is outside 50–500.',
            'Suggestion 3 was dropped: it repeats trial 1.',
            'Suggestion 4 was dropped: it repeats an earlier suggestion.',
        ])
        self.assertEqual(saved['error'], '')

        # Reloading the page shows the saved suggestions without calling the LLM again.
        with mock.patch('tuning.suggestions.get_llm', side_effect=no_llm) as get_llm:
            for _ in range(2):
                page = self.client.get(self.detail_url())
                self.assertContains(page, 'Try this', count=2)
                self.assertContains(page, self.good_reason)
                self.assertContains(page, 'it repeats trial 1')
                self.assertContains(page, '<input type="hidden" name="index" value="1">', html=True)
        get_llm.assert_not_called()

    def test_non_staff_see_saved_suggestions_without_buttons(self):
        self.login_staff()
        with mock.patch('tuning.suggestions.get_llm', return_value=FakeListChatModel(responses=[self.reply()])):
            self.client.post(self.url())
        page = Client().get(self.detail_url())
        self.assertContains(page, self.good_reason)
        self.assertNotContains(page, 'Try this')

    def test_llm_unavailable(self):
        self.login_staff()
        with mock.patch('tuning.suggestions.get_llm', return_value=RunnableLambda(no_llm)):
            response = self.client.post(self.url(), follow=True)
        self.assertContains(response, 'The LLM was unavailable (AssertionError).')
        self.assertNotContains(response, 'Try this')
        self.job.refresh_from_db()
        self.assertEqual(self.job.suggestions['configs'], [])

    def test_staff_only_post_only_and_csrf(self):
        with mock.patch('tuning.suggestions.get_llm', side_effect=no_llm) as get_llm:
            for client in (Client(), self.client):
                if client is self.client:
                    client.force_login(self.make_user('viewer', is_staff=False))
                response = client.post(self.url())
                self.assertRedirects(response, f'/admin/login/?next={self.url()}', fetch_redirect_response=False)
            staff = Client(enforce_csrf_checks=True)
            staff.force_login(self.make_user())
            self.assertEqual(staff.post(self.url()).status_code, 403)
            self.assertEqual(staff.get(self.url()).status_code, 405)
        get_llm.assert_not_called()
        self.job.refresh_from_db()
        self.assertEqual(self.job.suggestions, {})


@mock.patch('tuning.suggestions.get_llm', side_effect=no_llm)
class TryTests(TuningPagesTestCase):
    def url(self):
        return reverse('tuning:try', args=[self.job.pk])

    def save_suggestions(self, *configs):
        """Save (params, llm_reason) pairs on the job as if "Suggest next experiments" had run."""
        self.job.suggestions = {
            'created_at': '2026-10-18T12:00:00+00:00',
            'configs': [
                {'params': params, 'reasoning': reason, 'reason_check': Trial.REASON_PASSED,
                 'llm_reason': reason, 'tried_trial': None}
                for params, reason in configs
            ],
            'rejected': [],
            'error': '',
        }
        self.job.save(update_fields=['suggestions'])

    def try_index(self, index, client=None):
        return (client or self.client).post(self.url(), {'index': index}, follow=True)

    def test_runs_one_trial_and_adds_it_to_the_job(self, get_llm):
        self.login_staff()
        params = self.untried()
        direction = 'more' if params['n_estimators'] > self.best.params['n_estimators'] else 'fewer'
        self.save_suggestions((self.untried(max_features=None), 'Another idea.'), (params, f'Use {direction} trees.'))

        response = self.try_index(1)

        trial = Trial.objects.get(job=self.job, number=4)
        self.assertRedirects(response, reverse('tuning:detail', args=[self.job.pk]) + '#trial-4')
        self.assertEqual((trial.params, trial.proposed_by), (params, Trial.PROPOSED_BY_SUGGESTION))
        self.assertEqual((trial.reasoning, trial.reason_check), (f'Use {direction} trees.', Trial.REASON_PASSED))
        self.assertEqual(trial.run.experiment.name, 'heart-disease-rf-tuning-random')
        self.assertEqual(trial.run.data_hash, self.trials[0].run.data_hash)
        self.assertEqual(trial.run.params['random_state'], 42)
        self.assertTrue(Path(trial.run.checkpoint_path).exists())
        self.assertContains(response, f'Trial 4: CV accuracy {trial.cv_score:.4f}')
        self.job.refresh_from_db()
        best = max(Trial.objects.filter(job=self.job), key=lambda t: (t.cv_score, -t.number))
        self.assertEqual(self.job.best_trial, best)

        # The suggestion is marked as tried and loses its button; the other keeps its button.
        self.assertEqual([c['tried_trial'] for c in self.job.suggestions['configs']], [None, 4])
        self.assertContains(response, 'Tried as <a href="#trial-4">trial 4</a>.', html=True)
        self.assertContains(response, 'Try this', count=1)

    def test_wrong_reason_is_replaced(self, get_llm):
        self.login_staff()
        params = self.untried()
        wrong = 'fewer' if params['n_estimators'] > self.best.params['n_estimators'] else 'more'
        self.save_suggestions((params, f'Use {wrong} trees.'))
        self.try_index(0)
        trial = Trial.objects.get(job=self.job, number=4)
        self.assertEqual(trial.reason_check, Trial.REASON_FAILED)
        self.assertIn(f'compared with the best trial (trial {self.best.number})', trial.reasoning)

    def test_refuses_bad_indexes_and_params(self, get_llm):
        self.login_staff()
        self.save_suggestions(
            (self.trials[0].params, 'Repeat trial 1.'),
            ({**self.best.params, 'n_estimators': 9999}, 'Edited in the database.'),
        )
        cases = {
            0: 'These params have already been tried in this job.',
            1: 'invalid params (n_estimators=9999 is outside 50–500)',
            2: 'it is no longer saved on this job',
            -1: 'it is no longer saved on this job',
            'x': 'it is no longer saved on this job',
        }
        for index, message in cases.items():
            with self.subTest(index=index):
                self.assertContains(self.try_index(index), message)
        self.assertEqual(Trial.objects.filter(job=self.job).count(), 3)

    def test_refuses_changed_data(self, get_llm):
        self.login_staff()
        self.save_suggestions((self.untried(), 'Use more trees.'))
        original = self.data.read_bytes()
        self.addCleanup(self.data.write_bytes, original)
        self.data.write_bytes(original + b'50,200.0,150,1\n')
        self.assertContains(self.try_index(0), 'has changed since this job ran')
        self.assertEqual(Trial.objects.filter(job=self.job).count(), 3)

    def test_staff_only_post_only_and_csrf(self, get_llm):
        self.save_suggestions((self.untried(), 'Use more trees.'))
        non_staff = Client()
        non_staff.force_login(self.make_user('viewer', is_staff=False))
        for client in (Client(), non_staff):
            response = client.post(self.url(), {'index': 0})
            self.assertRedirects(response, f'/admin/login/?next={self.url()}', fetch_redirect_response=False)
        staff = Client(enforce_csrf_checks=True)
        staff.force_login(self.make_user())
        self.assertEqual(staff.post(self.url(), {'index': 0}).status_code, 403)
        self.assertEqual(staff.get(self.url()).status_code, 405)
        self.assertEqual(Trial.objects.filter(job=self.job).count(), 3)
        self.assertFalse(Run.objects.filter(version=4, experiment__name='heart-disease-rf-tuning-random').exists())
