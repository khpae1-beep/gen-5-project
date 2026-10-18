import io
import json
import random
import tempfile
from pathlib import Path
from unittest import mock

from django.core.management import CommandError, call_command
from django.test import TestCase
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.runnables import RunnableLambda

from experiments import training
from experiments.models import Run
from experiments.tests import write_heart_csv
from tuning.engine import run_job
from tuning.llm_proposer import propose_next
from tuning.models import Trial, TuningJob
from tuning.scoring import make_split
from tuning.search_space import (
    HEART_DISEASE_RF_SPACE as SPACE, InvalidParams, params_key, sample_params, sample_unique_params, validate_params,
)

VALID = {'n_estimators': 200, 'max_depth': 8, 'min_samples_split': 4, 'min_samples_leaf': 2, 'max_features': 'log2'}


def reply(params, reason='More trees and a depth limit should generalise better.'):
    return json.dumps({'params': params, 'reason': reason})


class SearchSpaceTests(TestCase):
    def test_accepts_and_normalises_values_in_the_space(self):
        self.assertEqual(validate_params(VALID, SPACE), VALID)
        self.assertEqual(
            validate_params({**VALID, 'n_estimators': 300.0, 'max_depth': 'None', 'max_features': None}, SPACE),
            {**VALID, 'n_estimators': 300, 'max_depth': None, 'max_features': None},
        )
        edges = {'n_estimators': 50, 'max_depth': 20, 'min_samples_split': 2, 'min_samples_leaf': 10, 'max_features': 'sqrt'}
        self.assertEqual(validate_params(edges, SPACE), edges)

    def test_rejects_values_outside_the_space(self):
        cases = {
            'n_estimators=501 is outside 50–500': {**VALID, 'n_estimators': 501},
            'max_depth=1 is outside 2–20': {**VALID, 'max_depth': 1},
            'min_samples_leaf=0 is outside 1–10': {**VALID, 'min_samples_leaf': 0},
            'min_samples_split cannot be None': {**VALID, 'min_samples_split': None},
            "max_features='auto' is not one of": {**VALID, 'max_features': 'auto'},
            'n_estimators must be a whole number': {**VALID, 'n_estimators': 120.5},
            "must be a whole number, got '200'": {**VALID, 'n_estimators': '200'},
            'got True': {**VALID, 'min_samples_leaf': True},
            'missing max_features': {k: v for k, v in VALID.items() if k != 'max_features'},
            'unknown criterion': {**VALID, 'criterion': 'gini'},
        }
        for message, params in cases.items():
            with self.subTest(message), self.assertRaisesMessage(InvalidParams, message):
                validate_params(params, SPACE)

    def test_random_samples_are_valid_and_reproducible(self):
        first = [sample_params(SPACE, rng) for rng in [random.Random(7)] for _ in range(50)]
        again = [sample_params(SPACE, rng) for rng in [random.Random(7)] for _ in range(50)]
        other = [sample_params(SPACE, rng) for rng in [random.Random(8)] for _ in range(50)]
        self.assertEqual(first, again)
        self.assertNotEqual(first, other)
        for params in first:
            self.assertEqual(validate_params(params, SPACE), params)

    def test_unique_samples_skip_tried_params(self):
        tried = sample_params(SPACE, random.Random(1))
        params = sample_unique_params(SPACE, random.Random(1), {params_key(tried)})
        self.assertNotEqual(params, tried)


class ProposerTests(TestCase):
    HISTORY = [
        {'number': 1, 'params': VALID, 'cv_score': 0.85, 'cv_std': 0.04},
        {'number': 2, 'params': {**VALID, 'n_estimators': 300}, 'cv_score': 0.861, 'cv_std': 0.03},
    ]
    NEW = {**VALID, 'n_estimators': 400, 'max_depth': None}

    def propose(self, llm, history=HISTORY, seed=42):
        return propose_next(SPACE, history, random.Random(seed), llm=llm)

    def test_valid_proposal(self):
        text = f'Here is my suggestion:\n```json\n{reply(self.NEW)}\n```'
        proposal = self.propose(FakeListChatModel(responses=[text]))

        self.assertEqual(proposal.proposed_by, Trial.PROPOSED_BY_LLM)
        self.assertEqual(proposal.params, self.NEW)
        self.assertEqual(proposal.reasoning, 'More trees and a depth limit should generalise better.')

    def test_prompt_has_space_and_cv_history_but_not_test_accuracy(self):
        prompts = []

        def fake_llm(prompt):
            prompts.append(prompt.to_string())
            return reply(self.NEW)

        self.propose(RunnableLambda(fake_llm))
        prompt = prompts[0]
        self.assertIn('- n_estimators: an integer from 50 to 500', prompt)
        self.assertIn('- max_depth: an integer from 2 to 20, or null (no limit)', prompt)
        self.assertIn('- max_features: one of "sqrt", "log2", null', prompt)
        self.assertIn(f'Trial 2: {json.dumps(self.HISTORY[1]["params"])} → CV accuracy 0.8610 ± 0.0300', prompt)
        self.assertIn('Best so far: trial 2 with CV accuracy 0.8610.', prompt)
        self.assertNotIn('test', prompt.lower())

    def test_uses_get_llm_with_temperature_zero(self):
        with mock.patch('tuning.llm_proposer.get_llm', return_value=FakeListChatModel(responses=[reply(self.NEW)])) as get_llm:
            proposal = propose_next(SPACE, self.HISTORY, random.Random(42))
        get_llm.assert_called_once_with(temperature=0)
        self.assertEqual(proposal.proposed_by, Trial.PROPOSED_BY_LLM)

    def test_falls_back_to_random_for_unusable_replies(self):
        cases = {
            'out of range': (reply({**VALID, 'n_estimators': 900}), "the LLM's params are invalid (n_estimators=900 is outside 50–500)"),
            'bad choice': (reply({**VALID, 'max_features': 'auto'}), "the LLM's params are invalid (max_features='auto'"),
            'missing param': (reply({'n_estimators': 200}), "the LLM's params are invalid (missing max_depth"),
            'not json': ('I would try more trees.', 'the LLM reply had no JSON object'),
            'broken json': ('{"params": {"n_estimators": 200,}', 'the LLM reply was not valid JSON'),
            'no params object': ('{"reason": "more trees"}', 'the LLM reply had no "params" object'),
            'duplicate': (reply(self.HISTORY[0]['params']), 'the LLM repeated the params of trial 1'),
        }
        for case, (text, why) in cases.items():
            with self.subTest(case):
                proposal = self.propose(FakeListChatModel(responses=[text]))
                self.assertEqual(proposal.proposed_by, Trial.PROPOSED_BY_FALLBACK)
                self.assertTrue(proposal.reasoning.startswith(f'Random fallback: {why}'), proposal.reasoning)
                self.assertEqual(validate_params(proposal.params, SPACE), proposal.params)
                self.assertNotIn(params_key(proposal.params), {params_key(t['params']) for t in self.HISTORY})

    def test_falls_back_when_llm_unavailable(self):
        def unavailable(_):
            raise ConnectionError('Ollama is not running')

        proposal = self.propose(RunnableLambda(unavailable))
        self.assertEqual(proposal.proposed_by, Trial.PROPOSED_BY_FALLBACK)
        self.assertEqual(proposal.reasoning, 'Random fallback: the LLM was unavailable (ConnectionError).')
        # The fallback is reproducible with the same seed.
        self.assertEqual(self.propose(RunnableLambda(unavailable)).params, proposal.params)

    def test_accepts_params_at_top_level_and_missing_reason(self):
        proposal = self.propose(FakeListChatModel(responses=[json.dumps(self.NEW)]))
        self.assertEqual((proposal.params, proposal.reasoning), (self.NEW, 'The LLM gave no reason.'))


class TuningTestCase(TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.tmp = Path(tmp.name)
        self.data = self.tmp / 'heart.csv'
        write_heart_csv(self.data)
        patcher = mock.patch.object(training, 'RUNS_DIR', self.tmp / 'runs')
        patcher.start()
        self.addCleanup(patcher.stop)

    def make_job(self, strategy='random', n_trials=3, seed=42):
        return TuningJob.objects.create(
            strategy=strategy, n_trials=n_trials, cv_folds=3, search_space=SPACE, seed=seed,
            experiment_name=f'heart-disease-rf-tuning-{strategy}',
        )


class RandomBaselineTests(TuningTestCase):
    def test_same_seed_gives_the_same_trials(self):
        split = make_split(*training.load_data(self.data))

        def trials(seed):
            job = run_job(self.make_job(seed=seed), split, self.data)
            return [(t.params, t.cv_score) for t in job.trials.all()]

        first, again, other = trials(42), trials(42), trials(7)
        self.assertEqual(first, again)
        self.assertNotEqual([p for p, _ in first], [p for p, _ in other])
        self.assertEqual(len({params_key(p) for p, _ in first}), 3)


class RunTuningCommandTests(TuningTestCase):
    def run_tuning(self, *args):
        out = io.StringIO()
        call_command('run_tuning', '--data', str(self.data), '--cv', '3', *args, stdout=out)
        return out.getvalue()

    def test_random_strategy_end_to_end(self):
        output = self.run_tuning('--strategy', 'random', '--trials', '3')

        job = TuningJob.objects.get()
        trials = list(job.trials.all())
        self.assertEqual(job.status, TuningJob.STATUS_COMPLETED)
        self.assertEqual((job.strategy, job.n_trials, job.cv_folds, job.seed), ('random', 3, 3, 42))
        self.assertEqual(job.search_space, SPACE)
        self.assertEqual([t.number for t in trials], [1, 2, 3])
        self.assertTrue(all(t.proposed_by == Trial.PROPOSED_BY_RANDOM for t in trials))

        # The best trial is chosen by CV accuracy only (the earliest wins ties), never test accuracy.
        best = max(trials, key=lambda t: (t.cv_score, -t.number))
        self.assertEqual(job.best_trial, best)
        self.assertEqual(job.best_cv_score, best.cv_score)

        # Every trial is a run of the tuning experiment, with a promotable checkpoint.
        runs = list(Run.objects.filter(experiment__name='heart-disease-rf-tuning-random').order_by('version'))
        self.assertEqual([t.run for t in trials], runs)
        for trial, run in zip(trials, runs):
            self.assertEqual(run.params, {**trial.params, 'test_size': 0.2, 'random_state': 42, 'cv_folds': 3})
            self.assertEqual(run.metrics['cv_accuracy'], trial.cv_score)
            self.assertEqual(run.metrics['test_accuracy'], trial.test_accuracy)
            self.assertTrue(Path(run.checkpoint_path).exists())
            self.assertIn(f'trial {trial.number} (random): CV accuracy', run.version_note)

        self.assertIn('Trial 1/3 [random]', output)
        self.assertIn('Random search: sampled uniformly from the search space.', output)
        self.assertIn(f'Best: trial {best.number} of 3 (found at trial {best.number})', output)
        self.assertIn(f'promote_run heart-disease-rf-tuning-random {best.run.version}', output)

    def test_llm_strategy_end_to_end_with_fake_llm(self):
        replies = [
            reply(VALID),                               # Trial 1: valid.
            reply({**VALID, 'n_estimators': 9999}),     # Trial 2: out of range.
            reply(VALID),                               # Trial 3: repeats trial 1.
            reply({**VALID, 'max_features': None}),     # Trial 4: valid.
        ]
        with mock.patch('tuning.llm_proposer.get_llm', return_value=FakeListChatModel(responses=replies)):
            output = self.run_tuning('--strategy', 'llm', '--trials', '4')

        job = TuningJob.objects.get()
        trials = list(job.trials.all())
        self.assertEqual(job.status, TuningJob.STATUS_COMPLETED)
        self.assertEqual(job.experiment_name, 'heart-disease-rf-tuning-llm')
        self.assertEqual(
            [t.proposed_by for t in trials],
            ['llm', 'random-fallback', 'random-fallback', 'llm'],
        )
        self.assertEqual(trials[0].params, VALID)
        self.assertEqual(trials[3].params, {**VALID, 'max_features': None})
        self.assertIn('n_estimators=9999 is outside 50–500', trials[1].reasoning)
        self.assertIn('repeated the params of trial 1', trials[2].reasoning)
        self.assertEqual(len({params_key(t.params) for t in trials}), 4)
        self.assertEqual(Run.objects.filter(experiment__name='heart-disease-rf-tuning-llm').count(), 4)
        self.assertIn('2 trials proposed by the LLM, 2 random fallbacks.', output)
        self.assertIn('Random fallback:', output)

    def test_errors(self):
        cases = {
            '--trials must be at least 1': ['--trials', '0'],
            '--cv must be at least 2': ['--cv', '1'],
            'not found': ['--data', str(self.tmp / 'missing.csv')],
        }
        for message, args in cases.items():
            with self.subTest(message), self.assertRaisesMessage(CommandError, message):
                call_command('run_tuning', '--strategy', 'random', '--data', str(self.data), *args, stdout=io.StringIO())
        self.assertFalse(TuningJob.objects.exists())

    def test_failed_job_is_recorded(self):
        # 200 folds cannot be made from 96 training rows.
        with self.assertRaisesMessage(CommandError, 'Tuning job'):
            call_command('run_tuning', '--strategy', 'random', '--data', str(self.data), '--cv', '200', stdout=io.StringIO())
        job = TuningJob.objects.get()
        self.assertEqual(job.status, TuningJob.STATUS_FAILED)
        self.assertIn('ValueError', job.error)
        self.assertFalse(job.trials.exists())
