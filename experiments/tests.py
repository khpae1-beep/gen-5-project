import hashlib
import io
import json
import subprocess
import tempfile
from pathlib import Path
from unittest import mock

from django.core.management import CommandError, call_command
from django.test import TestCase
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.runnables import RunnableLambda

from experiments import training
from experiments import promotion
from experiments.models import Experiment, Run
from experiments.tracking import current_git_commit, log_run
from experiments.versioning import fallback_note, format_value, write_version_note


class TempDirTestCase(TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.tmp = Path(tmp.name)


class LogRunTests(TempDirTestCase):
    def test_first_run_is_version_one_without_parent_or_changes(self):
        run = log_run('exp', {'n_estimators': 100}, {'test_accuracy': 0.75}, checkpoint_path='models/runs/exp-v1.joblib')

        self.assertEqual(run.version, 1)
        self.assertIsNone(run.parent)
        self.assertEqual(run.changes, {})
        self.assertEqual(run.checkpoint_path, 'models/runs/exp-v1.joblib')
        self.assertEqual(Experiment.objects.get().name, 'exp')

    def test_next_run_links_parent_and_records_changes(self):
        first = log_run('exp', {'n_estimators': 100, 'max_depth': None, 'seed': 42}, {'test_accuracy': 0.754})
        second = log_run('exp', {'n_estimators': 300, 'max_depth': 8, 'seed': 42}, {'test_accuracy': 0.787})

        self.assertEqual(second.version, 2)
        self.assertEqual(second.parent, first)
        self.assertEqual(second.changes, {
            'params': {
                'n_estimators': {'old': 100, 'new': 300},
                'max_depth': {'old': None, 'new': 8},
            },
            'metrics': {'test_accuracy': {'old': 0.754, 'new': 0.787}},
        })
        self.assertEqual(Experiment.objects.count(), 1)

    def test_changes_include_added_and_removed_keys(self):
        log_run('exp', {'a': 1, 'gone': 'x'}, {'m': 0.5})
        run = log_run('exp', {'a': 1, 'added': None}, {'m': 0.5, 'f1': 0.7})

        self.assertEqual(run.changes['params'], {
            'gone': {'old': 'x', 'new': None},
            'added': {'old': None, 'new': None},
        })
        self.assertEqual(run.changes['metrics'], {'f1': {'old': None, 'new': 0.7}})

    def test_versions_are_numbered_per_experiment(self):
        log_run('a', {}, {})
        log_run('a', {}, {})
        other = log_run('b', {}, {})
        third = log_run('a', {}, {})

        self.assertEqual((other.version, other.parent), (1, None))
        self.assertEqual(third.version, 3)
        self.assertEqual(third.parent.version, 2)

    def test_hashes_data_file(self):
        data = self.tmp / 'heart.csv'
        data.write_bytes(b'age,target\n50,1\n')

        run = log_run('exp', {}, {}, data_path=data)

        self.assertEqual(run.data_hash, hashlib.sha256(b'age,target\n50,1\n').hexdigest())
        self.assertEqual(log_run('exp', {}, {}).data_hash, '')

    def test_records_git_commit(self):
        result = subprocess.CompletedProcess([], 0, stdout='abc123\n')
        with mock.patch('experiments.tracking.subprocess.run', return_value=result):
            self.assertEqual(log_run('exp', {}, {}).git_commit, 'abc123')

    def test_git_commit_is_blank_without_git(self):
        failures = [FileNotFoundError('git'), subprocess.CalledProcessError(128, 'git'), subprocess.TimeoutExpired('git', 5)]
        for failure in failures:
            with self.subTest(type(failure).__name__):
                with mock.patch('experiments.tracking.subprocess.run', side_effect=failure):
                    self.assertEqual(current_git_commit(), '')


def failing_llm(_):
    raise ConnectionError('Ollama is not running')


class VersionNoteTests(TestCase):
    def setUp(self):
        log_run('exp', {'n_estimators': 100, 'max_depth': None}, {'train_accuracy': 1.0, 'test_accuracy': 0.7540983})
        self.run = log_run('exp', {'n_estimators': 300, 'max_depth': 8}, {'train_accuracy': 1.0, 'test_accuracy': 0.787})
        self.expected = 'v2: n_estimators 100→300, max_depth None→8; test accuracy 0.754→0.787'

    def test_saves_llm_note(self):
        llm = FakeListChatModel(responses=[self.expected])

        note, used_llm = write_version_note(self.run, llm=llm)

        self.assertTrue(used_llm)
        self.assertEqual(note, self.expected)
        self.run.refresh_from_db()
        self.assertEqual(self.run.version_note, self.expected)

    def test_cleans_up_llm_reply(self):
        llm = FakeListChatModel(responses=['"n_estimators 100→300; test accuracy 0.754→0.787"\n\nHope this helps!'])
        note, _ = write_version_note(self.run, llm=llm)
        self.assertEqual(note, 'v2: n_estimators 100→300; test accuracy 0.754→0.787')

    def test_uses_get_llm_with_temperature_zero(self):
        with mock.patch('experiments.versioning.get_llm', return_value=FakeListChatModel(responses=[self.expected])) as get_llm:
            _, used_llm = write_version_note(self.run)
        get_llm.assert_called_once_with(temperature=0)
        self.assertTrue(used_llm)

    def test_falls_back_when_llm_fails(self):
        note, used_llm = write_version_note(self.run, llm=RunnableLambda(failing_llm))

        self.assertFalse(used_llm)
        self.assertEqual(note, self.expected)
        self.run.refresh_from_db()
        self.assertEqual(self.run.version_note, self.expected)

    def test_falls_back_when_get_llm_fails(self):
        with mock.patch('experiments.versioning.get_llm', side_effect=ValueError('bad provider')):
            note, used_llm = write_version_note(self.run)
        self.assertFalse(used_llm)
        self.assertEqual(note, self.expected)

    def test_falls_back_when_llm_invents_or_reverses_changes(self):
        replies = {
            'empty': '   ',
            'invented number': 'v2: n_estimators 100→500; test accuracy 0.754→0.787',
            'reversed change': 'v2: n_estimators 300→100, max_depth None→8',
        }
        for case, reply in replies.items():
            with self.subTest(case):
                note, used_llm = write_version_note(self.run, llm=FakeListChatModel(responses=[reply]))
                self.assertFalse(used_llm)
                self.assertEqual(note, self.expected)

    def test_skips_llm_when_nothing_to_compare(self):
        first = self.run.parent
        unchanged = log_run('exp', dict(self.run.params), dict(self.run.metrics))
        llm = RunnableLambda(mock.Mock(side_effect=AssertionError('LLM should not be called')))

        self.assertEqual(
            write_version_note(first, llm=llm),
            ('v1: first run with n_estimators=100, max_depth=None; test accuracy 0.754', False),
        )
        self.assertEqual(
            write_version_note(unchanged, llm=llm),
            ('v3: no parameter changes; metrics unchanged', False),
        )

    def test_no_llm_option(self):
        with mock.patch('experiments.versioning.get_llm') as get_llm:
            note, used_llm = write_version_note(self.run, use_llm=False)
        get_llm.assert_not_called()
        self.assertEqual((note, used_llm), (self.expected, False))

    def test_fallback_shows_other_metrics_when_test_metrics_are_unchanged(self):
        run = log_run('exp', {'n_estimators': 300, 'max_depth': 8}, {'train_accuracy': 0.95, 'test_accuracy': 0.787})
        self.assertEqual(fallback_note(run), 'v3: no parameter changes; train accuracy 1.0→0.95')

    def test_format_value(self):
        self.assertEqual(
            [format_value(v) for v in (0.7540983, 0.75, 1.0, 300, None, 'gini')],
            ['0.754', '0.75', '1.0', '300', 'None', 'gini'],
        )


def write_heart_csv(path, rows=120):
    """A small, learnable heart-style dataset with a few duplicate rows."""
    import numpy as np
    import pandas as pd

    rng = np.random.default_rng(0)
    data = pd.DataFrame({
        'age': rng.integers(30, 80, rows),
        'chol': rng.integers(150, 350, rows).astype(float),
        'thalach': rng.integers(90, 200, rows),
    })
    data['target'] = (data['thalach'] - data['age'] > 70).astype(int)
    data.loc[3, 'chol'] = None  # Exercises the median imputer.
    pd.concat([data, data.head(5)]).to_csv(path, index=False)


class TrainExperimentTests(TempDirTestCase):
    def setUp(self):
        super().setUp()
        self.data = self.tmp / 'heart.csv'
        write_heart_csv(self.data)
        patcher = mock.patch.object(training, 'RUNS_DIR', self.tmp / 'runs')
        patcher.start()
        self.addCleanup(patcher.stop)

    def train(self, *args):
        out = io.StringIO()
        call_command('train_experiment', '--data', str(self.data), '--n-estimators', '10', *args, stdout=out)
        return out.getvalue()

    def test_trains_saves_and_logs_run(self):
        import joblib

        output = self.train('--no-llm')

        run = Run.objects.get()
        self.assertEqual(run.experiment.name, 'heart-disease-rf')
        self.assertEqual(run.version, 1)
        self.assertEqual(run.params, {
            'n_estimators': 10, 'max_depth': None, 'min_samples_split': 2, 'test_size': 0.2, 'random_state': 42,
        })
        for split in ('train', 'test'):
            for name in ('accuracy', 'precision', 'recall', 'f1'):
                self.assertTrue(0 <= run.metrics[f'{split}_{name}'] <= 1)
        self.assertGreater(run.metrics['test_accuracy'], 0.7)

        checkpoint = self.tmp / 'runs' / 'heart-disease-rf-v1.joblib'
        self.assertEqual(Path(run.checkpoint_path), checkpoint.resolve())
        model = joblib.load(checkpoint)
        self.assertEqual(model.named_steps['imputer'].strategy, 'median')
        self.assertEqual(model.named_steps['classifier'].n_estimators, 10)

        self.assertEqual(run.data_hash, hashlib.sha256(self.data.read_bytes()).hexdigest())
        self.assertGreater(run.duration_seconds, 0)
        self.assertTrue(run.version_note.startswith('v1: first run with n_estimators=10'))
        self.assertIn('120 rows after dropping duplicates', output)
        self.assertIn('96 training rows, 24 test rows', output)

    def test_second_run_records_changes_and_llm_note(self):
        self.train('--no-llm')
        llm = FakeListChatModel(responses=['v2: max_depth None→3'])
        with mock.patch('experiments.versioning.get_llm', return_value=llm):
            self.train('--max-depth', '3', '--experiment', 'heart-disease-rf')

        run = Run.objects.get(version=2)
        self.assertEqual(run.parent.version, 1)
        self.assertEqual(run.changes['params'], {'max_depth': {'old': None, 'new': 3}})
        self.assertEqual(run.version_note, 'v2: max_depth None→3')
        self.assertTrue((self.tmp / 'runs' / 'heart-disease-rf-v2.joblib').exists())

    def test_same_settings_reproduce_the_same_metrics(self):
        self.train('--no-llm')
        self.train('--no-llm')
        first, second = Run.objects.order_by('version')
        self.assertEqual(first.metrics, second.metrics)
        self.assertEqual(second.changes, {'params': {}, 'metrics': {}})

    def test_errors(self):
        no_target = self.tmp / 'no_target.csv'
        no_target.write_text('age,chol\n50,200\n')
        cases = {
            'missing file': ['--data', str(self.tmp / 'missing.csv')],
            'no target column': ['--data', str(no_target)],
            'bad test size': ['--data', str(self.data), '--test-size', '1.5'],
            'bad min samples split': ['--data', str(self.data), '--min-samples-split', '1'],
        }
        for case, args in cases.items():
            with self.subTest(case), self.assertRaises(CommandError):
                call_command('train_experiment', '--no-llm', *args, stdout=io.StringIO())
        self.assertFalse(Run.objects.exists())

    def test_finds_default_data_file(self):
        with mock.patch.object(training, 'DATA_PATHS', [self.tmp / 'nope.csv', self.data]):
            call_command('train_experiment', '--no-llm', '--n-estimators', '5', stdout=io.StringIO())
        self.assertEqual(Run.objects.count(), 1)


class PromoteRunTests(TempDirTestCase):
    def setUp(self):
        super().setUp()
        self.model_path = self.tmp / 'models' / 'heart_disease_model.joblib'
        self.info_path = self.tmp / 'models' / 'current_model.json'
        for name, value in (('PROMOTED_MODEL_PATH', self.model_path), ('CURRENT_MODEL_INFO_PATH', self.info_path)):
            patcher = mock.patch.object(promotion, name, value)
            patcher.start()
            self.addCleanup(patcher.stop)

    def make_run(self, content):
        run = log_run('heart-disease-rf', {'n': len(content)}, {'test_accuracy': 0.8})
        checkpoint = self.tmp / f'v{run.version}.joblib'
        checkpoint.write_bytes(content)
        run.checkpoint_path = str(checkpoint)
        run.version_note = f'v{run.version}: note'
        run.save()
        return run

    def promote(self, *args):
        out = io.StringIO()
        call_command('promote_run', *args, stdout=out)
        return out.getvalue()

    def test_copies_checkpoint_and_writes_current_model(self):
        self.make_run(b'model one')
        run = self.make_run(b'model two')
        self.make_run(b'model three')

        output = self.promote('heart-disease-rf', '2')

        self.assertIn('Promoted heart-disease-rf v2', output)
        self.assertEqual(self.model_path.read_bytes(), b'model two')
        info = json.loads(self.info_path.read_text())
        self.assertEqual(info['experiment'], 'heart-disease-rf')
        self.assertEqual(info['version'], 2)
        self.assertEqual(info['run_id'], run.pk)
        self.assertEqual(info['version_note'], 'v2: note')
        self.assertEqual(self.model_path.stat().st_mode & 0o777, 0o644)

    def test_promoting_again_replaces_files(self):
        self.make_run(b'model one')
        self.make_run(b'model two')
        self.promote('heart-disease-rf', '2')
        self.promote('heart-disease-rf', '1')

        self.assertEqual(self.model_path.read_bytes(), b'model one')
        self.assertEqual(json.loads(self.info_path.read_text())['version'], 1)
        self.assertEqual(sorted(p.name for p in self.model_path.parent.iterdir()), ['current_model.json', 'heart_disease_model.joblib'])

    def test_errors_leave_current_model_alone(self):
        run = self.make_run(b'model one')
        Run.objects.create(experiment=run.experiment, version=2, parent=run)  # No checkpoint.
        Run.objects.create(experiment=run.experiment, version=3, checkpoint_path=str(self.tmp / 'gone.joblib'))

        for args in (['heart-disease-rf', '9'], ['unknown', '1'], ['heart-disease-rf', '2'], ['heart-disease-rf', '3']):
            with self.subTest(args), self.assertRaises(CommandError):
                self.promote(*args)
        self.assertFalse(self.model_path.exists())
        self.assertFalse(self.info_path.exists())
