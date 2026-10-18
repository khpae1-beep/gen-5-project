import json
import random
import time
from pathlib import Path

import pika
from django.apps import apps
from django.conf import settings
from django.core.management.base import BaseCommand, CommandError
from django.db import DatabaseError
from django.utils import timezone

from experiments.training import load_data, split_data
from monitoring.rabbitmq import PREDICTIONS_QUEUE, open_channel

MODEL_PATH = settings.BASE_DIR / 'models' / 'heart_disease_model.joblib'
# Written by the experiments app's promote_run; says which run MODEL_PATH came from.
CURRENT_MODEL_INFO_PATH = settings.BASE_DIR / 'models' / 'current_model.json'
DEFAULT_MODEL_VERSION = 'v1'
# train_experiment's defaults, used when the promoted run's params are not available.
DEFAULT_TEST_SIZE = 0.2
DEFAULT_RANDOM_STATE = 42
# Checked in order; the first one that exists is used.
DATA_PATHS = [
    settings.BASE_DIR / 'ml' / 'data' / 'heart.csv',
    settings.BASE_DIR / 'data' / 'heart.csv',
    settings.BASE_DIR / 'heart.csv',
]
LABEL_COLUMN = 'target'
MODEL_NAME = 'heart_disease'

# Simulated predictions are right this often before drift starts.
SIMULATED_ACCURACY = 0.9
# Once drift starts, each prediction is flipped with this probability.
DRIFT_FLIP_RATE = 0.4


def build_message(model_name, model_version, prediction, actual, latency_ms):
    return {
        'model_name': model_name,
        'model_version': model_version,
        'prediction': int(prediction),
        'actual': None if actual is None else int(actual),
        'latency_ms': round(float(latency_ms), 3),
        'timestamp': timezone.now().isoformat(),
    }


def find_data_path():
    return next((path for path in DATA_PATHS if path.exists()), None)


def display_path(path):
    """Relative to the project when inside it, otherwise as given."""
    try:
        return str(Path(path).resolve().relative_to(settings.BASE_DIR.resolve()))
    except ValueError:
        return str(path)


def current_model_info():
    """The contents of current_model.json, or None if there is no usable file."""
    try:
        with open(CURRENT_MODEL_INFO_PATH, encoding='utf-8') as f:
            info = json.load(f)
    except (OSError, ValueError):
        return None
    if not isinstance(info, dict) or not isinstance(info.get('version'), int):
        return None
    return info


def promoted_version():
    """'v<N>' from current_model.json, or None if there is no usable file."""
    info = current_model_info()
    return f'v{info["version"]}' if info else None


def promoted_split():
    """(test_size, random_state, where they came from) for the promoted run's train/test split."""
    info = current_model_info()
    if info and apps.is_installed('experiments'):
        Run = apps.get_model('experiments', 'Run')
        try:
            run = Run.objects.filter(experiment__name=info.get('experiment'), version=info['version']).first()
        except DatabaseError:  # e.g. the experiments migrations have not been applied.
            run = None
        if run is not None:
            return (
                run.params.get('test_size', DEFAULT_TEST_SIZE),
                run.params.get('random_state', DEFAULT_RANDOM_STATE),
                f'{run.experiment.name} v{run.version}',
            )
    return DEFAULT_TEST_SIZE, DEFAULT_RANDOM_STATE, 'defaults'


class HeartDiseasePredictor:
    """Runs the trained heart disease model on random rows of heart.csv.

    With test_split=(test_size, random_state), only the held-out test rows of train_experiment's
    split are used (after dropping duplicates, as training does). With test_split=None, every row
    of the file is used, including the rows the model was trained on.
    """

    def __init__(self, model_path, data_path, version=DEFAULT_MODEL_VERSION, test_split=None):
        import joblib
        import pandas as pd

        self.version = version
        self.model = joblib.load(model_path)
        if test_split is None:
            data = pd.read_csv(data_path)
            if LABEL_COLUMN not in data.columns:
                raise CommandError(f'{data_path} has no {LABEL_COLUMN!r} column.')
            self.features = data.drop(columns=[LABEL_COLUMN])
            self.labels = data[LABEL_COLUMN]
            return

        try:
            features, labels = load_data(data_path)
            _, self.features, _, self.labels = split_data(features, labels, *test_split)
        except ValueError as exc:
            raise CommandError(f'Could not split {data_path} into train and test rows: {exc} Use --all-rows.') from exc

    def predict(self):
        i = random.randrange(len(self.features))
        row = self.features.iloc[[i]]
        start = time.perf_counter()
        prediction = int(self.model.predict(row)[0])
        latency_ms = (time.perf_counter() - start) * 1000
        return prediction, int(self.labels.iloc[i]), latency_ms


class SimulatedPredictor:
    """Stand-in used when the trained model or dataset is not available."""

    version = 'simulated'

    def predict(self):
        actual = random.randint(0, 1)
        prediction = actual if random.random() < SIMULATED_ACCURACY else 1 - actual
        latency_ms = max(1.0, random.gauss(20, 5))
        return prediction, actual, latency_ms


class Command(BaseCommand):
    help = (
        f'Publish predictions as JSON to the {PREDICTIONS_QUEUE!r} RabbitMQ queue. Uses '
        'models/heart_disease_model.joblib on random held-out test rows (or with --all-rows, any rows) of '
        '--data, or of heart.csv from ml/data/, data/ or the project root, when both exist; otherwise '
        'simulated predictions.'
    )

    def add_arguments(self, parser):
        parser.add_argument('--count', type=int, default=100, help='Number of predictions to publish (default 100).')
        parser.add_argument(
            '--drift',
            action='store_true',
            help='Halfway through, start flipping some predictions so accuracy drops.',
        )
        parser.add_argument('--delay', type=float, default=0.0, help='Seconds to wait between messages (default 0).')
        parser.add_argument(
            '--all-rows',
            action='store_true',
            help='Sample from every row of the CSV, including rows the model was trained on. By default only '
                 'the held-out test rows of the promoted run\'s train/test split are used.',
        )
        parser.add_argument(
            '--data',
            help='CSV with a "target" column to predict on. Default: ml/data/heart.csv, data/heart.csv or '
                 'heart.csv, whichever exists first.',
        )

    def handle(self, *args, **options):
        count = options['count']
        if count < 1:
            raise CommandError('--count must be at least 1.')

        predictor = self.get_predictor(options['data'], all_rows=options['all_rows'])
        drift_start = count // 2 if options['drift'] else None

        try:
            connection, channel = open_channel()
        except pika.exceptions.AMQPConnectionError as exc:
            raise CommandError(
                f'Could not connect to RabbitMQ at {settings.RABBITMQ_HOST!r}. Is it running (docker compose up -d)?'
            ) from exc

        correct = 0
        try:
            for i in range(count):
                prediction, actual, latency_ms = predictor.predict()
                if drift_start is not None and i >= drift_start and random.random() < DRIFT_FLIP_RATE:
                    prediction = 1 - prediction
                correct += prediction == actual

                message = build_message(MODEL_NAME, predictor.version, prediction, actual, latency_ms)
                channel.basic_publish(
                    exchange='',
                    routing_key=PREDICTIONS_QUEUE,
                    body=json.dumps(message),
                    properties=pika.BasicProperties(
                        content_type='application/json',
                        delivery_mode=pika.DeliveryMode.Persistent,
                    ),
                )
                if i == drift_start:
                    self.stdout.write(f'Drift started at message {i + 1}.')
                if options['delay']:
                    time.sleep(options['delay'])
        finally:
            connection.close()

        self.stdout.write(self.style.SUCCESS(
            f'Published {count} predictions to {PREDICTIONS_QUEUE!r} (accuracy {correct / count:.1%}).'
        ))

    def get_predictor(self, data=None, all_rows=False):
        if data:
            # An explicit CSV means real predictions are wanted, so don't fall back to simulated ones.
            data_path = Path(data)
            if not data_path.exists():
                raise CommandError(f'Data file {data} not found.')
            if not MODEL_PATH.exists():
                raise CommandError(
                    f'{display_path(MODEL_PATH)} not found. Train and promote a model first '
                    '(train_experiment, then promote_run).'
                )
        else:
            data_path = find_data_path()
        if MODEL_PATH.exists() and data_path:
            version = promoted_version() or DEFAULT_MODEL_VERSION
            self.stdout.write(f'Using {MODEL_PATH.name} ({version}) on rows from {display_path(data_path)}.')
            if all_rows:
                predictor = HeartDiseasePredictor(MODEL_PATH, data_path, version)
                self.stdout.write(
                    f'Mode: all rows (--all-rows), sampling {len(predictor.features)} rows. Most were in the '
                    'training data, so accuracy will look higher than on new data.'
                )
                return predictor
            test_size, random_state, source = promoted_split()
            predictor = HeartDiseasePredictor(MODEL_PATH, data_path, version, (test_size, random_state))
            self.stdout.write(
                f'Mode: held-out test rows, sampling {len(predictor.features)} rows not used for training '
                f'(test_size {test_size}, random_state {random_state} from {source}).'
            )
            return predictor
        self.stdout.write(f'{MODEL_PATH.name} or heart.csv not found; publishing simulated predictions.')
        return SimulatedPredictor()
