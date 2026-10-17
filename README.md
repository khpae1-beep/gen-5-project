# AutoMLOps Suite

Django project for the team's AutoMLOps Suite. The `config` project holds settings and URLs; the `core` app serves the home page and shared helpers such as `core/llm.py`. Each teammate adds their own app (deployments, monitoring, experiments, tuning).

## Setup

Requires Python 3.10 and Docker.

1. Create and activate a virtual environment:

   ```bash
   python3.10 -m venv .venv
   source .venv/bin/activate
   ```

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Copy the example environment file and edit values as needed:

   ```bash
   cp .env.example .env
   ```

4. Apply database migrations (SQLite):

   ```bash
   python manage.py migrate
   ```

5. Start RabbitMQ:

   ```bash
   docker compose up -d
   ```

   The management UI is at http://localhost:15672 (guest / guest).

6. Run the development server:

   ```bash
   python manage.py runserver
   ```

   Open http://127.0.0.1:8000/. With `DEBUG=True` the Django debug toolbar appears on the right.

## Environment variables

See `.env.example` for the full list:

| Variable | Purpose | Default |
| --- | --- | --- |
| `SECRET_KEY` | Django secret key | insecure dev key |
| `DEBUG` | Debug mode; also enables the debug toolbar | `True` |
| `RABBITMQ_HOST` | RabbitMQ host | `localhost` |
| `LLM_PROVIDER` | `openai` or `ollama`, used by `core.llm.get_llm()` | `ollama` |
| `LLM_MODEL` | Model name for the chosen provider | `llama3.2` / `gpt-4o-mini` |
| `OPENAI_API_KEY` | Required when `LLM_PROVIDER=openai` | none |
| `OLLAMA_BASE_URL` | Ollama server URL | `http://localhost:11434` |

## Monitoring

The `monitoring` app logs model predictions sent through RabbitMQ.

1. Start RabbitMQ and apply migrations:

   ```bash
   docker compose up -d
   python manage.py migrate
   ```

2. In one terminal, start the consumer. It saves each message from the `predictions` queue as a `PredictionLog` row; stop it with Ctrl+C:

   ```bash
   python manage.py consume_predictions
   ```

3. In another terminal, publish predictions:

   ```bash
   python manage.py publish_predictions --count 100
   python manage.py publish_predictions --count 100 --drift
   ```

   `--drift` makes accuracy drop halfway through the run. `--delay 0.5` waits between messages.

`publish_predictions` uses the real heart disease model only after you run Khaliyah's `ml/train_model.py`, which creates `models/heart_disease_model.joblib`. It predicts on random rows of `heart.csv`, taken from the first of `ml/data/heart.csv`, `data/heart.csv` or `heart.csv` in the project root that exists. Without the model or the CSV, it publishes simulated predictions.

Logged predictions and metric snapshots are visible in the Django admin.

## Experiment tracking

The `experiments` app trains the heart disease model as numbered runs of an experiment, so you can compare settings and choose which version monitoring uses. Apply migrations first (`python manage.py migrate`).

### Train a run

```bash
python manage.py train_experiment
```

This repeats the steps in Khaliyah's `ml/train_model.py` (drop duplicates, stratified train/test split, median imputer, scaler, random forest). It reads the first of `ml/data/heart.csv`, `data/heart.csv` or `heart.csv` that exists, or the CSV given with `--data`. It records train and test accuracy, precision, recall and F1, and saves the model to `models/runs/heart-disease-rf-v1.joblib`. The run is logged as v1 of the `heart-disease-rf` experiment.

To try different settings, pass them as options. Each call becomes the next version and records what changed from the previous one:

```bash
python manage.py train_experiment --n-estimators 300 --max-depth 8
```

```
Logged heart-disease-rf v2: v2: n_estimators 100→300, max_depth None→8; test accuracy 0.754→0.77, ...
```

| Option | Default |
| --- | --- |
| `--n-estimators` | `100` |
| `--max-depth` | none (unlimited) |
| `--min-samples-split` | `2` |
| `--test-size` | `0.2` |
| `--random-state` | `42` (used for the split and the forest) |
| `--experiment` | `heart-disease-rf` |
| `--data` | first of `ml/data/heart.csv`, `data/heart.csv`, `heart.csv` |
| `--no-llm` | off |

The version note is written by the LLM from `LLM_PROVIDER` (see `core/llm.py`). If the LLM is unavailable, gets the numbers wrong, or `--no-llm` is passed, the note is written from the changes in plain code instead. Runs, with their parameters, metrics, changes, data hash and git commit, are listed in the Django admin under Experiments.

Other code can log runs the same way:

```python
from experiments.tracking import log_run

run = log_run('heart-disease-rf', params={'n_estimators': 300}, metrics={'test_accuracy': 0.787},
              checkpoint_path='models/runs/my-model.joblib', data_path='ml/data/heart.csv')
```

### Promote a run

```bash
python manage.py promote_run heart-disease-rf 2
```

This copies that run's model to `models/heart_disease_model.joblib`, the file `publish_predictions` serves, and records the experiment and version in `models/current_model.json`. To roll back, promote an earlier version.

### The promoted version in monitoring

`publish_predictions` loads the model when it starts, so run it again after promoting. Each prediction it publishes then has the promoted version as its `model_version` (for example `v2`), and that version appears in the Version column of the recent predictions table at http://127.0.0.1:8000/monitoring/ and on `PredictionLog` rows in the admin. If `models/current_model.json` does not exist, the real model is published as `v1`.

To predict on a specific CSV, pass `--data`:

```bash
python manage.py publish_predictions --count 100 --data ml/data/heart.csv
```

With `--data`, a missing CSV or model is an error instead of a fallback to simulated predictions.

## Tests

```bash
python manage.py test
```
