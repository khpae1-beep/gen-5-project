# AutoMLOps Suite

Django project for the team's AutoMLOps Suite. The `config` project holds settings and URLs; the `core` app serves the home page and shared helpers such as `core/llm.py`. Each teammate adds their own app (deployments, monitoring, experiments, tuning).

## Setup

Requires Python 3.10, Docker and the Terraform CLI.

1. Create and activate a virtual environment:

   ```bash
   python3.10 -m venv .venv
   source .venv/bin/activate
   ```

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Install the Terraform CLI, which the deployment generator uses to format and validate the configuration it generates. On macOS:

   ```bash
   brew tap hashicorp/tap
   brew install hashicorp/tap/terraform
   ```

   For other systems, see https://developer.hashicorp.com/terraform/install. Check it with `terraform -version`.

4. Copy the example environment file and edit values as needed:

   ```bash
   cp .env.example .env
   ```

5. Apply database migrations (SQLite):

   ```bash
   python manage.py migrate
   ```

6. Start RabbitMQ:

   ```bash
   docker compose up -d
   ```

   The management UI is at http://localhost:15672 (guest / guest).

7. Run the development server:

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

The `monitoring` app logs model predictions sent through RabbitMQ, groups them into windows, flags anomalies and shows them on a dashboard.

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

   | Option | Effect |
   | --- | --- |
   | `--count N` | Number of predictions (default 100). |
   | `--drift` | Accuracy drops halfway through the run. |
   | `--delay 0.5` | Seconds to wait between messages. |
   | `--data path.csv` | Predict on this CSV instead of the first of `ml/data/heart.csv`, `data/heart.csv` or `heart.csv`. A missing CSV or model is then an error. |
   | `--all-rows` | Sample every row of the CSV instead of only the held-out test rows (see below). |

4. Compute metrics, then open the dashboard at http://127.0.0.1:8000/monitoring/ (with `python manage.py runserver` running):

   ```bash
   python manage.py compute_metrics
   ```

### Which model is published

`publish_predictions` serves `models/heart_disease_model.joblib`. Create it by training and promoting a run (see [Experiment tracking](#experiment-tracking)):

```bash
python manage.py train_experiment
python manage.py promote_run heart-disease-rf 1
```

Running Khaliyah's `ml/train_model.py` also creates that file. Without the model or a CSV, `publish_predictions` publishes simulated predictions (version `simulated`).

Each message's `model_version` is the promoted run's version (for example `v2`), or `v1` if no run has been promoted. `publish_predictions` loads the model when it starts, so run it again after promoting.

By default, it samples only the rows the model was **not** trained on. It repeats `train_experiment`'s split (drop duplicates, stratified split) with the promoted run's `test_size` and `random_state`, or 0.2 and 42 if there is no promoted run. Accuracy then matches the run's test accuracy (about 0.75 for the default settings). With `--all-rows`, most rows were in the training data, so accuracy looks much higher (about 0.95). The command prints which mode it used.

### Metrics and anomalies

`compute_metrics` groups each model's predictions into windows of `--window-size` predictions (default 10), in the order they were made, and saves one `MetricSnapshot` per window with accuracy, average latency and count. It replaces that model's earlier snapshots, and leaves the last window out until it is full. A window is an anomaly when:

- accuracy is more than 0.10 below the baseline, `MONITORING_BASELINE_ACCURACY` in `config/settings.py` (default 0.75, the model's test accuracy), or
- average latency is more than double the median of the earlier windows.

It then asks the LLM from `LLM_PROVIDER` for a short plain-English summary of the recent windows and saves it on the latest snapshot. If the LLM is unavailable, it saves a rule-based summary instead; `--no-llm` skips the summary.

### Dashboard

http://127.0.0.1:8000/monitoring/ shows, for one model (choose another from the menu when there are several):

- total predictions, overall accuracy, average latency and the number of anomalous windows;
- accuracy and latency per window, with the baseline as a dashed line and windows flagged by that chart's rule in red;
- the latest saved summary (the page never calls the LLM itself);
- the 20 most recent predictions, with their model version.

Run `compute_metrics` again to update the charts and summary. Predictions and snapshots are also in the Django admin.

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

`publish_predictions` loads the model when it starts, so run it again after promoting. Each prediction it publishes then has the promoted version as its `model_version` (for example `v2`), shown in the Version column of the recent predictions table at http://127.0.0.1:8000/monitoring/ and on `PredictionLog` rows in the admin. It samples that run's held-out test rows, so monitored accuracy is comparable with the run's test accuracy. See [Monitoring](#monitoring) for the options.

### Web pages

With `python manage.py runserver` running, http://127.0.0.1:8000/experiments/ lists each experiment with its number of runs, best test accuracy and promoted version. An experiment's page shows its run history: each version's changed settings, train and test accuracy and version note, plus a chart of train vs test accuracy per version (a large gap means overfitting). A run's page shows all its parameters, metrics, changes from the previous version, checkpoint, data hash and git commit. Its "Promote this version" button does the same as `promote_run`, but only for staff users: log in at http://127.0.0.1:8000/admin/ (create an account with `python manage.py createsuperuser`). The compare page puts two versions side by side, highlighting differences with +/- changes. Its "Explain the difference" button asks the LLM for a short explanation, checked against the real numbers, with a written-from-the-numbers fallback. Page loads never call the LLM.

## Hyperparameter tuning

The `tuning` app searches for better random forest settings. Each trial proposes a set of hyperparameters, scores them, and is logged as a run of an experiment, so it can be compared and promoted like any other run.

```bash
python manage.py run_tuning --strategy llm
python manage.py run_tuning --strategy random
```

| Option | Default | Effect |
| --- | --- | --- |
| `--strategy` | `llm` | `llm`: the LLM from `LLM_PROVIDER` proposes each trial from the earlier trials' scores. `random`: random search, the baseline to compare against. |
| `--trials` | `15` | Number of trials. |
| `--cv` | `5` | Cross-validation folds. |
| `--seed` | `42` | Seed for the train/test split, the folds, the forest and random proposals. The same seed gives the same random search. |
| `--data` | first of `ml/data/heart.csv`, `data/heart.csv`, `heart.csv` | CSV with a `target` column. |

The search space is `n_estimators` 50–500, `max_depth` 2–20 or none, `min_samples_split` 2–20, `min_samples_leaf` 1–10 and `max_features` `sqrt`, `log2` or none.

### How trials are scored

The data is split the same way as in `train_experiment` (drop duplicates, stratified 80/20 split). Each trial is scored by k-fold cross-validation accuracy on the **training rows only**, and the best trial is the one with the highest CV accuracy. Test accuracy is printed for reference but never used to choose settings, so it stays an honest estimate for new data.

### LLM proposals

With `--strategy llm`, the LLM sees the search space and every earlier trial's settings and CV score (never test accuracy). It replies with the next settings and a one-sentence reason.

- If the settings are outside the search space, repeat an earlier trial, or the LLM is unavailable, a random proposal is used instead. The reason then starts with "Random fallback:" and says why.
- If only the reason is wrong about what changed (for example "increase min_samples_split" when it went from 10 to 5), the LLM's settings are kept, but the reason is replaced by a code-written description of the change that notes the failed check.

Each trial prints its settings, CV accuracy and reason:

```
Trial 2/15 [llm] {"n_estimators": 200, "max_depth": 15, "min_samples_split": 5, "min_samples_leaf": 2, "max_features": "log2"}
  CV accuracy 0.8504 ± 0.0640 (best so far); test accuracy 0.7705 (reference only)
  Changed n_estimators 300→200, max_depth None→15, ... The LLM's reason failed the fact check (...).
```

At the end it prints the best trial, the trial number where it was found, and the command to promote it.

### Trials on the experiments pages

Trials are logged under the experiment `heart-disease-rf-tuning-llm` or `heart-disease-rf-tuning-random`, one version per trial (later jobs continue the numbering). At http://127.0.0.1:8000/experiments/ each trial shows its settings, train and test accuracy and, as its version note, the tuning job, trial number, CV accuracy and reason. Use the compare page to see two trials side by side. Tuning jobs and their trials are also listed in the Django admin under Tuning.

### Promote the best trial

`run_tuning` ends with the command for its best trial, for example:

```bash
python manage.py promote_run heart-disease-rf-tuning-llm 2
```

Staff users can also use the "Promote this version" button on the trial's run page. As with any promoted run, restart `publish_predictions` to serve it. It then samples the same held-out test rows the trial was scored against.

### Tuning pages

http://127.0.0.1:8000/tuning/ lists the tuning jobs with their strategy, number of trials, best CV accuracy ± std, the trial where it was found, that trial's test accuracy (reference only), status and date. A job's page charts the CV accuracy of each trial with a "best so far" line, and lists the trials with their proposer, settings, scores and reason, best trial highlighted; each trial links to its run on the experiments pages, where it can be promoted. The compare page (http://127.0.0.1:8000/tuning/compare/) charts the best CV accuracy so far for an LLM job against a random job, defaulting to the latest of each, with a summary of best score, trial found, time taken, random fallbacks and how many LLM reasons passed the fact check. On a job's page, staff users (log in at `/admin/`) can press "Suggest next experiments" to ask the LLM for 3 new configurations, each checked to be inside the search space and untried, with its reason fact-checked against the best trial. The suggestions are saved on the job, so reloading the page doesn't ask again. "Try this" on a suggestion runs one trial with those settings on the job's data and adds it to the job. Page loads never call the LLM.

## Tests

```bash
python manage.py test
```
