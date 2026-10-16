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

## Tests

```bash
python manage.py test
```
