#!/usr/bin/env bash
# Start the read-only demo on Render: a fresh copy of the demo database on every start, then gunicorn.
#
# Render settings:
#   Build command: pip install -r requirements.txt && python manage.py collectstatic --noinput
#   Start command: bin/render_start.sh
#   Environment:   DB_PATH (e.g. /tmp/demo.sqlite3), DJANGO_DEBUG=False, DJANGO_SECRET_KEY
set -euo pipefail

cd "$(dirname "$0")/.."

# Required so this script can never overwrite a local db.sqlite3 by accident.
: "${DB_PATH:?Set DB_PATH to where the demo database should be copied, e.g. /tmp/demo.sqlite3}"

mkdir -p "$(dirname "$DB_PATH")"
cp demo/demo.sqlite3 "$DB_PATH"

# The promoted-model record the experiments pages read (models/ is not in git). Only copied when
# missing, so a local models/current_model.json is never replaced.
mkdir -p models
[ -e models/current_model.json ] || cp demo/current_model.json models/current_model.json

python manage.py migrate --noinput

exec gunicorn config.wsgi:application --bind "0.0.0.0:${PORT:-8000}" --workers 1 --timeout 120
