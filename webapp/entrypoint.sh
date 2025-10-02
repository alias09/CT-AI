#!/bin/bash
set -e

# Удаляем файл server.pid, если он существует
if [ -f /webapp/tmp/pids/server.pid ]; then
  rm /webapp/tmp/pids/server.pid
fi

# Автоматическая инициализация БД (создание + миграции)
# rails db:prepare безопасно: создаёт БД (если нет) и применяет миграции
echo "[entrypoint] Running rails db:prepare..."
MAX_RETRIES=${DB_PREPARE_RETRIES:-20}
SLEEP_SECONDS=${DB_PREPARE_DELAY:-2}
COUNT=0
until bundle exec rails db:prepare; do
  COUNT=$((COUNT+1))
  if [ "$COUNT" -ge "$MAX_RETRIES" ]; then
    echo "[entrypoint] Failed to run db:prepare after ${MAX_RETRIES} attempts. Exiting." >&2
    exit 1
  fi
  echo "[entrypoint] DB not ready yet, retry ${COUNT}/${MAX_RETRIES} in ${SLEEP_SECONDS}s..."
  sleep "$SLEEP_SECONDS"
done
echo "[entrypoint] db:prepare done."

# Выполняем основную команду контейнера (CMD)
exec "$@"
