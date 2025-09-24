#!/bin/bash
set -e

# Удаляем файл server.pid, если он существует
if [ -f /webapp/tmp/pids/server.pid ]; then
  rm /webapp/tmp/pids/server.pid
fi

# Выполняем основную команду контейнера (CMD)
exec "$@"
