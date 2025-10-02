# CT-AI — анализ КТ-снимков (Rails + Python + ONNX)

Веб‑приложение для загрузки и анализа КТ‑снимков. Сервис выполняет бинарную классификацию (норма/патология) и построен на гибридной архитектуре: Ruby on Rails для оркестрации и UI, Python для ML‑инференса ONNX‑модели. Полностью контейнеризован (Docker Compose), фоновые задачи — Sidekiq, БД — PostgreSQL, конвертация DICOM — `dcmtk`.

## Ключевые возможности
- Массовая загрузка `.zip` архивов с DICOM‑срезами
- Автоопределение DICOM‑файлов по сигнатуре `DICM` (имя и расширение не важны)
- Конвертация DICOM → PNG (`dcm2img`), сортировка срезов по вероятности
- История исследований: статус, Study/Series UID, длительность обработки, итог
- Экспорт полной истории в `.xlsx`

## Архитектура
- Backend: Ruby 3.3, Rails 7
- База данных: PostgreSQL
- Фоновые задачи: Sidekiq + Redis
- Конвертация DICOM: системная утилита `dcmtk` (`dcm2img`)
- Инференс: Python + ONNXRuntime (изолированный venv в контейнере)
- Контейнеризация: Docker, Docker Compose

Критические узлы:
- `webapp/app/services/dicom_processor.rb` — пайплайн обработки (конвертация, запуск Python, сбор результатов)
- `webapp/lib/python/run_inference.py` — инференс ONNX‑модели, возврат JSON c метриками
- Модель: `webapp/vendor/assets/ml_models/model.onnx`

## Быстрый старт
Требуются Docker Desktop и Docker Compose.

1) Сборка и запуск:
```bash
docker-compose -f webapp/docker-compose.yml up -d --build
```

2) Инициализация БД — автоматически
При старте контейнера `web` скрипт `entrypoint.sh` выполняет `rails db:prepare` с ретраями: БД создаётся (если отсутствует) и применяются миграции. Ничего вручную делать не нужно.

Опционально (если нужно вручную):
```bash
docker-compose -f webapp/docker-compose.yml exec web bundle exec rails db:prepare
```

3) Откройте приложение: http://localhost:3000

### Последующие запуски
```bash
docker-compose -f webapp/docker-compose.yml up -d
```
Остановка:
```bash
docker-compose -f webapp/docker-compose.yml down
```

## Использование
- Загружайте `.zip`, содержащий одно или несколько исследований (папки/файлы DICOM).
- Имена файлов и отсутствие расширения `.dcm` — не проблема: проверяется сигнатура `DICM` на смещении 128 байт.
- После обработки исследование появится в Истории с итогом (патология/вероятность). Можно выгрузить отчёт `.xlsx`.

## Зависимости
Python‑зависимости описаны в `webapp/lib/python/requirements.txt`:
- `numpy`
- `Pillow`
- `onnxruntime`

Версии фиксируются внутри контейнера. Проверить фактические версии можно из контейнера `web`:
```bash
docker-compose -f webapp/docker-compose.yml exec web /webapp/lib/python/venv/bin/python -c "import numpy, PIL, onnxruntime as ort; print(numpy.__version__); print(PIL.__version__); print(ort.__version__)"
```

Ruby‑зависимости перечислены в `webapp/Gemfile` и устанавливаются при сборке образа.

## Разработка
- Изменения в Ruby‑зависимостях — правьте `webapp/Gemfile`, затем ребилд:
```bash
docker-compose -f webapp/docker-compose.yml up -d --build
```
- Изменения в Python‑зависимостях — `webapp/lib/python/requirements.txt`, затем ребилд той же командой.
- Логи Sidekiq:
```bash
docker-compose -f webapp/docker-compose.yml logs -f sidekiq
```

## Траблшутинг
- Завис статус "processing": проверьте логи `sidekiq`, убедитесь, что архив содержит валидные DICOM (можно проверить `dcmdump` внутри контейнера `web`).
- Если в путях есть пробелы/не ASCII — в конвертации используются экранированные пути.
- Полезно очистить тома при конфликте окружений:
```bash
docker-compose -f webapp/docker-compose.yml down -v
```
и затем пересобрать контейнеры.

## Структура репозитория (основное)
```
webapp/
  app/
    jobs/ (Sidekiq jobs)
    services/ (DicomProcessor)
    views/ (ERB/ui)
  lib/python/ (ML‑скрипты)
  vendor/assets/ml_models/model.onnx
  docker-compose.yml
DATASETS.md
PLAN.md
PROJECT_STATUS.md
```

Примечание: для развёртывания достаточно каталога `webapp/` — в нём есть всё необходимое (Dockerfile, docker-compose.yml, приложение и скрипты). Внешние файлы (README, PLAN, PROJECT_STATUS, DATASETS) носят справочный характер.

## Лицензия
Уточните лицензию проекта при публикации.

## Благодарности
Датасеты и источники перечислены в `DATASETS.md`.
