# TT KV Cache Compression for RAG

Проект подготавливает данные HotpotQA, строит для документов базу эмбеддингов и
запускает два RAG-эксперимента: исходный baseline и baseline со сжатием KV cache
тензорными сетями (Tensor Train, TT).

## Требования

- Windows, Linux или macOS с установленным `uv`.
- Python `3.13.16` (версия закреплена в `pyproject.toml`).
- Доступ в интернет для первоначальной загрузки моделей и датасета.
- CUDA GPU рекомендуется для запуска `Qwen3-8B`; если CUDA недоступна, код
  использует CPU, но генерация будет значительно медленнее.

## Установка

Команды ниже выполняются из корня репозитория.

```powershell
uv python install 3.13.16
uv sync --no-install-project
```

Зависимости и версии фиксируются в `uv.lock`. Флаг `--no-install-project`
устанавливает зависимости проекта; исходный код запускается непосредственно из
репозитория.

## Первичная подготовка данных

В `configs/config.toml` при первом запуске оставьте:

```toml
[constants]
is_first_run = true
```

Затем выполните:

```powershell
uv run --no-sync python main.py
```

Подготовка создаст нужные каталоги, скачает генеративную и embedding-модели,
сохранит заданную часть HotpotQA и построит документы с эмбеддингами. По
умолчанию используется `validation[:300]`. Результаты подготовки:

- `models/Qwen3-8B/` — генеративная модель;
- `models/Qwen3-Embedding-0.6B/` — embedding-модель;
- `datasets/hotpot_qa/` — сохранённый датасет;
- `datasets/hotpot_qa_documents/documents.jsonl` — тексты документов;
- `datasets/hotpot_qa_documents/documents.json` — индексы документов по
  `example_id`;
- `datasets/hotpot_qa_documents/documents.npy` — эмбеддинги документов.

После завершения проверьте наличие файлов и в существующей секции `[constants]`
файла `configs/config.toml` вручную измените `is_first_run` на `false`.

Пока метка остаётся `true`, каждый вызов `main.py` запускает первичную
подготовку. Оба эксперимента отказываются запускаться, если метка не сброшена.

## Настройки

Основные параметры находятся в `configs/config.toml`:

- `[generator]` — репозиторий, ревизия и локальный путь генеративной модели;
- `[embedding]` — embedding-модель и её локальный путь;
- `[dataset_raw]` — датасет, конфигурация, split и путь сохранения;
- `[dataset_prepared]` — каталог и файлы документов, индекса и эмбеддингов;
- `[constants]` — устройство, параметры пакетирования эмбеддингов и метка
  первичной подготовки.

Относительные пути разрешаются от корня проекта. По умолчанию поиск для каждого
вопроса ограничен документами с соответствующим `example_id`, а не общей базой
документов.

## Запуск экспериментов

### Исходный baseline

```powershell
uv run --no-sync python -m experiments.baseline
```

Для быстрой оценки первых 10 примеров:

```powershell
uv run --no-sync python -m experiments.baseline --limit 10
```

Доступны параметры `--top-k`, `--limit`, `--offset`, `--max-new-tokens`, `--seed`,
`--seed-per-example` и `--output-dir`. `--offset` задаёт начальный индекс, а
`--limit` — число примеров после него. С `--seed-per-example` каждый вопрос
получает seed `seed + индекс`, поэтому непересекающиеся части выборки можно
запускать параллельно. Например:

```powershell
uv run --no-sync python -m experiments.baseline --limit 10 --top-k 5 --max-new-tokens 64
```

### Baseline со сжатием TT KV cache

```powershell
uv run --no-sync python -m experiments.baseline_tt
```

Параметры `--top-k`, `--limit`, `--offset`, `--max-new-tokens`, `--seed`,
`--seed-per-example` и `--output-dir`
работают так же, как в исходном baseline. Дополнительно доступны:

- `--tt-epsilon` — точность TT-SVD, по умолчанию `0.01`;
- `--tt-block-size` — число токенов в TT-блоке, по умолчанию `16`.
- `--tt-core-quantization int8` — экспериментальное симметричное 8-битное
  хранение TT-ядер с отдельным масштабом для каждого ядра; по умолчанию `none`.

Например:

```powershell
uv run --no-sync python -m experiments.baseline_tt --limit 10 --tt-epsilon 0.01 --tt-block-size 16
```

TT-ядра и неполный плотный блок хранятся на CPU. Перед attention cache
восстанавливается в плотные K/V на устройстве модели. Поэтому compression ratio
описывает размер сохранённого cache и сам по себе не равен снижению пиковой
памяти GPU или ускорению генерации.

TT-SVD выполняет много разложений небольших матриц. При запуске
`baseline_tt.py` число потоков BLAS по умолчанию ограничено одним: на H100
это заметно сократило время разложений. Значения `OPENBLAS_NUM_THREADS`,
`OMP_NUM_THREADS` и `MKL_NUM_THREADS`, заданные до запуска, сохраняются.

## Результаты

Каждый запуск создаёт отдельную папку в `experiments/results/` с временной
меткой. В ней находятся:

- `predictions.jsonl` — предсказания, метрики и использованные документы для
  каждого примера;
- `summary.json` — агрегированные метрики, параметры запуска и пути к артефактам;
- `plots/quality_metrics.png` — answer EM/F1 и retrieval-метрики;
- `plots/generation_metrics.png` — задержка, throughput и пиковая CUDA-память;
- `plots/tt_cache_metrics.png` — размер TT cache, compression ratio, ошибка
  восстановления, TT-rank и время обработки (только TT-эксперимент).

Каталог результатов можно переопределить параметром `--output-dir`.
Для объединения запусков с `--seed-per-example` используйте
`python -m experiments.aggregate_shards --baseline PATH/summary.json --tt
PATH1/summary.json PATH2/summary.json --output-dir PATH`. Скрипт проверяет
покрытие индексов и сохраняет `comparison.json` и общий `tt_predictions.jsonl`.

Опубликованные прогоны находятся в `experiments/results/`: сырые предсказания,
JSON-сводки и графики сохранены по узлам и конфигурациям. Сводка парного
сравнения 300 вопросов — в `docs/full_300_comparison.json`, ход экспериментов
и ограничения результатов — в `docs/experiment_log.md`.

## Структура проекта

- `main.py` — входная точка первичной подготовки;
- `configs/config.toml` — параметры моделей, данных и подготовки;
- `setup/` — загрузка данных, создание документов и эмбеддингов;
- `experiments/baseline.py` — RAG baseline без сжатия cache;
- `experiments/baseline_tt.py` — RAG baseline с TT-сжатием KV cache;
- `experiments/tt_cache.py` — адаптер TT cache для Transformers;
- `experiments/plots.py` — построение и сохранение графиков;
- `metrics.py` — метрики качества ответов.
