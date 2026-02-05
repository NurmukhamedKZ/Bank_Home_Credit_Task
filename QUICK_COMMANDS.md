# CV Search - Быстрые команды

## Структура проекта

```
app/
├── main.py              # FastAPI приложение (точка входа)
├── models/              # Pydantic модели
│   ├── cv.py            # CVOutput, WorkExperience, Education
│   └── api.py           # SearchRequest, SearchResponse, etc.
├── api/                 # FastAPI эндпоинты
│   └── routes.py        # /search, /health, /
├── services/            # Бизнес-логика
│   ├── cv_parser.py     # CVParser - парсинг и эмбеддинги
│   ├── email_fetcher.py # EmailFetcher - получение из почты
│   └── search.py        # Поисковая логика
├── evaluation/          # Оценка качества
│   ├── metrics.py       # SearchMetrics (MAP, MRR, P@K, etc.)
│   ├── evaluator.py     # CVSearchEvaluator
│   └── analysis.py      # Визуализация и статистика
├── ui/                  # Streamlit интерфейсы
│   ├── frontend.py      # Поиск кандидатов
│   └── dashboard.py     # Dashboard метрик
├── scripts/             # CLI скрипты
│   ├── load_cvs.py      # Загрузка CV в Qdrant
│   ├── run_evaluation.py# Запуск оценки
│   ├── compare_modes.py # Сравнение режимов
│   └── fetch_emails.py  # Получение из почты
└── experiments/         # Эксперименты
    └── experiment_runner.py
```

## API сервер

```bash
# Запуск FastAPI сервера
uvicorn app.main:app --reload --port 8000

# Документация API
open http://localhost:8000/docs
```

## Streamlit UI

```bash
# Поиск кандидатов (требует запущенный API)
streamlit run app/ui/frontend.py

# Dashboard метрик
streamlit run app/ui/dashboard.py --server.port 8502
```

## CLI команды

```bash
# Загрузка CV в Qdrant (TF-IDF)
python -m app.scripts.load_cvs

# Загрузка CV в Qdrant (BM25)
python -m app.scripts.load_cvs --bm25

# Оценка качества поиска
python -m app.scripts.run_evaluation              # Hybrid (по умолчанию)
python -m app.scripts.run_evaluation --dense      # Dense-only
python -m app.scripts.run_evaluation --sparse     # Sparse-only
python -m app.scripts.run_evaluation --bm25       # С BM25

# Сравнение режимов поиска
python -m app.scripts.compare_modes               # Dense vs Sparse vs Hybrid
python -m app.scripts.compare_modes --sparse      # TF-IDF vs BM25

# Получение резюме из почты
python -m app.scripts.fetch_emails
```

## Режимы поиска

| Режим | Описание | Когда использовать |
|-------|----------|-------------------|
| `dense` | Семантический поиск (Voyage AI) | Поиск по смыслу |
| `sparse` | Keyword поиск (TF-IDF/BM25) | Точное соответствие |
| `hybrid` | Комбинация обоих | **Рекомендуется** |

## API примеры

```bash
# Health check
curl http://localhost:8000/health

# Поиск кандидатов
curl -X POST http://localhost:8000/search \
  -H "Content-Type: application/json" \
  -d '{
    "vacancy_text": "Python разработчик с опытом FastAPI, Docker, PostgreSQL",
    "search_mode": "hybrid",
    "top_k": 5
  }'
```

## Метрики качества

- **MAP** (Mean Average Precision) - основная метрика
- **MRR** (Mean Reciprocal Rank) - позиция первого релевантного
- **Precision@K** - точность в топ-K
- **Recall@K** - полнота в топ-K
- **NDCG@K** - учитывает порядок результатов
- **F1@K** - гармоническое среднее P и R
