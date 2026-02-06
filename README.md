# 🤖 AI Recruiting Agent

## 📋 Описание

Система для автоматического поиска релевантных кандидатов по описанию вакансии. Использует:
- **Векторный поиск** (Qdrant + Google Gemini embeddings)
- **Keyword поиск** (BM25, TF-IDF)
- **Гибридный поиск** (комбинация dense + sparse)
- **LLM анализ** (GPT-4 для детальной оценки соответствия)

---

## ✨ Возможности

### 🔍 Множественные методы поиска
- **Vector Search** - семантический поиск через embeddings (~0.2 сек)
- **ML Classifier** - TF-IDF + Logistic Regression (~1-2 сек)
- **LLM Analyzer** - GPT-4 с объяснениями (~15-20 сек)

### 📊 Визуализация и метрики
- **Dashboard** - сравнение методов поиска
- **Метрики качества** - MAP, MRR, Precision@K, Recall@K, NDCG

### 📧 Автоматизация
- **Email Fetcher** - автоматическое получение резюме из почты
- **CV Parser** - парсинг PDF, DOCX, TXT
- **Structured Extraction** - извлечение данных через LLM

---

### Stack
- **Backend**: FastAPI, Pydantic
- **ML/AI**: LangChain, Google Gemini, OpenAI GPT-4
- **Vector DB**: Qdrant
- **Embeddings**: Google Gemini Embedding, Voyage AI
- **Sparse**: BM25, TF-IDF
- **UI**: Streamlit

## 🚀 Быстрый старт

### Требования
- Python 3.13+
- Docker + Docker Compose (для production)
- API Keys: Google AI, OpenAI, Qdrant

### 1. Установка зависимостей

```bash
# С uv (рекомендуется)
curl -LsSf https://astral.sh/uv/install.sh | sh
uv sync
```

### 2. Настройка окружения

Создайте `.env` файл:

```env
# LLM
OPENAI_API_KEY=sk-...
GOOGLE_API_KEY=...

# Llama Parse
LLAMA_PARSE_API=..

# Vector DB
QDRANT_URL=https://...
QDRANT_API=...
QDRANT_COLLECTION_NAME=CVs_google

# Email (опционально)
EMAIL_ADDRESS=your@email.com
EMAIL_PASSWORD=app_password
IMAP_SERVER=imap.gmail.com

# Sparse Method
DEFAULT_SPARSE_METHOD=bm25  # или tfidf
```

### 3. Запуск через Docker (Production)

```bash
# Запуск всех сервисов
docker-compose up --build
```

**Доступ:**
- API: http://localhost:8000/docs
- Frontend: http://localhost:8501
- Dashboard: http://localhost:8502

### 4. Запуск локально (Development)

В отдельных терминалах:

```bash
# Terminal 1: Backend
uvicorn app.main:app --reload --port 8000

# Terminal 2: Frontend
streamlit run app/ui/frontend.py --server.port 8501

# Terminal 3: Dashboard
streamlit run app/ui/dashboard.py --server.port 8502

# Terminal 4: Email Fetcher (опционально)
python -m app.scripts.fetch_emails
```

## 📝 Использование

### Загрузка резюме

#### Из текстовых файлов
```bash
# Поместите .txt файлы в data/Parsed_CVs/
python -m app.scripts.load_cvs
```

#### Из JSON (структурированные)
```bash
# Загрузка готовых JSON в Qdrant
python -m app.scripts.load_jsons_to_qdrant

# С переобучением sparse модели
python -m app.scripts.load_jsons_to_qdrant --refit-sparse
```

#### Из почты
```bash
# Однократная проверка
python -m app.scripts.fetch_emails

# Непрерывный мониторинг (в Docker)
# Запускается автоматически при docker-compose up
```

### API Examples

#### Поиск кандидатов

```bash
curl -X POST "http://localhost:8000/search" \
  -H "Content-Type: application/json" \
  -d '{
    "vacancy_text": "Ищем Python разработчика с опытом FastAPI и PostgreSQL",
    "search_mode": "hybrid",
    "top_k": 5
  }'
```

#### ML Classifier

```bash
curl -X POST "http://localhost:8000/search/ml-classifier" \
  -H "Content-Type: application/json" \
  -d '{
    "vacancy_text": "Backend разработчик Python/Django",
    "top_k": 10,
    "threshold": 0.5
  }'
```

#### LLM Анализ

```bash
curl -X POST "http://localhost:8000/search/with-llm-analysis" \
  -H "Content-Type: application/json" \
  -d '{
    "vacancy_text": "Senior Python Developer",
    "search_mode": "hybrid",
    "top_k": 5
  }'
```

## 📊 Оценка качества

```bash
# Запуск evaluation
python -m app.scripts.run_evaluation --hybrid

# Сравнение методов
python -m app.scripts.compare_modes

# Оценка ML классификатора
python -m app.scripts.evaluate_ml_classifier
```

Результаты сохраняются в `metrics/` и отображаются в Dashboard.

## 🗂️ Структура проекта

```
Bank_Home_Credit_Task/
├── app/
│   ├── api/              # API endpoints
│   ├── core/             # Конфигурация
│   ├── models/           # Pydantic модели
│   ├── services/         # Бизнес-логика
│   │   ├── cv_parser.py      # Парсинг и эмбеддинги
│   │   ├── cv_pipeline.py    # Пайплайн обработки
│   │   ├── email_fetcher.py  # Email мониторинг
│   │   ├── llm_analyze.py    # LLM анализ
│   │   └── ml_classifier.py  # ML классификатор
│   ├── scripts/          # Утилиты и скрипты
│   ├── ui/              # Streamlit интерфейсы
│   └── main.py          # FastAPI приложение
├── data/
│   ├── Raw_CVs/         # Исходные файлы
│   ├── Parsed_CVs/      # Текстовые резюме
│   ├── CV_JSONs/        # Структурированные данные
│   └── models/          # Sparse модели
├── metrics/             # Результаты оценки
├── docker-compose.yml   # Оркестрация сервисов
├── Dockerfile          # Docker образ
└── README.md
```

## 🔧 Конфигурация

### Настройка sparse метода

В `.env`:
```env
DEFAULT_SPARSE_METHOD=bm25  # или tfidf
```

### Настройка embeddings

В `app/services/cv_parser.py`:
- Google Gemini: `models/gemini-embedding-001` (3072 dim)
- Voyage AI: `voyage-4-large` (1024 dim)

### Настройка Qdrant

- **Collection**: автоматически создается при первом запуске
- **Vectors**: dense (3072) + sparse (BM25)
- **Distance**: COSINE

## 📈 Метрики качества

| Метод | MAP | Precision@5 | Recall@5 | 
|-------|-----|-------------|----------|
| Dense | 0.85 | 0.82 | 0.76 | 
| BM25 | 0.72 | 0.68 | 0.71 | 
| Hybrid | **0.91** | **0.88** | **0.83** | 

