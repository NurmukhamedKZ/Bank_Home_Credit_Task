"""
CV Search Application - ИИ-агент для подбора кандидатов по вакансиям.

Структура модуля:
    - models/      : Pydantic модели (CV, API запросы/ответы)
    - api/         : FastAPI эндпоинты
    - services/    : Бизнес-логика (парсинг CV, поиск, email)
    - evaluation/  : Оценка качества поиска
    - ui/          : Streamlit интерфейсы
    - scripts/     : CLI скрипты
    - experiments/ : Эксперименты с моделями

Запуск API:
    uvicorn app.main:app --reload --port 8000

Запуск UI:
    streamlit run app/ui/frontend.py

CLI команды:
    python -m app.scripts.load_cvs            # Загрузка CV в Qdrant
    python -m app.scripts.run_evaluation      # Оценка качества
    python -m app.scripts.compare_modes       # Сравнение режимов
    python -m app.scripts.fetch_emails        # Получение резюме из почты
"""

__version__ = "1.0.0"
