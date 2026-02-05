"""
Главный модуль приложения - FastAPI сервер для поиска кандидатов.

Запуск:
    uvicorn app.main:app --reload --port 8000
    
Документация API:
    http://localhost:8000/docs
"""

from contextlib import asynccontextmanager
from fastapi import FastAPI

from app.api.routes import router, set_cv_parser, clear_cv_parser
from app.services.cv_parser import CVParser


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Инициализация и очистка ресурсов при запуске/остановке приложения"""
    print("🚀 Запуск API сервера...")
    print("📊 Инициализация CVParser...")
    
    # Инициализируем CVParser при старте
    cv_parser = CVParser(collection_name="CVs")
    set_cv_parser(cv_parser)
    
    # Проверяем соединение с Qdrant
    try:
        collection_info = cv_parser.qdrant_client.get_collection(cv_parser.collection_name)
        print(f"✅ Подключено к Qdrant. Документов в базе: {collection_info.points_count}")
    except Exception as e:
        print(f"⚠️ Ошибка подключения к Qdrant: {e}")
    
    # Проверяем TF-IDF
    if cv_parser._sparse_fitted:
        print(f"✅ {cv_parser.sparse_method.upper()} модель загружена")
    else:
        print(f"⚠️ {cv_parser.sparse_method.upper()} не обучен - sparse и hybrid поиск недоступны")
    
    print("✅ API готов к работе!")
    
    yield
    
    # Очистка при остановке
    print("👋 Остановка API сервера...")
    clear_cv_parser()


app = FastAPI(
    title="CV Search API",
    description="""
    API для поиска релевантных кандидатов по тексту вакансии.
    
    ## Возможности
    
    * **Семантический поиск** - через Voyage AI embeddings
    * **Keyword поиск** - через TF-IDF или BM25
    * **Гибридный поиск** - комбинация обоих методов (рекомендуется)
    
    ## Использование
    
    1. Отправьте POST запрос на `/search` с текстом вакансии
    2. Получите список релевантных кандидатов с оценками
    """,
    version="1.0.0",
    lifespan=lifespan
)

# Подключаем роуты
app.include_router(router)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
