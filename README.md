# CVParser - Парсер резюме с AI и векторным поиском

Класс для автоматической обработки резюме: парсинг PDF/DOCX, извлечение структурированных данных через LLM и сохранение в векторную базу данных Qdrant.

## 🚀 Возможности

- ✅ **Парсинг файлов**: PDF (через LlamaParse), DOCX, TXT
- ✅ **Структурирование данных**: Извлечение информации через GPT-4o-mini
- ✅ **Векторный поиск**: Hybrid search (dense + sparse embeddings)
- ✅ **Сохранение в Qdrant**: Автоматическое создание коллекции и индексов
- ✅ **Полный пайплайн**: Один метод для обработки от файла до базы данных

## 📦 Структура данных

Каждое CV преобразуется в структурированный JSON со следующими полями:

```python
{
    "full_name": str,
    "email": str,
    "phone": str,
    "links": [str],  # GitHub, LinkedIn, Portfolio
    "location": [str],
    "summary": str,  # Краткое профессиональное резюме
    "total_experience_months": int,
    "work_history": [
        {
            "role": str,
            "company": str,
            "start_date": str,  # YYYY-MM
            "end_date": str,  # YYYY-MM или "Present"
            "description": str,
            "technologies": [str]
        }
    ],
    "education": [
        {
            "institution": str,
            "degree": str,
            "year": str
        }
    ],
    "skills": [str],
    "languages": [str]
}
```

## 🔧 Установка зависимостей

```bash
pip install llama-parse
pip install langchain langchain-openai langchain-voyageai
pip install qdrant-client
pip install FlagEmbedding
pip install python-dotenv
```

## 🔑 Настройка переменных окружения

Создайте файл `.env` в корне проекта:

```env
# LlamaParse для парсинга PDF
LLAMA_PARSE_API=your_llama_parse_api_key

# OpenAI для структурирования данных
OPENAI_API_KEY=your_openai_api_key

# Qdrant для векторной базы данных
QDRANT_API=your_qdrant_api_key
QDRANT_URL=your_qdrant_url

# Voyage AI для dense embeddings
VOYAGE_API=your_voyage_api_key
```

## 💡 Использование

### Базовое использование (один файл)

```python
from Parse_pdf import CVParser

# Инициализация
parser = CVParser(collection_name="CVs")

# Обработка CV (полный пайплайн)
result = parser.process_cv("path/to/resume.pdf")

print(f"Имя: {result['full_name']}")
print(f"Email: {result['email']}")
print(f"Опыт: {result['total_experience_months']} месяцев")
```

### Пакетная обработка

```python
from pathlib import Path

parser = CVParser(collection_name="CVs")

# Обрабатываем все PDF в папке
cvs_folder = Path("data/CVs")
for pdf_file in cvs_folder.glob("*.pdf"):
    try:
        result = parser.process_cv(pdf_file)
        print(f"✅ {result['full_name']} обработан")
    except Exception as e:
        print(f"❌ Ошибка: {e}")
```

### Пошаговая обработка

Если нужен контроль на каждом шаге:

```python
parser = CVParser(collection_name="CVs")

# Шаг 1: Парсим файл
full_text = parser.parse_file("resume.pdf")

# Шаг 2: Извлекаем структурированные данные
cv_data = parser.extract_cv_data(full_text)

# Шаг 3: Создаем текст для поиска
searchable_text = parser.create_searchable_text(cv_data)

# Шаг 4: Создаем эмбеддинги
dense_vec, sparse_idx, sparse_val = parser.create_embeddings(searchable_text)

# Шаг 5: Сохраняем в Qdrant
point_id = parser.save_to_qdrant(
    cv_data=cv_data,
    full_text=full_text,
    dense_vector=dense_vec,
    sparse_indices=sparse_idx,
    sparse_values=sparse_val
)
```

### Поиск в Qdrant

```python
parser = CVParser(collection_name="CVs")

# Создаем эмбеддинг для запроса
query = "Python developer with FastAPI and PostgreSQL"
dense_vec, sparse_idx, sparse_val = parser.create_embeddings(query)

# Hybrid search
results = parser.qdrant_client.query_points(
    collection_name="CVs",
    query=dense_vec,
    using="default",
    limit=5,
    with_payload=True
)

# Выводим результаты
for point in results.points:
    print(f"{point.payload['full_name']} - Score: {point.score:.4f}")
    print(f"Skills: {', '.join(point.payload['skills'][:5])}")
```

## 🏗️ Архитектура

```
CVParser
├── parse_file()              # Парсинг PDF/DOCX/TXT
├── extract_cv_data()         # Структурирование через LLM
├── create_searchable_text()  # Оптимизация текста для поиска
├── create_embeddings()       # Dense + Sparse векторы
├── save_to_qdrant()         # Сохранение в БД
└── process_cv()             # Полный пайплайн (всё вместе)
```

## 🔍 Модели и технологии

- **Парсинг PDF**: LlamaParse (с OCR и распознаванием структуры)
- **Структурирование**: GPT-4o-mini через LangChain
- **Dense Embeddings**: Voyage AI (voyage-4-large, 1024 dim)
- **Sparse Embeddings**: BGE-M3 (BM25-like)
- **Векторная БД**: Qdrant (Hybrid Search)

## 📊 Производительность

- Парсинг PDF: ~10-30 секунд (зависит от размера)
- Структурирование LLM: ~5-10 секунд
- Эмбеддинги: ~2-5 секунд
- **Итого**: ~20-45 секунд на одно CV

## 🛠️ Расширение

### Добавление поддержки DOCX

```python
def parse_docx(self, file_path: str | Path) -> str:
    from docx import Document
    
    doc = Document(file_path)
    full_text = []
    
    for paragraph in doc.paragraphs:
        full_text.append(paragraph.text)
    
    return "\n".join(full_text)
```

### Кастомизация System Prompt

```python
parser = CVParser(collection_name="CVs")

# Изменяем промпт
parser.system_prompt = """
Your custom instructions for CV parsing...
"""

# Пересоздаем цепочку
parser.prompt = ChatPromptTemplate.from_messages([
    ("system", parser.system_prompt),
    ("user", "Resume:\n\n{text}")
])
parser.chain = parser.prompt | parser.structured_llm
```

### Добавление новых полей в CVOutput

```python
class CVOutput(BaseModel):
    # Существующие поля...
    
    # Добавляем новое поле
    certifications: List[str] = Field(
        default_factory=list,
        description="Professional certifications"
    )
```

## 🐛 Troubleshooting

### Ошибка: "Collection already exists"
Коллекция создается автоматически. Если хотите пересоздать:

```python
parser.qdrant_client.delete_collection("CVs")
parser._ensure_collection(1024)
```

### Ошибка: "File not found"
Проверьте путь к файлу:

```python
from pathlib import Path
file_path = Path("data/CVs/resume.pdf")
print(f"Существует: {file_path.exists()}")
print(f"Абсолютный путь: {file_path.absolute()}")
```

### Медленная обработка
- Используйте batch processing для нескольких CV
- Кэшируйте модели эмбеддингов
- Рассмотрите параллельную обработку

## 📝 Лицензия

MIT

## 👨‍💻 Автор

Создано на основе jupyter notebook `research/CV_parser.ipynb`
