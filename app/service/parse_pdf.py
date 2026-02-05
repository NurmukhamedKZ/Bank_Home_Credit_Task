from dotenv import load_dotenv
from pathlib import Path
from typing import List, Optional, Dict
import os
import uuid
import json
import pickle

# LlamaParse для парсинга PDF
from llama_parse import LlamaParse

# LangChain для структурированного парсинга
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

# Pydantic модели для структуры CV
from pydantic import BaseModel, Field

# Qdrant для векторного хранилища
from qdrant_client import QdrantClient, models

# Эмбеддинги
from langchain_voyageai import VoyageAIEmbeddings
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

load_dotenv()

# API ключи
LLAMA_PARSE_API = os.getenv("LLAMA_PARSE_API")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
QDRANT_API = os.getenv("QDRANT_API")
QDRANT_URL = os.getenv("QDRANT_URL")
VOYAGE_API = os.getenv("VOYAGE_API")


# ==================== PYDANTIC МОДЕЛИ ====================

class WorkExperience(BaseModel):
    """Структура для опыта работы"""
    role: str = Field(description="Job title, e.g. 'Senior Python Developer'")
    company: str = Field(description="Company name")
    start_date: str = Field(description="Start date usually in YYYY-MM format")
    end_date: str = Field(description="End date in YYYY-MM format or 'Present'")
    description: str = Field(description="Short summary of responsibilities and achievements")
    technologies: List[str] = Field(description="Specific tools used in this role")


class Education(BaseModel):
    """Структура для образования"""
    institution: str
    degree: str = Field(description="Degree, e.g. 'Bachelor in Computer Science'")
    year: str = Field(description="Year of graduation")


class CVOutput(BaseModel):
    """Основная модель структурированного CV"""
    full_name: str = Field(description="Candidate's full name")
    email: Optional[str] = Field(description="Email address")
    phone: Optional[str] = Field(description="Phone number")
    links: List[str] = Field(default_factory=list, description="URLs to LinkedIn, GitHub, Portfolio")
    location: List[str] = Field(default_factory=list, description="Location of the candidate")
    
    summary: str = Field(description="A brief professional summary of the candidate")
    
    total_experience_months: int = Field(description="Total work experience in months")
    
    work_history: List[WorkExperience] = Field(default_factory=list, description="List of work experiences")
    education: List[Education] = Field(default_factory=list)
    
    skills: List[str] = Field(default_factory=list, description="List of technical/hard skills")
    languages: List[str] = Field(default_factory=list, description="Languages spoken and proficiency level")


# ==================== ОСНОВНОЙ КЛАСС ====================

class CVParser:
    """
    Класс для парсинга резюме из различных форматов (PDF, DOCX),
    извлечения структурированных данных через LLM и сохранения в Qdrant.
    """
    
    def __init__(
        self,
        collection_name: str = "CVs",
        dense_model_name: str = "voyage-4-large",
        dense_output_dim: int = 1024,
        raw_cvs_folder: str | Path = None,
        json_cvs_folder: str | Path = None,
        parsed_cvs_folder: str | Path = None
    ):
        """
        Инициализация парсера
        
        Args:
            collection_name: Название коллекции в Qdrant
            dense_model_name: Название модели для dense embeddings
            dense_output_dim: Размерность dense векторов
            raw_cvs_folder: Папка для сохранения raw текстов CV (default: data/Raw_CVs)
            json_cvs_folder: Папка для сохранения JSON файлов (default: data/CV_JSONs)
        """
        self.collection_name = collection_name
        
        # Настройка папок для сохранения
        project_root = Path(__file__).parent.parent.parent
        self.raw_cvs_folder = Path(raw_cvs_folder) if raw_cvs_folder else project_root / "data" / "Raw_CVs"
        self.json_cvs_folder = Path(json_cvs_folder) if json_cvs_folder else project_root / "data" / "CV_JSONs"
        self.parsed_cvs_folder = Path(parsed_cvs_folder) if parsed_cvs_folder else project_root / "data" / "Parsed_CVs"
        
        # Создаем папки если их нет
        self.raw_cvs_folder.mkdir(parents=True, exist_ok=True)
        self.json_cvs_folder.mkdir(parents=True, exist_ok=True)
        self.parsed_cvs_folder.mkdir(parents=True, exist_ok=True)
        
        print(f"📁 Raw CVs folder: {self.raw_cvs_folder}")
        print(f"📁 JSON CVs folder: {self.json_cvs_folder}")
        print(f"📁 Parsed CVs folder: {self.parsed_cvs_folder}")
        
        # Инициализируем парсеры для разных форматов
        self.pdf_parser = LlamaParse(
            api_key=LLAMA_PARSE_API,
            parse_mode="parse_page_with_llm",
            result_type="markdown",
            high_res_ocr=True,
        )
        
        # LLM для структурированного парсинга
        self.llm = ChatOpenAI(
            model="gpt-4o-mini",
            api_key=OPENAI_API_KEY,
            temperature=0
        )
        
        self.structured_llm = self.llm.with_structured_output(CVOutput)
        
        # System prompt для парсинга CV
        self.system_prompt = """
You are an expert technical recruiter and CV parser.
Your task is to extract structured data from the provided resume text.

CRITICAL RULES:
1. Be precise with dates and names.
2. If a specific field is missing, leave it as None or an empty list.
3. For 'work_history', try to split distinct roles even if they are in the same company.
4. Extract ALL technical skills mentioned.
5. In 'total_experience_months', calculate the sum of all work durations.
"""
        
        # Создаем prompt для LLM
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", self.system_prompt),
            ("user", "Resume:\n\n{text}")
        ])
        
        # Цепочка для парсинга
        self.chain = self.prompt | self.structured_llm
        
        # Модели для эмбеддингов
        self.dense_model = VoyageAIEmbeddings(
            voyage_api_key=VOYAGE_API,
            model=dense_model_name,
            output_dimension=dense_output_dim
        )
        
        # TF-IDF для sparse embeddings
        self.sparse_model = TfidfVectorizer(
            max_features=10000,  # Максимум 10k наиболее важных слов
            ngram_range=(1, 2),  # Uni-grams и bi-grams
            min_df=1,  # Минимальная частота документа
            sublinear_tf=True,  # Использовать логарифмическую шкалу для TF
            lowercase=True,  # Приводить к нижнему регистру
            stop_words='english'  # Удалять английские стоп-слова (можно добавить русские)
        )
        
        # Флаг, был ли обучен TF-IDF
        self._tfidf_fitted = False
        # Хранилище документов для обучения TF-IDF
        self._tfidf_corpus = []
        
        # Путь к сохраненной TF-IDF модели
        self.tfidf_model_path = project_root / "data" / "models" / f"tfidf_{collection_name}.pkl"
        self.tfidf_model_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Автоматически загружаем TF-IDF модель если она существует
        if self.tfidf_model_path.exists():
            self.load_tfidf_model()
            print(f"✅ TF-IDF модель загружена из: {self.tfidf_model_path.name}")
        
        # Qdrant клиент
        self.qdrant_client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API)
        
        # Создаем коллекцию если её нет
        self._ensure_collection(dense_output_dim)
    
    def _ensure_collection(self, vector_size: int):
        """Создает коллекцию в Qdrant если её нет"""
        if not self.qdrant_client.collection_exists(self.collection_name):
            self.qdrant_client.create_collection(
                collection_name=self.collection_name,
                vectors_config={
                    "default": models.VectorParams(
                        size=vector_size,
                        distance=models.Distance.COSINE
                    )
                },
                sparse_vectors_config={
                    "sparse": models.SparseVectorParams(
                        index=models.SparseIndexParams(on_disk=True)
                    )
                }
            )
            print(f"✅ Коллекция '{self.collection_name}' создана")
    
    def parse_pdf(self, file_path: str | Path) -> str:
        """
        Парсит PDF файл в текст
        
        Args:
            file_path: Путь к PDF файлу
            
        Returns:
            Текст резюме
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"Файл не найден: {file_path}")
        
        print(f"📄 Парсинг PDF: {file_path.name}")
        parsed_documents = self.pdf_parser.load_data(str(file_path))
        
        # Объединяем все страницы
        full_text = "\n\n".join([doc.text for doc in parsed_documents])
        
        print(f"✅ PDF успешно распарсен ({len(parsed_documents)} страниц)")
        return full_text
    
    def parse_docx(self, file_path: str | Path) -> str:
        """
        Парсит DOCX файл в текст
        
        Args:
            file_path: Путь к DOCX файлу
            
        Returns:
            Текст резюме
        """
        # TODO: Реализовать парсинг DOCX
        # Можно использовать python-docx или LlamaParse
        raise NotImplementedError("DOCX парсинг будет добавлен позже")
    
    def parse_file(self, file_path: str | Path) -> str:
        """
        Автоматически определяет тип файла и парсит его
        
        Args:
            file_path: Путь к файлу
            
        Returns:
            Текст резюме
        """
        file_path = Path(file_path)
        suffix = file_path.suffix.lower()
        
        if suffix == '.pdf':
            return self.parse_pdf(file_path)
        elif suffix in ['.docx', '.doc']:
            return self.parse_docx(file_path)
        elif suffix == '.txt':
            return file_path.read_text(encoding='utf-8')
        else:
            raise ValueError(f"Неподдерживаемый формат файла: {suffix}")
    
    def extract_cv_data(self, text: str) -> CVOutput:
        """
        Извлекает структурированные данные из текста резюме через LLM
        
        Args:
            text: Текст резюме
            
        Returns:
            Структурированные данные CV
        """
        print("🤖 Извлечение структурированных данных через LLM...")
        cv_data = self.chain.invoke({"text": text})
        print(f"✅ Данные извлечены: {cv_data.full_name}")
        return cv_data
    
    def create_searchable_text(self, cv: CVOutput) -> str:
        """
        Создает текст для векторного поиска из структурированных данных
        
        Args:
            cv: Структурированные данные CV
            
        Returns:
            Оптимизированный текст для поиска
        """
        main_info = f"Candidate for {cv.work_history[0].role if cv.work_history else 'Professional'}. "
        skills = f"Main Skills: {', '.join(cv.skills)}. "
        
        # Добавляем описание последних мест работы
        exp_descriptions = []
        for work in cv.work_history:
            exp_descriptions.append(f"{work.role} at {work.company}: {work.description}")
        
        experience_text = " Experience summary: " + " | ".join(exp_descriptions)
        
        search_text = main_info + skills + cv.summary + experience_text
        return search_text.lower()
    
    def create_embeddings(self, text: str) -> tuple[List[float], List[int], List[float]]:
        """
        Создает dense и sparse эмбеддинги для текста
        
        Args:
            text: Текст для эмбеддинга
            
        Returns:
            Tuple (dense_vector, sparse_indices, sparse_values)
        """
        print("🔢 Создание эмбеддингов...")
        
        # Dense embedding
        dense_vector = self.dense_model.embed_documents([text])[0]
        
        # Sparse embedding с TF-IDF
        # Добавляем текст в корпус если еще не обучен
        if not self._tfidf_fitted:
            self._tfidf_corpus.append(text)
            # Обучаем TF-IDF на накопленном корпусе
            self.sparse_model.fit(self._tfidf_corpus)
            self._tfidf_fitted = True
            print(f"   📚 TF-IDF обучен на {len(self._tfidf_corpus)} документах")
            # Сохраняем модель после первого обучения
            self.save_tfidf_model()
        else:
            # Добавляем в корпус для будущего переобучения
            self._tfidf_corpus.append(text)
        
        # Трансформируем текст в TF-IDF вектор
        tfidf_vector = self.sparse_model.transform([text])
        
        # Конвертируем sparse matrix в индексы и значения
        # tfidf_vector это scipy.sparse.csr_matrix
        tfidf_coo = tfidf_vector.tocoo()  # Конвертируем в COO формат
        
        # Извлекаем ненулевые элементы
        sparse_indices = tfidf_coo.col.tolist()  # Индексы столбцов (слов)
        sparse_values = tfidf_coo.data.tolist()  # Значения TF-IDF
        
        print(f"✅ Эмбеддинги созданы (TF-IDF: {len(sparse_indices)} ненулевых элементов)")
        return dense_vector, sparse_indices, sparse_values
    
    def refit_tfidf(self, auto_save: bool = True):
        """
        Переобучает TF-IDF на всем накопленном корпусе
        Полезно вызвать после обработки нескольких документов
        
        Args:
            auto_save: Автоматически сохранить модель после обучения
        """
        if len(self._tfidf_corpus) > 0:
            print(f"🔄 Переобучение TF-IDF на {len(self._tfidf_corpus)} документах...")
            self.sparse_model.fit(self._tfidf_corpus)
            self._tfidf_fitted = True
            print("✅ TF-IDF переобучен")
            
            # Автоматически сохраняем модель
            if auto_save:
                self.save_tfidf_model()
        else:
            print("⚠️  Корпус пуст, нечего обучать")
    
    def save_tfidf_model(self):
        """
        Сохраняет обученную TF-IDF модель и корпус на диск
        """
        if not self._tfidf_fitted:
            print("⚠️  TF-IDF не обучен, нечего сохранять")
            return
        
        # Сохраняем модель и корпус в один файл
        model_data = {
            'sparse_model': self.sparse_model,
            'corpus': self._tfidf_corpus,
            'fitted': self._tfidf_fitted,
            'vocabulary_size': len(self.sparse_model.vocabulary_) if hasattr(self.sparse_model, 'vocabulary_') else 0
        }
        
        with open(self.tfidf_model_path, 'wb') as f:
            pickle.dump(model_data, f)
        
        vocab_size = model_data['vocabulary_size']
        print(f"💾 TF-IDF модель сохранена: {self.tfidf_model_path.name}")
        print(f"   📊 Словарь: {vocab_size} слов, Корпус: {len(self._tfidf_corpus)} документов")
    
    def load_tfidf_model(self) -> bool:
        """
        Загружает сохраненную TF-IDF модель с диска
        
        Returns:
            True если загрузка успешна, False иначе
        """
        if not self.tfidf_model_path.exists():
            return False
        
        try:
            with open(self.tfidf_model_path, 'rb') as f:
                model_data = pickle.load(f)
            
            self.sparse_model = model_data['sparse_model']
            self._tfidf_corpus = model_data['corpus']
            self._tfidf_fitted = model_data['fitted']
            
            vocab_size = model_data.get('vocabulary_size', 0)
            print(f"📂 TF-IDF загружен: {vocab_size} слов, {len(self._tfidf_corpus)} документов")
            
            return True
        except Exception as e:
            print(f"⚠️  Ошибка загрузки TF-IDF модели: {e}")
            self._tfidf_fitted = False
            self._tfidf_corpus = []
            return False
    
    def create_sparse_query(self, query_text: str) -> tuple[List[int], List[float]]:
        """
        Создает sparse query вектор для поиска (TF-IDF)
        
        Args:
            query_text: Текст запроса (вакансия)
            
        Returns:
            Tuple (sparse_indices, sparse_values)
        """
        if not self._tfidf_fitted:
            raise ValueError("TF-IDF не обучен! Сначала обработайте хотя бы один документ.")
        
        # Трансформируем query в TF-IDF вектор
        query_vector = self.sparse_model.transform([query_text.lower()])
        query_coo = query_vector.tocoo()
        
        sparse_indices = query_coo.col.tolist()
        sparse_values = query_coo.data.tolist()
        
        return sparse_indices, sparse_values
    
    def cv_to_payload(self, cv: CVOutput, full_text: str, source_file: str = None) -> dict:
        """
        Преобразует CVOutput в payload для Qdrant
        
        Args:
            cv: Структурированные данные CV
            full_text: Полный текст резюме
            
        Returns:
            Словарь с данными для payload
        """
        # Преобразуем вложенные объекты в словари
        work_history_dicts = [
            {
                "role": work.role,
                "company": work.company,
                "start_date": work.start_date,
                "end_date": work.end_date,
                "description": work.description,
                "technologies": work.technologies
            }
            for work in cv.work_history
        ]
        
        education_dicts = [
            {
                "institution": edu.institution,
                "degree": edu.degree,
                "year": edu.year
            }
            for edu in cv.education
        ]
        
        payload = {
            "full_content": full_text,
            "full_name": cv.full_name,
            "email": cv.email,
            "phone": cv.phone,
            "links": cv.links,
            "location": cv.location,
            "summary": cv.summary,
            "total_experience_months": cv.total_experience_months,
            "work_history": work_history_dicts,
            "education": education_dicts,
            "skills": cv.skills,
            "languages": cv.languages,
            "source_file": source_file}
        return payload
    
    def save_to_qdrant(
        self,
        cv_data: CVOutput,
        full_text: str,
        dense_vector: List[float],
        sparse_indices: List[int],
        sparse_values: List[float],
        point_id: Optional[str] = None,
        source_file: str = None
    ) -> str:
        """
        Сохраняет CV в Qdrant
        
        Args:
            cv_data: Структурированные данные CV
            full_text: Полный текст резюме
            dense_vector: Dense эмбеддинг
            sparse_indices: Индексы sparse эмбеддинга
            sparse_values: Значения sparse эмбеддинга
            point_id: ID точки (если None, генерируется автоматически)
            source_file: Имя исходного файла (для идентификации в метриках)
            
        Returns:
            ID сохраненной точки
        """
        if point_id is None:
            point_id = str(uuid.uuid4())
        
        payload = self.cv_to_payload(cv_data, full_text, source_file)
        
        point = models.PointStruct(
            id=point_id,
            vector={
                "default": dense_vector,
                "sparse": models.SparseVector(indices=sparse_indices, values=sparse_values)
            },
            payload=payload
        )
        
        print(f"💾 Сохранение в Qdrant...")
        self.qdrant_client.upsert(
            collection_name=self.collection_name,
            points=[point],
            wait=True
        )
        
        print(f"✅ CV сохранено в Qdrant (ID: {point_id})")
        return point_id
    
    def save_raw_text(self, full_text: str, original_filename: str) -> Path:
        """
        Сохраняет raw текст резюме в файл
        
        Args:
            full_text: Полный текст резюме
            original_filename: Оригинальное имя файла
            
        Returns:
            Путь к сохраненному файлу
        """
        # Создаем имя файла (без расширения оригинала + .txt)
        base_name = Path(original_filename).stem
        output_file = self.parsed_cvs_folder / f"{base_name}.txt"
        
        # Сохраняем текст
        output_file.write_text(full_text, encoding='utf-8')
        print(f"💾 Raw текст сохранен: {output_file.name}")
        
        return output_file
    
    def save_json(self, cv_data: CVOutput, original_filename: str) -> Path:
        """
        Сохраняет структурированные данные CV в JSON файл
        
        Args:
            cv_data: Структурированные данные CV
            original_filename: Оригинальное имя файла
            
        Returns:
            Путь к сохраненному файлу
        """
        # Создаем имя файла
        base_name = Path(original_filename).stem
        output_file = self.json_cvs_folder / f"{base_name}.json"
        
        # Конвертируем Pydantic модель в dict
        cv_dict = cv_data.model_dump()
        
        # Сохраняем в JSON с красивым форматированием
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(cv_dict, f, ensure_ascii=False, indent=2)
        
        print(f"💾 JSON сохранен: {output_file.name}")
        
        return output_file
    
    def process_cv(self, file_path: str | Path) -> Dict:
        """
        Полный пайплайн обработки CV: парсинг -> структурирование -> файлы -> эмбеддинги -> Qdrant
        
        Args:
            file_path: Путь к файлу резюме
            
        Returns:
            Словарь с результатами обработки
        """
        file_path = Path(file_path)
        
        print(f"\n{'='*60}")
        print(f"🚀 Начало обработки: {file_path.name}")
        print(f"{'='*60}\n")
        
        # 1. Парсим файл
        full_text = self.parse_file(file_path)
        
        # 2. Извлекаем структурированные данные
        cv_data = self.extract_cv_data(full_text)
        
        # 3. Сохраняем промежуточные результаты в файлы
        print("\n📝 Сохранение промежуточных файлов...")
        raw_file = self.save_raw_text(full_text, file_path.name)
        json_file = self.save_json(cv_data, file_path.name)
        
        # 4. Создаем текст для поиска
        searchable_text = self.create_searchable_text(cv_data)
        
        # 5. Создаем эмбеддинги
        dense_vector, sparse_indices, sparse_values = self.create_embeddings(searchable_text)
        
        # 6. Сохраняем в Qdrant с именем исходного файла
        # Используем stem (без расширения) для сопоставления
        source_file_stem = file_path.stem
        point_id = self.save_to_qdrant(
            cv_data=cv_data,
            full_text=full_text,
            dense_vector=dense_vector,
            sparse_indices=sparse_indices,
            sparse_values=sparse_values,
            source_file=source_file_stem
        )
        
        print(f"\n{'='*60}")
        print(f"✅ Обработка завершена успешно!")
        print(f"📄 Raw текст: {raw_file}")
        print(f"📋 JSON: {json_file}")
        print(f"☁️  Qdrant ID: {point_id}")
        print(f"{'='*60}\n")
        
        return {
            "point_id": point_id,
            "full_name": cv_data.full_name,
            "email": cv_data.email,
            "total_experience_months": cv_data.total_experience_months,
            "skills_count": len(cv_data.skills),
            "cv_data": cv_data,
            "raw_file": str(raw_file),
            "json_file": str(json_file)
        }


# ==================== LEGACY ФУНКЦИЯ (для обратной совместимости) ====================

def parse_pdf(file_name: str) -> str:
    """
    Legacy функция для парсинга PDF (оставлена для обратной совместимости)
    
    Args:
        file_name: Имя файла в папке data/CVs/
        
    Returns:
        Текст резюме
    """
    parser = LlamaParse(
        api_key=LLAMA_PARSE_API,
        parse_mode="parse_page_with_llm",
        result_type="markdown",
        high_res_ocr=True,
    )

    dir_path = Path(__file__)
    pdf_path = dir_path.parent.parent.parent / "data" / "CVs" / file_name

    if pdf_path.exists():
        print(f"Парсинг PDF: {pdf_path}")
        parsed_documents = parser.load_data(str(pdf_path))
        
        full_cv = ""
        for doc in parsed_documents:
            full_cv += doc.text
        
        return full_cv
    else:
        raise FileNotFoundError(f"Файл не найден: {pdf_path}")