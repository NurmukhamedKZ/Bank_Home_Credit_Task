"""
CVParser - класс для парсинга резюме из различных форматов (PDF, DOCX),
извлечения структурированных данных через LLM и сохранения в Qdrant.
"""

from pathlib import Path
from typing import List, Optional, Dict
import uuid
import json
import pickle
import re

# LlamaParse для парсинга PDF
from llama_parse import LlamaParse

# LangChain для структурированного парсинга
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

# Qdrant для векторного хранилища
from qdrant_client import QdrantClient, models

# Эмбеддинги
from langchain_voyageai import VoyageAIEmbeddings
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from rank_bm25 import BM25Okapi

# Pydantic модели
from app.models.cv import CVOutput

# Конфигурация
from app.core.config import (
    LLAMA_PARSE_API,
    OPENAI_API_KEY,
    QDRANT_API,
    QDRANT_URL,
    QDRANT_COLLECTION_NAME,
    VOYAGE_API,
    DEFAULT_SPARSE_METHOD,
)


class CVParser:
    """
    Класс для парсинга резюме из различных форматов (PDF, DOCX),
    извлечения структурированных данных через LLM и сохранения в Qdrant.
    """
    
    def __init__(
        self,
        collection_name: str = None,
        dense_model_name: str = "voyage-4-large",
        dense_output_dim: int = 1024,
        sparse_method: str = None,
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
            sparse_method: Метод для sparse embeddings - "tfidf" или "bm25"
            raw_cvs_folder: Папка для сохранения raw текстов CV (default: data/Raw_CVs)
            json_cvs_folder: Папка для сохранения JSON файлов (default: data/CV_JSONs)
            parsed_cvs_folder: Папка для parsed текстов (default: data/Parsed_CVs)
        """
        self.collection_name = collection_name or QDRANT_COLLECTION_NAME
        self.sparse_method = (sparse_method or DEFAULT_SPARSE_METHOD).lower()
        
        if self.sparse_method not in ["tfidf", "bm25"]:
            raise ValueError(f"sparse_method должен быть 'tfidf' или 'bm25', получено: {sparse_method}")
        
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
        
        # Sparse embeddings: TF-IDF или BM25
        if self.sparse_method == "tfidf":
            self.sparse_model = TfidfVectorizer(
                max_features=10000,
                ngram_range=(1, 2),
                min_df=1,
                sublinear_tf=True,
                lowercase=True,
                stop_words='english'
            )
            self._use_bm25 = False
        else:  # bm25
            self.sparse_model = None  # BM25 инициализируется после обучения
            self._use_bm25 = True
            self._bm25_tokenizer = self._default_tokenizer
        
        # Флаг обученности
        self._sparse_fitted = False
        # Корпус документов
        self._sparse_corpus = []
        # Токенизированный корпус для BM25
        self._tokenized_corpus = []
        
        # Путь к сохраненной модели
        model_filename = f"{self.sparse_method}_{collection_name}.pkl"
        self.sparse_model_path = project_root / "data" / "models" / model_filename
        self.sparse_model_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Автоматически загружаем модель если существует
        if self.sparse_model_path.exists():
            self.load_sparse_model()
            print(f"✅ {self.sparse_method.upper()} модель загружена из: {self.sparse_model_path.name}")
        
        # Qdrant клиент
        self.qdrant_client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API)
        
        # Создаем коллекцию если её нет
        self._ensure_collection(dense_output_dim)
    
    @staticmethod
    def _default_tokenizer(text: str) -> List[str]:
        """Простой токенизатор для BM25"""
        text = text.lower()
        tokens = re.findall(r'\b\w+\b', text)
        return tokens
    
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
        print(f"🔢 Создание эмбеддингов ({self.sparse_method.upper()})...")
        
        # Dense embedding
        dense_vector = self.dense_model.embed_documents([text])[0]
        
        # Sparse embedding
        if self._use_bm25:
            sparse_indices, sparse_values = self._create_bm25_embedding(text)
        else:
            sparse_indices, sparse_values = self._create_tfidf_embedding(text)
        
        print(f"✅ Эмбеддинги созданы ({self.sparse_method.upper()}: {len(sparse_indices)} ненулевых элементов)")
        return dense_vector, sparse_indices, sparse_values
    
    def _create_tfidf_embedding(self, text: str) -> tuple[List[int], List[float]]:
        """Создает TF-IDF sparse embedding"""
        if not self._sparse_fitted:
            self._sparse_corpus.append(text)
            self.sparse_model.fit(self._sparse_corpus)
            self._sparse_fitted = True
            print(f"   📚 TF-IDF обучен на {len(self._sparse_corpus)} документах")
            self.save_sparse_model()
        else:
            self._sparse_corpus.append(text)
        
        tfidf_vector = self.sparse_model.transform([text])
        tfidf_coo = tfidf_vector.tocoo()
        
        sparse_indices = tfidf_coo.col.tolist()
        sparse_values = tfidf_coo.data.tolist()
        
        return sparse_indices, sparse_values
    
    def _create_bm25_embedding(self, text: str) -> tuple[List[int], List[float]]:
        """Создает BM25 sparse embedding"""
        tokens = self._bm25_tokenizer(text)
        
        if not self._sparse_fitted:
            self._sparse_corpus.append(text)
            self._tokenized_corpus.append(tokens)
            self.sparse_model = BM25Okapi(self._tokenized_corpus)
            self._sparse_fitted = True
            print(f"   📚 BM25 обучен на {len(self._tokenized_corpus)} документах")
            self.save_sparse_model()
        else:
            self._sparse_corpus.append(text)
            self._tokenized_corpus.append(tokens)
        
        scores = self.sparse_model.get_scores(tokens)
        
        sparse_indices = []
        sparse_values = []
        
        all_tokens = []
        for doc_tokens in self._tokenized_corpus:
            all_tokens.extend(doc_tokens)
        vocab = {token: idx for idx, token in enumerate(sorted(set(all_tokens)))}
        
        token_counts = {}
        for token in tokens:
            if token in vocab:
                token_counts[token] = token_counts.get(token, 0) + 1
        
        for token, count in token_counts.items():
            if token in vocab:
                sparse_indices.append(vocab[token])
                sparse_values.append(float(count))
        
        return sparse_indices, sparse_values
    
    def refit_sparse(self, auto_save: bool = True):
        """
        Переобучает sparse модель (TF-IDF или BM25) на всем накопленном корпусе
        """
        if len(self._sparse_corpus) == 0:
            print("⚠️  Корпус пуст, нечего обучать")
            return
        
        print(f"🔄 Переобучение {self.sparse_method.upper()} на {len(self._sparse_corpus)} документах...")
        
        if self._use_bm25:
            self._tokenized_corpus = [self._bm25_tokenizer(text) for text in self._sparse_corpus]
            self.sparse_model = BM25Okapi(self._tokenized_corpus)
        else:
            self.sparse_model.fit(self._sparse_corpus)
        
        self._sparse_fitted = True
        print(f"✅ {self.sparse_method.upper()} переобучен")
        
        if auto_save:
            self.save_sparse_model()
    
    def refit_tfidf(self, auto_save: bool = True):
        """Алиас для refit_sparse (обратная совместимость)"""
        self.refit_sparse(auto_save)
    
    def save_sparse_model(self):
        """Сохраняет обученную sparse модель и корпус на диск"""
        if not self._sparse_fitted:
            print(f"⚠️  {self.sparse_method.upper()} не обучен, нечего сохранять")
            return
        
        model_data = {
            'sparse_model': self.sparse_model,
            'corpus': self._sparse_corpus,
            'tokenized_corpus': self._tokenized_corpus if self._use_bm25 else None,
            'fitted': self._sparse_fitted,
            'method': self.sparse_method,
            'use_bm25': self._use_bm25
        }
        
        if not self._use_bm25 and hasattr(self.sparse_model, 'vocabulary_'):
            model_data['vocabulary_size'] = len(self.sparse_model.vocabulary_)
        elif self._use_bm25:
            all_tokens = set()
            for tokens in self._tokenized_corpus:
                all_tokens.update(tokens)
            model_data['vocabulary_size'] = len(all_tokens)
        
        with open(self.sparse_model_path, 'wb') as f:
            pickle.dump(model_data, f)
        
        vocab_size = model_data.get('vocabulary_size', 0)
        print(f"💾 {self.sparse_method.upper()} модель сохранена: {self.sparse_model_path.name}")
        print(f"   📊 Словарь: {vocab_size} слов, Корпус: {len(self._sparse_corpus)} документов")
    
    def load_sparse_model(self) -> bool:
        """Загружает сохраненную sparse модель с диска"""
        if not self.sparse_model_path.exists():
            return False
        
        try:
            with open(self.sparse_model_path, 'rb') as f:
                model_data = pickle.load(f)
            
            saved_method = model_data.get('method', 'tfidf')
            if saved_method != self.sparse_method:
                print(f"⚠️  Несовпадение метода: сохранено '{saved_method}', ожидается '{self.sparse_method}'")
                return False
            
            self.sparse_model = model_data['sparse_model']
            self._sparse_corpus = model_data['corpus']
            self._sparse_fitted = model_data['fitted']
            
            if self._use_bm25:
                self._tokenized_corpus = model_data.get('tokenized_corpus', [])
            
            vocab_size = model_data.get('vocabulary_size', 0)
            print(f"📂 {self.sparse_method.upper()} загружен: {vocab_size} слов, {len(self._sparse_corpus)} документов")
            
            return True
        except Exception as e:
            print(f"⚠️  Ошибка загрузки {self.sparse_method.upper()} модели: {e}")
            self._sparse_fitted = False
            self._sparse_corpus = []
            self._tokenized_corpus = []
            return False
    
    def save_tfidf_model(self):
        """Алиас для save_sparse_model (обратная совместимость)"""
        self.save_sparse_model()
    
    def load_tfidf_model(self) -> bool:
        """Алиас для load_sparse_model (обратная совместимость)"""
        return self.load_sparse_model()
    
    def create_sparse_query(self, query_text: str) -> tuple[List[int], List[float]]:
        """Создает sparse query вектор для поиска"""
        if not self._sparse_fitted:
            raise ValueError(f"{self.sparse_method.upper()} не обучен!")
        
        if self._use_bm25:
            return self._create_bm25_query(query_text)
        else:
            return self._create_tfidf_query(query_text)
    
    def _create_tfidf_query(self, query_text: str) -> tuple[List[int], List[float]]:
        """Создает TF-IDF query вектор"""
        query_vector = self.sparse_model.transform([query_text.lower()])
        query_coo = query_vector.tocoo()
        
        return query_coo.col.tolist(), query_coo.data.tolist()
    
    def _create_bm25_query(self, query_text: str) -> tuple[List[int], List[float]]:
        """Создает BM25 query вектор"""
        query_tokens = self._bm25_tokenizer(query_text)
        scores = self.sparse_model.get_scores(query_tokens)
        
        all_tokens = []
        for doc_tokens in self._tokenized_corpus:
            all_tokens.extend(doc_tokens)
        vocab = {token: idx for idx, token in enumerate(sorted(set(all_tokens)))}
        
        sparse_indices = []
        sparse_values = []
        
        token_counts = {}
        for token in query_tokens:
            if token in vocab:
                token_counts[token] = token_counts.get(token, 0) + 1
        
        for token, count in token_counts.items():
            if token in vocab:
                sparse_indices.append(vocab[token])
                sparse_values.append(float(count))
        
        return sparse_indices, sparse_values
    
    def cv_to_payload(self, cv: CVOutput, full_text: str, source_file: str = None) -> dict:
        """Преобразует CVOutput в payload для Qdrant"""
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
        
        return {
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
            "source_file": source_file
        }
    
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
        """Сохраняет CV в Qdrant"""
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
        """Сохраняет raw текст резюме в файл"""
        base_name = Path(original_filename).stem
        output_file = self.parsed_cvs_folder / f"{base_name}.txt"
        output_file.write_text(full_text, encoding='utf-8')
        print(f"💾 Raw текст сохранен: {output_file.name}")
        return output_file
    
    def save_json(self, cv_data: CVOutput, original_filename: str) -> Path:
        """Сохраняет структурированные данные CV в JSON файл"""
        base_name = Path(original_filename).stem
        output_file = self.json_cvs_folder / f"{base_name}.json"
        
        cv_dict = cv_data.model_dump()
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(cv_dict, f, ensure_ascii=False, indent=2)
        
        print(f"💾 JSON сохранен: {output_file.name}")
        return output_file
    
    def process_cv(self, file_path: str | Path) -> Dict:
        """
        Полный пайплайн обработки CV: парсинг -> структурирование -> файлы -> эмбеддинги -> Qdrant
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
        
        # 6. Сохраняем в Qdrant
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


# Legacy функция для обратной совместимости
def parse_pdf(file_name: str) -> str:
    """Legacy функция для парсинга PDF"""
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
