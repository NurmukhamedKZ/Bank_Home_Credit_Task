"""
CVPipeline - автоматический пайплайн обработки резюме.

Полный цикл:
    Email → Сохранение по типу → Парсинг → Структурирование → JSON → Qdrant (BM25)

Использование:
    pipeline = CVPipeline()
    pipeline.process_file("/path/to/resume.pdf")   # Один файл
    pipeline.run_email_watcher(interval=60)         # Мониторинг почты
"""

import time
import shutil
import traceback
from pathlib import Path
from typing import List, Dict, Optional
from datetime import datetime

from app.services.cv_parser import CVParser
from app.services.email_fetcher import EmailFetcher
from app.core.config import QDRANT_COLLECTION_NAME


class CVPipeline:
    """
    Автоматический пайплайн обработки резюме от получения до индексации в Qdrant.
    
    Этапы:
        1. Получение файла (из email или вручную)
        2. Сортировка по типу (PDF → PDF_CVs, DOCX → DOCX_CVs, TXT → Parsed_CVs)
        3. Парсинг в текст (PDF/DOCX → текст)
        4. Сохранение текста в Parsed_CVs
        5. Структурирование через LLM → CVOutput
        6. Сохранение JSON в CV_JSONs
        7. Создание эмбеддингов (dense + BM25)
        8. Сохранение в Qdrant
    """
    
    def __init__(
        self,
        collection_name: str = None,
        sparse_method: str = "bm25"
    ):
        """
        Args:
            collection_name: Название коллекции в Qdrant
            sparse_method: Метод sparse embeddings ("bm25" или "tfidf")
        """
        self.collection_name = collection_name or QDRANT_COLLECTION_NAME
        self.sparse_method = sparse_method
        
        # Пути к папкам
        self.project_root = Path(__file__).parent.parent.parent
        self.pdf_cvs_folder = self.project_root / "data" / "PDF_CVs"
        self.docx_cvs_folder = self.project_root / "data" / "DOCX_CVs"
        self.parsed_cvs_folder = self.project_root / "data" / "Parsed_CVs"
        self.json_cvs_folder = self.project_root / "data" / "CV_JSONs"
        
        # Создаем папки
        for folder in [self.pdf_cvs_folder, self.docx_cvs_folder, self.parsed_cvs_folder, self.json_cvs_folder]:
            folder.mkdir(parents=True, exist_ok=True)
        
        # CVParser для парсинга и сохранения в Qdrant
        print("🚀 Инициализация CVPipeline...")
        self.parser = CVParser(
            collection_name=collection_name,
            sparse_method=sparse_method
        )
        
        # Статистика
        self.stats = {
            "processed": 0,
            "failed": 0,
            "skipped": 0
        }
        
        print(f"📁 PDF_CVs: {self.pdf_cvs_folder}")
        print(f"📁 DOCX_CVs: {self.docx_cvs_folder}")
        print(f"📁 Parsed_CVs: {self.parsed_cvs_folder}")
        print(f"📁 CV_JSONs: {self.json_cvs_folder}")
        print(f"✅ CVPipeline готов (коллекция: {collection_name}, метод: {sparse_method})\n")
    
    def _get_existing_source_files(self) -> set:
        """Получает множество source_file из Qdrant для проверки дубликатов"""
        try:
            scroll_result = self.parser.qdrant_client.scroll(
                collection_name=self.collection_name,
                limit=10000,
                with_payload=True,
                with_vectors=False
            )
            return {
                point.payload.get("source_file", "")
                for point in scroll_result[0]
                if point.payload.get("source_file")
            }
        except Exception:
            return set()
    
    def _sort_file(self, file_path: Path) -> Path:
        """
        Сортирует файл по типу в соответствующую папку.
        
        Args:
            file_path: Путь к исходному файлу
            
        Returns:
            Путь к файлу в новой папке
        """
        ext = file_path.suffix.lower()
        
        if ext == ".pdf":
            target_folder = self.pdf_cvs_folder
        elif ext in [".docx", ".doc"]:
            target_folder = self.docx_cvs_folder
        elif ext == ".txt":
            target_folder = self.parsed_cvs_folder
        else:
            # Неизвестный формат — сохраняем как есть в Raw_CVs
            target_folder = self.project_root / "data" / "Raw_CVs"
            target_folder.mkdir(parents=True, exist_ok=True)
        
        target_path = target_folder / file_path.name
        
        # Если файл уже в нужной папке — не копируем
        if file_path.resolve() == target_path.resolve():
            return target_path
        
        # Если файл с таким именем уже есть — добавляем timestamp
        if target_path.exists():
            stem = file_path.stem
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            target_path = target_folder / f"{stem}_{timestamp}{ext}"
        
        shutil.copy2(file_path, target_path)
        print(f"   📂 Сохранён в: {target_folder.name}/{target_path.name}")
        
        return target_path
    
    def _parse_to_text(self, file_path: Path) -> str:
        """
        Парсит файл в текст. PDF/DOCX через LlamaParse, TXT напрямую.
        
        Args:
            file_path: Путь к файлу
            
        Returns:
            Извлечённый текст
        """
        ext = file_path.suffix.lower()
        
        if ext == ".txt":
            return file_path.read_text(encoding="utf-8")
        elif ext == ".pdf":
            return self.parser.parse_pdf(file_path)
        elif ext in [".docx", ".doc"]:
            return self.parser.parse_file(file_path)
        else:
            raise ValueError(f"Неподдерживаемый формат: {ext}")
    
    def process_file(self, file_path: str | Path, skip_existing: bool = True) -> Optional[Dict]:
        """
        Полный пайплайн обработки одного файла.
        
        Этапы:
            1. Сортировка по типу (PDF/DOCX/TXT → соответствующая папка)
            2. Парсинг в текст
            3. Сохранение текста в Parsed_CVs
            4. Структурирование через LLM → CVOutput
            5. Сохранение JSON в CV_JSONs
            6. Эмбеддинги (dense + BM25)
            7. Сохранение в Qdrant
        
        Args:
            file_path: Путь к файлу резюме
            skip_existing: Пропускать файлы уже загруженные в Qdrant
            
        Returns:
            Словарь с результатами обработки или None если ошибка
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            print(f"❌ Файл не найден: {file_path}")
            return None
        
        print(f"\n{'='*60}")
        print(f"🚀 Обработка: {file_path.name}")
        print(f"{'='*60}")
        
        # Проверяем дубликаты
        if skip_existing:
            existing = self._get_existing_source_files()
            if file_path.stem in existing:
                print(f"   ⏭️  Уже в Qdrant, пропускаем")
                self.stats["skipped"] += 1
                return None
        
        try:
            # 1. Сортировка по типу
            print("   📂 Шаг 1: Сортировка по типу...")
            sorted_path = self._sort_file(file_path)
            
            # 2. Парсинг в текст
            print("   📄 Шаг 2: Парсинг в текст...")
            full_text = self._parse_to_text(sorted_path)
            
            if not full_text or not full_text.strip():
                print("   ⚠️  Пустой текст, пропускаем")
                self.stats["failed"] += 1
                return None
            
            print(f"      ✅ Извлечено {len(full_text)} символов")
            
            # 3. Сохранение текста в Parsed_CVs
            parsed_txt_path = self.parsed_cvs_folder / f"{file_path.stem}.txt"
            if not parsed_txt_path.exists() or sorted_path.suffix.lower() != ".txt":
                parsed_txt_path.write_text(full_text, encoding="utf-8")
                print(f"   💾 Шаг 3: Текст сохранён → Parsed_CVs/{parsed_txt_path.name}")
            else:
                print(f"   💾 Шаг 3: Текст уже в Parsed_CVs")
            
            # 4. Структурирование через LLM
            print("   🤖 Шаг 4: Извлечение структурированных данных (LLM)...")
            cv_data = self.parser.extract_cv_data(full_text)
            print(f"      ✅ {cv_data.full_name} | {cv_data.total_experience_months} мес. | {len(cv_data.skills)} навыков")
            
            # 5. Сохранение JSON
            print("   📋 Шаг 5: Сохранение JSON...")
            json_path = self.parser.save_json(cv_data, file_path.name)
            print(f"      ✅ → CV_JSONs/{json_path.name}")
            
            # 6. Создание эмбеддингов
            print(f"   🔢 Шаг 6: Создание эмбеддингов ({self.sparse_method.upper()})...")
            searchable_text = self.parser.create_searchable_text(cv_data)
            dense_vector, sparse_indices, sparse_values = self.parser.create_embeddings(searchable_text)
            print(f"      ✅ Dense: {len(dense_vector)} dims, Sparse: {len(sparse_indices)} non-zero")
            
            # 7. Сохранение в Qdrant
            print("   ☁️  Шаг 7: Сохранение в Qdrant...")
            point_id = self.parser.save_to_qdrant(
                cv_data=cv_data,
                full_text=full_text,
                dense_vector=dense_vector,
                sparse_indices=sparse_indices,
                sparse_values=sparse_values,
                source_file=file_path.stem
            )
            
            self.stats["processed"] += 1
            
            print(f"\n   ✅ ГОТОВО: {cv_data.full_name}")
            print(f"      Qdrant ID: {point_id}")
            print(f"{'='*60}\n")
            
            return {
                "file": file_path.name,
                "point_id": point_id,
                "full_name": cv_data.full_name,
                "email": cv_data.email,
                "experience_months": cv_data.total_experience_months,
                "skills_count": len(cv_data.skills),
                "json_file": str(json_path),
                "parsed_text": str(parsed_txt_path)
            }
        
        except Exception as e:
            print(f"\n   ❌ ОШИБКА: {e}")
            traceback.print_exc()
            self.stats["failed"] += 1
            return None
    
    def process_files(self, file_paths: List[Path], skip_existing: bool = True) -> List[Dict]:
        """
        Обрабатывает несколько файлов.
        
        Args:
            file_paths: Список путей к файлам
            skip_existing: Пропускать уже загруженные
            
        Returns:
            Список результатов обработки
        """
        results = []
        total = len(file_paths)
        
        print(f"\n📦 Обработка {total} файлов...\n")
        
        for i, fp in enumerate(file_paths, 1):
            print(f"[{i}/{total}]", end="")
            result = self.process_file(fp, skip_existing=skip_existing)
            if result:
                results.append(result)
        
        # Переобучаем sparse модель на полном корпусе
        if results:
            print(f"\n🔄 Переобучение {self.sparse_method.upper()} на всём корпусе...")
            self.parser.refit_sparse()
        
        self._print_stats()
        return results
    
    def process_from_email(
        self,
        folder: str = "INBOX",
        search_criteria: str = "UNSEEN",
        mark_as_read: bool = True
    ) -> List[Dict]:
        """
        Получает резюме из email и обрабатывает их через полный пайплайн.
        
        Args:
            folder: Папка почты
            search_criteria: Критерий поиска
            mark_as_read: Помечать как прочитанное
            
        Returns:
            Список результатов обработки
        """
        print(f"\n{'='*60}")
        print("📧 ПОЛУЧЕНИЕ РЕЗЮМЕ ИЗ ПОЧТЫ")
        print(f"{'='*60}\n")
        
        # Получаем файлы из почты
        with EmailFetcher() as fetcher:
            saved_files = fetcher.fetch_resumes(
                folder=folder,
                search_criteria=search_criteria,
                save_text_body=True,
                mark_as_read=mark_as_read
            )
        
        if not saved_files:
            print("\n📭 Новых резюме не найдено")
            return []
        
        print(f"\n📬 Получено {len(saved_files)} файлов из почты")
        
        # Обрабатываем каждый файл через пайплайн
        file_paths = [Path(f) for f in saved_files]
        return self.process_files(file_paths)
    
    def run_email_watcher(
        self,
        interval: int = 60,
        folder: str = "INBOX",
        mark_as_read: bool = True
    ):
        """
        Непрерывный мониторинг почты. Проверяет новые письма каждые N секунд.
        
        Args:
            interval: Интервал проверки в секундах (default: 60)
            folder: Папка почты
            mark_as_read: Помечать как прочитанное
        """
        print(f"""
╔═══════════════════════════════════════════════════════════════╗
║         АВТОМАТИЧЕСКИЙ МОНИТОРИНГ ПОЧТЫ                       ║
║  Интервал: {interval:>3} сек | Папка: {folder:<10}                        ║
║  Нажмите Ctrl+C для остановки                                 ║
╚═══════════════════════════════════════════════════════════════╝
        """)
        
        cycle = 0
        
        try:
            while True:
                cycle += 1
                now = datetime.now().strftime("%H:%M:%S")
                print(f"\n[{now}] 🔄 Проверка #{cycle}...")
                
                try:
                    results = self.process_from_email(
                        folder=folder,
                        search_criteria="UNSEEN",
                        mark_as_read=mark_as_read
                    )
                    
                    if results:
                        print(f"[{now}] ✅ Обработано {len(results)} новых резюме")
                    else:
                        print(f"[{now}] 📭 Новых резюме нет")
                
                except Exception as e:
                    print(f"[{now}] ⚠️  Ошибка: {e}")
                
                print(f"[{now}] 💤 Следующая проверка через {interval} сек...")
                time.sleep(interval)
        
        except KeyboardInterrupt:
            print(f"\n\n{'='*60}")
            print("🛑 Мониторинг остановлен")
            self._print_stats()
            print(f"{'='*60}\n")
    
    def _print_stats(self):
        """Выводит статистику"""
        print(f"\n{'='*60}")
        print("📊 СТАТИСТИКА ПАЙПЛАЙНА")
        print(f"{'='*60}")
        print(f"✅ Обработано: {self.stats['processed']}")
        print(f"⏭️  Пропущено: {self.stats['skipped']}")
        print(f"❌ Ошибок: {self.stats['failed']}")
        print(f"{'='*60}")
