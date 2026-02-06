#!/usr/bin/env python3
"""
Загрузка CV в Qdrant.

Использование:
    python -m app.scripts.load_cvs           # TXT файлы с TF-IDF
    python -m app.scripts.load_cvs --bm25    # TXT файлы с BM25
    python -m app.scripts.load_cvs --pdf     # PDF файлы
"""

import sys
from pathlib import Path


def process_txt_cvs(
    txt_folder: Path,
    collection_name: str = "CVs",
    skip_existing: bool = False,
    sparse_method: str = "tfidf"
) -> dict:
    """Обрабатывает TXT файлы и загружает в Qdrant"""
    from app.services.cv_parser import CVParser
    
    if not txt_folder.exists():
        raise FileNotFoundError(f"Папка не найдена: {txt_folder}")
    
    print(f"🚀 Инициализация CVParser (коллекция: {collection_name}, метод: {sparse_method})")
    parser = CVParser(collection_name=collection_name, sparse_method=sparse_method)
    
    txt_files = list(txt_folder.glob("*.txt"))
    
    if not txt_files:
        print(f"⚠️  TXT файлы не найдены в {txt_folder}")
        return {"success": 0, "failed": 0, "skipped": 0, "results": []}
    
    print(f"📁 Найдено TXT файлов: {len(txt_files)}\n")
    
    # Получаем список уже загруженных CV
    existing_names = set()
    if skip_existing:
        try:
            scroll_result = parser.qdrant_client.scroll(
                collection_name=collection_name,
                limit=1000,
                with_payload=True,
                with_vectors=False
            )
            existing_names = {
                point.payload.get('full_name', '') 
                for point in scroll_result[0]
            }
            print(f"ℹ️  Уже в Qdrant: {len(existing_names)} CV\n")
        except Exception as e:
            print(f"⚠️  Не удалось получить список из Qdrant: {e}\n")
    
    results = []
    failed = []
    skipped = []
    
    for i, txt_file in enumerate(txt_files, 1):
        print(f"\n{'='*60}")
        print(f"[{i}/{len(txt_files)}] Обработка: {txt_file.name}")
        print(f"{'='*60}")
        
        try:
            print("📄 Чтение текста из файла...")
            full_text = txt_file.read_text(encoding='utf-8')
            
            if not full_text.strip():
                print(f"⚠️  Файл пуст, пропускаем")
                skipped.append({"file": txt_file.name, "reason": "empty_file"})
                continue
            
            print(f"   ✅ Прочитано {len(full_text)} символов")
            
            print("🤖 Извлечение структурированных данных...")
            cv_data = parser.extract_cv_data(full_text)
            print(f"   ✅ {cv_data.full_name}")
            
            # Сохраняем JSON с тем же именем что и оригинальный файл
            print("📋 Сохранение JSON...")
            json_file = parser.save_json(cv_data, txt_file.name)
            print(f"   ✅ {json_file.name}")
            
            if skip_existing and cv_data.full_name in existing_names:
                print(f"   ⏭️  CV уже в Qdrant, пропускаем")
                skipped.append({
                    "file": txt_file.name,
                    "full_name": cv_data.full_name,
                    "reason": "already_exists"
                })
                continue
            
            print("🔍 Создание текста для поиска...")
            searchable_text = parser.create_searchable_text(cv_data)
            
            print("🔢 Создание эмбеддингов...")
            dense_vector, sparse_indices, sparse_values = parser.create_embeddings(searchable_text)
            
            print("💾 Сохранение в Qdrant...")
            point_id = parser.save_to_qdrant(
                cv_data=cv_data,
                full_text=full_text,
                dense_vector=dense_vector,
                sparse_indices=sparse_indices,
                sparse_values=sparse_values,
                source_file=txt_file.stem
            )
            
            results.append({
                "file": txt_file.name,
                "status": "success",
                "point_id": point_id,
                "full_name": cv_data.full_name,
                "email": cv_data.email,
                "experience_months": cv_data.total_experience_months,
                "skills_count": len(cv_data.skills),
                "json_file": str(json_file)
            })
            
            print(f"\n✅ Успешно обработано: {cv_data.full_name}")
            
        except Exception as e:
            print(f"\n❌ Ошибка при обработке {txt_file.name}: {e}")
            import traceback
            traceback.print_exc()
            failed.append({
                "file": txt_file.name,
                "status": "failed",
                "error": str(e)
            })
            continue
    
    # Переобучаем sparse модель
    if len(results) > 0:
        print(f"\n{'='*60}")
        print(f"🔄 Переобучение {parser.sparse_method.upper()} на всем корпусе...")
        parser.refit_sparse()
    
    # Статистика
    print(f"\n{'='*60}")
    print("📊 ИТОГОВАЯ СТАТИСТИКА")
    print(f"{'='*60}")
    print(f"✅ Успешно: {len(results)}")
    print(f"⏭️  Пропущено: {len(skipped)}")
    print(f"❌ Ошибок: {len(failed)}")
    
    return {
        "success": len(results),
        "failed": len(failed),
        "skipped": len(skipped),
        "total": len(txt_files),
        "results": results,
        "errors": failed
    }


def main():
    """Основная функция"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Загрузка CV в Qdrant")
    parser.add_argument("--tfidf", action="store_true", help="Использовать TF-IDF вместо BM25")
    parser.add_argument("--pdf", action="store_true", help="Обрабатывать PDF файлы")
    parser.add_argument("--skip-existing", action="store_true", default=True, help="Пропускать существующие CV")
    parser.add_argument("--no-skip", action="store_true", help="Не пропускать существующие CV")
    
    args = parser.parse_args()
    
    project_root = Path(__file__).parent.parent.parent
    sparse_method = "tfidf" if args.tfidf else None  # None = дефолт BM25 из конфига
    skip_existing = not args.no_skip
    
    from app.core.config import DEFAULT_SPARSE_METHOD
    display_method = sparse_method or DEFAULT_SPARSE_METHOD
    
    print(f"""
╔═══════════════════════════════════════════════════════════════╗
║         ЗАГРУЗКА CV В QDRANT                                  ║
║  Sparse метод: {display_method.upper():<10}                                  ║
╚═══════════════════════════════════════════════════════════════╝
    """)
    
    if args.pdf:
        cvs_folder = project_root / "data" / "PDF_CVs"
        print(f"📁 Папка с PDF: {cvs_folder}")
        # TODO: Реализовать обработку PDF
        print("⚠️  Обработка PDF пока не реализована в этом скрипте")
    else:
        parsed_cvs_folder = project_root / "data" / "Parsed_CVs"
        print(f"📁 Папка с TXT: {parsed_cvs_folder}")
        
        from app.core.config import QDRANT_COLLECTION_NAME
        
        summary = process_txt_cvs(
            txt_folder=parsed_cvs_folder,
            collection_name=QDRANT_COLLECTION_NAME,
            skip_existing=skip_existing,
            sparse_method=sparse_method
        )
        
        print(f"""
╔═══════════════════════════════════════════════════════════════╗
║                    ЗАГРУЗКА ЗАВЕРШЕНА                         ║
╚═══════════════════════════════════════════════════════════════╝

✅ Успешно: {summary['success']}
⏭️  Пропущено: {summary['skipped']}
❌ Ошибок: {summary['failed']}

💡 Теперь можно запустить оценку:
   python -m app.scripts.run_evaluation --hybrid
        """)


if __name__ == "__main__":
    main()
