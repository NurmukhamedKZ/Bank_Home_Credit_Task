"""
Скрипт для пакетной обработки резюме и сохранения в Qdrant
Использование: python app/process_cvs.py
"""

import sys
from pathlib import Path
from typing import List

# Добавляем путь к модулям
sys.path.insert(0, str(Path(__file__).parent))

from service.parse_pdf import CVParser


def process_txt_cvs(
    txt_folder: str | Path,
    collection_name: str = "CVs",
    skip_existing: bool = False
) -> dict:
    """
    Обрабатывает уже распарсенные резюме из TXT файлов и загружает в Qdrant
    Быстрее чем process_all_cvs, т.к. не требует парсинга PDF
    
    Args:
        txt_folder: Папка с .txt файлами резюме
        collection_name: Название коллекции в Qdrant
        skip_existing: Пропускать CV которые уже есть в Qdrant
        
    Returns:
        Словарь с результатами обработки
    """
    txt_folder = Path(txt_folder)
    
    if not txt_folder.exists():
        raise FileNotFoundError(f"Папка не найдена: {txt_folder}")
    
    # Инициализируем парсер
    print(f"🚀 Инициализация CVParser (коллекция: {collection_name})")
    parser = CVParser(collection_name=collection_name)
    
    # Находим все .txt файлы
    txt_files = list(txt_folder.glob("*.txt"))
    
    if not txt_files:
        print(f"⚠️  TXT файлы не найдены в {txt_folder}")
        return {"success": 0, "failed": 0, "skipped": 0, "results": []}
    
    print(f"📁 Найдено TXT файлов: {len(txt_files)}\n")
    
    # Получаем список уже загруженных CV если нужно
    existing_names = set()
    if skip_existing:
        try:
            # Получаем все точки из Qdrant
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
    
    # Обрабатываем каждый файл
    results = []
    failed = []
    skipped = []
    
    for i, txt_file in enumerate(txt_files, 1):
        print(f"\n{'='*60}")
        print(f"[{i}/{len(txt_files)}] Обработка: {txt_file.name}")
        print(f"{'='*60}")
        
        try:
            # 1. Читаем текст из файла
            print("📄 Чтение текста из файла...")
            full_text = txt_file.read_text(encoding='utf-8')
            
            if not full_text.strip():
                print(f"⚠️  Файл пуст, пропускаем")
                skipped.append({
                    "file": txt_file.name,
                    "reason": "empty_file"
                })
                continue
            
            print(f"   ✅ Прочитано {len(full_text)} символов")
            
            # 2. Извлекаем структурированные данные через LLM
            print("🤖 Извлечение структурированных данных...")
            cv_data = parser.extract_cv_data(full_text)
            print(f"   ✅ {cv_data.full_name}")
            
            # Проверяем не загружен ли уже
            if skip_existing and cv_data.full_name in existing_names:
                print(f"   ⏭️  CV уже в Qdrant, пропускаем")
                skipped.append({
                    "file": txt_file.name,
                    "full_name": cv_data.full_name,
                    "reason": "already_exists"
                })
                continue
            
            # 3. Создаем текст для поиска
            print("🔍 Создание текста для поиска...")
            searchable_text = parser.create_searchable_text(cv_data)
            print(f"   ✅ Создан ({len(searchable_text)} символов)")
            
            # 4. Создаем эмбеддинги
            print("🔢 Создание эмбеддингов...")
            dense_vector, sparse_indices, sparse_values = parser.create_embeddings(searchable_text)
            print(f"   ✅ Dense: {len(dense_vector)} dim, Sparse: {len(sparse_indices)} элементов")
            
            # 5. Сохраняем в Qdrant с именем файла для идентификации
            print("💾 Сохранение в Qdrant...")
            point_id = parser.save_to_qdrant(
                cv_data=cv_data,
                full_text=full_text,
                dense_vector=dense_vector,
                sparse_indices=sparse_indices,
                sparse_values=sparse_values,
                source_file=txt_file.stem  # Имя без расширения
            )
            
            results.append({
                "file": txt_file.name,
                "status": "success",
                "point_id": point_id,
                "full_name": cv_data.full_name,
                "email": cv_data.email,
                "experience_months": cv_data.total_experience_months,
                "skills_count": len(cv_data.skills)
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
    
    # После всех CV переобучаем TF-IDF на всем корпусе
    if len(results) > 0:
        print(f"\n{'='*60}")
        print("🔄 Переобучение TF-IDF на всем корпусе...")
        parser.refit_tfidf()
        print(f"✅ TF-IDF переобучен на {len(parser._tfidf_corpus)} документах")
    
    # Итоговая статистика
    print(f"\n{'='*60}")
    print("📊 ИТОГОВАЯ СТАТИСТИКА")
    print(f"{'='*60}")
    print(f"✅ Успешно обработано: {len(results)}")
    print(f"⏭️  Пропущено: {len(skipped)}")
    print(f"❌ Ошибок: {len(failed)}")
    print(f"📦 Всего файлов: {len(txt_files)}")
    
    if results:
        print(f"\n📋 Обработанные CV:")
        for r in results:
            print(f"  • {r['full_name']}")
            print(f"    Email: {r['email']}")
            print(f"    Опыт: {r['experience_months']} мес.")
            print(f"    Навыков: {r['skills_count']}")
            print(f"    Qdrant ID: {r['point_id']}")
            print()
    
    if skipped:
        print(f"\n⏭️  Пропущенные файлы:")
        for s in skipped:
            reason = "уже в базе" if s.get('reason') == 'already_exists' else s.get('reason', 'неизвестно')
            print(f"  • {s['file']}: {reason}")
    
    if failed:
        print(f"\n⚠️  Файлы с ошибками:")
        for f in failed:
            print(f"  • {f['file']}: {f['error']}")
    
    # Статистика коллекции
    print(f"\n{'='*60}")
    try:
        collection_info = parser.qdrant_client.get_collection(collection_name)
        print(f"📊 Коллекция '{collection_name}':")
        print(f"   Всего CV в базе: {collection_info.points_count}")
        print(f"   Размерность векторов: {collection_info.config.params.vectors['default'].size}")
    except Exception as e:
        print(f"⚠️  Не удалось получить статистику: {e}")
    
    print(f"{'='*60}\n")
    
    return {
        "success": len(results),
        "failed": len(failed),
        "skipped": len(skipped),
        "total": len(txt_files),
        "results": results,
        "errors": failed
    }


def process_all_cvs(
    cvs_folder: str | Path,
    collection_name: str = "CVs",
    file_extensions: List[str] = None
) -> dict:
    """
    Обрабатывает все резюме из указанной папки
    
    Args:
        cvs_folder: Папка с резюме
        collection_name: Название коллекции в Qdrant
        file_extensions: Список расширений для обработки (по умолчанию ['.pdf'])
        
    Returns:
        Словарь с результатами обработки
    """
    if file_extensions is None:
        file_extensions = ['.pdf']
    
    cvs_folder = Path(cvs_folder)
    
    if not cvs_folder.exists():
        raise FileNotFoundError(f"Папка не найдена: {cvs_folder}")
    
    # Инициализируем парсер
    print(f"🚀 Инициализация CVParser (коллекция: {collection_name})")
    parser = CVParser(collection_name=collection_name)
    
    # Находим все файлы
    all_files = []
    for ext in file_extensions:
        all_files.extend(cvs_folder.glob(f"*{ext}"))
    
    if not all_files:
        print(f"⚠️  Файлы не найдены в {cvs_folder}")
        return {"success": 0, "failed": 0, "results": []}
    
    print(f"📁 Найдено файлов: {len(all_files)}\n")
    
    # Обрабатываем каждый файл
    results = []
    failed = []
    
    for i, file_path in enumerate(all_files, 1):
        print(f"\n{'='*60}")
        print(f"[{i}/{len(all_files)}] Обработка: {file_path.name}")
        print(f"{'='*60}")
        
        try:
            result = parser.process_cv(file_path)
            results.append({
                "file": file_path.name,
                "status": "success",
                "point_id": result["point_id"],
                "full_name": result["full_name"],
                "email": result["email"],
                "experience_months": result["total_experience_months"],
                "skills_count": result["skills_count"],
                "raw_file": result.get("raw_file"),
                "json_file": result.get("json_file")
            })
            
            print(f"\n✅ Успешно обработано: {result['full_name']}")
            
        except Exception as e:
            print(f"\n❌ Ошибка при обработке {file_path.name}: {e}")
            failed.append({
                "file": file_path.name,
                "status": "failed",
                "error": str(e)
            })
            continue
    
    # Итоговая статистика
    print(f"\n{'='*60}")
    print("📊 ИТОГОВАЯ СТАТИСТИКА")
    print(f"{'='*60}")
    print(f"✅ Успешно обработано: {len(results)}")
    print(f"❌ Ошибок: {len(failed)}")
    print(f"📦 Всего файлов: {len(all_files)}")
    
    if results:
        print(f"\n📋 Обработанные CV:")
        for r in results:
            print(f"  • {r['full_name']}")
            print(f"    Email: {r['email']}")
            print(f"    Опыт: {r['experience_months']} мес.")
            print(f"    Навыков: {r['skills_count']}")
            print(f"    📄 Raw: {Path(r.get('raw_file', 'N/A')).name if r.get('raw_file') else 'N/A'}")
            print(f"    📋 JSON: {Path(r.get('json_file', 'N/A')).name if r.get('json_file') else 'N/A'}")
            print(f"    ☁️  Qdrant ID: {r['point_id']}")
            print()
    
    if failed:
        print(f"\n⚠️  Файлы с ошибками:")
        for f in failed:
            print(f"  • {f['file']}: {f['error']}")
    
    # Статистика коллекции
    print(f"\n{'='*60}")
    try:
        collection_info = parser.qdrant_client.get_collection(collection_name)
        print(f"📊 Коллекция '{collection_name}':")
        print(f"   Всего CV в базе: {collection_info.points_count}")
        print(f"   Размерность векторов: {collection_info.config.params.vectors['default'].size}")
    except Exception as e:
        print(f"⚠️  Не удалось получить статистику: {e}")
    
    print(f"{'='*60}\n")
    
    return {
        "success": len(results),
        "failed": len(failed),
        "total": len(all_files),
        "results": results,
        "errors": failed
    }


def main():
    """Основная функция"""
    
    # Пути к папкам
    project_root = Path(__file__).parent.parent
    cvs_folder = project_root / "data" / "CVs"
    parsed_cvs_folder = project_root / "data" / "Parsed_CVs"
    
    print("="*60)
    print("🎯 ПАКЕТНАЯ ОБРАБОТКА РЕЗЮМЕ")
    print("="*60)
    print("\nВыберите режим обработки:")
    print("  1. Обработать PDF файлы (медленно, полный пайплайн)")
    print("  2. Загрузить из TXT файлов (быстро, только в Qdrant)")
    print("  3. Обе папки (PDF + TXT)")
    print("="*60)
    
    choice = input("\nВведите номер (или 'q' для выхода): ").strip()
    
    if choice == 'q':
        print("Выход.")
        return
    
    try:
        if choice == '1':
            # Обработка PDF
            print(f"\n📁 Папка с PDF: {cvs_folder}")
            print(f"☁️  Коллекция Qdrant: CVs\n")
            
            summary = process_all_cvs(
                cvs_folder=cvs_folder,
                collection_name="CVs",
                file_extensions=['.pdf']
            )
            
        elif choice == '2':
            # Загрузка из TXT
            print(f"\n📁 Папка с TXT: {parsed_cvs_folder}")
            print(f"☁️  Коллекция Qdrant: CVs")
            
            skip = input("\nПропускать уже загруженные CV? (y/n): ").strip().lower()
            skip_existing = skip == 'y'
            print()
            
            summary = process_txt_cvs(
                txt_folder=parsed_cvs_folder,
                collection_name="CVs",
                skip_existing=skip_existing
            )
            
        elif choice == '3':
            # Обе папки
            print(f"\n📁 Сначала обработаем PDF из: {cvs_folder}")
            print(f"📁 Затем загрузим TXT из: {parsed_cvs_folder}\n")
            
            # PDF
            summary1 = process_all_cvs(
                cvs_folder=cvs_folder,
                collection_name="CVs",
                file_extensions=['.pdf']
            )
            
            # TXT (пропускаем уже загруженные)
            print("\n" + "="*60)
            print("Переходим к TXT файлам...")
            print("="*60 + "\n")
            
            summary2 = process_txt_cvs(
                txt_folder=parsed_cvs_folder,
                collection_name="CVs",
                skip_existing=True
            )
            
            summary = {
                "success": summary1['success'] + summary2['success'],
                "failed": summary1['failed'] + summary2['failed'],
                "skipped": summary2.get('skipped', 0)
            }
            
        else:
            print("❌ Неверный выбор!")
            return
        
        print("\n🎉 ОБРАБОТКА ЗАВЕРШЕНА!")
        print(f"✅ Успешно: {summary['success']}")
        if summary.get('skipped', 0) > 0:
            print(f"⏭️  Пропущено: {summary['skipped']}")
        print(f"❌ Ошибок: {summary['failed']}")
        
    except Exception as e:
        print(f"\n❌ Критическая ошибка: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
