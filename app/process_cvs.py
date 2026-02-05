"""
Скрипт для пакетной обработки резюме и сохранения в Qdrant
Использование: python app/process_cvs.py
"""

import sys
from pathlib import Path
from typing import List

# Добавляем путь к модулям
sys.path.insert(0, str(Path(__file__).parent))

from service.Parse_pdf import CVParser


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
    
    # Путь к папке с резюме
    project_root = Path(__file__).parent.parent
    cvs_folder = project_root / "data" / "CVs"
    
    print("="*60)
    print("🎯 ПАКЕТНАЯ ОБРАБОТКА РЕЗЮМЕ")
    print("="*60)
    print(f"Папка: {cvs_folder}")
    print(f"Коллекция Qdrant: CVs")
    print("="*60 + "\n")
    
    try:
        summary = process_all_cvs(
            cvs_folder=cvs_folder,
            collection_name="CVs",
            file_extensions=['.pdf', '.txt']  # Можно добавить .docx
        )
        
        print("\n🎉 ОБРАБОТКА ЗАВЕРШЕНА!")
        print(f"✅ Успешно: {summary['success']}")
        print(f"❌ Ошибок: {summary['failed']}")
        
    except Exception as e:
        print(f"\n❌ Критическая ошибка: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
