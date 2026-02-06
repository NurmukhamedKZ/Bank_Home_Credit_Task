#!/usr/bin/env python3
"""
Запуск автоматического пайплайна обработки резюме.

Режимы:
    python -m app.scripts.run_pipeline --watch          # Мониторинг почты (каждые 60 сек)
    python -m app.scripts.run_pipeline --watch -i 120   # Мониторинг (каждые 120 сек)
    python -m app.scripts.run_pipeline --once            # Одноразовая проверка почты
    python -m app.scripts.run_pipeline --file resume.pdf # Обработать один файл
    python -m app.scripts.run_pipeline --folder ./cvs    # Обработать папку с файлами
"""

import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        description="Автоматический пайплайн обработки резюме"
    )
    
    # Режимы работы
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--watch", action="store_true", help="Непрерывный мониторинг почты")
    mode.add_argument("--once", action="store_true", help="Одноразовая проверка почты")
    mode.add_argument("--file", type=str, help="Обработать один файл")
    mode.add_argument("--folder", type=str, help="Обработать все файлы в папке")
    
    # Настройки
    parser.add_argument("-i", "--interval", type=int, default=60, help="Интервал проверки почты в секундах (default: 60)")
    parser.add_argument("--collection", type=str, default=None, help="Название коллекции Qdrant (default: из .env)")
    parser.add_argument("--no-skip", action="store_true", help="Не пропускать существующие CV")
    
    args = parser.parse_args()
    
    from app.services.cv_pipeline import CVPipeline
    
    pipeline = CVPipeline(
        collection_name=args.collection,  # None = возьмёт из .env QDRANT_COLLECTION_NAME
    )
    
    skip_existing = not args.no_skip
    
    # ========== МОНИТОРИНГ ПОЧТЫ ==========
    if args.watch:
        pipeline.run_email_watcher(interval=args.interval)
    
    # ========== ОДНОРАЗОВАЯ ПРОВЕРКА ==========
    elif args.once:
        print(f"""
╔═══════════════════════════════════════════════════════════════╗
║         ОДНОРАЗОВАЯ ПРОВЕРКА ПОЧТЫ                            ║
╚═══════════════════════════════════════════════════════════════╝
        """)
        
        results = pipeline.process_from_email()
        
        if results:
            print(f"\n✅ Обработано {len(results)} резюме:")
            for r in results:
                print(f"   • {r['full_name']} ({r['file']})")
        else:
            print("\n📭 Новых резюме не найдено")
    
    # ========== ОДИН ФАЙЛ ==========
    elif args.file:
        file_path = Path(args.file)
        
        print(f"""
╔═══════════════════════════════════════════════════════════════╗
║         ОБРАБОТКА ФАЙЛА                                       ║
║  Файл: {file_path.name:<53} ║
╚═══════════════════════════════════════════════════════════════╝
        """)
        
        result = pipeline.process_file(file_path, skip_existing=skip_existing)
        
        if result:
            print(f"\n✅ Файл обработан:")
            print(f"   Имя: {result['full_name']}")
            print(f"   Email: {result.get('email', '-')}")
            print(f"   Опыт: {result['experience_months']} мес.")
            print(f"   Навыков: {result['skills_count']}")
            print(f"   JSON: {result['json_file']}")
            print(f"   Qdrant ID: {result['point_id']}")
        else:
            print("\n❌ Файл не обработан (ошибка или уже существует)")
    
    # ========== ПАПКА ==========
    elif args.folder:
        folder = Path(args.folder)
        
        if not folder.exists():
            print(f"❌ Папка не найдена: {folder}")
            return
        
        # Собираем все поддерживаемые файлы
        extensions = ["*.pdf", "*.docx", "*.doc", "*.txt"]
        files = []
        for ext in extensions:
            files.extend(folder.glob(ext))
        
        if not files:
            print(f"⚠️  Нет файлов в {folder}")
            return
        
        print(f"""
╔═══════════════════════════════════════════════════════════════╗
║         ОБРАБОТКА ПАПКИ                                       ║
║  Папка: {folder.name:<52} ║
║  Файлов: {len(files):<51} ║
╚═══════════════════════════════════════════════════════════════╝
        """)
        
        results = pipeline.process_files(files, skip_existing=skip_existing)
        
        if results:
            print(f"\n✅ Обработано {len(results)} файлов:")
            for r in results:
                print(f"   • {r['full_name']} ({r['file']})")


if __name__ == "__main__":
    main()
