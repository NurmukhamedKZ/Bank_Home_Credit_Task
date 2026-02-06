#!/usr/bin/env python3
"""
Загрузка структурированных резюме из JSON в Qdrant.

Скрипт берет готовые JSON файлы из папки CV_JSONs,
создает векторные представления и сохраняет в Qdrant.

Использование:
    python -m app.scripts.load_jsons_to_qdrant [--refit-sparse]
"""

import json
from pathlib import Path
import argparse
from typing import List

from app.services.cv_parser import CVParser
from app.models.cv import CVOutput, WorkExperience, Education
from app.core.config import QDRANT_COLLECTION_NAME


def json_to_cv_output(data: dict) -> CVOutput:
    """Преобразование JSON в CVOutput модель"""
    
    # Преобразуем work_history
    work_history = []
    for work in data.get("work_history", []):
        work_history.append(WorkExperience(
            role=work.get("role", ""),
            company=work.get("company", ""),
            start_date=work.get("start_date"),
            end_date=work.get("end_date"),
            description=work.get("description"),
            technologies=work.get("technologies", [])
        ))
    
    # Преобразуем education
    education = []
    for edu in data.get("education", []):
        education.append(Education(
            institution=edu.get("institution", ""),
            degree=edu.get("degree"),
            year=edu.get("year")
        ))
    
    # Создаем CVOutput
    return CVOutput(
        full_name=data.get("full_name", "Unknown"),
        email=data.get("email"),
        phone=data.get("phone"),
        links=data.get("links", []),
        location=data.get("location", []),
        summary=data.get("summary", ""),
        total_experience_months=data.get("total_experience_months", 0),
        work_history=work_history,
        education=education,
        skills=data.get("skills", []),
        languages=data.get("languages", [])
    )


def load_json_files(json_folder: Path, needing_file: str) -> List[tuple[Path, dict]]:
    """Загрузка всех JSON файлов из папки"""
    json_files = []
    
    if not json_folder.exists():
        print(f"❌ Папка не найдена: {json_folder}")
        return json_files
    
    for json_file in sorted(json_folder.glob("*.json")):
        try:
            if needing_file != json_file.name:
                continue
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            json_files.append((json_file, data))
        except Exception as e:
            print(f"⚠️  Ошибка чтения {json_file.name}: {e}")
    
    return json_files


def main():
    parser = argparse.ArgumentParser(description="Загрузка JSON резюме в Qdrant")
    parser.add_argument(
        "--refit-sparse",
        action="store_true",
        help="Переобучить sparse модель (BM25/TF-IDF) после загрузки"
    )
    args = parser.parse_args()
    

    needing_file = "AI_engineer_3.json"

    print(f"""
╔═══════════════════════════════════════════════════════════════╗
║         ЗАГРУЗКА JSON РЕЗЮМЕ В QDRANT                         ║
╚═══════════════════════════════════════════════════════════════╝
    """)
    
    # Инициализируем CVParser
    cv_parser = CVParser(collection_name=QDRANT_COLLECTION_NAME)
    
    # Путь к JSON файлам
    json_folder = cv_parser.json_cvs_folder
    print(f"📁 Папка с JSON: {json_folder}")
    
    # Загружаем JSON файлы
    json_files = load_json_files(json_folder, needing_file=needing_file)
    
    if not json_files:
        print("\n❌ JSON файлы не найдены")
        return
    
    print(f"📋 Найдено файлов: {len(json_files)}\n")
    
    # Статистика
    success_count = 0
    skip_count = 0
    error_count = 0
    
    # Обрабатываем каждый JSON
    for idx, (json_file, data) in enumerate(json_files, 1):
        print(f"\n{'='*60}")
        print(f"[{idx}/{len(json_files)}] Обработка: {json_file.name}")
        print(f"{'='*60}")
        
        try:
            # Преобразуем в CVOutput
            cv_data = json_to_cv_output(data)
            print(f"👤 Кандидат: {cv_data.full_name}")
            
            # Создаем текст для поиска
            print("🔍 Создание текста для поиска...")
            searchable_text = cv_parser.create_searchable_text(cv_data)
            
            # Создаем эмбеддинги
            dense_vector, sparse_indices, sparse_values = cv_parser.create_embeddings(searchable_text)
            
            # Получаем source_file из имени JSON
            source_file = json_file.stem
            
            # Сохраняем в Qdrant
            # Используем полный текст из JSON если есть
            full_text = data.get("full_content", searchable_text)
            
            point_id = cv_parser.save_to_qdrant(
                cv_data=cv_data,
                full_text=full_text,
                dense_vector=dense_vector,
                sparse_indices=sparse_indices,
                sparse_values=sparse_values,
                source_file=source_file
            )
            
            print(f"✅ Успешно загружено в Qdrant (ID: {point_id})")
            success_count += 1
            
        except Exception as e:
            print(f"❌ Ошибка: {e}")
            error_count += 1
            continue
    
    # Переобучаем sparse модель если нужно
    if args.refit_sparse and cv_parser._sparse_corpus:
        print(f"\n{'='*60}")
        print("🔄 Переобучение sparse модели...")
        cv_parser.refit_sparse(auto_save=True)
    
    # Итоговая статистика
    print(f"\n{'='*60}")
    print("📊 ИТОГОВАЯ СТАТИСТИКА")
    print(f"{'='*60}")
    print(f"✅ Успешно загружено: {success_count}")
    print(f"⏭️  Пропущено: {skip_count}")
    print(f"❌ Ошибок: {error_count}")
    
    # Проверяем что в Qdrant
    try:
        collection_info = cv_parser.qdrant_client.get_collection(cv_parser.collection_name)
        print(f"\n☁️  Всего документов в Qdrant: {collection_info.points_count}")
    except Exception as e:
        print(f"\n⚠️  Не удалось получить информацию о коллекции: {e}")
    
    print(f"""
╔═══════════════════════════════════════════════════════════════╗
║                    ЗАГРУЗКА ЗАВЕРШЕНА                         ║
╚═══════════════════════════════════════════════════════════════╝

✅ Успешно: {success_count}
❌ Ошибок: {error_count}
    """)


if __name__ == "__main__":
    main()
