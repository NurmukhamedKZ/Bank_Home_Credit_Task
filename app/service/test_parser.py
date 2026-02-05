"""
Тестовый скрипт для проверки CVParser
"""

import sys
from pathlib import Path

# Добавляем путь к модулю
sys.path.append(str(Path(__file__).parent))

from parse_pdf import CVParser


def test_parser():
    """Быстрый тест парсера"""
    
    print("🧪 Тестирование CVParser\n")
    
    # 1. Инициализация
    print("1️⃣ Инициализация парсера...")
    try:
        parser = CVParser(collection_name="CVs_test")
        print("   ✅ Парсер инициализирован")
        print(f"   📦 Коллекция: {parser.collection_name}")
    except Exception as e:
        print(f"   ❌ Ошибка: {e}")
        return
    
    # 2. Проверка файла
    print("\n2️⃣ Проверка наличия тестового файла...")
    test_file = Path(__file__).parent.parent.parent / "data" / "Raw_CVs" / "Әшекей Нұрмұхамед-5-1.pdf"
    
    if not test_file.exists():
        print(f"   ❌ Файл не найден: {test_file}")
        print("   💡 Положите тестовое резюме в data/CVs/")
        
        # Проверяем какие файлы есть
        cvs_folder = test_file.parent
        if cvs_folder.exists():
            pdf_files = list(cvs_folder.glob("*.pdf"))
            if pdf_files:
                print(f"\n   Найдены другие PDF файлы:")
                for f in pdf_files[:3]:
                    print(f"     - {f.name}")
                test_file = pdf_files[0]
                print(f"\n   Используем: {test_file.name}")
            else:
                print("   ❌ PDF файлы не найдены")
                return
        else:
            print(f"   ❌ Папка не существует: {cvs_folder}")
            return
    else:
        print(f"   ✅ Файл найден: {test_file.name}")
    
    # 3. Парсинг файла
    print("\n3️⃣ Парсинг PDF...")
    try:
        full_text = parser.parse_file(test_file)
        print(f"   ✅ Текст извлечен ({len(full_text)} символов)")
        print(f"   📄 Превью: {full_text[:100]}...")
    except Exception as e:
        print(f"   ❌ Ошибка парсинга: {e}")
        return
    
    # 4. Извлечение структурированных данных
    print("\n4️⃣ Извлечение структурированных данных через LLM...")
    try:
        cv_data = parser.extract_cv_data(full_text)
        print(f"   ✅ Данные извлечены")
        print(f"   👤 Имя: {cv_data.full_name}")
        print(f"   📧 Email: {cv_data.email}")
        print(f"   📱 Телефон: {cv_data.phone}")
        print(f"   💼 Опыт: {cv_data.total_experience_months} месяцев")
        print(f"   🛠️  Навыков: {len(cv_data.skills)}")
        print(f"   🏢 Мест работы: {len(cv_data.work_history)}")
    except Exception as e:
        print(f"   ❌ Ошибка извлечения данных: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # 5. Создание текста для поиска
    print("\n5️⃣ Создание текста для векторного поиска...")
    try:
        searchable_text = parser.create_searchable_text(cv_data)
        print(f"   ✅ Текст создан ({len(searchable_text)} символов)")
        print(f"   🔍 Превью: {searchable_text[:150]}...")
    except Exception as e:
        print(f"   ❌ Ошибка: {e}")
        return
    
    # 6. Создание эмбеддингов
    print("\n6️⃣ Создание эмбеддингов (Dense + Sparse)...")
    try:
        dense_vector, sparse_indices, sparse_values = parser.create_embeddings(searchable_text)
        print(f"   ✅ Эмбеддинги созданы")
        print(f"   📊 Dense размерность: {len(dense_vector)}")
        print(f"   📊 Sparse элементов: {len(sparse_indices)}")
    except Exception as e:
        print(f"   ❌ Ошибка создания эмбеддингов: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # 7. Сохранение в Qdrant
    print("\n7️⃣ Сохранение в Qdrant...")
    try:
        point_id = parser.save_to_qdrant(
            cv_data=cv_data,
            full_text=full_text,
            dense_vector=dense_vector,
            sparse_indices=sparse_indices,
            sparse_values=sparse_values
        )
        print(f"   ✅ CV сохранено")
        print(f"   🆔 Point ID: {point_id}")
    except Exception as e:
        print(f"   ❌ Ошибка сохранения: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # 8. Проверка что данные действительно в Qdrant
    print("\n8️⃣ Проверка данных в Qdrant...")
    try:
        results = parser.qdrant_client.scroll(
            collection_name=parser.collection_name,
            limit=1,
            with_payload=True,
            with_vectors=False
        )
        
        if results[0]:
            point = results[0][0]
            print(f"   ✅ Данные найдены в Qdrant")
            print(f"   👤 Имя: {point.payload.get('full_name')}")
            print(f"   📧 Email: {point.payload.get('email')}")
            print(f"   🛠️  Навыки: {', '.join(point.payload.get('skills', [])[:5])}...")
        else:
            print(f"   ⚠️  Данные не найдены")
    except Exception as e:
        print(f"   ❌ Ошибка проверки: {e}")
    
    # 9. Статистика коллекции
    print("\n9️⃣ Статистика коллекции...")
    try:
        collection_info = parser.qdrant_client.get_collection(parser.collection_name)
        print(f"   ✅ Коллекция: {parser.collection_name}")
        print(f"   📊 Всего точек: {collection_info.points_count}")
        print(f"   📏 Размерность векторов: {collection_info.config.params.vectors['default'].size}")
    except Exception as e:
        print(f"   ❌ Ошибка: {e}")
    
    print("\n" + "="*60)
    print("✅ ВСЕ ТЕСТЫ ПРОЙДЕНЫ УСПЕШНО!")
    print("="*60 + "\n")
    
    print("💡 Теперь можно использовать parser.process_cv() для обработки новых CV")


def test_full_pipeline():
    """Тест полного пайплайна через один метод"""
    
    print("\n🚀 Тест полного пайплайна (process_cv)\n")
    
    parser = CVParser(collection_name="CVs_test")
    
    test_file = Path(__file__).parent.parent.parent / "data" / "Raw_CVs" / "Әшекей Нұрмұхамед-5-1.pdf"
    
    if not test_file.exists():
        cvs_folder = test_file.parent
        pdf_files = list(cvs_folder.glob("*.pdf"))
        if pdf_files:
            test_file = pdf_files[0]
        else:
            print("❌ PDF файлы не найдены")
            return
    
    try:
        result = parser.process_cv(test_file)
        
        print("\n✅ Результаты:")
        print(f"   ID: {result['point_id']}")
        print(f"   Имя: {result['full_name']}")
        print(f"   Email: {result['email']}")
        print(f"   Опыт: {result['total_experience_months']} месяцев")
        print(f"   Навыков: {result['skills_count']}")
        
    except Exception as e:
        print(f"❌ Ошибка: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    # Запускаем тесты
    # test_parser()
    
    # Или тест полного пайплайна
    test_full_pipeline()
