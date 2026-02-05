"""
Главный модуль приложения для обработки резюме.
"""

from app.service import EmailFetcher


def fetch_resumes_from_email():
    """Получение резюме из почты и сохранение в data/CVs."""
    print("Запуск получения резюме из почты...")
    print("=" * 50)
    
    with EmailFetcher() as fetcher:
        saved_files = fetcher.fetch_resumes(
            folder="INBOX",           # Папка почты
            search_criteria="UNSEEN", # Только непрочитанные
            save_text_body=True,      # Сохранять текстовые резюме
            mark_as_read=True         # Помечать как прочитанные
        )
    
    if saved_files:
        print("\nСохраненные файлы:")
        for filepath in saved_files:
            print(f"  - {filepath}")
    else:
        print("\nНовых резюме не найдено.")
    
    return saved_files


if __name__ == "__main__":
    fetch_resumes_from_email()
