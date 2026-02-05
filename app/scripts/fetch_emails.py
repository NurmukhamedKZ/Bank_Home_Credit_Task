#!/usr/bin/env python3
"""
Получение резюме из почты.

Использование:
    python -m app.scripts.fetch_emails
"""

from app.services.email_fetcher import EmailFetcher


def main():
    """Основная функция"""
    print(f"""
╔═══════════════════════════════════════════════════════════════╗
║         ПОЛУЧЕНИЕ РЕЗЮМЕ ИЗ ПОЧТЫ                             ║
╚═══════════════════════════════════════════════════════════════╝
    """)
    
    print("Запуск получения резюме из почты...")
    print("=" * 50)
    
    with EmailFetcher() as fetcher:
        saved_files = fetcher.fetch_resumes(
            folder="INBOX",
            search_criteria="UNSEEN",
            save_text_body=True,
            mark_as_read=True
        )
    
    if saved_files:
        print("\nСохраненные файлы:")
        for filepath in saved_files:
            print(f"  - {filepath}")
    else:
        print("\nНовых резюме не найдено.")
    
    return saved_files


if __name__ == "__main__":
    main()
