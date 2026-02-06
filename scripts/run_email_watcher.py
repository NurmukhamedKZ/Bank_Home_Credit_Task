#!/usr/bin/env python3
"""
Email Watcher - непрерывный мониторинг почты для получения резюме.

Запускается в Docker контейнере и проверяет почту с заданным интервалом.
Интервал настраивается через переменную окружения EMAIL_CHECK_INTERVAL (секунды).
"""

import os
import time
import signal
import sys
from datetime import datetime

# Добавляем корень проекта в path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.services.email_fetcher import EmailFetcher


# Флаг для graceful shutdown
running = True


def signal_handler(signum, frame):
    """Обработчик сигналов для корректного завершения"""
    global running
    print(f"\n[{datetime.now()}] Получен сигнал завершения. Останавливаем...")
    running = False


def main():
    """Основной цикл мониторинга почты"""
    global running
    
    # Регистрируем обработчики сигналов
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Интервал проверки из переменной окружения (по умолчанию 60 секунд)
    check_interval = int(os.environ.get("EMAIL_CHECK_INTERVAL", "60"))
    
    print(f"""
╔═══════════════════════════════════════════════════════════════╗
║         EMAIL WATCHER - МОНИТОРИНГ ПОЧТЫ                      ║
╠═══════════════════════════════════════════════════════════════╣
║  Интервал проверки: {check_interval:>4} секунд                            ║
║  Для остановки: Ctrl+C или docker-compose stop               ║
╚═══════════════════════════════════════════════════════════════╝
    """)
    
    while running:
        try:
            print(f"\n[{datetime.now()}] Проверка почты...")
            
            with EmailFetcher() as fetcher:
                saved_files = fetcher.fetch_resumes(
                    folder="INBOX",
                    search_criteria="UNSEEN",
                    save_text_body=True,
                    mark_as_read=True
                )
            
            if saved_files:
                print(f"[{datetime.now()}] Сохранено {len(saved_files)} новых резюме:")
                for filepath in saved_files:
                    print(f"  ✓ {filepath}")
            else:
                print(f"[{datetime.now()}] Новых резюме не найдено")
            
        except Exception as e:
            print(f"[{datetime.now()}] Ошибка: {e}")
        
        # Ждем следующей проверки (с возможностью прерывания)
        for _ in range(check_interval):
            if not running:
                break
            time.sleep(1)
    
    print(f"\n[{datetime.now()}] Email Watcher остановлен")


if __name__ == "__main__":
    main()
