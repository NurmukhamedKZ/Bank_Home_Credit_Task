#!/usr/bin/env python3
"""
Быстрый запуск оценки качества поиска CV
"""

import sys
from pathlib import Path

# Добавляем путь к модулям
sys.path.append(str(Path(__file__).parent))

from evaluate_search import main as evaluate_main

if __name__ == "__main__":
    print("""
╔═══════════════════════════════════════════════════════════════╗
║                 CV SEARCH QUALITY EVALUATION                  ║
╚═══════════════════════════════════════════════════════════════╝
    """)
    
    # Запускаем оценку
    df, results = evaluate_main()
    
    print("""
╔═══════════════════════════════════════════════════════════════╗
║                     EVALUATION COMPLETE                       ║
╚═══════════════════════════════════════════════════════════════╝
    """)
