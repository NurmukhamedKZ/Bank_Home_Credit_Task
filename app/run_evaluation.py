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
    import sys
    
    print("""
╔═══════════════════════════════════════════════════════════════╗
║                 CV SEARCH QUALITY EVALUATION                  ║
╚═══════════════════════════════════════════════════════════════╝
    """)
    
    # Проверяем аргументы
    search_mode = "hybrid"  # По умолчанию
    
    if len(sys.argv) > 1:
        arg = sys.argv[1]
        if arg in ["--dense", "--dense-only"]:
            search_mode = "dense"
            print("ℹ️  Режим: Dense-only (Voyage AI)\n")
        elif arg in ["--sparse", "--sparse-only", "--tfidf"]:
            search_mode = "sparse"
            print("ℹ️  Режим: Sparse-only (TF-IDF)\n")
        elif arg == "--hybrid":
            search_mode = "hybrid"
            print("ℹ️  Режим: Hybrid Search (Dense + TF-IDF)\n")
        else:
            print(f"⚠️  Неизвестный режим: {arg}")
            print("\nДоступные режимы:")
            print("  --dense     Dense-only (Voyage AI)")
            print("  --sparse    Sparse-only (TF-IDF)")
            print("  --hybrid    Hybrid (Dense + TF-IDF) [по умолчанию]")
            sys.exit(1)
    
    # Запускаем оценку
    df, results = evaluate_main(search_mode=search_mode)
    
    print("""
╔═══════════════════════════════════════════════════════════════╗
║                     EVALUATION COMPLETE                       ║
╚═══════════════════════════════════════════════════════════════╝
    """)
    
    print("\n💡 Подсказка:")
    print("   Сравните все режимы: python app/compare_search_modes.py")
    print("   Dense-only:  python app/run_evaluation.py --dense")
    print("   Sparse-only: python app/run_evaluation.py --sparse")
    print("   Hybrid:      python app/run_evaluation.py --hybrid\n")
