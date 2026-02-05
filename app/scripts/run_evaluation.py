#!/usr/bin/env python3
"""
Запуск оценки качества поиска CV.

Использование:
    python -m app.scripts.run_evaluation              # Hybrid по умолчанию
    python -m app.scripts.run_evaluation --dense      # Dense-only
    python -m app.scripts.run_evaluation --sparse     # Sparse-only
    python -m app.scripts.run_evaluation --bm25       # Использовать BM25
"""

import sys
from pathlib import Path


def main():
    """Основная функция"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Оценка качества поиска CV")
    parser.add_argument("--dense", action="store_true", help="Dense-only режим")
    parser.add_argument("--sparse", action="store_true", help="Sparse-only режим")
    parser.add_argument("--hybrid", action="store_true", help="Hybrid режим (по умолчанию)")
    parser.add_argument("--bm25", action="store_true", help="Использовать BM25 вместо TF-IDF")
    parser.add_argument("--tfidf", action="store_true", help="Использовать TF-IDF (по умолчанию)")
    
    args = parser.parse_args()
    
    # Определяем режим поиска
    if args.dense:
        search_mode = "dense"
    elif args.sparse:
        search_mode = "sparse"
    else:
        search_mode = "hybrid"
    
    # Определяем sparse метод
    sparse_method = "bm25" if args.bm25 else "tfidf"
    
    print(f"""
╔═══════════════════════════════════════════════════════════════╗
║                 CV SEARCH QUALITY EVALUATION                  ║
╚═══════════════════════════════════════════════════════════════╝

ℹ️  Режим: {search_mode.upper()}
ℹ️  Sparse метод: {sparse_method.upper()}
    """)
    
    from app.services.cv_parser import CVParser
    from app.evaluation.evaluator import CVSearchEvaluator
    
    print("🚀 Инициализация CVParser...")
    cv_parser = CVParser(collection_name="CVs_BM25", sparse_method=sparse_method)
    
    print("📊 Создание оценщика...")
    evaluator = CVSearchEvaluator(cv_parser)
    
    # Полная оценка
    df, results = evaluator.evaluate_all(top_k=10, search_mode=search_mode)
    
    # Детальные результаты
    evaluator.print_detailed_results(results)
    
    # Матрица путаницы
    print(f"\n{'='*60}")
    print("🎭 МАТРИЦА ПУТАНИЦЫ (найденные CV по типам)")
    print(f"{'='*60}\n")
    confusion = evaluator.generate_confusion_matrix(results)
    print(confusion.to_string())
    
    # Сохраняем результаты
    print(f"\n{'='*60}")
    evaluator.save_results(df, results)
    print(f"{'='*60}")
    
    print(f"""
╔═══════════════════════════════════════════════════════════════╗
║                     EVALUATION COMPLETE                       ║
╚═══════════════════════════════════════════════════════════════╝

💡 Подсказка:
   Сравните все режимы: python -m app.scripts.compare_modes
    """)
    
    return df, results


if __name__ == "__main__":
    main()
