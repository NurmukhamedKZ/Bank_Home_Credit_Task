#!/usr/bin/env python3
"""
Сравнение режимов поиска: Dense-only, Sparse-only, Hybrid.

Использование:
    python -m app.scripts.compare_modes           # Сравнение всех режимов
    python -m app.scripts.compare_modes --sparse  # Сравнение TF-IDF vs BM25
"""

import sys
from pathlib import Path
import pandas as pd
from datetime import datetime


def compare_search_modes():
    """Сравнивает все три режима поиска"""
    from app.services.cv_parser import CVParser
    from app.evaluation.evaluator import CVSearchEvaluator
    from app.core.config import QDRANT_COLLECTION_NAME
    
    print(f"""
╔═══════════════════════════════════════════════════════════════╗
║              СРАВНЕНИЕ РЕЖИМОВ ПОИСКА                         ║
║  Dense-only | Sparse-only (TF-IDF) | Hybrid                   ║
╚═══════════════════════════════════════════════════════════════╝
    """)
    
    parser = CVParser(collection_name=QDRANT_COLLECTION_NAME)
    evaluator = CVSearchEvaluator(parser)
    
    has_sparse = parser._sparse_fitted
    if not has_sparse:
        print("⚠️  TF-IDF не обучен!")
        print("   Запустите сначала: python -m app.scripts.load_cvs\n")
    
    # Dense-only
    print("="*70)
    print("1️⃣  РЕЖИМ: Dense-only (Voyage AI)")
    print("="*70)
    
    df_dense, results_dense = evaluator.evaluate_all(top_k=10, search_mode="dense")
    
    # Sparse-only
    df_sparse = None
    if has_sparse:
        print("\n" + "="*70)
        print("2️⃣  РЕЖИМ: Sparse-only (TF-IDF)")
        print("="*70)
        
        df_sparse, _ = evaluator.evaluate_all(top_k=10, search_mode="sparse")
    
    # Hybrid
    df_hybrid = None
    if has_sparse:
        print("\n" + "="*70)
        print("3️⃣  РЕЖИМ: Hybrid Search (Dense + TF-IDF)")
        print("="*70)
        
        df_hybrid, _ = evaluator.evaluate_all(top_k=10, search_mode="hybrid")
    
    # Сравнение
    print("\n" + "="*70)
    print("📊 СРАВНЕНИЕ РЕЗУЛЬТАТОВ")
    print("="*70 + "\n")
    
    metrics_to_compare = ['precision@5', 'recall@10', 'map', 'mrr', 'ndcg@5', 'f1@5']
    
    comparison_data = []
    for metric in metrics_to_compare:
        if metric not in df_dense.columns:
            continue
        
        row = {'Метрика': metric.upper()}
        
        dense_avg = df_dense[metric].mean()
        row['Dense'] = f"{dense_avg:.3f}"
        
        if df_sparse is not None and metric in df_sparse.columns:
            sparse_avg = df_sparse[metric].mean()
            row['Sparse'] = f"{sparse_avg:.3f}"
            diff = ((sparse_avg - dense_avg) / dense_avg * 100) if dense_avg > 0 else 0
            row['Sparse vs Dense'] = f"{diff:+.1f}%"
        
        if df_hybrid is not None and metric in df_hybrid.columns:
            hybrid_avg = df_hybrid[metric].mean()
            row['Hybrid'] = f"{hybrid_avg:.3f}"
            diff = ((hybrid_avg - dense_avg) / dense_avg * 100) if dense_avg > 0 else 0
            row['Hybrid vs Dense'] = f"{diff:+.1f}%"
        
        comparison_data.append(row)
    
    comparison_df = pd.DataFrame(comparison_data)
    print(comparison_df.to_string(index=False))
    
    # Рекомендация
    print("\n" + "="*70)
    print("🎯 РЕКОМЕНДАЦИИ")
    print("="*70 + "\n")
    
    avg_dense_map = df_dense['map'].mean()
    map_scores = {'Dense': avg_dense_map}
    
    if df_sparse is not None:
        map_scores['Sparse'] = df_sparse['map'].mean()
    if df_hybrid is not None:
        map_scores['Hybrid'] = df_hybrid['map'].mean()
    
    best_mode = max(map_scores, key=map_scores.get)
    best_score = map_scores[best_mode]
    
    print(f"📊 Средний MAP по режимам:")
    for mode, score in map_scores.items():
        emoji = "🏆" if mode == best_mode else "  "
        print(f"   {emoji} {mode:<15} {score:.3f}")
    
    print(f"\n✅ ЛУЧШИЙ РЕЖИМ: {best_mode} (MAP: {best_score:.3f})")
    
    # Сохраняем
    output_dir = Path("evaluation_results")
    output_dir.mkdir(exist_ok=True, parents=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    comparison_df.to_csv(output_dir / f"comparison_all_modes_{timestamp}.csv", index=False)
    print(f"\n💾 Результаты сохранены: comparison_all_modes_{timestamp}.csv")
    
    return comparison_df


def compare_sparse_methods():
    """Сравнивает TF-IDF и BM25"""
    from app.services.cv_parser import CVParser
    from app.evaluation.evaluator import CVSearchEvaluator
    
    print(f"""
╔═══════════════════════════════════════════════════════════════╗
║        СРАВНЕНИЕ SPARSE МЕТОДОВ: TF-IDF vs BM25               ║
╚═══════════════════════════════════════════════════════════════╝
    """)
    
    results_all = {}
    
    # TF-IDF
    print("="*70)
    print("🔤 TF-IDF ТЕСТЫ")
    print("="*70 + "\n")
    
    parser_tfidf = CVParser(collection_name=QDRANT_COLLECTION_NAME, sparse_method="tfidf")
    
    if not parser_tfidf._sparse_fitted:
        print("⚠️  TF-IDF модель не найдена")
    else:
        evaluator_tfidf = CVSearchEvaluator(parser_tfidf)
        
        print("1️⃣  TF-IDF Sparse-only")
        df_tfidf_sparse, _ = evaluator_tfidf.evaluate_all(top_k=10, search_mode="sparse")
        results_all['tfidf_sparse'] = df_tfidf_sparse
        
        print("\n2️⃣  TF-IDF Hybrid")
        df_tfidf_hybrid, _ = evaluator_tfidf.evaluate_all(top_k=10, search_mode="hybrid")
        results_all['tfidf_hybrid'] = df_tfidf_hybrid
    
    # BM25
    print("\n" + "="*70)
    print("🎯 BM25 ТЕСТЫ")
    print("="*70 + "\n")
    
    parser_bm25 = CVParser(collection_name=QDRANT_COLLECTION_NAME, sparse_method="bm25")
    
    if not parser_bm25._sparse_fitted:
        print("⚠️  BM25 модель не найдена")
    else:
        evaluator_bm25 = CVSearchEvaluator(parser_bm25)
        
        print("3️⃣  BM25 Sparse-only")
        df_bm25_sparse, _ = evaluator_bm25.evaluate_all(top_k=10, search_mode="sparse")
        results_all['bm25_sparse'] = df_bm25_sparse
        
        print("\n4️⃣  BM25 Hybrid")
        df_bm25_hybrid, _ = evaluator_bm25.evaluate_all(top_k=10, search_mode="hybrid")
        results_all['bm25_hybrid'] = df_bm25_hybrid
    
    if len(results_all) == 0:
        print("\n❌ Нет данных для сравнения!")
        return
    
    # Сравнение
    print("\n" + "="*70)
    print("📊 СРАВНЕНИЕ РЕЗУЛЬТАТОВ")
    print("="*70 + "\n")
    
    metrics_to_compare = ['precision@5', 'recall@10', 'map', 'mrr', 'ndcg@5', 'f1@5']
    
    comparison_data = []
    for metric in metrics_to_compare:
        row = {'Метрика': metric.upper()}
        
        for method_name, df in results_all.items():
            if df is not None and metric in df.columns:
                row[method_name.replace('_', ' ').title()] = f"{df[metric].mean():.3f}"
        
        comparison_data.append(row)
    
    comparison_df = pd.DataFrame(comparison_data)
    print(comparison_df.to_string(index=False))
    
    # Лучший метод
    best_map = 0
    best_method = None
    
    for method_name, df in results_all.items():
        if df is not None and 'map' in df.columns:
            map_value = df['map'].mean()
            if map_value > best_map:
                best_map = map_value
                best_method = method_name
    
    if best_method:
        print(f"\n🏆 Лучший метод: {best_method.replace('_', ' ').upper()} (MAP: {best_map:.3f})")
    
    return comparison_df


def main():
    """Основная функция"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Сравнение режимов поиска")
    parser.add_argument("--sparse", action="store_true", help="Сравнить TF-IDF vs BM25")
    
    args = parser.parse_args()
    
    if args.sparse:
        compare_sparse_methods()
    else:
        compare_search_modes()


if __name__ == "__main__":
    main()
