#!/usr/bin/env python3
"""
Сравнение всех методов поиска: Dense, Sparse (BM25), Hybrid, ML Classifier.

Использование:
    python -m app.scripts.compare_all_methods
"""

import sys
from pathlib import Path
import pandas as pd
from datetime import datetime


def compare_all_methods():
    """Сравнивает все реализованные методы поиска"""
    from app.services.cv_parser import CVParser
    from app.evaluation.evaluator import CVSearchEvaluator
    from app.services.ml_classifier import MLClassifier, build_training_data_from_ground_truth
    from app.core.config import QDRANT_COLLECTION_NAME
    
    print(f"""
╔═══════════════════════════════════════════════════════════════╗
║         СРАВНЕНИЕ ВСЕХ МЕТОДОВ ПОИСКА                         ║
║  Dense | Sparse (BM25) | Hybrid | ML Classifier               ║
╚═══════════════════════════════════════════════════════════════╝
    """)
    
    parser = CVParser(collection_name=QDRANT_COLLECTION_NAME, sparse_method="bm25")
    evaluator = CVSearchEvaluator(parser)
    
    has_sparse = parser._sparse_fitted
    if not has_sparse:
        print("⚠️  BM25 не обучен!")
        print("   Запустите: python -m app.scripts.load_cvs --bm25\n")
    
    all_results = {}
    
    # 1. Dense-only
    print("="*70)
    print("1️⃣  Dense-only (Voyage AI)")
    print("="*70)
    
    df_dense, results_dense = evaluator.evaluate_all(top_k=10, search_mode="dense")
    all_results['Dense (Voyage AI)'] = df_dense
    
    # 2. Sparse-only (BM25)
    if has_sparse:
        print("\n" + "="*70)
        print("2️⃣  Sparse-only (BM25)")
        print("="*70)
        
        df_sparse, _ = evaluator.evaluate_all(top_k=10, search_mode="sparse")
        all_results['Sparse (BM25)'] = df_sparse
    
    # 3. Hybrid
    if has_sparse:
        print("\n" + "="*70)
        print("3️⃣  Hybrid (Dense + BM25)")
        print("="*70)
        
        df_hybrid, _ = evaluator.evaluate_all(top_k=10, search_mode="hybrid")
        all_results['Hybrid (Dense+BM25)'] = df_hybrid
    
    # 4. ML Classifier
    print("\n" + "="*70)
    print("4️⃣  ML Classifier (TF-IDF + Logistic)")
    print("="*70)
    
    # Проверяем наличие обученной модели
    model_path = Path("data/models/ml_classifier_evaluation.pkl")
    
    if model_path.exists():
        print(f"📂 Загрузка модели: {model_path}")
        classifier = MLClassifier.load(model_path)
    else:
        print("🤖 Обучение ML классификатора...")
        
        vacancy_texts, cv_texts, labels = build_training_data_from_ground_truth(
            evaluator,
            negative_ratio=1.5
        )
        
        classifier = MLClassifier(
            model_type='logistic',
            tfidf_max_features=5000,
            tfidf_ngram_range=(1, 2)
        )
        
        classifier.fit(vacancy_texts, cv_texts, labels, validation_split=0.2, verbose=False)
        
        model_path.parent.mkdir(parents=True, exist_ok=True)
        classifier.save(model_path)
    
    # Оценка ML классификатора
    from app.scripts.evaluate_ml_classifier import evaluate_ml_classifier
    
    df_ml, _ = evaluate_ml_classifier(
        model_path=str(model_path),
        save_results=False
    )
    
    all_results['ML Classifier'] = df_ml
    
    # Сравнение результатов
    print("\n" + "="*70)
    print("📊 СРАВНЕНИЕ ВСЕХ МЕТОДОВ")
    print("="*70 + "\n")
    
    metrics_to_compare = ['precision@5', 'recall@10', 'map', 'mrr', 'ndcg@5', 'f1@5']
    
    comparison_data = []
    for metric in metrics_to_compare:
        row = {'Метрика': metric.upper()}
        
        for method_name, df in all_results.items():
            if df is not None and metric in df.columns:
                avg_value = df[metric].mean()
                row[method_name] = f"{avg_value:.3f}"
        
        comparison_data.append(row)
    
    comparison_df = pd.DataFrame(comparison_data)
    print(comparison_df.to_string(index=False))
    
    # Сохранение сравнения
    output_dir = Path("evaluation_results")
    output_dir.mkdir(exist_ok=True, parents=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    comparison_file = output_dir / f"comparison_all_methods_{timestamp}.csv"
    comparison_df.to_csv(comparison_file, index=False)
    
    print(f"\n💾 Сравнение сохранено: {comparison_file}")
    
    # Лучший метод по MAP
    print("\n" + "="*70)
    print("🏆 ЛУЧШИЙ МЕТОД ПО MAP")
    print("="*70 + "\n")
    
    map_scores = {}
    for method_name, df in all_results.items():
        if df is not None and 'map' in df.columns:
            map_scores[method_name] = df['map'].mean()
    
    for method, score in sorted(map_scores.items(), key=lambda x: x[1], reverse=True):
        emoji = "🥇" if method == max(map_scores, key=map_scores.get) else "  "
        print(f"{emoji} {method:<30} MAP: {score:.4f}")
    
    best_method = max(map_scores, key=map_scores.get)
    print(f"\n✅ ЛУЧШИЙ МЕТОД: {best_method}")
    
    # Рекомендации
    print("\n" + "="*70)
    print("💡 РЕКОМЕНДАЦИИ")
    print("="*70 + "\n")
    
    print("🔍 Dense-only: Быстро, хорошо для семантического поиска")
    if has_sparse:
        print("📝 Sparse (BM25): Отлично для точного совпадения ключевых слов")
        print("🔀 Hybrid: Комбинация преимуществ обоих методов")
    print("🤖 ML Classifier: Supervised learning, интерпретируемо, без API")
    
    print(f"\n{'='*70}")
    print("✅ СРАВНЕНИЕ ЗАВЕРШЕНО")
    print(f"{'='*70}\n")
    
    return comparison_df


def main():
    """Основная функция"""
    compare_all_methods()


if __name__ == "__main__":
    main()
