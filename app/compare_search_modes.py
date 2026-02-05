#!/usr/bin/env python3
"""
Сравнение всех режимов поиска: Dense-only, Sparse-only, Hybrid
Запускает оценку в трех режимах и сравнивает результаты
"""

import sys
from pathlib import Path
import pandas as pd

sys.path.append(str(Path(__file__).parent))

from service.parse_pdf import CVParser
from evaluate_search import CVSearchEvaluator


def compare_search_modes():
    """Сравнивает все три режима поиска"""
    
    print("""
╔═══════════════════════════════════════════════════════════════╗
║              СРАВНЕНИЕ РЕЖИМОВ ПОИСКА                         ║
║  Dense-only | Sparse-only (TF-IDF) | Hybrid                   ║
╚═══════════════════════════════════════════════════════════════╝
    """)
    
    parser = CVParser(collection_name="CVs")
    evaluator = CVSearchEvaluator(parser)
    
    # Проверка TF-IDF
    has_tfidf = parser._tfidf_fitted
    if not has_tfidf:
        print("⚠️  TF-IDF не обучен!")
        print("   Запустите сначала: python app/load_txt_to_qdrant.py")
        print("   Sparse и Hybrid режимы будут недоступны.\n")
        print("   Сравним только Dense vs что получится...\n")
    else:
        print(f"✅ TF-IDF обучен на {len(parser._tfidf_corpus)} документах\n")
    
    # ========== Режим 1: Dense-only ==========
    print("="*70)
    print("1️⃣  РЕЖИМ: Dense-only (Voyage AI)")
    print("="*70)
    
    df_dense, results_dense = evaluator.evaluate_all(top_k=10, search_mode="dense")
    
    # ========== Режим 2: Sparse-only (если доступен) ==========
    df_sparse = None
    results_sparse = None
    
    if has_tfidf:
        print("\n" + "="*70)
        print("2️⃣  РЕЖИМ: Sparse-only (TF-IDF)")
        print("="*70)
        
        df_sparse, results_sparse = evaluator.evaluate_all(top_k=10, search_mode="sparse")
    
    # ========== Режим 3: Hybrid (если доступен) ==========
    df_hybrid = None
    results_hybrid = None
    
    if has_tfidf:
        print("\n" + "="*70)
        print("3️⃣  РЕЖИМ: Hybrid Search (Dense + TF-IDF)")
        print("="*70)
        
        df_hybrid, results_hybrid = evaluator.evaluate_all(top_k=10, search_mode="hybrid")
    
    # ========== Сравнение ==========
    print("\n" + "="*70)
    print("📊 СРАВНЕНИЕ РЕЗУЛЬТАТОВ")
    print("="*70 + "\n")
    
    # Средние метрики
    metrics_to_compare = ['precision@5', 'recall@10', 'map', 'mrr', 'ndcg@5', 'f1@5']
    
    comparison_data = []
    for metric in metrics_to_compare:
        if metric not in df_dense.columns:
            continue
        
        row = {'Метрика': metric.upper()}
        
        # Dense
        dense_avg = df_dense[metric].mean()
        row['Dense'] = f"{dense_avg:.3f}"
        
        # Sparse (если есть)
        if df_sparse is not None and metric in df_sparse.columns:
            sparse_avg = df_sparse[metric].mean()
            row['Sparse (TF-IDF)'] = f"{sparse_avg:.3f}"
            sparse_vs_dense = ((sparse_avg - dense_avg) / dense_avg * 100) if dense_avg > 0 else 0
            row['Sparse vs Dense'] = f"{sparse_vs_dense:+.1f}%"
        
        # Hybrid (если есть)
        if df_hybrid is not None and metric in df_hybrid.columns:
            hybrid_avg = df_hybrid[metric].mean()
            row['Hybrid'] = f"{hybrid_avg:.3f}"
            hybrid_vs_dense = ((hybrid_avg - dense_avg) / dense_avg * 100) if dense_avg > 0 else 0
            row['Hybrid vs Dense'] = f"{hybrid_vs_dense:+.1f}%"
        
        comparison_data.append(row)
    
    comparison_df = pd.DataFrame(comparison_data)
    print(comparison_df.to_string(index=False))
    
    # Детальное сравнение по вакансиям
    print("\n" + "="*70)
    print("📋 ДЕТАЛЬНОЕ СРАВНЕНИЕ ПО ВАКАНСИЯМ (MAP)")
    print("="*70 + "\n")
    
    for vacancy in df_dense['vacancy']:
        dense_map = df_dense[df_dense['vacancy'] == vacancy]['map'].values[0]
        
        line = f"{vacancy:<20} | Dense: {dense_map:.3f}"
        
        # Sparse (если есть)
        if df_sparse is not None:
            sparse_map = df_sparse[df_sparse['vacancy'] == vacancy]['map'].values[0]
            sparse_improvement = ((sparse_map - dense_map) / dense_map * 100) if dense_map > 0 else 0
            emoji_sparse = "📈" if sparse_improvement > 0 else "📉" if sparse_improvement < 0 else "➡️"
            line += f" | Sparse: {sparse_map:.3f} {emoji_sparse}{sparse_improvement:+.0f}%"
        
        # Hybrid (если есть)
        if df_hybrid is not None:
            hybrid_map = df_hybrid[df_hybrid['vacancy'] == vacancy]['map'].values[0]
            hybrid_improvement = ((hybrid_map - dense_map) / dense_map * 100) if dense_map > 0 else 0
            emoji_hybrid = "📈" if hybrid_improvement > 0 else "📉" if hybrid_improvement < 0 else "➡️"
            line += f" | Hybrid: {hybrid_map:.3f} {emoji_hybrid}{hybrid_improvement:+.0f}%"
        
        print(line)
    
    # Итоговый вердикт
    print("\n" + "="*70)
    print("🎯 РЕКОМЕНДАЦИИ")
    print("="*70 + "\n")
    
    avg_dense_map = df_dense['map'].mean()
    
    # Собираем все MAP значения
    map_scores = {'Dense': avg_dense_map}
    
    if df_sparse is not None:
        avg_sparse_map = df_sparse['map'].mean()
        map_scores['Sparse'] = avg_sparse_map
    
    if df_hybrid is not None:
        avg_hybrid_map = df_hybrid['map'].mean()
        map_scores['Hybrid'] = avg_hybrid_map
    
    # Находим лучший режим
    best_mode = max(map_scores, key=map_scores.get)
    best_score = map_scores[best_mode]
    
    print(f"📊 Средний MAP по режимам:")
    for mode, score in map_scores.items():
        emoji = "🏆" if mode == best_mode else "  "
        improvement = ((score - avg_dense_map) / avg_dense_map * 100) if avg_dense_map > 0 and mode != 'Dense' else 0
        if mode == 'Dense':
            print(f"   {emoji} {mode:<15} {score:.3f}")
        else:
            print(f"   {emoji} {mode:<15} {score:.3f}  ({improvement:+.1f}%)")
    
    print(f"\n✅ ЛУЧШИЙ РЕЖИМ: {best_mode} (MAP: {best_score:.3f})")
    
    # Детальная рекомендация
    if best_mode == "Hybrid":
        improvement = ((best_score - avg_dense_map) / avg_dense_map * 100) if avg_dense_map > 0 else 0
        if improvement > 10:
            print(f"   💡 Hybrid значительно лучше (+{improvement:.1f}%) - РЕКОМЕНДУЕТСЯ!")
        else:
            print(f"   💡 Hybrid немного лучше (+{improvement:.1f}%) - можно использовать")
    elif best_mode == "Sparse":
        print(f"   💡 TF-IDF работает лучше всего - интересный результат!")
        print(f"   💡 Возможно стоит увеличить max_features в TF-IDF")
    else:
        print(f"   💡 Dense достаточно хорош - Hybrid не даст большого улучшения")
    
    print("="*70 + "\n")
    
    # Сохраняем результаты
    output_dir = Path("evaluation_results")
    output_dir.mkdir(exist_ok=True, parents=True)
    
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    comparison_df.to_csv(output_dir / f"comparison_all_modes_{timestamp}.csv", index=False)
    print(f"💾 Сравнение сохранено: {output_dir}/comparison_all_modes_{timestamp}.csv\n")
    
    # Возвращаем все доступные результаты
    return {
        'dense': (df_dense, results_dense),
        'sparse': (df_sparse, results_sparse) if df_sparse is not None else None,
        'hybrid': (df_hybrid, results_hybrid) if df_hybrid is not None else None,
        'comparison': comparison_df
    }


if __name__ == "__main__":
    compare_search_modes()
