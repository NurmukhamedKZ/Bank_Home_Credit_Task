#!/usr/bin/env python3
"""
Оценка качества ML классификатора на вакансиях.

Использование:
    python -m app.scripts.evaluate_ml_classifier
    python -m app.scripts.evaluate_ml_classifier --model-path models/my_classifier.pkl
"""

import sys
from pathlib import Path
import json
from datetime import datetime
from typing import Dict, List
import pandas as pd


def evaluate_ml_classifier(
    model_path: str = None,
    threshold: float = 0.5,
    save_results: bool = True
):
    """
    Оценивает ML классификатор на всех вакансиях
    
    Args:
        model_path: Путь к модели (если None - обучит новую)
        threshold: Порог для бинарной классификации (default: 0.5)
        save_results: Сохранять ли результаты
    """
    from app.services.cv_parser import CVParser
    from app.evaluation.evaluator import CVSearchEvaluator
    from app.services.ml_classifier import MLClassifier, build_training_data_from_ground_truth
    from app.evaluation.metrics import SearchMetrics
    
    print(f"""
╔═══════════════════════════════════════════════════════════════╗
║         ОЦЕНКА ML КЛАССИФИКАТОРА                              ║
╚═══════════════════════════════════════════════════════════════╝
    """)
    
    # Инициализация
    from app.core.config import QDRANT_COLLECTION_NAME
    
    print("🚀 Инициализация CVParser и Evaluator...")
    parser = CVParser(collection_name=QDRANT_COLLECTION_NAME)
    evaluator = CVSearchEvaluator(parser)
    
    # Загрузка или обучение модели
    if model_path and Path(model_path).exists():
        print(f"📂 Загрузка модели: {model_path}")
        classifier = MLClassifier.load(model_path)
    else:
        print("🤖 Обучение новой модели...")
        
        # Построение обучающей выборки
        vacancy_texts, cv_texts, labels = build_training_data_from_ground_truth(
            evaluator,
            negative_ratio=1.5
        )
        
        # Обучение
        classifier = MLClassifier(
            model_type='logistic',
            tfidf_max_features=5000,
            tfidf_ngram_range=(1, 2)
        )
        
        classifier.fit(vacancy_texts, cv_texts, labels, validation_split=0.1, verbose=True)
        
        # Сохранение модели
        model_dir = Path("data/models")
        model_dir.mkdir(parents=True, exist_ok=True)
        model_path = model_dir / "ml_classifier_evaluation.pkl"
        classifier.save(model_path)
    
    print(f"\n{'='*70}")
    print("ОЦЕНКА НА ВАКАНСИЯХ")
    print(f"{'='*70}\n")
    
    results = []
    
    # Для каждой вакансии
    for vacancy_name in evaluator.vacancies.keys():
        print(f"Обработка: {vacancy_name}...", end=' ')
        
        try:
            vacancy_text = evaluator.vacancies[vacancy_name]
            relevant_cvs = evaluator.ground_truth[vacancy_name]
            
            # Получаем все CV
            all_cvs = list(evaluator.cvs_folder.glob("*.txt"))
            
            # Предсказания для всех CV
            cv_scores = []
            
            for cv_path in all_cvs:
                cv_text = cv_path.read_text(encoding='utf-8')
                cv_name = cv_path.stem
                
                # Вероятность релевантности
                probability = classifier.predict_proba(vacancy_text, cv_text)
                
                cv_scores.append((cv_name, probability))
            
            # Сортируем по вероятности (descending)
            cv_scores.sort(key=lambda x: x[1], reverse=True)
            
            # Получаем ранжированный список
            retrieved_ids = [cv_id for cv_id, _ in cv_scores]
            
            # Вычисляем метрики
            metrics = {}
            
            # Precision@K
            for k in [1, 3, 5, 8, 10]:
                if k <= len(retrieved_ids):
                    metrics[f'precision@{k}'] = SearchMetrics.precision_at_k(
                        relevant_cvs, retrieved_ids, k
                    )
            
            # Recall@K
            for k in [1, 3, 5, 8, 10]:
                if k <= len(retrieved_ids):
                    metrics[f'recall@{k}'] = SearchMetrics.recall_at_k(
                        relevant_cvs, retrieved_ids, k
                    )
            
            # F1@K
            for k in [1, 3, 5, 8, 10]:
                if k <= len(retrieved_ids):
                    metrics[f'f1@{k}'] = SearchMetrics.f1_at_k(
                        relevant_cvs, retrieved_ids, k
                    )
            
            # MAP
            metrics['map'] = SearchMetrics.average_precision(relevant_cvs, retrieved_ids)
            
            # MRR
            metrics['mrr'] = SearchMetrics.mean_reciprocal_rank(relevant_cvs, retrieved_ids)
            
            # NDCG@K
            for k in [1, 3, 5, 8, 10]:
                if k <= len(retrieved_ids):
                    metrics[f'ndcg@{k}'] = SearchMetrics.ndcg_at_k(
                        relevant_cvs, retrieved_ids, k
                    )
            
            # Топ-10 с scores
            top_10 = [
                {
                    'cv_id': cv_id,
                    'score': float(score),
                    'relevant': cv_id in relevant_cvs
                }
                for cv_id, score in cv_scores[:10]
            ]
            
            result = {
                'vacancy': vacancy_name,
                'relevant_count': len(relevant_cvs),
                'metrics': metrics,
                'retrieved': top_10
            }
            
            results.append(result)
            
            print(f"✅ MAP: {metrics['map']:.3f}")
            
        except Exception as e:
            print(f"❌ Ошибка: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    if len(results) == 0:
        print("\n⚠️  НЕТ РЕЗУЛЬТАТОВ")
        return
    
    # Создаем DataFrame
    rows = []
    for r in results:
        row = {'vacancy': r['vacancy'], 'relevant_count': r['relevant_count']}
        row.update(r['metrics'])
        rows.append(row)
    
    df = pd.DataFrame(rows)
    
    # Выводим статистику
    print(f"\n{'='*70}")
    print("📊 СРЕДНИЕ МЕТРИКИ")
    print(f"{'='*70}\n")
    
    metric_cols = [col for col in df.columns if col not in ['vacancy', 'relevant_count']]
    if len(metric_cols) > 0:
        summary = df[metric_cols].describe().loc[['mean', 'std', 'min', 'max']]
        print(summary.to_string())
    
    # Сохранение результатов
    if save_results:
        print(f"\n{'='*70}")
        output_dir = Path("evaluation_results")
        output_dir.mkdir(exist_ok=True, parents=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # CSV с метриками
        csv_path = output_dir / f"ml_classifier_{timestamp}.csv"
        df.to_csv(csv_path, index=False)
        print(f"💾 Метрики сохранены: {csv_path}")
        
        # JSON с детальными результатами
        json_path = output_dir / f"ml_classifier_detailed_{timestamp}.json"
        
        # Конвертируем sets в lists
        results_serializable = []
        for r in results:
            r_copy = r.copy()
            results_serializable.append(r_copy)
        
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(results_serializable, f, ensure_ascii=False, indent=2)
        
        print(f"💾 Детали сохранены: {json_path}")
        print(f"{'='*70}")
    
    # Детальные результаты
    print(f"\n{'='*70}")
    print("📋 ДЕТАЛЬНЫЕ РЕЗУЛЬТАТЫ ПО ВАКАНСИЯМ")
    print(f"{'='*70}\n")
    
    for result in results:
        print(f"\n🎯 Вакансия: {result['vacancy']}")
        print(f"   Релевантных CV: {result['relevant_count']}")
        print(f"   MAP: {result['metrics']['map']:.3f}")
        print(f"   MRR: {result['metrics']['mrr']:.3f}")
        print(f"   Precision@5: {result['metrics'].get('precision@5', 0):.3f}")
        print(f"   Recall@10: {result['metrics'].get('recall@10', 0):.3f}")
        
        print(f"\n   Топ-5 найденных CV:")
        for i, item in enumerate(result['retrieved'][:5], 1):
            is_relevant = "✅" if item['relevant'] else "❌"
            print(f"      {i}. {item['cv_id']:<30} (score: {item['score']:.4f}) {is_relevant}")
    
    print(f"\n{'='*70}")
    print("✅ ОЦЕНКА ЗАВЕРШЕНА")
    print(f"{'='*70}\n")
    
    return df, results


def main():
    """Основная функция"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Оценка ML классификатора")
    parser.add_argument("--model-path", type=str, help="Путь к сохраненной модели")
    parser.add_argument("--threshold", type=float, default=0.5, help="Порог классификации (default: 0.5)")
    parser.add_argument("--no-save", action="store_true", help="Не сохранять результаты")
    
    args = parser.parse_args()
    
    evaluate_ml_classifier(
        model_path=args.model_path,
        threshold=args.threshold,
        save_results=not args.no_save
    )


if __name__ == "__main__":
    main()
