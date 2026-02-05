"""
Оценка качества поиска CV по вакансиям.
"""

from pathlib import Path
from typing import List, Dict, Set, Tuple
import pandas as pd
from datetime import datetime
import json

from qdrant_client import models
from qdrant_client.models import Prefetch

from app.services.cv_parser import CVParser
from app.evaluation.metrics import SearchMetrics


class CVSearchEvaluator:
    """
    Оценка качества поиска CV по вакансиям
    Автоматически строит ground truth на основе названий файлов
    """
    
    def __init__(
        self,
        parser: CVParser,
        vacancies_folder: str | Path = None,
        cvs_folder: str | Path = None
    ):
        """
        Args:
            parser: Инициализированный CVParser с доступом к Qdrant
            vacancies_folder: Папка с текстами вакансий
            cvs_folder: Папка с распарсенными CV
        """
        self.parser = parser
        
        # Определяем корень проекта
        project_root = Path(__file__).parent.parent.parent
        
        # Устанавливаем пути к данным
        if vacancies_folder is None:
            self.vacancies_folder = project_root / "data" / "vacancy"
        else:
            self.vacancies_folder = Path(vacancies_folder)
        
        if cvs_folder is None:
            self.cvs_folder = project_root / "data" / "Parsed_CVs"
        else:
            self.cvs_folder = Path(cvs_folder)
        
        # Загружаем данные
        self.vacancies = self._load_vacancies()
        self.ground_truth = self._build_ground_truth()
        
        print(f"📊 Загружено вакансий: {len(self.vacancies)}")
        print(f"📊 Ground truth построен для {len(self.ground_truth)} вакансий")
        
        # Проверка на пустые данные
        if len(self.vacancies) == 0:
            print(f"\n⚠️  ВНИМАНИЕ: Вакансии не найдены!")
            print(f"   Папка: {self.vacancies_folder}")
        
        if len(self.ground_truth) == 0:
            print(f"\n⚠️  ВНИМАНИЕ: CV не найдены!")
            print(f"   Папка: {self.cvs_folder}")
        
        # Проверка Qdrant
        try:
            collection_info = self.parser.qdrant_client.get_collection(self.parser.collection_name)
            print(f"📊 CV в Qdrant: {collection_info.points_count}")
            
            if collection_info.points_count == 0:
                print(f"\n⚠️  ВНИМАНИЕ: Коллекция Qdrant пуста!")
        except Exception as e:
            print(f"\n⚠️  Ошибка при проверке Qdrant: {e}")
    
    def _load_vacancies(self) -> Dict[str, str]:
        """Загружает все вакансии из папки"""
        vacancies = {}
        
        for file in self.vacancies_folder.glob("*.txt"):
            vacancies[file.stem] = file.read_text(encoding='utf-8')
        
        return vacancies
    
    def _build_ground_truth(self) -> Dict[str, Set[str]]:
        """
        Строит ground truth на основе названий файлов
        
        Логика: для вакансии "AI_engineer_1" релевантными считаются
        все CV с префиксом "AI_engineer_" (AI_engineer_1, AI_engineer_2, etc.)
        """
        ground_truth = {}
        
        for vacancy_name in self.vacancies.keys():
            # Извлекаем базовое имя (без номера)
            parts = vacancy_name.split('_')
            if len(parts) >= 2 and parts[-1].isdigit():
                base_name = '_'.join(parts[:-1])
            else:
                base_name = vacancy_name
            
            # Находим все CV с таким префиксом
            relevant_cvs = {
                f.stem for f in self.cvs_folder.glob(f"{base_name}_*.txt")
            }
            
            ground_truth[vacancy_name] = relevant_cvs
        
        return ground_truth
    
    def search_cvs(
        self,
        query_text: str,
        top_k: int = 10,
        search_mode: str = "hybrid"
    ) -> List[Tuple[str, float, str]]:
        """
        Поиск CV через Qdrant с поддержкой разных режимов
        
        Args:
            query_text: Текст запроса (вакансия)
            top_k: Количество результатов
            search_mode: Режим поиска - "dense", "sparse", или "hybrid"
            
        Returns:
            List[(cv_identifier, score, full_name)]
        """
        # Валидация режима
        if search_mode not in ["dense", "sparse", "hybrid"]:
            raise ValueError(f"Неверный search_mode: {search_mode}")
        
        # Проверка доступности sparse метода
        if search_mode in ["sparse", "hybrid"] and not self.parser._sparse_fitted:
            print(f"   ⚠️  TF-IDF не обучен, fallback на dense-only...")
            search_mode = "dense"
        
        # ========== DENSE-ONLY SEARCH ==========
        if search_mode == "dense":
            print("   🔍 Dense-only search (Voyage AI)...")
            
            dense_query = self.parser.dense_model.embed_documents([query_text])[0]
            
            results = self.parser.qdrant_client.query_points(
                collection_name=self.parser.collection_name,
                query=dense_query,
                using="default",
                limit=top_k,
                with_payload=True,
                with_vectors=False
            )
        
        # ========== SPARSE-ONLY SEARCH ==========
        elif search_mode == "sparse":
            print("   🔍 Sparse-only search (TF-IDF)...")
            
            sparse_indices, sparse_values = self.parser.create_sparse_query(query_text)
            sparse_query_vector = models.SparseVector(
                indices=sparse_indices,
                values=sparse_values
            )
            
            results = self.parser.qdrant_client.query_points(
                collection_name=self.parser.collection_name,
                query=sparse_query_vector,
                using="sparse",
                limit=top_k,
                with_payload=True,
                with_vectors=False
            )
        
        # ========== HYBRID SEARCH ==========
        elif search_mode == "hybrid":
            print("   🔍 Hybrid search (Dense + TF-IDF)...")
            
            dense_query = self.parser.dense_model.embed_documents([query_text])[0]
            sparse_indices, sparse_values = self.parser.create_sparse_query(query_text)
            sparse_query_vector = models.SparseVector(
                indices=sparse_indices,
                values=sparse_values
            )
            
            results = self.parser.qdrant_client.query_points(
                collection_name=self.parser.collection_name,
                prefetch=[
                    Prefetch(
                        query=dense_query,
                        using="default",
                        limit=top_k * 2
                    ),
                    Prefetch(
                        query=sparse_query_vector,
                        using="sparse",
                        limit=top_k * 2
                    )
                ],
                query=models.FusionQuery(fusion=models.Fusion.RRF),
                limit=top_k,
                with_payload=True,
                with_vectors=False
            )
        
        # Извлекаем идентификаторы
        cv_results = []
        for point in results.points:
            cv_identifier = point.payload.get('source_file', point.payload.get('full_name', 'Unknown'))
            score = point.score
            full_name = point.payload.get('full_name', 'Unknown')
            cv_results.append((cv_identifier, score, full_name))
        
        return cv_results
    
    def evaluate_single_vacancy(
        self,
        vacancy_name: str,
        top_k: int = 10,
        search_mode: str = "hybrid"
    ) -> Dict:
        """Оценка качества поиска для одной вакансии"""
        if vacancy_name not in self.vacancies:
            raise ValueError(f"Вакансия '{vacancy_name}' не найдена")
        
        vacancy_text = self.vacancies[vacancy_name]
        relevant_cvs = self.ground_truth[vacancy_name]
        
        # Поиск
        retrieved_results = self.search_cvs(vacancy_text, top_k, search_mode)
        retrieved_ids = [cv_id for cv_id, _, _ in retrieved_results]
        
        # Вычисляем метрики
        metrics = {}
        
        # Precision@K
        for k in [1, 3, 5, 8, 10]:
            if k <= top_k:
                metrics[f'precision@{k}'] = SearchMetrics.precision_at_k(
                    relevant_cvs, retrieved_ids, k
                )
        
        # Recall@K
        for k in [1, 3, 5, 8, 10]:
            if k <= top_k:
                metrics[f'recall@{k}'] = SearchMetrics.recall_at_k(
                    relevant_cvs, retrieved_ids, k
                )
        
        # F1@K
        for k in [1, 3, 5, 8, 10]:
            if k <= top_k:
                metrics[f'f1@{k}'] = SearchMetrics.f1_at_k(
                    relevant_cvs, retrieved_ids, k
                )
        
        # MAP
        metrics['map'] = SearchMetrics.average_precision(relevant_cvs, retrieved_ids)
        
        # MRR
        metrics['mrr'] = SearchMetrics.mean_reciprocal_rank(relevant_cvs, retrieved_ids)
        
        # NDCG@K
        for k in [1, 3, 5, 8, 10]:
            if k <= top_k:
                metrics[f'ndcg@{k}'] = SearchMetrics.ndcg_at_k(
                    relevant_cvs, retrieved_ids, k
                )
        
        return {
            'vacancy': vacancy_name,
            'retrieved': retrieved_results[:5],
            'relevant': relevant_cvs,
            'relevant_count': len(relevant_cvs),
            'metrics': metrics
        }
    
    def evaluate_all(self, top_k: int = 10, search_mode: str = "hybrid") -> Tuple[pd.DataFrame, List[Dict]]:
        """Полная оценка всех вакансий"""
        results = []
        
        mode_names = {
            "dense": "Dense-only (Voyage AI)",
            "sparse": "Sparse-only (TF-IDF)",
            "hybrid": "Hybrid (Dense + TF-IDF)"
        }
        mode_display = mode_names.get(search_mode, search_mode)
        
        print(f"\n{'='*60}")
        print(f"🔍 ОЦЕНКА КАЧЕСТВА ПОИСКА - {mode_display}")
        print(f"{'='*60}\n")
        
        for vacancy_name in self.vacancies.keys():
            print(f"Обработка: {vacancy_name}...", end=' ')
            try:
                result = self.evaluate_single_vacancy(vacancy_name, top_k, search_mode)
                results.append(result)
                print(f"✅ MAP: {result['metrics']['map']:.3f}")
            except Exception as e:
                print(f"❌ Ошибка: {e}")
                continue
        
        # Конвертируем в DataFrame
        rows = []
        for r in results:
            row = {'vacancy': r['vacancy'], 'relevant_count': r['relevant_count']}
            row.update(r['metrics'])
            rows.append(row)
        
        df = pd.DataFrame(rows)
        
        if len(df) == 0:
            print(f"\n⚠️  НЕТ РЕЗУЛЬТАТОВ ДЛЯ ОЦЕНКИ")
            return df, results
        
        # Выводим статистику
        print(f"\n{'='*60}")
        print("📊 СРЕДНИЕ МЕТРИКИ ПО ВСЕМ ВАКАНСИЯМ")
        print(f"{'='*60}\n")
        
        metric_cols = [col for col in df.columns if col not in ['vacancy', 'relevant_count']]
        if len(metric_cols) > 0:
            summary = df[metric_cols].describe().loc[['mean', 'std', 'min', 'max']]
            print(summary.to_string())
        
        print(f"\n{'='*60}\n")
        
        return df, results
    
    def save_results(
        self,
        df: pd.DataFrame,
        results: List[Dict],
        output_dir: str | Path = "evaluation_results"
    ):
        """Сохраняет результаты оценки"""
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True, parents=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # CSV с метриками
        csv_path = output_dir / f"metrics_{timestamp}.csv"
        df.to_csv(csv_path, index=False)
        print(f"💾 Метрики сохранены: {csv_path}")
        
        # JSON с детальными результатами
        json_path = output_dir / f"detailed_{timestamp}.json"
        
        results_serializable = []
        for r in results:
            r_copy = r.copy()
            r_copy['relevant'] = list(r['relevant'])
            results_serializable.append(r_copy)
        
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(results_serializable, f, ensure_ascii=False, indent=2)
        
        print(f"💾 Детали сохранены: {json_path}")
        
        return csv_path, json_path
    
    def generate_confusion_matrix(self, results: List[Dict]) -> pd.DataFrame:
        """Генерирует матрицу путаницы"""
        position_types = sorted(set(
            '_'.join(v.split('_')[:-1]) for v in self.vacancies.keys()
        ))
        
        matrix = pd.DataFrame(
            0,
            index=position_types,
            columns=position_types
        )
        
        for result in results:
            vacancy_name = result['vacancy']
            vacancy_type = '_'.join(vacancy_name.split('_')[:-1])
            
            for item in result['retrieved']:
                cv_id = item[0] if isinstance(item, (tuple, list)) else item
                
                cv_parts = cv_id.split('_')
                if len(cv_parts) >= 2:
                    cv_type = '_'.join(cv_parts[:-1])
                    if cv_type in position_types:
                        matrix.loc[vacancy_type, cv_type] += 1
        
        return matrix
    
    def print_detailed_results(self, results: List[Dict]):
        """Выводит детальные результаты по каждой вакансии"""
        print(f"\n{'='*60}")
        print("📋 ДЕТАЛЬНЫЕ РЕЗУЛЬТАТЫ ПО ВАКАНСИЯМ")
        print(f"{'='*60}\n")
        
        for result in results:
            print(f"\n🎯 Вакансия: {result['vacancy']}")
            print(f"   Релевантных CV: {result['relevant_count']}")
            print(f"   MAP: {result['metrics']['map']:.3f}")
            print(f"   MRR: {result['metrics']['mrr']:.3f}")
            print(f"   Precision@5: {result['metrics'].get('precision@5', 0):.3f}")
            print(f"   Recall@10: {result['metrics'].get('recall@10', 0):.3f}")
            
            print(f"\n   Топ-5 найденных CV:")
            for i, item in enumerate(result['retrieved'], 1):
                if len(item) == 3:
                    cv_id, score, full_name = item
                else:
                    cv_id, score = item
                    full_name = cv_id
                
                is_relevant = "✅" if cv_id in result['relevant'] else "❌"
                display_name = f"{full_name} [{cv_id}]" if full_name != cv_id else cv_id
                print(f"      {i}. {display_name:<40} (score: {score:.4f}) {is_relevant}")
