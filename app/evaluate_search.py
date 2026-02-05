"""
Система оценки качества поиска резюме по вакансиям
Поддерживает различные метрики IR (Information Retrieval)
"""

from pathlib import Path
from typing import List, Dict, Set, Optional, Tuple
import numpy as np
import pandas as pd
from datetime import datetime
import json

from service.parse_pdf import CVParser


class SearchMetrics:
    """Вычисление метрик качества поиска"""
    
    @staticmethod
    def precision_at_k(relevant: Set[str], retrieved: List[str], k: int) -> float:
        """Precision@K: доля релевантных документов в топ-K"""
        if k == 0:
            return 0.0
        top_k = set(retrieved[:k])
        return len(top_k & relevant) / k
    
    @staticmethod
    def recall_at_k(relevant: Set[str], retrieved: List[str], k: int) -> float:
        """Recall@K: доля найденных релевантных документов из всех релевантных"""
        if len(relevant) == 0:
            return 0.0
        top_k = set(retrieved[:k])
        return len(top_k & relevant) / len(relevant)
    
    @staticmethod
    def f1_at_k(relevant: Set[str], retrieved: List[str], k: int) -> float:
        """F1-score@K: гармоническое среднее precision и recall"""
        p = SearchMetrics.precision_at_k(relevant, retrieved, k)
        r = SearchMetrics.recall_at_k(relevant, retrieved, k)
        if p + r == 0:
            return 0.0
        return 2 * (p * r) / (p + r)
    
    @staticmethod
    def average_precision(relevant: Set[str], retrieved: List[str]) -> float:
        """Average Precision: учитывает порядок релевантных результатов"""
        if len(relevant) == 0:
            return 0.0
        
        avg_precision = 0.0
        num_relevant = 0
        
        for i, doc_id in enumerate(retrieved, 1):
            if doc_id in relevant:
                num_relevant += 1
                precision_at_i = num_relevant / i
                avg_precision += precision_at_i
        
        return avg_precision / len(relevant)
    
    @staticmethod
    def mean_reciprocal_rank(relevant: Set[str], retrieved: List[str]) -> float:
        """MRR: позиция первого релевантного результата"""
        for i, doc_id in enumerate(retrieved, 1):
            if doc_id in relevant:
                return 1.0 / i
        return 0.0
    
    @staticmethod
    def ndcg_at_k(relevant: Set[str], retrieved: List[str], k: int) -> float:
        """NDCG@K: нормализованный дисконтированный кумулятивный выигрыш"""
        # Упрощенная версия: релевантность либо 1, либо 0
        dcg = 0.0
        for i, doc_id in enumerate(retrieved[:k], 1):
            if doc_id in relevant:
                dcg += 1.0 / np.log2(i + 1)
        
        # Ideal DCG
        idcg = sum(1.0 / np.log2(i + 1) for i in range(1, min(len(relevant), k) + 1))
        
        return dcg / idcg if idcg > 0 else 0.0


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
        project_root = Path(__file__).parent.parent
        
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
            print(f"   Существует: {self.vacancies_folder.exists()}")
            if self.vacancies_folder.exists():
                files = list(self.vacancies_folder.glob("*.txt"))
                print(f"   Файлов .txt: {len(files)}")
        
        if len(self.ground_truth) == 0:
            print(f"\n⚠️  ВНИМАНИЕ: CV не найдены!")
            print(f"   Папка: {self.cvs_folder}")
            print(f"   Существует: {self.cvs_folder.exists()}")
            if self.cvs_folder.exists():
                files = list(self.cvs_folder.glob("*.txt"))
                print(f"   Файлов .txt: {len(files)}")
        
        # Проверка Qdrant
        try:
            collection_info = self.parser.qdrant_client.get_collection(self.parser.collection_name)
            print(f"📊 CV в Qdrant: {collection_info.points_count}")
            
            if collection_info.points_count == 0:
                print(f"\n⚠️  ВНИМАНИЕ: Коллекция Qdrant пуста!")
                print(f"   Запустите: python app/process_cvs.py")
        except Exception as e:
            print(f"\n⚠️  Ошибка при проверке Qdrant: {e}")
            print(f"   Коллекция '{self.parser.collection_name}' может не существовать")
    
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
        
        Returns:
            {vacancy_name: set(relevant_cv_names)}
        """
        ground_truth = {}
        
        for vacancy_name in self.vacancies.keys():
            # Извлекаем базовое имя (без номера)
            # "AI_engineer_1" -> "AI_engineer"
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
        use_hybrid: bool = True
    ) -> List[Tuple[str, float]]:
        """
        Поиск CV через Qdrant
        
        Args:
            query_text: Текст запроса (вакансия)
            top_k: Количество результатов
            use_hybrid: Использовать hybrid search (dense + sparse)
            
        Returns:
            List[(cv_identifier, score)]
        """
        # Создаем dense embedding
        query_vector = self.parser.dense_model.embed_documents([query_text])[0]
        
        # Поиск в Qdrant
        results = self.parser.qdrant_client.query_points(
            collection_name=self.parser.collection_name,
            query=query_vector,
            using="default",
            limit=top_k,
            with_payload=True,
            with_vectors=False
        )
        
        # Извлекаем идентификаторы (используем source_file для сопоставления с ground truth)
        cv_results = []
        for point in results.points:
            # Приоритет: source_file (имя файла) для сопоставления, или full_name для отображения
            cv_identifier = point.payload.get('source_file', point.payload.get('full_name', 'Unknown'))
            score = point.score
            # Сохраняем и ID и имя для отображения
            full_name = point.payload.get('full_name', 'Unknown')
            cv_results.append((cv_identifier, score, full_name))
        
        return cv_results
    
    def evaluate_single_vacancy(
        self,
        vacancy_name: str,
        top_k: int = 10,
        use_hybrid: bool = True
    ) -> Dict:
        """
        Оценка качества поиска для одной вакансии
        
        Returns:
            {
                'vacancy': str,
                'retrieved': List[Tuple[str, float]],
                'relevant': Set[str],
                'metrics': Dict[str, float]
            }
        """
        if vacancy_name not in self.vacancies:
            raise ValueError(f"Вакансия '{vacancy_name}' не найдена")
        
        vacancy_text = self.vacancies[vacancy_name]
        relevant_cvs = self.ground_truth[vacancy_name]
        
        # Поиск
        retrieved_results = self.search_cvs(vacancy_text, top_k, use_hybrid)
        # retrieved_results теперь содержит (cv_id, score, full_name)
        retrieved_ids = [cv_id for cv_id, _, _ in retrieved_results]
        
        # Вычисляем метрики
        metrics = {}
        
        # Precision@K для разных K
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
        
        # MAP (Mean Average Precision)
        metrics['map'] = SearchMetrics.average_precision(relevant_cvs, retrieved_ids)
        
        # MRR (Mean Reciprocal Rank)
        metrics['mrr'] = SearchMetrics.mean_reciprocal_rank(relevant_cvs, retrieved_ids)
        
        # NDCG@K
        for k in [1, 3, 5, 8, 10]:
            if k <= top_k:
                metrics[f'ndcg@{k}'] = SearchMetrics.ndcg_at_k(
                    relevant_cvs, retrieved_ids, k
                )
        
        return {
            'vacancy': vacancy_name,
            'retrieved': retrieved_results[:5],  # Топ-5 для анализа
            'relevant': relevant_cvs,
            'relevant_count': len(relevant_cvs),
            'metrics': metrics
        }
    
    def evaluate_all(self, top_k: int = 10) -> Tuple[pd.DataFrame, List[Dict]]:
        """
        Полная оценка всех вакансий
        
        Returns:
            (DataFrame с метриками, список детальных результатов)
        """
        results = []
        
        print(f"\n{'='*60}")
        print("🔍 ОЦЕНКА КАЧЕСТВА ПОИСКА")
        print(f"{'='*60}\n")
        
        for vacancy_name in self.vacancies.keys():
            print(f"Обработка: {vacancy_name}...", end=' ')
            try:
                result = self.evaluate_single_vacancy(vacancy_name, top_k)
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
        
        # Проверка на пустые результаты
        if len(df) == 0:
            print(f"\n{'='*60}")
            print("⚠️  НЕТ РЕЗУЛЬТАТОВ ДЛЯ ОЦЕНКИ")
            print(f"{'='*60}\n")
            print("Возможные причины:")
            print("  1. Не найдены вакансии в папке data/vacancy/")
            print("  2. Не найдены CV в папке data/Parsed_CVs/")
            print("  3. CV не загружены в Qdrant")
            print("\nРешение:")
            print("  - Проверьте наличие файлов в указанных папках")
            print("  - Запустите: python app/process_cvs.py")
            print(f"\n{'='*60}\n")
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
        
        # Конвертируем sets в lists для JSON
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
        """
        Генерирует матрицу путаницы: какие типы CV находятся для каких вакансий
        
        Returns:
            DataFrame с матрицей путаницы
        """
        # Типы позиций
        position_types = sorted(set(
            '_'.join(v.split('_')[:-1]) for v in self.vacancies.keys()
        ))
        
        # Создаем матрицу
        matrix = pd.DataFrame(
            0,
            index=position_types,
            columns=position_types
        )
        
        for result in results:
            vacancy_name = result['vacancy']
            vacancy_type = '_'.join(vacancy_name.split('_')[:-1])
            
            # Считаем найденные CV по типам
            for item in result['retrieved']:
                # item может быть (cv_id, score, full_name) или (cv_id, score)
                cv_id = item[0] if isinstance(item, (tuple, list)) else item
                
                # Извлекаем тип CV из ID
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
                # item может быть (cv_id, score, full_name) или (cv_id, score)
                if len(item) == 3:
                    cv_id, score, full_name = item
                else:
                    cv_id, score = item
                    full_name = cv_id
                
                is_relevant = "✅" if cv_id in result['relevant'] else "❌"
                display_name = f"{full_name} [{cv_id}]" if full_name != cv_id else cv_id
                print(f"      {i}. {display_name:<40} (score: {score:.4f}) {is_relevant}")


def main():
    """Основная функция для запуска оценки"""
    from service.parse_pdf import CVParser
    
    print("🚀 Инициализация CVParser...")
    parser = CVParser(collection_name="CVs")
    
    print("📊 Создание оценщика...")
    evaluator = CVSearchEvaluator(parser)
    
    # Полная оценка
    df, results = evaluator.evaluate_all(top_k=10)
    
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
    print(f"{'='*60}\n")
    
    return df, results


if __name__ == "__main__":
    main()
