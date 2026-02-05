"""
Метрики качества поиска (Information Retrieval metrics).
"""

from typing import List, Set
import numpy as np


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
