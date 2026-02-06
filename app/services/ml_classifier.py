"""
ML Classifier - классификатор на базе TF-IDF для сопоставления CV с вакансиями.

Supervised learning подход:
- Обучается на размеченных данных (вакансия → релевантные CV)
- Использует TF-IDF векторы как фичи
- Поддерживает несколько алгоритмов (LogisticRegression, SVM, RandomForest)
"""

import pickle
from pathlib import Path
from typing import List, Tuple, Optional, Dict
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import pandas as pd

from app.models.cv import CVOutput


class MLClassifier:
    """
    ML классификатор для определения релевантности кандидата вакансии.
    
    Использует TF-IDF векторы для представления текстов и
    классический ML алгоритм для предсказания релевантности.
    """
    
    SUPPORTED_MODELS = {
        'logistic': LogisticRegression,
        'svm': SVC,
        'random_forest': RandomForestClassifier
    }
    
    def __init__(
        self,
        model_type: str = 'logistic',
        tfidf_max_features: int = 5000,
        tfidf_ngram_range: Tuple[int, int] = (1, 2),
        model_params: Optional[Dict] = None
    ):
        """
        Args:
            model_type: Тип классификатора ('logistic', 'svm', 'random_forest')
            tfidf_max_features: Максимум фичей для TF-IDF
            tfidf_ngram_range: N-граммы для TF-IDF
            model_params: Дополнительные параметры для модели
        """
        if model_type not in self.SUPPORTED_MODELS:
            raise ValueError(f"model_type должен быть один из: {list(self.SUPPORTED_MODELS.keys())}")
        
        self.model_type = model_type
        self.tfidf_max_features = tfidf_max_features
        self.tfidf_ngram_range = tfidf_ngram_range
        
        # TF-IDF векторайзер
        self.vectorizer = TfidfVectorizer(
            max_features=tfidf_max_features,
            ngram_range=tfidf_ngram_range,
            lowercase=True,
            stop_words='english',
            min_df=2,
            max_df=0.95
        )
        
        # Модель классификатора
        model_class = self.SUPPORTED_MODELS[model_type]
        
        if model_params is None:
            model_params = self._get_default_params(model_type)
        
        self.classifier = model_class(**model_params)
        
        # Флаги обученности
        self._vectorizer_fitted = False
        self._classifier_fitted = False
        
        # Статистика обучения
        self.training_stats = {}
    
    @staticmethod
    def _get_default_params(model_type: str) -> Dict:
        """Параметры по умолчанию для каждого типа модели"""
        defaults = {
            'logistic': {
                'C': 1.0,
                'max_iter': 1000,
                'random_state': 42,
                'class_weight': 'balanced'  # Для несбалансированных классов
            },
            'svm': {
                'C': 1.0,
                'kernel': 'rbf',
                'probability': True,  # Для predict_proba
                'random_state': 42,
                'class_weight': 'balanced'
            },
            'random_forest': {
                'n_estimators': 100,
                'max_depth': 10,
                'random_state': 42,
                'class_weight': 'balanced'
            }
        }
        return defaults.get(model_type, {})
    
    def prepare_training_data(
        self,
        vacancy_texts: List[str],
        cv_texts: List[str],
        labels: List[int]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Подготавливает данные для обучения
        
        Args:
            vacancy_texts: Список текстов вакансий
            cv_texts: Список текстов CV
            labels: Метки релевантности (0 - не релевантно, 1 - релевантно)
            
        Returns:
            (X, y) - фичи и метки
        """
        if len(vacancy_texts) != len(cv_texts) or len(vacancy_texts) != len(labels):
            raise ValueError("Размеры vacancy_texts, cv_texts и labels должны совпадать")
        
        # Комбинируем вакансию и CV в один текст для векторизации
        combined_texts = [
            f"{vacancy} [SEP] {cv}"
            for vacancy, cv in zip(vacancy_texts, cv_texts)
        ]
        
        print(f"📊 Подготовка {len(combined_texts)} пар (вакансия, CV)...")
        
        # TF-IDF векторизация
        if not self._vectorizer_fitted:
            print(f"   🔢 Обучение TF-IDF vectorizer...")
            X = self.vectorizer.fit_transform(combined_texts)
            self._vectorizer_fitted = True
            print(f"   ✅ Vocabulary size: {len(self.vectorizer.vocabulary_)}")
        else:
            X = self.vectorizer.transform(combined_texts)
        
        y = np.array(labels)
        
        # Статистика классов
        unique, counts = np.unique(y, return_counts=True)
        class_dist = dict(zip(unique, counts))
        print(f"   📊 Распределение классов: {class_dist}")
        
        return X, y
    
    def fit(
        self,
        vacancy_texts: List[str],
        cv_texts: List[str],
        labels: List[int],
        validation_split: float = 0.2,
        verbose: bool = True
    ):
        """
        Обучает классификатор
        
        Args:
            vacancy_texts: Тексты вакансий
            cv_texts: Тексты CV
            labels: Метки (0/1)
            validation_split: Доля данных для валидации
            verbose: Выводить ли подробную информацию
        """
        if verbose:
            print(f"\n{'='*70}")
            print(f"ОБУЧЕНИЕ ML КЛАССИФИКАТОРА ({self.model_type.upper()})")
            print(f"{'='*70}\n")
        
        # Подготовка данных
        X, y = self.prepare_training_data(vacancy_texts, cv_texts, labels)
        
        # Train/validation split
        X_train, X_val, y_train, y_val = train_test_split(
            X, y,
            test_size=validation_split,
            random_state=42,
            stratify=y
        )
        
        if verbose:
            print(f"\n📚 Размеры данных:")
            print(f"   Train: {X_train.shape[0]} samples")
            print(f"   Validation: {X_val.shape[0]} samples")
            print(f"   Features: {X_train.shape[1]}")
        
        # Обучение
        if verbose:
            print(f"\n🤖 Обучение {self.model_type} классификатора...")
        
        self.classifier.fit(X_train, y_train)
        self._classifier_fitted = True
        
        # Оценка на train
        train_score = self.classifier.score(X_train, y_train)
        
        # Оценка на validation
        val_score = self.classifier.score(X_val, y_val)
        y_val_pred = self.classifier.predict(X_val)
        
        # Статистика
        self.training_stats = {
            'train_accuracy': train_score,
            'val_accuracy': val_score,
            'train_size': X_train.shape[0],
            'val_size': X_val.shape[0],
            'n_features': X_train.shape[1]
        }
        
        if verbose:
            print(f"\n{'='*70}")
            print("РЕЗУЛЬТАТЫ ОБУЧЕНИЯ")
            print(f"{'='*70}")
            print(f"📊 Train Accuracy: {train_score:.4f}")
            print(f"📊 Validation Accuracy: {val_score:.4f}")
            
            # Classification report
            print(f"\n📋 Classification Report (Validation):")
            print(classification_report(y_val, y_val_pred, target_names=['Not Relevant', 'Relevant']))
            
            # Confusion matrix
            cm = confusion_matrix(y_val, y_val_pred)
            print(f"\n📊 Confusion Matrix:")
            print(f"                  Predicted")
            print(f"                  0    1")
            print(f"Actual    0     {cm[0][0]:4d} {cm[0][1]:4d}")
            print(f"          1     {cm[1][0]:4d} {cm[1][1]:4d}")
            
            # ROC AUC (если есть predict_proba)
            if hasattr(self.classifier, 'predict_proba'):
                y_val_proba = self.classifier.predict_proba(X_val)[:, 1]
                roc_auc = roc_auc_score(y_val, y_val_proba)
                print(f"\n📊 ROC AUC: {roc_auc:.4f}")
                self.training_stats['roc_auc'] = roc_auc
        
        print(f"\n✅ Обучение завершено!")
    
    def predict(
        self,
        vacancy_text: str,
        cv_text: str
    ) -> int:
        """
        Предсказывает релевантность (0 или 1)
        
        Args:
            vacancy_text: Текст вакансии
            cv_text: Текст CV
            
        Returns:
            0 (не релевантно) или 1 (релевантно)
        """
        if not self._classifier_fitted:
            raise ValueError("Классификатор не обучен! Вызовите fit() сначала.")
        
        combined = f"{vacancy_text} [SEP] {cv_text}"
        X = self.vectorizer.transform([combined])
        
        return int(self.classifier.predict(X)[0])
    
    def predict_proba(
        self,
        vacancy_text: str,
        cv_text: str
    ) -> float:
        """
        Предсказывает вероятность релевантности
        
        Args:
            vacancy_text: Текст вакансии
            cv_text: Текст CV
            
        Returns:
            Вероятность релевантности (0.0 - 1.0)
        """
        if not self._classifier_fitted:
            raise ValueError("Классификатор не обучен! Вызовите fit() сначала.")
        
        if not hasattr(self.classifier, 'predict_proba'):
            raise ValueError(f"{self.model_type} не поддерживает predict_proba")
        
        combined = f"{vacancy_text} [SEP] {cv_text}"
        X = self.vectorizer.transform([combined])
        
        return float(self.classifier.predict_proba(X)[0][1])
    
    def predict_batch(
        self,
        vacancy_texts: List[str],
        cv_texts: List[str]
    ) -> List[int]:
        """Batch предсказание для нескольких пар"""
        if not self._classifier_fitted:
            raise ValueError("Классификатор не обучен!")
        
        combined = [f"{v} [SEP] {c}" for v, c in zip(vacancy_texts, cv_texts)]
        X = self.vectorizer.transform(combined)
        
        return self.classifier.predict(X).tolist()
    
    def predict_proba_batch(
        self,
        vacancy_texts: List[str],
        cv_texts: List[str]
    ) -> List[float]:
        """Batch предсказание вероятностей"""
        if not self._classifier_fitted:
            raise ValueError("Классификатор не обучен!")
        
        if not hasattr(self.classifier, 'predict_proba'):
            raise ValueError(f"{self.model_type} не поддерживает predict_proba")
        
        combined = [f"{v} [SEP] {c}" for v, c in zip(vacancy_texts, cv_texts)]
        X = self.vectorizer.transform(combined)
        
        return self.classifier.predict_proba(X)[:, 1].tolist()
    
    def save(self, filepath: str | Path):
        """Сохраняет модель на диск"""
        filepath = Path(filepath)
        
        model_data = {
            'model_type': self.model_type,
            'vectorizer': self.vectorizer,
            'classifier': self.classifier,
            'vectorizer_fitted': self._vectorizer_fitted,
            'classifier_fitted': self._classifier_fitted,
            'training_stats': self.training_stats,
            'tfidf_max_features': self.tfidf_max_features,
            'tfidf_ngram_range': self.tfidf_ngram_range
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"💾 Модель сохранена: {filepath}")
    
    @classmethod
    def load(cls, filepath: str | Path) -> 'MLClassifier':
        """Загружает модель с диска"""
        filepath = Path(filepath)
        
        if not filepath.exists():
            raise FileNotFoundError(f"Файл не найден: {filepath}")
        
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        # Создаем экземпляр
        instance = cls(
            model_type=model_data['model_type'],
            tfidf_max_features=model_data['tfidf_max_features'],
            tfidf_ngram_range=model_data['tfidf_ngram_range']
        )
        
        # Восстанавливаем состояние
        instance.vectorizer = model_data['vectorizer']
        instance.classifier = model_data['classifier']
        instance._vectorizer_fitted = model_data['vectorizer_fitted']
        instance._classifier_fitted = model_data['classifier_fitted']
        instance.training_stats = model_data['training_stats']
        
        print(f"📂 Модель загружена: {filepath}")
        return instance
    
    def get_feature_importance(self, top_n: int = 20) -> pd.DataFrame:
        """
        Возвращает важность фичей (только для некоторых моделей)
        
        Args:
            top_n: Количество топ фичей
            
        Returns:
            DataFrame с фичами и их важностью
        """
        if not self._classifier_fitted:
            raise ValueError("Классификатор не обучен!")
        
        feature_names = self.vectorizer.get_feature_names_out()
        
        if self.model_type == 'logistic':
            # Для логистической регрессии - коэффициенты
            importances = self.classifier.coef_[0]
        elif self.model_type == 'random_forest':
            # Для случайного леса - feature_importances
            importances = self.classifier.feature_importances_
        else:
            raise ValueError(f"{self.model_type} не поддерживает feature importance")
        
        # Сортируем по абсолютному значению
        indices = np.argsort(np.abs(importances))[::-1][:top_n]
        
        df = pd.DataFrame({
            'feature': feature_names[indices],
            'importance': importances[indices]
        })
        
        return df


def build_training_data_from_ground_truth(
    evaluator,
    negative_ratio: float = 1.0
) -> Tuple[List[str], List[str], List[int]]:
    """
    Строит обучающую выборку из ground truth evaluator'а
    
    Args:
        evaluator: CVSearchEvaluator с ground truth
        negative_ratio: Соотношение негативных примеров к позитивным
        
    Returns:
        (vacancy_texts, cv_texts, labels)
    """
    from pathlib import Path
    import random
    
    vacancy_texts = []
    cv_texts = []
    labels = []
    
    print(f"🏗️  Построение обучающей выборки из ground truth...")
    
    # Для каждой вакансии
    for vacancy_name, relevant_cvs in evaluator.ground_truth.items():
        if vacancy_name not in evaluator.vacancies:
            continue
        
        vacancy_text = evaluator.vacancies[vacancy_name]
        
        # Позитивные примеры (релевантные CV)
        for cv_name in relevant_cvs:
            cv_path = evaluator.cvs_folder / f"{cv_name}.txt"
            if cv_path.exists():
                cv_text = cv_path.read_text(encoding='utf-8')
                
                vacancy_texts.append(vacancy_text)
                cv_texts.append(cv_text)
                labels.append(1)  # Релевантно
        
        # Негативные примеры (нерелевантные CV)
        all_cvs = set(f.stem for f in evaluator.cvs_folder.glob("*.txt"))
        irrelevant_cvs = all_cvs - relevant_cvs
        
        # Сэмплируем негативные примеры
        n_negative = int(len(relevant_cvs) * negative_ratio)
        sampled_irrelevant = random.sample(
            list(irrelevant_cvs),
            min(n_negative, len(irrelevant_cvs))
        )
        
        for cv_name in sampled_irrelevant:
            cv_path = evaluator.cvs_folder / f"{cv_name}.txt"
            if cv_path.exists():
                cv_text = cv_path.read_text(encoding='utf-8')
                
                vacancy_texts.append(vacancy_text)
                cv_texts.append(cv_text)
                labels.append(0)  # Не релевантно
    
    print(f"   ✅ Создано {len(labels)} примеров")
    print(f"      Позитивных: {sum(labels)}")
    print(f"      Негативных: {len(labels) - sum(labels)}")
    
    return vacancy_texts, cv_texts, labels


# ==================== ТЕСТОВАЯ ФУНКЦИЯ ====================

def test_ml_classifier():
    """Тестирование ML классификатора"""
    from app.services.cv_parser import CVParser
    from app.evaluation.evaluator import CVSearchEvaluator
    
    print("""
╔═══════════════════════════════════════════════════════════════╗
║         ТЕСТ ML КЛАССИФИКАТОРА (TF-IDF + ML)                 ║
╚═══════════════════════════════════════════════════════════════╝
    """)
    
    # Инициализация
    from app.core.config import QDRANT_COLLECTION_NAME
    
    print("🚀 Инициализация CVParser и Evaluator...")
    parser = CVParser(collection_name=QDRANT_COLLECTION_NAME)
    evaluator = CVSearchEvaluator(parser)
    
    # Построение обучающей выборки
    vacancy_texts, cv_texts, labels = build_training_data_from_ground_truth(
        evaluator,
        negative_ratio=1.5  # 1.5 негативных на 1 позитивный
    )
    
    # Создание и обучение классификатора
    classifier = MLClassifier(
        model_type='logistic',
        tfidf_max_features=5000,
        tfidf_ngram_range=(1, 2)
    )
    
    classifier.fit(vacancy_texts, cv_texts, labels, validation_split=0.2)
    
    # Сохранение модели
    model_path = Path("data/models/ml_classifier_logistic.pkl")
    model_path.parent.mkdir(parents=True, exist_ok=True)
    classifier.save(model_path)
    
    # Тест предсказания
    print(f"\n{'='*70}")
    print("ТЕСТ ПРЕДСКАЗАНИЯ")
    print(f"{'='*70}\n")
    
    if len(evaluator.vacancies) > 0:
        test_vacancy_name = list(evaluator.vacancies.keys())[0]
        test_vacancy_text = evaluator.vacancies[test_vacancy_name]
        
        print(f"Вакансия: {test_vacancy_name}")
        
        # Топ-5 CV
        all_cvs = list(evaluator.cvs_folder.glob("*.txt"))[:5]
        
        for cv_path in all_cvs:
            cv_text = cv_path.read_text(encoding='utf-8')
            
            prediction = classifier.predict(test_vacancy_text, cv_text)
            probability = classifier.predict_proba(test_vacancy_text, cv_text)
            
            relevant = "✅ РЕЛЕВАНТНО" if prediction == 1 else "❌ НЕ РЕЛЕВАНТНО"
            
            print(f"   {cv_path.stem}: {relevant} (вероятность: {probability:.3f})")
    
    print(f"\n✅ ТЕСТ ЗАВЕРШЕН!")


if __name__ == "__main__":
    test_ml_classifier()
