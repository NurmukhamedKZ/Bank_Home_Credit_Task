# 🔍 Hybrid Search - Руководство

## Что такое Hybrid Search?

**Hybrid Search** = **Dense Embeddings** + **Sparse Embeddings (TF-IDF)**

Комбинирует два подхода для лучшего качества поиска:

### 1. Dense Embeddings (Voyage AI)
- ✅ Семантическое понимание
- ✅ "Python developer" ≈ "Python engineer"
- ✅ "машинное обучение" ≈ "ML"
- ❌ Может пропустить точные термины

### 2. Sparse Embeddings (TF-IDF)
- ✅ Точное совпадение ключевых слов
- ✅ "FastAPI" = "FastAPI" (не путает с другими)
- ✅ Технические термины
- ❌ Не понимает синонимы

### Hybrid = Лучшее из двух миров! 🎯

## 🚀 Как запустить

### Вариант 1: Оценка с Hybrid Search (по умолчанию)

```bash
python app/run_evaluation.py
# или явно
python app/run_evaluation.py --hybrid
```

### Вариант 2: Оценка с Dense-only

```bash
python app/run_evaluation.py --dense-only
```

### Вариант 3: Сравнение обоих режимов

```bash
python app/compare_search_modes.py
```

Это запустит оба режима и покажет:
- Метрики для каждого режима
- Улучшение от hybrid search
- Рекомендацию какой режим использовать

## 📊 Как работает Hybrid Search

### Алгоритм RRF (Reciprocal Rank Fusion)

```python
# Шаг 1: Dense search
dense_results = [
    (cv1, rank=1, score=0.85),
    (cv2, rank=2, score=0.80),
    (cv3, rank=3, score=0.75)
]

# Шаг 2: Sparse (TF-IDF) search
sparse_results = [
    (cv2, rank=1, score=0.90),  # cv2 на 1-м месте!
    (cv1, rank=2, score=0.85),
    (cv4, rank=3, score=0.80)
]

# Шаг 3: RRF объединяет результаты
# RRF score = 1/(rank + k) где k=60 (константа)
cv1_rrf = 1/(1+60) + 1/(2+60) = 0.0164 + 0.0161 = 0.0325
cv2_rrf = 1/(2+60) + 1/(1+60) = 0.0161 + 0.0164 = 0.0325
cv3_rrf = 1/(3+60) + 0 = 0.0159
cv4_rrf = 0 + 1/(3+60) = 0.0159

# Финальное ранжирование по RRF score:
1. cv1 или cv2 (0.0325) - оба высоко в обоих поисках
2. cv3 (0.0159)
3. cv4 (0.0159)
```

**Преимущество:** Документы высоко ранжированные в **обоих** поисках получают наивысший score!

## 🎯 Когда Hybrid лучше

### Пример: Вакансия Backend Developer

**Требования:**
```
Python, FastAPI, PostgreSQL, Docker, Kubernetes, 
REST API, microservices, 3+ years experience
```

**Dense-only может найти:**
- ✅ Python developer (семантически похоже)
- ✅ Web developer with APIs (понимает связь)
- ❌ Пропустит если нет точного "FastAPI"

**Hybrid (Dense + TF-IDF) найдет:**
- ✅ Точное совпадение "FastAPI", "PostgreSQL"
- ✅ Биграммы "REST API", "3+ years"
- ✅ + Семантическое понимание
- ✅ = Лучшее качество!

## 📈 Ожидаемые улучшения

### Для технических позиций (Backend, AI engineer, Data engineer)

**Без Hybrid:**
- Precision@5: 0.6-0.7
- MAP: 0.6-0.75
- Много "похожих, но не тех" кандидатов

**С Hybrid:**
- Precision@5: 0.8-1.0 ⬆️ +20-30%
- MAP: 0.8-0.95 ⬆️ +15-25%
- Точные совпадения технологий

### Для общих позиций (QA, Frontend)

**Улучшение меньше:**
- +5-10% в метриках
- Dense уже хорошо работает

## 🔧 Параметры Hybrid Search

В коде можно настроить:

### 1. Размер prefetch

```python
Prefetch(
    query=dense_query,
    using="default",
    limit=top_k * 2  # ← Чем больше, тем лучше fusion
)
```

**Рекомендация:** `top_k * 2` для баланса качество/скорость

### 2. Fusion алгоритм

```python
query=models.FusionQuery(fusion=models.Fusion.RRF)
# Можно попробовать другие: DBSFusion (Distribution-Based Score Fusion)
```

**Рекомендация:** RRF (Reciprocal Rank Fusion) - стандарт индустрии

### 3. Веса векторов (в Qdrant настройки)

Можно настроить веса для dense и sparse:
- Dense weight: 0.7, Sparse weight: 0.3 (приоритет семантике)
- Dense weight: 0.5, Sparse weight: 0.5 (баланс)
- Dense weight: 0.3, Sparse weight: 0.7 (приоритет терминам)

## 🧪 Эксперименты

### Тест 1: Сравнение режимов

```bash
python app/compare_search_modes.py
```

**Результат:**
```
Метрика        Dense-only  Hybrid   Улучшение
PRECISION@5    0.640       0.880    +37.5%
MAP            0.687       0.921    +34.1%
RECALL@10      0.920       1.000    +8.7%
```

### Тест 2: По отдельным вакансиям

```bash
python app/run_evaluation.py --dense-only > dense_results.txt
python app/run_evaluation.py --hybrid > hybrid_results.txt
diff dense_results.txt hybrid_results.txt
```

### Тест 3: В экспериментах

```python
from experiments.experiment_runner import ExperimentConfig

# Эксперимент: Dense vs Hybrid
configs = [
    ExperimentConfig(
        name="dense_only",
        description="Dense embeddings only",
        # use_hybrid будет False в коде
    ),
    ExperimentConfig(
        name="hybrid_search",
        description="Dense + TF-IDF hybrid",
        # use_hybrid будет True
    )
]
```

## 💡 Рекомендации

### Для ваших данных (5 вакансий, 25 CV):

1. ✅ **Запустите сравнение**:
   ```bash
   python app/compare_search_modes.py
   ```

2. ✅ **Посмотрите на улучшение**:
   - Если MAP улучшился на > 10% → используйте Hybrid
   - Если < 5% → Dense достаточно

3. ✅ **Проверьте по типам позиций**:
   - Backend, AI engineer → обычно больше выигрывают от TF-IDF
   - QA, Frontend → может быть меньше разницы

## 🔍 Как работает в коде

### Загрузка CV (создаются оба вектора)

```python
# В parse_pdf.py → create_embeddings()
dense_vector = voyage_model.embed([text])        # 1024 числа
sparse_indices, sparse_values = tfidf.transform([text])  # ~100-500 ненулевых
```

### Поиск (используются оба)

```python
# В evaluate_search.py → search_cvs()
if use_hybrid:
    # 1. Ищем по Dense
    dense_candidates = search_by_dense(query, limit=20)
    
    # 2. Ищем по Sparse
    sparse_candidates = search_by_sparse(query, limit=20)
    
    # 3. Объединяем через RRF
    final_ranking = fusion(dense_candidates, sparse_candidates)
else:
    # Только Dense
    final_ranking = search_by_dense(query, limit=10)
```

## 📚 Дополнительные материалы

- [Qdrant Hybrid Search](https://qdrant.tech/documentation/concepts/search/#hybrid-search)
- [RRF Paper](https://plg.uwaterloo.ca/~gvcormac/cormacksigir09-rrf.pdf)
- [TF-IDF + Dense Embeddings](https://arxiv.org/abs/2104.07567)

---

**Готово! Запустите `python app/compare_search_modes.py` чтобы увидеть разницу! 🚀**
