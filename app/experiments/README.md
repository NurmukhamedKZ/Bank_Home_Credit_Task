# 🧪 Эксперименты с поиском CV

Система для A/B тестирования различных конфигураций поиска резюме.

## 🚀 Быстрый старт

```bash
# Запуск экспериментов
python run_experiments.py

# Или из корня проекта
python app/run_experiments.py
```

## 📁 Структура

```
experiments/
├── experiment_runner.py    # Основной класс для экспериментов
├── configs/               # Конфигурации экспериментов
│   ├── example_baseline.json
│   └── example_custom.json
└── results/              # Результаты экспериментов
    ├── *.json            # Детальные результаты
    └── comparison_*.csv  # Сравнительные таблицы
```

## 🎯 Готовые конфигурации

1. **Baseline** - Базовая конфигурация
   - TF-IDF: 10k features, unigrams+bigrams
   - Voyage-4-large embeddings

2. **Trigrams** - С триграммами
   - TF-IDF: 15k features, unigrams+bigrams+trigrams
   - Лучше для технических фраз

3. **Lightweight** - Облегченная
   - TF-IDF: 5k features, unigrams+bigrams
   - Быстрее, меньше памяти

4. **Detailed Prompt** - Детальный промпт
   - Акцент на технические навыки
   - Более глубокий парсинг

## 📊 Создание своей конфигурации

### Вариант 1: Через JSON

```json
{
  "name": "my_experiment",
  "description": "My custom configuration",
  "dense_model": "voyage-4-large",
  "dense_output_dim": 1024,
  "tfidf_max_features": 12000,
  "tfidf_ngram_range": [1, 3],
  "tfidf_min_df": 2,
  "system_prompt": "Custom prompt here...",
  "collection_name": "CVs_my_experiment"
}
```

Сохраните в `configs/my_experiment.json`.

### Вариант 2: Через код

```python
from experiments.experiment_runner import ExperimentConfig, ExperimentRunner

config = ExperimentConfig(
    name="my_experiment",
    description="Testing higher n-grams",
    tfidf_max_features=15000,
    tfidf_ngram_range=(1, 4),
    system_prompt="Focus on technical details..."
)

runner = ExperimentRunner()
result = runner.run_experiment(config)
```

## 🔬 Параметры для экспериментов

### TF-IDF параметры

| Параметр | Рекомендуется | Описание |
|----------|---------------|----------|
| `tfidf_max_features` | 10000 | Размер словаря |
| `tfidf_ngram_range` | (1, 2) | N-граммы |
| `tfidf_min_df` | 1-2 | Мин. частота |

### Dense Embeddings

| Параметр | Значение | Описание |
|----------|----------|----------|
| `dense_model` | "voyage-4-large" | Модель |
| `dense_output_dim` | 1024 | Размерность |

### System Prompt

Кастомизируйте под вашу задачу:
- Акцент на конкретные навыки
- Специфика индустрии
- Детализация извлечения

## 📈 Результаты

После запуска в `results/` появятся:

1. **Детальный JSON** - Полные результаты эксперимента
2. **Comparison CSV** - Сравнение с другими экспериментами

### Пример результата

```
Experiment     | MAP   | Precision@5 | Recall@10
---------------|-------|-------------|----------
baseline       | 0.923 | 0.880       | 1.000
trigrams       | 0.945 | 0.920       | 1.000
lightweight    | 0.901 | 0.840       | 0.980
```

## 💡 Best Practices

1. **Начните с baseline** для сравнения
2. **Меняйте один параметр** за раз
3. **Документируйте** изменения в description
4. **Сохраняйте** все результаты для анализа
5. **Используйте** статистические тесты для валидации

## 🎓 Примеры экспериментов

### Эксперимент 1: Влияние n-грамм

```python
configs = [
    ExperimentConfig(name="unigrams", tfidf_ngram_range=(1, 1)),
    ExperimentConfig(name="bigrams", tfidf_ngram_range=(1, 2)),
    ExperimentConfig(name="trigrams", tfidf_ngram_range=(1, 3)),
]
runner.run_multiple_experiments(configs)
```

### Эксперимент 2: Размер словаря

```python
configs = [
    ExperimentConfig(name="vocab_5k", tfidf_max_features=5000),
    ExperimentConfig(name="vocab_10k", tfidf_max_features=10000),
    ExperimentConfig(name="vocab_20k", tfidf_max_features=20000),
]
runner.run_multiple_experiments(configs)
```

### Эксперимент 3: System Prompt

```python
prompts = {
    "basic": "Extract CV data accurately.",
    "detailed": "Focus on technical skills and projects...",
    "industry": "Emphasize fintech experience and compliance..."
}

configs = [
    ExperimentConfig(name=name, system_prompt=prompt)
    for name, prompt in prompts.items()
]
runner.run_multiple_experiments(configs)
```

## 🔍 Анализ результатов

### Загрузка и сравнение

```python
import pandas as pd
import json

# Загружаем результаты
with open("results/baseline_*.json") as f:
    baseline = json.load(f)

with open("results/trigrams_*.json") as f:
    trigrams = json.load(f)

# Сравниваем MAP
print(f"Baseline MAP: {baseline['metrics_summary']['mean']['map']:.3f}")
print(f"Trigrams MAP: {trigrams['metrics_summary']['mean']['map']:.3f}")

# Improvement
improvement = (trigrams_map - baseline_map) / baseline_map * 100
print(f"Improvement: {improvement:.1f}%")
```

## 🆘 Troubleshooting

**Ошибка**: "Collection not found"
- Убедитесь что CV загружены в Qdrant
- Проверьте `collection_name` в конфиге

**Низкие метрики**:
- Проверьте quality парсинга CV
- Увеличьте `tfidf_max_features`
- Попробуйте другой system prompt

**Долгая обработка**:
- Используйте `reuse_collection=True` для повторных экспериментов
- Уменьшите `tfidf_max_features`

## 📚 Дополнительно

См. также:
- `app/EVALUATION_GUIDE.md` - Полное руководство
- `app/evaluate_search.py` - Код оценки метрик
- `app/run_experiments.py` - Скрипт запуска
