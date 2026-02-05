# LLM Analyzer - Анализ соответствия кандидата вакансии

## Описание

`LLMAnalyzer` использует GPT-4 для глубокого анализа соответствия кандидата вакансии с детальными объяснениями.

## Возможности

- 📊 **Оценка релевантности** (0.0 - 1.0) на основе 4 критериев
- ✅ **Сильные стороны** - что подходит идеально
- ⚠️ **Слабые стороны** - где есть пробелы
- 🎯 **Ключевые совпадения** - конкретные навыки и опыт
- ❌ **Отсутствующие требования** - что нужно докачать
- 💡 **Рекомендация** - нанимать или нет
- 💭 **Детальное обоснование** - почему такая оценка

## Критерии оценки

1. **Technical skills** (40%) - Соответствие технических навыков
2. **Experience level** (25%) - Годы опыта, уровень сеньорности
3. **Domain fit** (20%) - Опыт в индустрии, типы проектов
4. **Soft skills** (15%) - Коммуникация, лидерство, командная работа

## Быстрый старт

### 1. Тестовый запуск

```bash
# Запуск с примером встроенных данных
python test_llm_analyzer.py
```

Это проанализирует тестового кандидата и покажет все возможности анализатора.

### 2. Использование в коде

```python
from app.services.llm_analyze import LLMAnalyzer
from app.models.cv import CVOutput

# Инициализация
analyzer = LLMAnalyzer(
    model="gpt-4o",      # или "gpt-4o-mini" для экономии
    temperature=0.3       # 0.0-1.0, меньше = более стабильно
)

# Анализ одного кандидата
analysis = analyzer.analyze_match(cv_data, vacancy_text)

print(f"Score: {analysis.relevance_score}")
print(f"Recommendation: {analysis.recommendation}")
print(f"Summary: {analysis.summary}")

# Сильные стороны
for strength in analysis.strengths:
    print(f"✅ {strength}")

# Слабые стороны
for weakness in analysis.weaknesses:
    print(f"⚠️ {weakness}")
```

### 3. Анализ нескольких кандидатов

```python
# Список кандидатов
candidates = [cv1, cv2, cv3]

# Анализируем всех
results = analyzer.analyze_multiple(
    candidates=candidates,
    vacancy_text=vacancy_text,
    show_progress=True
)

# Сортируем по score
sorted_results = sorted(
    results,
    key=lambda x: x[1].relevance_score,
    reverse=True
)

# Топ-3 кандидата
for cv, analysis in sorted_results[:3]:
    print(f"{cv.full_name}: {analysis.relevance_score:.3f}")
```

## Структура MatchAnalysis

```python
class MatchAnalysis(BaseModel):
    relevance_score: float         # 0.0 - 1.0
    overall_assessment: str        # excellent/good/moderate/poor
    summary: str                   # Краткое резюме
    strengths: List[str]           # 3-5 сильных сторон
    weaknesses: List[str]          # 2-4 слабых стороны
    key_matches: List[str]         # 3-5 ключевых совпадений
    missing_requirements: List[str]# 2-4 отсутствующих требования
    recommendation: str            # strongly_recommend/recommend/consider/not_recommend
    reasoning: str                 # Детальное обоснование
```

## Интерпретация score

| Score | Уровень | Описание |
|-------|---------|----------|
| 0.9-1.0 | 🌟 Exceptional | Превосходит требования |
| 0.75-0.89 | ✅ Strong | Соответствует большинству требований |
| 0.6-0.74 | 👍 Good | Соответствует основным требованиям |
| 0.4-0.59 | ⚠️ Moderate | Есть пробелы в ключевых требованиях |
| 0.2-0.39 | ❌ Weak | Значительные расхождения |
| 0.0-0.19 | 🚫 Poor | Не подходит |

```python
# Получить интерпретацию
interpretation = analyzer.get_score_interpretation(0.85)
print(interpretation['label'])        # "✅ Сильное совпадение"
print(interpretation['description'])  # Описание
```

## Интеграция в приложение

### Добавление в API

```python
# В app/api/routes.py
from app.services.llm_analyze import LLMAnalyzer

analyzer = LLMAnalyzer()

@router.post("/search/with-analysis")
async def search_with_llm_analysis(request: SearchRequest):
    # Обычный поиск
    candidates = search_candidates(...)
    
    # LLM анализ для топ-5
    for candidate in candidates[:5]:
        cv_data = CVOutput(**candidate.dict())
        analysis = analyzer.analyze_match(cv_data, request.vacancy_text)
        candidate.llm_score = analysis.relevance_score
        candidate.llm_reasoning = analysis.reasoning
    
    return candidates
```

### Использование в Streamlit

```python
# В app/ui/frontend.py
if st.button("Анализ с LLM"):
    analyzer = LLMAnalyzer()
    
    for candidate in top_candidates:
        with st.expander(f"🤖 LLM Анализ: {candidate['full_name']}"):
            analysis = analyzer.analyze_match(...)
            
            st.metric("LLM Score", f"{analysis.relevance_score:.3f}")
            st.write(f"**{analysis.summary}**")
            
            col1, col2 = st.columns(2)
            with col1:
                st.write("✅ **Сильные стороны:**")
                for s in analysis.strengths:
                    st.write(f"- {s}")
            
            with col2:
                st.write("⚠️ **Слабые стороны:**")
                for w in analysis.weaknesses:
                    st.write(f"- {w}")
```

## Стоимость использования

GPT-4o pricing (на февраль 2024):
- Input: $5 / 1M tokens
- Output: $15 / 1M tokens

Примерная стоимость на 1 анализ:
- ~500-1000 tokens input (CV + vacancy)
- ~300-500 tokens output (analysis)
- **~$0.01-0.02 за анализ**

Для экономии используйте `gpt-4o-mini` (в 10 раз дешевле).

## Настройка

```python
# Более детальный анализ
analyzer = LLMAnalyzer(
    model="gpt-4o",
    temperature=0.3  # Меньше = более консистентно
)

# Более креативный (но менее стабильный)
analyzer = LLMAnalyzer(
    model="gpt-4o",
    temperature=0.7
)

# Экономичный вариант
analyzer = LLMAnalyzer(
    model="gpt-4o-mini",  # В 10 раз дешевле
    temperature=0.3
)
```

## Примеры использования

### Пример 1: Быстрая оценка топ-10

```python
from app.services.cv_parser import CVParser
from app.services.llm_analyze import LLMAnalyzer

# Получаем топ-10 из Qdrant
parser = CVParser()
results = search_candidates(parser, vacancy_text, top_k=10)

# LLM анализ
analyzer = LLMAnalyzer(model="gpt-4o-mini")  # экономия

for candidate in results[:5]:  # анализируем только топ-5
    analysis = analyzer.analyze_match(candidate, vacancy_text)
    
    if analysis.relevance_score >= 0.75:
        print(f"🌟 {candidate.full_name}: {analysis.relevance_score:.3f}")
        print(f"   {analysis.summary}")
```

### Пример 2: Сравнение кандидатов

```python
# Анализ нескольких кандидатов
results = analyzer.analyze_multiple(candidates, vacancy_text)

# Группировка по уровню
excellent = [r for r in results if r[1].relevance_score >= 0.9]
strong = [r for r in results if 0.75 <= r[1].relevance_score < 0.9]
good = [r for r in results if 0.6 <= r[1].relevance_score < 0.75]

print(f"Exceptional: {len(excellent)}")
print(f"Strong: {len(strong)}")
print(f"Good: {len(good)}")
```

## Лимиты и рекомендации

1. **Rate limits**: OpenAI имеет лимиты на количество запросов
   - Tier 1: 500 RPM (requests per minute)
   - Добавьте задержки между запросами при batch обработке

2. **Batch processing**: Для анализа большого числа кандидатов
   ```python
   import time
   
   for i, cv in enumerate(candidates):
       analysis = analyzer.analyze_match(cv, vacancy)
       results.append(analysis)
       
       if (i + 1) % 10 == 0:
           time.sleep(1)  # Пауза каждые 10 запросов
   ```

3. **Кэширование**: Сохраняйте результаты анализа
   ```python
   import json
   
   # Сохранить
   with open(f"analysis_{cv.full_name}.json", "w") as f:
       json.dump(analysis.dict(), f)
   
   # Загрузить
   with open(f"analysis_{cv.full_name}.json") as f:
       cached = MatchAnalysis(**json.load(f))
   ```

## Troubleshooting

### OpenAI API key не найден
```bash
# Добавьте в .env
OPENAI_API_KEY=sk-...
```

### Rate limit exceeded
```python
# Добавьте retry logic
from tenacity import retry, wait_exponential

@retry(wait=wait_exponential(multiplier=1, min=4, max=60))
def analyze_with_retry(cv, vacancy):
    return analyzer.analyze_match(cv, vacancy)
```

### Токены превышают лимит
```python
# Сократите CV (только основное)
cv_short = CVOutput(
    full_name=cv.full_name,
    summary=cv.summary,
    skills=cv.skills[:20],  # Только топ-20 навыков
    work_history=cv.work_history[:3],  # Только последние 3 места
    total_experience_months=cv.total_experience_months
)
```

## Следующие шаги

1. ✅ Протестируйте на примере: `python test_llm_analyzer.py`
2. 📊 Интегрируйте в API (добавить новый эндпоинт)
3. 🎨 Добавьте в Streamlit UI
4. 💾 Реализуйте кэширование результатов
5. 📈 Соберите метрики качества LLM оценок
