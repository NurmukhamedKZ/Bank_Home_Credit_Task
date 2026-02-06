"""
Streamlit фронтенд для поиска кандидатов по вакансии.

Поддерживает три метода поиска:
    1. Vector Search (Dense/Sparse/Hybrid)
    2. ML Classifier (TF-IDF + Logistic Regression)
    3. LLM Analyzer (GPT-4 с объяснениями)

Запуск:
    streamlit run app/ui/frontend.py

Требует запущенного API сервера:
    uvicorn app.main:app --port 8000
"""

import streamlit as st
import requests
from typing import Optional
import os

# Конфигурация - читаем из переменных окружения для Docker
API_URL = os.environ.get("API_URL", "http://localhost:8000")

# Настройка страницы
st.set_page_config(
    page_title="CV Search - Поиск кандидатов",
    page_icon="🔍",
    layout="wide"
)


# ==================== API ВЫЗОВЫ ====================

def check_api_health() -> Optional[dict]:
    """Проверка доступности API"""
    try:
        response = requests.get(f"{API_URL}/health", timeout=5)
        if response.status_code == 200:
            return response.json()
        return None
    except requests.exceptions.RequestException:
        return None


def api_vector_search(vacancy_text: str, search_mode: str, top_k: int) -> Optional[dict]:
    """Vector Search через /search"""
    try:
        response = requests.post(
            f"{API_URL}/search",
            json={"vacancy_text": vacancy_text, "search_mode": search_mode, "top_k": top_k},
            timeout=60
        )
        if response.status_code == 200:
            return response.json()
        st.error(f"Ошибка API: {response.status_code} - {response.text}")
        return None
    except requests.exceptions.RequestException as e:
        st.error(f"Ошибка подключения: {e}")
        return None


def api_ml_classifier(vacancy_text: str, top_k: int, threshold: float) -> Optional[dict]:
    """ML Classifier через /search/ml-classifier"""
    try:
        response = requests.post(
            f"{API_URL}/search/ml-classifier",
            json={"vacancy_text": vacancy_text, "top_k": top_k, "threshold": threshold},
            timeout=60
        )
        if response.status_code == 200:
            return response.json()
        st.error(f"Ошибка API: {response.status_code} - {response.text}")
        return None
    except requests.exceptions.RequestException as e:
        st.error(f"Ошибка подключения: {e}")
        return None


def api_llm_analysis(vacancy_text: str, search_mode: str, top_k: int) -> Optional[dict]:
    """LLM Analyzer через /search/with-llm-analysis"""
    try:
        response = requests.post(
            f"{API_URL}/search/with-llm-analysis",
            json={"vacancy_text": vacancy_text, "search_mode": search_mode, "top_k": top_k},
            timeout=180
        )
        if response.status_code == 200:
            return response.json()
        st.error(f"Ошибка API: {response.status_code} - {response.text}")
        return None
    except requests.exceptions.RequestException as e:
        st.error(f"Ошибка подключения: {e}")
        return None


# ==================== УТИЛИТЫ ====================

def format_experience(months: int) -> str:
    """Форматирование опыта работы"""
    if months < 12:
        return f"{months} мес."
    years = months // 12
    remaining_months = months % 12
    if remaining_months == 0:
        return f"{years} лет" if years > 4 else f"{years} года"
    return f"{years} г. {remaining_months} мес."


# ==================== РЕНДЕРИНГ КАРТОЧЕК ====================

def render_candidate_card(candidate: dict):
    """Базовая карточка кандидата"""
    score = candidate.get("score", 0)
    score_color = "green" if score >= 0.7 else ("orange" if score >= 0.5 else "red")

    col1, col2 = st.columns([4, 1])
    with col1:
        st.markdown(f"### {candidate.get('rank', '?')}. {candidate.get('full_name', 'Неизвестно')}")
    with col2:
        st.markdown(f"**Score:** :{score_color}[{score:.4f}]")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Опыт работы", format_experience(candidate.get("total_experience_months", 0)))
    with col2:
        st.metric("Навыков", len(candidate.get("skills", [])))
    with col3:
        loc = ", ".join(candidate.get("location", [])[:2]) or "Не указана"
        st.metric("Локация", loc)

    contacts = []
    if candidate.get("email"):
        contacts.append(f"📧 {candidate['email']}")
    if candidate.get("phone"):
        contacts.append(f"📱 {candidate['phone']}")
    if contacts:
        st.markdown(" | ".join(contacts))

    if candidate.get("summary"):
        summary = candidate["summary"]
        st.markdown(f"**О кандидате:** {summary[:300]}{'...' if len(summary) > 300 else ''}")

    if candidate.get("skills"):
        skills = candidate["skills"][:15]
        st.markdown("**Навыки:** " + ", ".join(f"`{s}`" for s in skills))
        if len(candidate["skills"]) > 15:
            st.caption(f"... и ещё {len(candidate['skills']) - 15} навыков")

    if candidate.get("links"):
        links_md = " | ".join([f"[{l.split('//')[-1][:30]}]({l})" for l in candidate["links"][:3]])
        st.markdown(f"**Ссылки:** {links_md}")

    with st.expander("📋 Подробная информация"):
        if candidate.get("languages"):
            st.markdown(f"**Языки:** {', '.join(candidate['languages'])}")

        if candidate.get("work_history"):
            st.markdown("#### История работы")
            for i, work in enumerate(candidate["work_history"][:5], 1):
                st.markdown(
                    f"**{i}. {work.get('role', 'Должность')}** @ {work.get('company', 'Компания')}  \n"
                    f"📅 {work.get('start_date', '?')} — {work.get('end_date', '?')}  \n"
                    f"{work.get('description', '')[:200]}"
                )
                if work.get("technologies"):
                    st.markdown("Технологии: " + ", ".join(f"`{t}`" for t in work["technologies"][:10]))
                st.divider()

        if candidate.get("source_file"):
            st.caption(f"ID файла: {candidate['source_file']}")


def render_ml_badge(candidate: dict):
    """Бейдж ML классификатора"""
    ml_prob = candidate.get("ml_probability", 0)
    ml_pred = candidate.get("ml_prediction", 0)

    if ml_pred == 1:
        st.success(f"ML: Релевантен (p={ml_prob:.3f})")
    else:
        st.warning(f"ML: Не релевантен (p={ml_prob:.3f})")


def render_llm_analysis(llm: dict):
    """Блок LLM анализа"""
    if not llm:
        return

    score = llm.get("relevance_score", 0)
    score_color = "green" if score >= 0.75 else ("orange" if score >= 0.5 else "red")

    st.markdown(f"#### 🤖 LLM Анализ")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("LLM Score", f"{score:.3f}")
    with col2:
        assessment_map = {
            "excellent": "🌟 Отличный",
            "good": "✅ Хороший",
            "moderate": "⚠️ Средний",
            "poor": "❌ Слабый"
        }
        st.metric("Оценка", assessment_map.get(llm.get("overall_assessment", ""), llm.get("overall_assessment", "")))
    with col3:
        rec_map = {
            "strongly_recommend": "🟢 Рекомендую",
            "recommend": "🟡 Рассмотреть",
            "consider": "🟠 Возможно",
            "not_recommend": "🔴 Не рекомендую"
        }
        st.metric("Рекомендация", rec_map.get(llm.get("recommendation", ""), llm.get("recommendation", "")))

    st.markdown(f"**Резюме:** {llm.get('summary', '')}")

    col_left, col_right = st.columns(2)

    with col_left:
        st.markdown("**✅ Сильные стороны:**")
        for s in llm.get("strengths", []):
            st.markdown(f"- {s}")

        st.markdown("**🎯 Ключевые совпадения:**")
        for m in llm.get("key_matches", []):
            st.markdown(f"- {m}")

    with col_right:
        st.markdown("**⚠️ Слабые стороны:**")
        for w in llm.get("weaknesses", []):
            st.markdown(f"- {w}")

        st.markdown("**❌ Отсутствующие требования:**")
        for m in llm.get("missing_requirements", []):
            st.markdown(f"- {m}")

    with st.expander("💭 Детальное обоснование"):
        st.markdown(llm.get("reasoning", ""))


# ==================== ОСНОВНАЯ ФУНКЦИЯ ====================

def main():
    st.title("🔍 Поиск кандидатов по вакансии")
    st.markdown("Введите текст вакансии и выберите метод поиска.")

    health = check_api_health()

    # ==================== SIDEBAR ====================
    with st.sidebar:
        st.header("⚙️ Настройки")

        if health:
            st.success("✅ API подключен")
            st.caption(f"Документов: {health.get('documents_count', 0)}")
            st.caption(f"Sparse: {'✅' if health.get('sparse_fitted') else '❌'} ({health.get('sparse_method', 'tfidf')})")
        else:
            st.error("❌ API недоступен")
            st.code("uvicorn app.main:app --port 8000")

        st.divider()

        # Выбор метода поиска
        search_method = st.radio(
            "Метод поиска",
            options=["vector", "ml_classifier", "llm_analysis"],
            format_func=lambda x: {
                "vector": "🔀 Vector Search (быстрый)",
                "ml_classifier": "🤖 ML Classifier (TF-IDF)",
                "llm_analysis": "🧠 LLM Analyzer (GPT-4)"
            }[x],
            help="""
**Vector Search** - семантический + keyword поиск через Qdrant (~0.2 сек)

**ML Classifier** - supervised learning на TF-IDF (~1-2 сек)

**LLM Analyzer** - GPT-4 с детальным анализом топ-5 (~15-20 сек)
            """
        )

        st.divider()

        # Параметры для каждого метода
        if search_method == "vector":
            search_mode = st.selectbox(
                "Режим",
                options=["hybrid", "dense", "sparse"],
                format_func=lambda x: {
                    "hybrid": "🔀 Hybrid (рекомендуется)",
                    "dense": "🧠 Dense (семантический)",
                    "sparse": "📝 Sparse (ключевые слова)"
                }[x]
            )
            top_k = st.slider("Количество кандидатов", 1, 30, 10)

        elif search_method == "ml_classifier":
            top_k = st.slider("Количество кандидатов", 1, 30, 10)
            threshold = st.slider("Порог релевантности", 0.0, 1.0, 0.5, 0.05,
                                  help="Кандидаты с вероятностью выше порога считаются релевантными")

        elif search_method == "llm_analysis":
            search_mode = st.selectbox(
                "Режим поиска",
                options=["hybrid", "dense", "sparse"],
                format_func=lambda x: {
                    "hybrid": "🔀 Hybrid",
                    "dense": "🧠 Dense",
                    "sparse": "📝 Sparse"
                }[x]
            )
            top_k = st.slider("Количество кандидатов", 1, 15, 5)
            st.warning("⏱ LLM анализ занимает ~3-5 сек на кандидата")

        st.divider()
        st.markdown("### 📖 Методы поиска")
        st.markdown("""
| Метод | Скорость | Объяснения |
|-------|----------|------------|
| Vector | ⚡⚡⚡ | ❌ |
| ML | ⚡⚡ | ❌ |
| LLM | 🐌 | ✅ |
        """)

    # ==================== MAIN AREA ====================
    if not health:
        st.warning("⚠️ API сервер не запущен.")
        st.code("uvicorn app.main:app --port 8000", language="bash")
        return

    vacancy_text = st.text_area(
        "Текст вакансии",
        height=200,
        placeholder="Вставьте сюда текст вакансии...\n\nНапример:\nИщем Python разработчика с опытом FastAPI, PostgreSQL, Docker..."
    )

    col1, col2, _ = st.columns([1, 1, 2])
    with col1:
        search_button = st.button("🔍 Найти кандидатов", type="primary", use_container_width=True)
    with col2:
        if st.button("🗑️ Очистить", use_container_width=True):
            st.rerun()

    # ==================== ПОИСК ====================
    if search_button:
        if not vacancy_text or len(vacancy_text.strip()) < 10:
            st.warning("⚠️ Введите текст вакансии (минимум 10 символов)")
            return

        # Выбираем API в зависимости от метода
        if search_method == "vector":
            with st.spinner("🔍 Vector Search..."):
                result = api_vector_search(vacancy_text, search_mode, top_k)
            method_label = f"🔀 Vector Search ({search_mode})"

        elif search_method == "ml_classifier":
            with st.spinner("🤖 ML классификация..."):
                result = api_ml_classifier(vacancy_text, top_k, threshold)
            method_label = "🤖 ML Classifier"

        elif search_method == "llm_analysis":
            with st.spinner("🧠 LLM анализ (это может занять до минуты)..."):
                result = api_llm_analysis(vacancy_text, search_mode, top_k)
            method_label = "🧠 LLM Analyzer"

        if not result:
            return

        # ==================== РЕЗУЛЬТАТЫ ====================
        st.divider()

        # Метаданные
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Найдено кандидатов", result.get("results_count", 0))
        with col2:
            st.metric("Метод поиска", method_label)
        with col3:
            if search_method == "ml_classifier":
                st.metric("Релевантных", f"{result.get('relevant_count', 0)} / {result.get('results_count', 0)}")
            elif search_method == "llm_analysis":
                st.metric("LLM проанализировано", result.get("llm_analyzed_count", 0))
            else:
                st.metric("Запрос", f"{len(vacancy_text)} символов")

        st.divider()

        candidates = result.get("candidates", [])
        if not candidates:
            st.info("😔 Кандидаты не найдены.")
            return

        st.subheader(f"👥 Топ-{len(candidates)} кандидатов")

        # Табы: карточки / таблица
        tab1, tab2 = st.tabs(["📋 Карточки", "📊 Таблица"])

        with tab1:
            for candidate in candidates:
                with st.container():
                    # Базовая карточка
                    render_candidate_card(candidate)

                    # ML бейдж (для ML метода)
                    if search_method == "ml_classifier":
                        render_ml_badge(candidate)

                    # LLM анализ (для LLM метода)
                    if search_method == "llm_analysis" and candidate.get("llm_analysis"):
                        render_llm_analysis(candidate["llm_analysis"])

                    st.divider()

        with tab2:
            table_data = []
            for c in candidates:
                row = {
                    "Ранг": c.get("rank"),
                    "Score": f"{c.get('score', 0):.4f}",
                    "Имя": c.get("full_name", ""),
                    "Опыт": format_experience(c.get("total_experience_months", 0)),
                    "Навыков": len(c.get("skills", [])),
                    "Email": c.get("email", "-"),
                    "Локация": ", ".join(c.get("location", [])[:2]) or "-"
                }

                if search_method == "ml_classifier":
                    row["ML Prob"] = f"{c.get('ml_probability', 0):.3f}"
                    row["Релевантен"] = "✅" if c.get("ml_prediction") == 1 else "❌"

                if search_method == "llm_analysis" and c.get("llm_analysis"):
                    llm = c["llm_analysis"]
                    row["LLM Score"] = f"{llm.get('relevance_score', 0):.3f}"
                    row["Рекомендация"] = llm.get("recommendation", "-")

                table_data.append(row)

            st.dataframe(table_data, use_container_width=True, hide_index=True)

        # Raw JSON
        with st.expander("🔧 Raw JSON Response"):
            st.json(result)


if __name__ == "__main__":
    main()
