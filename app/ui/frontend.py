"""
Streamlit фронтенд для поиска кандидатов по вакансии.

Запуск:
    streamlit run app/ui/frontend.py

Требует запущенного API сервера:
    uvicorn app.main:app --port 8000
"""

import streamlit as st
import requests
from typing import Optional

# Конфигурация
API_URL = "http://localhost:8000"

# Настройка страницы
st.set_page_config(
    page_title="CV Search - Поиск кандидатов",
    page_icon="🔍",
    layout="wide"
)


def check_api_health() -> Optional[dict]:
    """Проверка доступности API"""
    try:
        response = requests.get(f"{API_URL}/health", timeout=5)
        if response.status_code == 200:
            return response.json()
        return None
    except requests.exceptions.RequestException:
        return None


def search_candidates(vacancy_text: str, search_mode: str, top_k: int) -> Optional[dict]:
    """Поиск кандидатов через API"""
    try:
        response = requests.post(
            f"{API_URL}/search",
            json={
                "vacancy_text": vacancy_text,
                "search_mode": search_mode,
                "top_k": top_k
            },
            timeout=60
        )
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"Ошибка API: {response.status_code} - {response.text}")
            return None
    except requests.exceptions.RequestException as e:
        st.error(f"Ошибка подключения к API: {e}")
        return None


def format_experience(months: int) -> str:
    """Форматирование опыта работы"""
    if months < 12:
        return f"{months} мес."
    years = months // 12
    remaining_months = months % 12
    if remaining_months == 0:
        return f"{years} лет" if years > 4 else f"{years} года"
    return f"{years} г. {remaining_months} мес."


def render_candidate_card(candidate: dict, expanded: bool = False):
    """Отображение карточки кандидата"""
    
    # Определяем цвет badge по score
    score = candidate.get("score", 0)
    if score >= 0.7:
        score_color = "green"
    elif score >= 0.5:
        score_color = "orange"
    else:
        score_color = "red"
    
    # Заголовок с именем и score
    col1, col2 = st.columns([4, 1])
    with col1:
        st.markdown(f"### {candidate.get('rank', '?')}. {candidate.get('full_name', 'Неизвестно')}")
    with col2:
        st.markdown(f"**Score:** :{score_color}[{score:.4f}]")
    
    # Основная информация
    col1, col2, col3 = st.columns(3)
    
    with col1:
        experience = candidate.get("total_experience_months", 0)
        st.metric("Опыт работы", format_experience(experience))
    
    with col2:
        skills_count = len(candidate.get("skills", []))
        st.metric("Навыков", skills_count)
    
    with col3:
        if candidate.get("location"):
            st.metric("Локация", ", ".join(candidate["location"][:2]))
        else:
            st.metric("Локация", "Не указана")
    
    # Контакты
    contacts = []
    if candidate.get("email"):
        contacts.append(f"📧 {candidate['email']}")
    if candidate.get("phone"):
        contacts.append(f"📱 {candidate['phone']}")
    if contacts:
        st.markdown(" | ".join(contacts))
    
    # Summary
    if candidate.get("summary"):
        st.markdown(f"**О кандидате:** {candidate['summary'][:300]}{'...' if len(candidate.get('summary', '')) > 300 else ''}")
    
    # Навыки
    if candidate.get("skills"):
        skills = candidate["skills"][:15]
        st.markdown("**Навыки:** " + ", ".join(f"`{skill}`" for skill in skills))
        if len(candidate["skills"]) > 15:
            st.caption(f"... и ещё {len(candidate['skills']) - 15} навыков")
    
    # Ссылки
    if candidate.get("links"):
        links_md = " | ".join([f"[{link.split('//')[-1][:30]}]({link})" for link in candidate["links"][:3]])
        st.markdown(f"**Ссылки:** {links_md}")
    
    # Детальная информация (раскрываемая)
    with st.expander("📋 Подробная информация"):
        
        # Языки
        if candidate.get("languages"):
            st.markdown(f"**Языки:** {', '.join(candidate['languages'])}")
        
        # История работы
        if candidate.get("work_history"):
            st.markdown("#### История работы")
            for i, work in enumerate(candidate["work_history"][:5], 1):
                st.markdown(f"""
**{i}. {work.get('role', 'Должность')}** @ {work.get('company', 'Компания')}  
📅 {work.get('start_date', '?')} — {work.get('end_date', '?')}  
{work.get('description', '')[:200]}{'...' if len(work.get('description', '')) > 200 else ''}
""")
                if work.get("technologies"):
                    st.markdown("Технологии: " + ", ".join(f"`{t}`" for t in work["technologies"][:10]))
                st.divider()
        
        # Source file
        if candidate.get("source_file"):
            st.caption(f"ID файла: {candidate['source_file']}")


def main():
    """Основная функция приложения"""
    
    # Заголовок
    st.title("🔍 Поиск кандидатов по вакансии")
    st.markdown("Введите текст вакансии, чтобы найти подходящих кандидатов из базы резюме.")
    
    # Проверка API
    health = check_api_health()
    
    # Статус в sidebar
    with st.sidebar:
        st.header("⚙️ Настройки")
        
        # Статус API
        if health:
            st.success(f"✅ API подключен")
            st.caption(f"Документов в базе: {health.get('documents_count', 0)}")
            st.caption(f"Sparse: {'✅' if health.get('sparse_fitted') else '❌'} ({health.get('sparse_method', 'tfidf')})")
        else:
            st.error("❌ API недоступен")
            st.caption("Запустите сервер:")
            st.code("uvicorn app.main:app --port 8000")
        
        st.divider()
        
        # Режим поиска
        search_mode = st.selectbox(
            "Режим поиска",
            options=["hybrid", "dense", "sparse"],
            format_func=lambda x: {
                "hybrid": "🔀 Hybrid (рекомендуется)",
                "dense": "🧠 Dense (семантический)",
                "sparse": "📝 Sparse (ключевые слова)"
            }.get(x, x),
            help="""
            **Hybrid** - комбинация семантического и keyword поиска (лучшее качество)
            
            **Dense** - поиск по смыслу через Voyage AI embeddings
            
            **Sparse** - поиск по ключевым словам через TF-IDF
            """
        )
        
        # Количество результатов
        top_k = st.slider(
            "Количество кандидатов",
            min_value=1,
            max_value=30,
            value=10,
            help="Сколько кандидатов показать в результатах"
        )
        
        st.divider()
        
        # Навигация
        st.markdown("### 🔗 Навигация")
        st.markdown("""
        - [📊 Dashboard метрик](http://localhost:8502)
        """)
        st.caption("Запустите dashboard:")
        st.code("streamlit run app/ui/dashboard.py --server.port 8502", language="bash")
        
        st.divider()
        
        # Информация
        st.markdown("### 📖 Как использовать")
        st.markdown("""
        1. Вставьте текст вакансии
        2. Выберите режим поиска
        3. Нажмите "Найти кандидатов"
        4. Изучите результаты
        """)
    
    # Основная область
    if not health:
        st.warning("⚠️ API сервер не запущен. Пожалуйста, запустите его для работы приложения.")
        st.code("uvicorn app.main:app --port 8000", language="bash")
        return
    
    # Текстовое поле для вакансии
    vacancy_text = st.text_area(
        "Текст вакансии",
        height=200,
        placeholder="""Вставьте сюда полный текст вакансии...

Например:
Ищем Python разработчика с опытом работы от 3 лет.
Требования:
- Python, FastAPI, Django
- PostgreSQL, Redis
- Docker, Kubernetes
- Опыт с ML/AI будет преимуществом

Обязанности:
- Разработка backend сервисов
- Code review
- Написание документации
"""
    )
    
    # Кнопка поиска
    col1, col2, col3 = st.columns([1, 1, 2])
    with col1:
        search_button = st.button("🔍 Найти кандидатов", type="primary", use_container_width=True)
    with col2:
        clear_button = st.button("🗑️ Очистить", use_container_width=True)
    
    if clear_button:
        st.rerun()
    
    # Выполнение поиска
    if search_button:
        if not vacancy_text or len(vacancy_text.strip()) < 10:
            st.warning("⚠️ Введите текст вакансии (минимум 10 символов)")
            return
        
        with st.spinner("🔍 Поиск кандидатов..."):
            result = search_candidates(vacancy_text, search_mode, top_k)
        
        if result:
            # Показываем результаты
            st.divider()
            
            # Метаданные поиска
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Найдено кандидатов", result.get("results_count", 0))
            with col2:
                mode_display = {
                    "hybrid": "🔀 Hybrid",
                    "dense": "🧠 Dense",
                    "sparse": "📝 Sparse"
                }.get(result.get("search_mode"), result.get("search_mode"))
                st.metric("Режим поиска", mode_display)
            with col3:
                st.metric("Запрос", f"{len(vacancy_text)} символов")
            
            st.divider()
            
            # Кандидаты
            candidates = result.get("candidates", [])
            
            if not candidates:
                st.info("😔 Кандидаты не найдены. Попробуйте изменить текст вакансии.")
                return
            
            st.subheader(f"👥 Топ-{len(candidates)} кандидатов")
            
            # Табы для разных представлений
            tab1, tab2 = st.tabs(["📋 Карточки", "📊 Таблица"])
            
            with tab1:
                # Карточки кандидатов
                for candidate in candidates:
                    with st.container():
                        render_candidate_card(candidate)
                        st.divider()
            
            with tab2:
                # Табличное представление
                table_data = []
                for c in candidates:
                    table_data.append({
                        "Ранг": c.get("rank"),
                        "Score": f"{c.get('score', 0):.4f}",
                        "Имя": c.get("full_name", ""),
                        "Опыт": format_experience(c.get("total_experience_months", 0)),
                        "Навыков": len(c.get("skills", [])),
                        "Email": c.get("email", "-"),
                        "Локация": ", ".join(c.get("location", [])[:2]) or "-"
                    })
                
                st.dataframe(
                    table_data,
                    use_container_width=True,
                    hide_index=True
                )
            
            # JSON для отладки
            with st.expander("🔧 Raw JSON Response"):
                st.json(result)


if __name__ == "__main__":
    main()
