"""
Dashboard для визуализации метрик качества поиска.

Запуск:
    streamlit run app/ui/dashboard.py --server.port 8502
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path
import json

# Настройка страницы
st.set_page_config(
    page_title="CV Search - Метрики",
    page_icon="📊",
    layout="wide"
)

# Путь к папке с метриками
METRICS_FOLDER = Path(__file__).parent.parent.parent / "metrics"


def load_metrics_files() -> dict:
    """Загрузка всех JSON файлов метрик"""
    metrics_data = {}
    
    if not METRICS_FOLDER.exists():
        return metrics_data
    
    json_files = sorted(METRICS_FOLDER.glob("*.json"))
    
    for json_file in json_files:
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            if data and isinstance(data, list):
                display_name = json_file.stem.replace('_', ' ').replace('-', ' ').title()
                
                metrics_data[display_name] = {
                    "path": json_file,
                    "raw_data": data,
                    "filename": json_file.stem
                }
        except Exception as e:
            st.warning(f"Ошибка загрузки {json_file.name}: {e}")
    
    return metrics_data


def json_to_dataframe(data: list) -> pd.DataFrame:
    """Преобразование JSON данных в DataFrame"""
    rows = []
    for item in data:
        row = {
            'vacancy': item.get('vacancy', ''),
            'relevant_count': item.get('relevant_count', 0)
        }
        metrics = item.get('metrics', {})
        row.update(metrics)
        rows.append(row)
    
    return pd.DataFrame(rows)


def get_method_color(method_name: str) -> str:
    """Цвет для метода поиска"""
    colors = {
        'voyage': '#1f77b4',
        'tfidf': '#ff7f0e',
        'td_idf': '#ff7f0e',
        'hybrid': '#2ca02c',
        'bm25': '#9467bd',
    }
    
    method_lower = method_name.lower()
    for key, color in colors.items():
        if key in method_lower:
            if 'voyage' in method_lower and ('tfidf' in method_lower or 'bm25' in method_lower):
                return '#2ca02c'  # hybrid
            return color
    return '#9467bd'


def get_method_type(method_name: str) -> str:
    """Определение типа метода поиска"""
    method_lower = method_name.lower()
    
    if 'voyage' in method_lower and ('tfidf' in method_lower or 'tf_idf' in method_lower or 'bm25' in method_lower):
        return "🔀 Hybrid"
    elif 'voyage' in method_lower:
        return "🧠 Dense"
    elif 'tfidf' in method_lower or 'td_idf' in method_lower:
        return "📝 TF-IDF"
    elif 'bm25' in method_lower:
        return "📝 BM25"
    return "❓ Unknown"


def create_comparison_chart(all_data: dict, metric: str) -> go.Figure:
    """Создание графика сравнения методов"""
    
    fig = go.Figure()
    
    for method_name, data in all_data.items():
        df = json_to_dataframe(data['raw_data'])
        if metric not in df.columns:
            continue
        
        color = get_method_color(method_name)
        
        fig.add_trace(go.Bar(
            name=method_name,
            x=df['vacancy'],
            y=df[metric],
            marker_color=color,
            text=[f"{v:.2f}" for v in df[metric]],
            textposition='outside'
        ))
    
    fig.update_layout(
        title=f"Сравнение методов: {metric}",
        xaxis_title="Вакансия",
        yaxis_title=metric,
        yaxis_range=[0, 1.1],
        barmode='group',
        height=500,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    return fig


def create_radar_comparison(all_data: dict) -> go.Figure:
    """Радарная диаграмма сравнения методов"""
    
    radar_metrics = ['map', 'mrr', 'precision@5', 'recall@5', 'ndcg@5', 'f1@5']
    
    fig = go.Figure()
    
    for method_name, data in all_data.items():
        df = json_to_dataframe(data['raw_data'])
        
        avg_values = []
        available_metrics = []
        
        for m in radar_metrics:
            if m in df.columns:
                avg_values.append(df[m].mean())
                available_metrics.append(m)
        
        if not available_metrics:
            continue
        
        avg_values.append(avg_values[0])
        available_metrics.append(available_metrics[0])
        
        color = get_method_color(method_name)
        method_type = get_method_type(method_name)
        
        fig.add_trace(go.Scatterpolar(
            r=avg_values,
            theta=available_metrics,
            fill='toself',
            name=f"{method_type} {method_name}",
            line_color=color,
            opacity=0.7
        ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1]
            )
        ),
        showlegend=True,
        title="Сравнение методов поиска (средние значения)",
        height=500
    )
    
    return fig


def create_heatmap_by_method(data: list, method_name: str) -> go.Figure:
    """Тепловая карта для одного метода"""
    
    df = json_to_dataframe(data)
    metric_cols = [col for col in df.columns if col not in ['vacancy', 'relevant_count']]
    
    if not metric_cols:
        return None
    
    heatmap_data = df[metric_cols].values
    
    fig = go.Figure(data=go.Heatmap(
        z=heatmap_data,
        x=metric_cols,
        y=df['vacancy'].tolist(),
        colorscale='RdYlGn',
        zmin=0,
        zmax=1,
        text=[[f"{val:.2f}" for val in row] for row in heatmap_data],
        texttemplate="%{text}",
        textfont={"size": 9},
        hovertemplate="Вакансия: %{y}<br>Метрика: %{x}<br>Значение: %{z:.3f}<extra></extra>"
    ))
    
    fig.update_layout(
        title=f"Тепловая карта: {method_name}",
        xaxis_title="Метрика",
        yaxis_title="Вакансия",
        height=350
    )
    
    return fig


def main():
    """Основная функция dashboard"""
    
    st.title("📊 Dashboard метрик качества поиска")
    st.markdown("Сравнение различных методов поиска кандидатов по вакансиям")
    
    all_metrics = load_metrics_files()
    
    if not all_metrics:
        st.warning("⚠️ Файлы метрик не найдены")
        st.info(f"Ожидаемая папка: `{METRICS_FOLDER}`")
        st.markdown("""
        Запустите оценку для создания метрик:
        ```bash
        python -m app.scripts.run_evaluation --hybrid
        ```
        """)
        return
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Настройки")
        
        st.markdown("### 📁 Загруженные методы")
        for name in all_metrics.keys():
            method_type = get_method_type(name)
            st.markdown(f"• {method_type} **{name}**")
        
        st.divider()
        
        st.markdown("### 🔗 Навигация")
        st.markdown("""
        - [🔍 Поиск кандидатов](http://localhost:8501)
        """)
    
    # ==================== SUMMARY ====================
    st.header("📈 Сводное сравнение методов")
    
    summary_data = []
    for method_name, data in all_metrics.items():
        df = json_to_dataframe(data['raw_data'])
        method_type = get_method_type(method_name)
        
        summary_data.append({
            'Метод': f"{method_type} {method_name}",
            'MAP': df['map'].mean() if 'map' in df.columns else 0,
            'MRR': df['mrr'].mean() if 'mrr' in df.columns else 0,
            'Precision@5': df['precision@5'].mean() if 'precision@5' in df.columns else 0,
            'Recall@5': df['recall@5'].mean() if 'recall@5' in df.columns else 0,
            'NDCG@5': df['ndcg@5'].mean() if 'ndcg@5' in df.columns else 0,
        })
    
    summary_df = pd.DataFrame(summary_data)
    
    cols = st.columns(len(all_metrics))
    for i, (method_name, data) in enumerate(all_metrics.items()):
        df = json_to_dataframe(data['raw_data'])
        method_type = get_method_type(method_name)
        
        with cols[i]:
            st.markdown(f"### {method_type}")
            st.caption(method_name)
            
            if 'map' in df.columns:
                st.metric("MAP", f"{df['map'].mean():.3f}")
            
            if 'precision@5' in df.columns:
                st.metric("Precision@5", f"{df['precision@5'].mean():.3f}")
            
            if 'recall@5' in df.columns:
                st.metric("Recall@5", f"{df['recall@5'].mean():.3f}")
    
    st.divider()
    
    # ==================== RADAR CHART ====================
    st.header("🎯 Радарная диаграмма")
    
    radar_fig = create_radar_comparison(all_metrics)
    st.plotly_chart(radar_fig, use_container_width=True)
    
    st.divider()
    
    # ==================== COMPARISON BY METRIC ====================
    st.header("📊 Сравнение по метрикам")
    
    available_metrics = ['map', 'mrr', 'precision@1', 'precision@3', 'precision@5', 
                        'recall@5', 'recall@10', 'f1@5', 'ndcg@5', 'ndcg@10']
    
    col1, col2 = st.columns([1, 3])
    with col1:
        selected_metric = st.selectbox(
            "Выберите метрику",
            options=available_metrics,
            index=0
        )
    
    comparison_fig = create_comparison_chart(all_metrics, selected_metric)
    st.plotly_chart(comparison_fig, use_container_width=True)
    
    st.divider()
    
    # ==================== HEATMAPS ====================
    st.header("🔥 Тепловые карты по методам")
    
    heatmap_cols = st.columns(len(all_metrics))
    
    for i, (method_name, data) in enumerate(all_metrics.items()):
        with heatmap_cols[i]:
            heatmap_fig = create_heatmap_by_method(data['raw_data'], method_name)
            if heatmap_fig:
                st.plotly_chart(heatmap_fig, use_container_width=True)
    
    st.divider()
    
    # ==================== DATA TABLE ====================
    st.header("📋 Полные данные")
    
    table_method = st.selectbox(
        "Метод для отображения",
        options=list(all_metrics.keys()),
        key="table_method"
    )
    
    df = json_to_dataframe(all_metrics[table_method]['raw_data'])
    
    display_df = df.copy()
    for col in display_df.columns:
        if col not in ['vacancy', 'relevant_count']:
            display_df[col] = display_df[col].apply(lambda x: f"{x:.4f}" if pd.notna(x) else "-")
    
    st.dataframe(display_df, use_container_width=True, hide_index=True)
    
    csv_data = df.to_csv(index=False)
    st.download_button(
        label="📥 Скачать CSV",
        data=csv_data,
        file_name=f"metrics_{all_metrics[table_method]['filename']}.csv",
        mime="text/csv"
    )


if __name__ == "__main__":
    main()
