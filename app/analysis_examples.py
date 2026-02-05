"""
Примеры анализа результатов оценки качества поиска
Визуализация, статистические тесты, сравнение экспериментов
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
from typing import List, Dict
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats


def load_experiment_results(result_file: str | Path) -> Dict:
    """Загружает результаты эксперимента из JSON"""
    with open(result_file, 'r', encoding='utf-8') as f:
        return json.load(f)


def compare_two_experiments(result1: Dict, result2: Dict) -> pd.DataFrame:
    """
    Сравнивает два эксперимента
    
    Returns:
        DataFrame с метриками и улучшениями
    """
    metrics1 = result1['metrics_summary']['mean']
    metrics2 = result2['metrics_summary']['mean']
    
    comparison = []
    for metric in metrics1.keys():
        val1 = metrics1[metric]
        val2 = metrics2[metric]
        improvement = ((val2 - val1) / val1 * 100) if val1 > 0 else 0
        
        comparison.append({
            'metric': metric,
            f"{result1['config']['name']}": val1,
            f"{result2['config']['name']}": val2,
            'improvement_%': improvement
        })
    
    df = pd.DataFrame(comparison)
    return df


def plot_metrics_comparison(results: List[Dict], output_path: str = None):
    """
    Создает визуализацию сравнения метрик
    
    Args:
        results: Список результатов экспериментов
        output_path: Путь для сохранения графика
    """
    # Извлекаем данные
    data = []
    for result in results:
        row = {'experiment': result['config']['name']}
        row.update(result['metrics_summary']['mean'])
        data.append(row)
    
    df = pd.DataFrame(data)
    
    # Создаем графики
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Comparison of Search Quality Metrics', fontsize=16, fontweight='bold')
    
    metrics = ['precision@5', 'recall@10', 'map', 'mrr', 'ndcg@5', 'f1@5']
    
    for idx, metric in enumerate(metrics):
        if metric not in df.columns:
            continue
        
        row = idx // 3
        col = idx % 3
        ax = axes[row, col]
        
        # Bar plot
        df.plot(x='experiment', y=metric, kind='bar', ax=ax, legend=False)
        ax.set_title(metric.upper(), fontweight='bold')
        ax.set_xlabel('')
        ax.set_ylabel('Score')
        ax.set_ylim([0, 1.0])
        ax.grid(axis='y', alpha=0.3)
        
        # Добавляем значения на барах
        for container in ax.containers:
            ax.bar_label(container, fmt='%.3f', padding=3)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"📊 График сохранен: {output_path}")
    else:
        plt.show()
    
    return fig


def plot_confusion_matrix(results: List[Dict], output_path: str = None):
    """
    Визуализация матрицы путаницы
    
    Args:
        results: Результаты оценки
        output_path: Путь для сохранения
    """
    # Строим матрицу
    from evaluate_search import CVSearchEvaluator
    from service.parse_pdf import CVParser
    
    parser = CVParser()
    evaluator = CVSearchEvaluator(parser)
    
    confusion = evaluator.generate_confusion_matrix(results)
    
    # Визуализация
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        confusion,
        annot=True,
        fmt='g',
        cmap='YlOrRd',
        cbar_kws={'label': 'Number of CVs retrieved'}
    )
    plt.title('Confusion Matrix: Retrieved CV Types per Vacancy', fontweight='bold', pad=20)
    plt.xlabel('Retrieved CV Type', fontweight='bold')
    plt.ylabel('Vacancy Type', fontweight='bold')
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"📊 Матрица путаницы сохранена: {output_path}")
    else:
        plt.show()


def statistical_significance_test(results1: List[float], results2: List[float]) -> Dict:
    """
    Тест статистической значимости различий
    
    Args:
        results1: Метрики первого эксперимента (по вакансиям)
        results2: Метрики второго эксперимента (по вакансиям)
        
    Returns:
        Результаты t-теста
    """
    # Paired t-test (сравниваем на одних и тех же вакансиях)
    t_stat, p_value = stats.ttest_rel(results1, results2)
    
    # Cohen's d (размер эффекта)
    mean_diff = np.mean(results2) - np.mean(results1)
    pooled_std = np.sqrt((np.std(results1)**2 + np.std(results2)**2) / 2)
    cohens_d = mean_diff / pooled_std if pooled_std > 0 else 0
    
    return {
        't_statistic': t_stat,
        'p_value': p_value,
        'significant': p_value < 0.05,
        'cohens_d': cohens_d,
        'effect_size': (
            'small' if abs(cohens_d) < 0.5 else
            'medium' if abs(cohens_d) < 0.8 else
            'large'
        ),
        'mean_improvement': mean_diff
    }


def analyze_per_vacancy_performance(results: List[Dict]) -> pd.DataFrame:
    """
    Анализ производительности по каждой вакансии
    
    Returns:
        DataFrame с метриками по вакансиям
    """
    rows = []
    
    for result in results:
        for vacancy_result in result['detailed_results']:
            row = {
                'experiment': result['config']['name'],
                'vacancy': vacancy_result['vacancy']
            }
            row.update(vacancy_result['metrics'])
            rows.append(row)
    
    df = pd.DataFrame(rows)
    return df


def find_difficult_vacancies(per_vacancy_df: pd.DataFrame, metric: str = 'map') -> pd.DataFrame:
    """
    Находит вакансии, которые труднее всего подобрать
    
    Args:
        per_vacancy_df: DataFrame с метриками по вакансиям
        metric: Метрика для оценки сложности
        
    Returns:
        DataFrame с ранжированием по сложности
    """
    # Группируем по вакансиям и считаем среднюю метрику
    difficulty = per_vacancy_df.groupby('vacancy')[metric].mean().reset_index()
    difficulty['difficulty_score'] = 1 - difficulty[metric]  # Чем ниже метрика, тем сложнее
    difficulty = difficulty.sort_values('difficulty_score', ascending=False)
    
    return difficulty


def plot_per_vacancy_performance(per_vacancy_df: pd.DataFrame, output_path: str = None):
    """
    Визуализация производительности по вакансиям
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # MAP по вакансиям
    map_by_vacancy = per_vacancy_df.pivot_table(
        values='map',
        index='vacancy',
        columns='experiment'
    )
    
    map_by_vacancy.plot(kind='bar', ax=axes[0])
    axes[0].set_title('MAP by Vacancy', fontweight='bold')
    axes[0].set_xlabel('Vacancy')
    axes[0].set_ylabel('MAP Score')
    axes[0].legend(title='Experiment')
    axes[0].grid(axis='y', alpha=0.3)
    
    # Precision@5 по вакансиям
    p5_by_vacancy = per_vacancy_df.pivot_table(
        values='precision@5',
        index='vacancy',
        columns='experiment'
    )
    
    p5_by_vacancy.plot(kind='bar', ax=axes[1])
    axes[1].set_title('Precision@5 by Vacancy', fontweight='bold')
    axes[1].set_xlabel('Vacancy')
    axes[1].set_ylabel('Precision@5 Score')
    axes[1].legend(title='Experiment')
    axes[1].grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"📊 График по вакансиям сохранен: {output_path}")
    else:
        plt.show()


def generate_full_report(
    results: List[Dict],
    output_dir: str | Path = "analysis_reports"
):
    """
    Генерирует полный отчет с анализом
    
    Args:
        results: Список результатов экспериментов
        output_dir: Папка для сохранения отчета
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    print("\n" + "="*70)
    print("📊 ГЕНЕРАЦИЯ ПОЛНОГО ОТЧЕТА")
    print("="*70 + "\n")
    
    # 1. Общее сравнение метрик
    print("1. Сравнение метрик...")
    plot_metrics_comparison(
        results,
        output_path=output_dir / "metrics_comparison.png"
    )
    
    # 2. Анализ по вакансиям
    print("2. Анализ по вакансиям...")
    per_vacancy_df = analyze_per_vacancy_performance(results)
    per_vacancy_df.to_csv(output_dir / "per_vacancy_metrics.csv", index=False)
    
    plot_per_vacancy_performance(
        per_vacancy_df,
        output_path=output_dir / "per_vacancy_performance.png"
    )
    
    # 3. Сложность вакансий
    print("3. Анализ сложности вакансий...")
    difficulty = find_difficult_vacancies(per_vacancy_df)
    difficulty.to_csv(output_dir / "vacancy_difficulty.csv", index=False)
    
    print("\n📋 Самые сложные вакансии:")
    print(difficulty.head().to_string(index=False))
    
    # 4. Статистическая значимость (если >= 2 эксперимента)
    if len(results) >= 2:
        print("\n4. Статистический анализ...")
        
        # Сравниваем первый эксперимент с каждым последующим
        baseline = results[0]
        baseline_maps = [v['metrics']['map'] for v in baseline['detailed_results']]
        
        sig_tests = []
        for i, result in enumerate(results[1:], 1):
            result_maps = [v['metrics']['map'] for v in result['detailed_results']]
            test_result = statistical_significance_test(baseline_maps, result_maps)
            
            sig_tests.append({
                'comparison': f"{baseline['config']['name']} vs {result['config']['name']}",
                'p_value': test_result['p_value'],
                'significant': test_result['significant'],
                'mean_improvement': test_result['mean_improvement'],
                'cohens_d': test_result['cohens_d'],
                'effect_size': test_result['effect_size']
            })
        
        sig_df = pd.DataFrame(sig_tests)
        sig_df.to_csv(output_dir / "statistical_tests.csv", index=False)
        
        print("\n📊 Результаты статистических тестов:")
        print(sig_df.to_string(index=False))
    
    # 5. Итоговая таблица
    print("\n5. Итоговая сводка...")
    summary_rows = []
    for result in results:
        row = {
            'Experiment': result['config']['name'],
            'Description': result['config']['description']
        }
        metrics = result['metrics_summary']['mean']
        row.update({
            'MAP': f"{metrics['map']:.3f}",
            'P@5': f"{metrics['precision@5']:.3f}",
            'R@10': f"{metrics['recall@10']:.3f}",
            'MRR': f"{metrics['mrr']:.3f}"
        })
        summary_rows.append(row)
    
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(output_dir / "summary.csv", index=False)
    
    print("\n" + "="*70)
    print("📊 ИТОГОВАЯ СВОДКА")
    print("="*70)
    print(summary_df.to_string(index=False))
    
    print("\n" + "="*70)
    print(f"✅ Отчет сохранен в: {output_dir}")
    print("="*70 + "\n")
    
    return output_dir


# Примеры использования
def example_basic_analysis():
    """Пример базового анализа"""
    results_dir = Path("app/experiments/results")
    
    # Находим последние результаты
    json_files = sorted(results_dir.glob("*.json"), key=lambda x: x.stat().st_mtime)
    
    if len(json_files) < 2:
        print("⚠️  Нужно минимум 2 эксперимента для сравнения")
        return
    
    # Загружаем последние 2
    result1 = load_experiment_results(json_files[-2])
    result2 = load_experiment_results(json_files[-1])
    
    # Сравниваем
    comparison = compare_two_experiments(result1, result2)
    print("\n📊 Сравнение экспериментов:")
    print(comparison.to_string(index=False))
    
    # Статистический тест
    maps1 = [v['metrics']['map'] for v in result1['detailed_results']]
    maps2 = [v['metrics']['map'] for v in result2['detailed_results']]
    
    test = statistical_significance_test(maps1, maps2)
    print(f"\n📈 Статистическая значимость:")
    print(f"   p-value: {test['p_value']:.4f}")
    print(f"   Значимо: {'Да ✅' if test['significant'] else 'Нет ❌'}")
    print(f"   Effect size: {test['effect_size']}")


def example_full_analysis():
    """Пример полного анализа всех экспериментов"""
    results_dir = Path("app/experiments/results")
    
    # Загружаем все результаты
    all_results = []
    for json_file in results_dir.glob("*.json"):
        if json_file.stem.startswith("comparison"):
            continue
        all_results.append(load_experiment_results(json_file))
    
    if not all_results:
        print("⚠️  Результаты экспериментов не найдены")
        return
    
    print(f"Найдено {len(all_results)} экспериментов")
    
    # Генерируем полный отчет
    report_dir = generate_full_report(all_results)
    
    print(f"\n✅ Анализ завершен! Смотрите: {report_dir}")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "full":
        example_full_analysis()
    else:
        example_basic_analysis()
