"""
Пайплайн для экспериментов с разными моделями, промптами и параметрами поиска
Позволяет сравнивать разные конфигурации и находить оптимальные
"""

from pathlib import Path
from typing import Dict, List, Any, Optional
import json
import pandas as pd
from datetime import datetime
import sys

# Добавляем путь к родительской папке
sys.path.append(str(Path(__file__).parent.parent))

from service.parse_pdf import CVParser
from evaluate_search import CVSearchEvaluator


class ExperimentConfig:
    """Конфигурация эксперимента"""
    
    def __init__(
        self,
        name: str,
        description: str,
        dense_model: str = "voyage-4-large",
        dense_output_dim: int = 1024,
        tfidf_max_features: int = 10000,
        tfidf_ngram_range: tuple = (1, 2),
        tfidf_min_df: int = 1,
        system_prompt: Optional[str] = None,
        collection_name: str = "CVs_experiment"
    ):
        """
        Args:
            name: Название эксперимента
            description: Описание эксперимента
            dense_model: Модель для dense embeddings
            dense_output_dim: Размерность dense векторов
            tfidf_max_features: Максимум фичей для TF-IDF
            tfidf_ngram_range: N-граммы для TF-IDF
            tfidf_min_df: Минимальная частота документа для TF-IDF
            system_prompt: Кастомный system prompt для парсинга CV
            collection_name: Название коллекции в Qdrant
        """
        self.name = name
        self.description = description
        self.dense_model = dense_model
        self.dense_output_dim = dense_output_dim
        self.tfidf_max_features = tfidf_max_features
        self.tfidf_ngram_range = tfidf_ngram_range
        self.tfidf_min_df = tfidf_min_df
        self.system_prompt = system_prompt
        self.collection_name = collection_name
    
    def to_dict(self) -> Dict:
        """Конвертирует конфигурацию в словарь"""
        return {
            'name': self.name,
            'description': self.description,
            'dense_model': self.dense_model,
            'dense_output_dim': self.dense_output_dim,
            'tfidf_max_features': self.tfidf_max_features,
            'tfidf_ngram_range': self.tfidf_ngram_range,
            'tfidf_min_df': self.tfidf_min_df,
            'system_prompt': self.system_prompt,
            'collection_name': self.collection_name
        }
    
    @classmethod
    def from_dict(cls, config_dict: Dict) -> 'ExperimentConfig':
        """Создает конфигурацию из словаря"""
        return cls(**config_dict)
    
    @classmethod
    def from_json(cls, json_path: str | Path) -> 'ExperimentConfig':
        """Загружает конфигурацию из JSON файла"""
        with open(json_path, 'r', encoding='utf-8') as f:
            config_dict = json.load(f)
        return cls.from_dict(config_dict)


class ExperimentRunner:
    """Запускает эксперименты и сравнивает результаты"""
    
    def __init__(self, experiments_dir: str | Path = "app/experiments"):
        """
        Args:
            experiments_dir: Папка с экспериментами
        """
        self.experiments_dir = Path(experiments_dir)
        self.results_dir = self.experiments_dir / "results"
        self.configs_dir = self.experiments_dir / "configs"
        
        # Создаем папки
        self.results_dir.mkdir(exist_ok=True, parents=True)
        self.configs_dir.mkdir(exist_ok=True, parents=True)
        
        self.all_results = []
    
    def run_experiment(
        self,
        config: ExperimentConfig,
        cvs_to_process: Optional[List[Path]] = None,
        reuse_collection: bool = False
    ) -> Dict:
        """
        Запускает один эксперимент
        
        Args:
            config: Конфигурация эксперимента
            cvs_to_process: Список CV для обработки (если None, используются все)
            reuse_collection: Использовать существующую коллекцию (не обрабатывать CV заново)
            
        Returns:
            Результаты эксперимента
        """
        print(f"\n{'='*70}")
        print(f"🧪 ЭКСПЕРИМЕНТ: {config.name}")
        print(f"📝 {config.description}")
        print(f"{'='*70}\n")
        
        timestamp = datetime.now()
        
        # Инициализируем parser с конфигурацией
        print("⚙️  Инициализация CVParser...")
        parser = CVParser(
            collection_name=config.collection_name,
            dense_model_name=config.dense_model,
            dense_output_dim=config.dense_output_dim
        )
        
        # Применяем кастомный system prompt если указан
        if config.system_prompt:
            print("📝 Применение кастомного system prompt...")
            parser.system_prompt = config.system_prompt
            from langchain_core.prompts import ChatPromptTemplate
            parser.prompt = ChatPromptTemplate.from_messages([
                ("system", parser.system_prompt),
                ("user", "Resume:\n\n{text}")
            ])
            parser.chain = parser.prompt | parser.structured_llm
        
        # Настраиваем TF-IDF
        from sklearn.feature_extraction.text import TfidfVectorizer
        parser.sparse_model = TfidfVectorizer(
            max_features=config.tfidf_max_features,
            ngram_range=config.tfidf_ngram_range,
            min_df=config.tfidf_min_df,
            sublinear_tf=True,
            lowercase=True,
            stop_words='english'
        )
        parser._tfidf_fitted = False
        parser._tfidf_corpus = []
        
        # Обрабатываем CV если нужно
        if not reuse_collection:
            print("\n📄 Обработка CV...")
            
            if cvs_to_process is None:
                # Используем все CV из data/CVs
                cvs_folder = Path("data/CVs")
                cvs_to_process = list(cvs_folder.glob("*.pdf"))
            
            if not cvs_to_process:
                print("⚠️  CV для обработки не найдены!")
            else:
                for i, cv_path in enumerate(cvs_to_process, 1):
                    print(f"  [{i}/{len(cvs_to_process)}] {cv_path.name}...", end=' ')
                    try:
                        parser.process_cv(cv_path)
                        print("✅")
                    except Exception as e:
                        print(f"❌ {e}")
                
                # Переобучаем TF-IDF на всем корпусе
                print("\n🔄 Переобучение TF-IDF на всем корпусе...")
                parser.refit_tfidf()
        else:
            print("ℹ️  Используется существующая коллекция")
        
        # Оценка качества поиска
        print("\n📊 Оценка качества поиска...")
        evaluator = CVSearchEvaluator(parser)
        df, results = evaluator.evaluate_all(top_k=10)
        
        # Сохраняем результаты
        experiment_result = {
            'config': config.to_dict(),
            'timestamp': timestamp.isoformat(),
            'metrics_summary': {
                'mean': df[[col for col in df.columns if col not in ['vacancy', 'relevant_count']]].mean().to_dict(),
                'std': df[[col for col in df.columns if col not in ['vacancy', 'relevant_count']]].std().to_dict()
            },
            'per_vacancy_metrics': df.to_dict(orient='records'),
            'detailed_results': results
        }
        
        # Сохраняем в файл
        result_file = self.results_dir / f"{config.name}_{timestamp.strftime('%Y%m%d_%H%M%S')}.json"
        
        # Конвертируем sets в lists для JSON
        serializable_result = self._make_serializable(experiment_result)
        
        with open(result_file, 'w', encoding='utf-8') as f:
            json.dump(serializable_result, f, ensure_ascii=False, indent=2)
        
        print(f"\n💾 Результаты сохранены: {result_file}")
        
        self.all_results.append(experiment_result)
        
        return experiment_result
    
    def _make_serializable(self, obj):
        """Рекурсивно конвертирует объект в JSON-сериализуемый формат"""
        if isinstance(obj, dict):
            return {k: self._make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_serializable(item) for item in obj]
        elif isinstance(obj, set):
            return list(obj)
        elif isinstance(obj, tuple):
            return list(obj)
        else:
            return obj
    
    def run_multiple_experiments(
        self,
        configs: List[ExperimentConfig],
        cvs_to_process: Optional[List[Path]] = None
    ) -> pd.DataFrame:
        """
        Запускает несколько экспериментов и сравнивает результаты
        
        Args:
            configs: Список конфигураций для тестирования
            cvs_to_process: CV для обработки
            
        Returns:
            DataFrame со сравнением метрик
        """
        print(f"\n{'='*70}")
        print(f"🔬 ЗАПУСК {len(configs)} ЭКСПЕРИМЕНТОВ")
        print(f"{'='*70}\n")
        
        comparison_rows = []
        
        for i, config in enumerate(configs, 1):
            print(f"\n[{i}/{len(configs)}]")
            
            try:
                result = self.run_experiment(
                    config,
                    cvs_to_process=cvs_to_process,
                    reuse_collection=(i > 1)  # Первый раз обрабатываем, потом используем
                )
                
                # Добавляем в сравнение
                row = {
                    'experiment': config.name,
                    'description': config.description
                }
                row.update(result['metrics_summary']['mean'])
                comparison_rows.append(row)
                
            except Exception as e:
                print(f"❌ Ошибка в эксперименте {config.name}: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        # Создаем сравнительную таблицу
        comparison_df = pd.DataFrame(comparison_rows)
        
        # Сохраняем сравнение
        comparison_file = self.results_dir / f"comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        comparison_df.to_csv(comparison_file, index=False)
        
        print(f"\n{'='*70}")
        print("📊 СРАВНЕНИЕ ЭКСПЕРИМЕНТОВ")
        print(f"{'='*70}\n")
        print(comparison_df.to_string(index=False))
        
        print(f"\n💾 Сравнение сохранено: {comparison_file}")
        
        return comparison_df
    
    def create_default_configs(self) -> List[ExperimentConfig]:
        """Создает набор стандартных конфигураций для экспериментов"""
        configs = [
            # Базовая конфигурация
            ExperimentConfig(
                name="baseline",
                description="Базовая конфигурация (Voyage-4-large, TF-IDF unigrams+bigrams)",
                dense_model="voyage-4-large",
                dense_output_dim=1024,
                tfidf_max_features=10000,
                tfidf_ngram_range=(1, 2),
                collection_name="CVs_baseline"
            ),
            
            # Больше n-грамм
            ExperimentConfig(
                name="trigrams",
                description="TF-IDF с tri-grams для лучшего захвата фраз",
                dense_model="voyage-4-large",
                dense_output_dim=1024,
                tfidf_max_features=15000,
                tfidf_ngram_range=(1, 3),
                collection_name="CVs_trigrams"
            ),
            
            # Меньше фичей (быстрее)
            ExperimentConfig(
                name="lightweight",
                description="Облегченная версия - меньше фичей TF-IDF",
                dense_model="voyage-4-large",
                dense_output_dim=1024,
                tfidf_max_features=5000,
                tfidf_ngram_range=(1, 2),
                collection_name="CVs_lightweight"
            ),
            
            # Кастомный промпт
            ExperimentConfig(
                name="detailed_prompt",
                description="Детальный промпт с акцентом на технические навыки",
                dense_model="voyage-4-large",
                dense_output_dim=1024,
                tfidf_max_features=10000,
                tfidf_ngram_range=(1, 2),
                system_prompt="""
You are an expert technical recruiter and CV parser specializing in IT positions.
Your task is to extract structured data from the provided resume text.

CRITICAL FOCUS AREAS:
1. Technical Skills: Extract ALL programming languages, frameworks, tools, and technologies
2. Work Experience: Be precise with dates, calculate total months accurately
3. Projects: Capture specific technologies and achievements
4. Education: Include degrees, institutions, and graduation years

EXTRACTION RULES:
- For 'skills', extract both hard skills (technical) and important soft skills
- For 'work_history', split distinct roles even if same company
- In 'total_experience_months', sum ALL work durations carefully
- For technologies, be specific (e.g., "Python 3.10", "FastAPI", not just "Python")
- Extract version numbers when mentioned

QUALITY STANDARDS:
- Prefer explicit information over assumptions
- If field is missing, leave as None or empty list
- Maintain exact terminology from CV (don't paraphrase technical terms)
""",
                collection_name="CVs_detailed_prompt"
            )
        ]
        
        # Сохраняем конфигурации
        for config in configs:
            config_file = self.configs_dir / f"{config.name}.json"
            with open(config_file, 'w', encoding='utf-8') as f:
                json.dump(config.to_dict(), f, ensure_ascii=False, indent=2)
            print(f"💾 Конфигурация сохранена: {config_file}")
        
        return configs


def main():
    """Основная функция для запуска экспериментов"""
    
    runner = ExperimentRunner()
    
    print("🔧 Создание стандартных конфигураций...")
    configs = runner.create_default_configs()
    
    print(f"\n📋 Создано {len(configs)} конфигураций:")
    for config in configs:
        print(f"  • {config.name}: {config.description}")
    
    # Выбираем какие эксперименты запустить
    print("\n" + "="*70)
    print("Выберите эксперименты для запуска:")
    print("  1. Baseline")
    print("  2. Trigrams")
    print("  3. Lightweight")
    print("  4. Detailed Prompt")
    print("  5. Все эксперименты")
    print("="*70)
    
    choice = input("\nВведите номер (или 'q' для выхода): ").strip()
    
    if choice == 'q':
        print("Выход.")
        return
    
    if choice == '5':
        selected_configs = configs
    else:
        try:
            idx = int(choice) - 1
            if 0 <= idx < len(configs):
                selected_configs = [configs[idx]]
            else:
                print("Неверный выбор!")
                return
        except ValueError:
            print("Неверный ввод!")
            return
    
    # Запускаем эксперименты
    comparison_df = runner.run_multiple_experiments(selected_configs)
    
    print("\n✅ Все эксперименты завершены!")
    
    return comparison_df


if __name__ == "__main__":
    main()
