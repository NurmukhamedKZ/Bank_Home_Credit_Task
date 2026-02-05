#!/usr/bin/env python3
"""
Быстрый запуск экспериментов с разными конфигурациями
"""

import sys
from pathlib import Path

# Добавляем путь к модулям
sys.path.append(str(Path(__file__).parent))

from experiments.experiment_runner import main as experiments_main

if __name__ == "__main__":
    print("""
╔═══════════════════════════════════════════════════════════════╗
║              CV SEARCH EXPERIMENTS RUNNER                     ║
║  Test different models, prompts, and search configurations    ║
╚═══════════════════════════════════════════════════════════════╝
    """)
    
    # Запускаем эксперименты
    experiments_main()
