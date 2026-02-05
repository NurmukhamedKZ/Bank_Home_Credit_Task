#!/usr/bin/env python3
"""
Тестовый скрипт для LLM анализатора.

Использование:
    python test_llm_analyzer.py
"""

from app.services.llm_analyze import test_analyzer_with_sample_data

if __name__ == "__main__":
    test_analyzer_with_sample_data()
