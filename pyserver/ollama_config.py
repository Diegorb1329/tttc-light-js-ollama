#!/usr/bin/env python3
"""
Configuración para el adaptador Ollama
"""

import os
from dotenv import load_dotenv

load_dotenv()

# Configuración de Ollama
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_DEFAULT_MODEL = os.getenv("OLLAMA_DEFAULT_MODEL", "qwen3:8b")

# Flag para usar Ollama en lugar de OpenAI
USE_OLLAMA = os.getenv("USE_OLLAMA", "true").lower() == "true"

# Mapeo de modelos OpenAI a Ollama (para compatibilidad)
MODEL_MAPPING = {
    "gpt-4o-mini": "qwen3:8b",
    "gpt-4-turbo-preview": "qwen3:8b", 
    "gpt-4o": "qwen3:8b",
    "gpt-3.5-turbo": "qwen3:8b",
    # Agregar más mapeos según sea necesario
}

def get_ollama_model(openai_model: str) -> str:
    """Convertir nombre de modelo OpenAI a modelo Ollama equivalente"""
    return MODEL_MAPPING.get(openai_model, OLLAMA_DEFAULT_MODEL)

def should_use_ollama() -> bool:
    """Determinar si se debe usar Ollama"""
    return USE_OLLAMA

print(f"🦙 Configuración Ollama: USE_OLLAMA={USE_OLLAMA}, BASE_URL={OLLAMA_BASE_URL}, DEFAULT_MODEL={OLLAMA_DEFAULT_MODEL}") 