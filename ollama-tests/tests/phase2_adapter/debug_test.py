#!/usr/bin/env python3
"""
Test de debug para el adaptador Ollama
"""

import sys
import os
sys.path.append('pyserver')

try:
    print("1. Importando ollama_config...")
    import ollama_config
    print(f"   ✅ USE_OLLAMA: {ollama_config.USE_OLLAMA}")
    print(f"   ✅ BASE_URL: {ollama_config.OLLAMA_BASE_URL}")
    print(f"   ✅ DEFAULT_MODEL: {ollama_config.OLLAMA_DEFAULT_MODEL}")
    
    print("\n2. Importando adaptador...")
    from ollama_openai_adapter import create_client
    print("   ✅ Adaptador importado correctamente")
    
    print("\n3. Creando cliente...")
    client = create_client()
    print("   ✅ Cliente creado correctamente")
    
    print("\n4. Verificando health check...")
    health = client.adapter.health_check()
    print(f"   ✅ Health: {health}")
    
    if health["ollama_available"]:
        print("\n5. Probando chat simple...")
        response = client.chat.create(
            model="qwen3:8b",
            messages=[{"role": "user", "content": "Hola, ¿cómo estás?"}],
            think=False
        )
        print(f"   ✅ Respuesta: {response.choices[0].message.content[:100]}...")
        print(f"   ✅ Tokens: {response.usage.total_tokens}")
    
    print("\n🎉 ¡Todos los tests pasaron! El adaptador funciona correctamente.")
    
except Exception as e:
    print(f"\n❌ Error: {str(e)}")
    import traceback
    traceback.print_exc() 