#!/usr/bin/env python3
"""
Test básico de conectividad con Ollama
Fase 1: Verificar que podemos comunicarnos con el servidor Ollama
"""

import requests
import json
import time

# Configuración
OLLAMA_BASE_URL = "http://localhost:11434"
MODEL_NAME = "qwen3:8b"

def test_ollama_connection():
    """Probar conexión básica con Ollama"""
    try:
        response = requests.get(f"{OLLAMA_BASE_URL}/api/tags")
        if response.status_code == 200:
            models = response.json()
            print("✅ Conexión exitosa con Ollama")
            print(f"📋 Modelos disponibles: {len(models['models'])}")
            for model in models['models']:
                print(f"   - {model['name']} ({model['size']} bytes)")
            return True
        else:
            print(f"❌ Error de conexión: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Error de conexión: {e}")
        return False

def test_simple_chat():
    """Probar una conversación simple con el modelo"""
    try:
        print(f"\n🧠 Probando conversación con {MODEL_NAME}...")
        
        payload = {
            "model": MODEL_NAME,
            "messages": [
                {
                    "role": "user", 
                    "content": "Hello! Can you tell me what you are in one sentence?"
                }
            ],
            "stream": False
        }
        
        print("📤 Enviando mensaje...")
        start_time = time.time()
        
        response = requests.post(
            f"{OLLAMA_BASE_URL}/api/chat",
            json=payload,
            headers={"Content-Type": "application/json"}
        )
        
        end_time = time.time()
        response_time = end_time - start_time
        
        if response.status_code == 200:
            result = response.json()
            assistant_message = result['message']['content']
            print(f"✅ Respuesta recibida en {response_time:.2f} segundos")
            print(f"🤖 Respuesta: {assistant_message}")
            return True
        else:
            print(f"❌ Error en chat: {response.status_code}")
            print(f"📄 Respuesta: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Error en chat: {e}")
        return False

def test_streaming_chat():
    """Probar chat con streaming (como lo hace OpenAI)"""
    try:
        print(f"\n🌊 Probando streaming con {MODEL_NAME}...")
        
        payload = {
            "model": MODEL_NAME,
            "messages": [
                {
                    "role": "user", 
                    "content": "Count from 1 to 5, one number per line."
                }
            ],
            "stream": True
        }
        
        print("📤 Enviando mensaje con streaming...")
        
        response = requests.post(
            f"{OLLAMA_BASE_URL}/api/chat",
            json=payload,
            headers={"Content-Type": "application/json"},
            stream=True
        )
        
        if response.status_code == 200:
            print("🤖 Respuesta streaming:")
            full_response = ""
            for line in response.iter_lines():
                if line:
                    try:
                        data = json.loads(line.decode('utf-8'))
                        if 'message' in data and 'content' in data['message']:
                            content = data['message']['content']
                            print(content, end='', flush=True)
                            full_response += content
                        if data.get('done', False):
                            break
                    except json.JSONDecodeError:
                        continue
            print(f"\n✅ Streaming completado")
            print(f"📝 Respuesta completa: {full_response}")
            return True
        else:
            print(f"❌ Error en streaming: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ Error en streaming: {e}")
        return False

if __name__ == "__main__":
    print("🚀 INICIANDO PRUEBAS DE OLLAMA - FASE 1")
    print("=" * 50)
    
    # Test 1: Conexión básica
    print("\n1️⃣ Test de conexión básica")
    if not test_ollama_connection():
        print("❌ Falló la conexión básica. Saliendo...")
        exit(1)
    
    # Test 2: Chat simple
    print("\n2️⃣ Test de chat simple")
    if not test_simple_chat():
        print("❌ Falló el chat simple")
    
    # Test 3: Streaming
    print("\n3️⃣ Test de streaming")
    if not test_streaming_chat():
        print("❌ Falló el streaming")
    
    print("\n" + "=" * 50)
    print("🎉 PRUEBAS COMPLETADAS - FASE 1") 