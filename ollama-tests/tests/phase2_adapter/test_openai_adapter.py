#!/usr/bin/env python3
"""
Test para el Adaptador OpenAI-Compatible
Verifica que el adaptador simule correctamente la interfaz de OpenAI
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ollama_openai_adapter import create_client, OllamaOpenAIAdapter
import time

def test_client_creation():
    """Test: Crear cliente compatible con OpenAI"""
    try:
        client = create_client()
        print("✅ Cliente creado exitosamente")
        print(f"   - Adaptador: {type(client.adapter).__name__}")
        print(f"   - Chat: {type(client.chat).__name__}")
        print(f"   - Models: {type(client.models).__name__}")
        return True
    except Exception as e:
        print(f"❌ Error creando cliente: {e}")
        return False

def test_health_check():
    """Test: Verificar salud del adaptador"""
    try:
        adapter = OllamaOpenAIAdapter()
        health = adapter.health_check()
        
        print(f"✅ Health check completado")
        print(f"   - Status: {health['status']}")
        print(f"   - Ollama disponible: {health['ollama_available']}")
        print(f"   - Modelos: {health.get('models_count', 0)}")
        
        return health['status'] == 'healthy'
    except Exception as e:
        print(f"❌ Error en health check: {e}")
        return False

def test_list_models():
    """Test: Listar modelos en formato OpenAI"""
    try:
        client = create_client()
        models = client.models.list()
        
        print("✅ Listado de modelos exitoso")
        print(f"   - Formato: {models['object']}")
        print(f"   - Cantidad: {len(models['data'])}")
        
        for model in models['data']:
            print(f"   - {model['id']} (owned_by: {model['owned_by']})")
        
        # Verificar estructura OpenAI
        assert models['object'] == 'list'
        assert 'data' in models
        assert len(models['data']) > 0
        
        for model in models['data']:
            assert 'id' in model
            assert 'object' in model
            assert 'created' in model
            assert 'owned_by' in model
            
        return True
    except Exception as e:
        print(f"❌ Error listando modelos: {e}")
        return False

def test_chat_completion_simple():
    """Test: Chat completion simple (no streaming)"""
    try:
        client = create_client()
        
        print("🧠 Probando chat completion simple...")
        start_time = time.time()
        
        response = client.chat.create(
            messages=[
                {"role": "user", "content": "Say 'Hello World' and nothing else."}
            ],
            model="qwen3:8b",
            stream=False
        )
        
        end_time = time.time()
        duration = end_time - start_time
        
        print(f"✅ Chat completion completado en {duration:.2f}s")
        print(f"   - ID: {response.id}")
        print(f"   - Object: {response.object}")
        print(f"   - Model: {response.model}")
        print(f"   - Choices: {len(response.choices)}")
        print(f"   - Mensaje: {response.choices[0].message.content[:100]}...")
        print(f"   - Tokens: {response.usage.total_tokens} total")
        
        # Verificar estructura OpenAI
        assert response.object == "chat.completion"
        assert len(response.choices) > 0
        assert response.choices[0].message.role == "assistant"
        assert len(response.choices[0].message.content) > 0
        assert response.usage.total_tokens > 0
        
        return True
    except Exception as e:
        print(f"❌ Error en chat completion: {e}")
        return False

def test_chat_completion_streaming():
    """Test: Chat completion con streaming"""
    try:
        client = create_client()
        
        print("🌊 Probando chat completion streaming...")
        start_time = time.time()
        
        stream = client.chat.create(
            messages=[
                {"role": "user", "content": "Count from 1 to 3, one number per line."}
            ],
            model="qwen3:8b",
            stream=True
        )
        
        chunks_received = 0
        full_content = ""
        
        for chunk in stream:
            chunks_received += 1
            
            # Verificar estructura del chunk
            assert 'id' in chunk
            assert 'object' in chunk
            assert chunk['object'] == 'chat.completion.chunk'
            assert 'choices' in chunk
            
            if chunk['choices'] and chunk['choices'][0].get('delta', {}).get('content'):
                content = chunk['choices'][0]['delta']['content']
                full_content += content
                print(content, end='', flush=True)
            
            # Verificar chunk final
            if chunk['choices'] and chunk['choices'][0].get('finish_reason') == 'stop':
                break
        
        end_time = time.time()
        duration = end_time - start_time
        
        print(f"\n✅ Streaming completado en {duration:.2f}s")
        print(f"   - Chunks recibidos: {chunks_received}")
        print(f"   - Contenido total: {len(full_content)} caracteres")
        print(f"   - Respuesta: {full_content[:100]}...")
        
        assert chunks_received > 0
        assert len(full_content) > 0
        
        return True
    except Exception as e:
        print(f"❌ Error en streaming: {e}")
        return False

def test_openai_compatibility():
    """Test: Verificar compatibilidad con sintaxis OpenAI"""
    try:
        client = create_client()
        
        print("🔄 Probando compatibilidad con sintaxis OpenAI...")
        
        # Sintaxis idéntica a OpenAI
        response = client.chat.create(
            model="qwen3:8b",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "What is 2+2?"}
            ],
            temperature=0.7,
            max_tokens=50
        )
        
        print("✅ Sintaxis OpenAI compatible")
        print(f"   - Respuesta: {response.choices[0].message.content}")
        print(f"   - Tokens usados: {response.usage.total_tokens}")
        
        # Verificar que la respuesta tenga sentido
        content = response.choices[0].message.content.lower()
        assert any(word in content for word in ['4', 'four', 'cuatro']), "La respuesta debería contener la respuesta a 2+2"
        
        return True
    except Exception as e:
        print(f"❌ Error en compatibilidad OpenAI: {e}")
        return False

def run_all_tests():
    """Ejecutar todos los tests de la fase 2"""
    print("🚀 INICIANDO TESTS DE FASE 2: ADAPTADOR OPENAI")
    print("=" * 60)
    
    tests = [
        ("Creación de cliente", test_client_creation),
        ("Health check", test_health_check),
        ("Listado de modelos", test_list_models),
        ("Chat completion simple", test_chat_completion_simple),
        ("Chat completion streaming", test_chat_completion_streaming),
        ("Compatibilidad OpenAI", test_openai_compatibility)
    ]
    
    passed = 0
    failed = 0
    
    for test_name, test_func in tests:
        print(f"\n🧪 {test_name}:")
        try:
            if test_func():
                passed += 1
                print(f"✅ {test_name}: PASSED")
            else:
                failed += 1
                print(f"❌ {test_name}: FAILED")
        except Exception as e:
            failed += 1
            print(f"❌ {test_name}: ERROR - {e}")
    
    print("\n" + "=" * 60)
    print(f"📊 RESULTADOS FASE 2:")
    print(f"✅ Tests exitosos: {passed}")
    print(f"❌ Tests fallidos: {failed}")
    print(f"📈 Porcentaje de éxito: {(passed/(passed+failed)*100):.1f}%")
    
    return failed == 0

if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1) 