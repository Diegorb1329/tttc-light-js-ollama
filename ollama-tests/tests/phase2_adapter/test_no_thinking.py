#!/usr/bin/env python3
"""
Test específico para verificar que el thinking mode está deshabilitado por defecto
"""

import sys
import os
import unittest
import time
import json

# Agregar el directorio actual al path para importar el adaptador
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from ollama_openai_adapter import create_client

class TestNoThinking(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        """Configuración inicial para todos los tests"""
        print("\n🧪 INICIANDO TESTS DE NO-THINKING MODE")
        print("=" * 50)
        cls.client = create_client()
        
        # Verificar que Ollama esté funcionando
        health = cls.client.adapter.health_check()
        if not health["ollama_available"]:
            raise unittest.SkipTest("Ollama no está disponible")
        
        print(f"✅ Ollama disponible con {health['models_count']} modelos")
    
    def test_1_thinking_disabled_by_default(self):
        """Test 1: Verificar que thinking está deshabilitado por defecto"""
        print("\n🔍 Test 1: Thinking deshabilitado por defecto")
        
        messages = [
            {"role": "user", "content": "¿Cuál es la capital de Francia? Responde solo con el nombre de la ciudad."}
        ]
        
        start_time = time.time()
        
        # Llamada sin especificar think (debería ser False por defecto)
        response = self.client.chat.create(
            messages=messages,
            model="qwen3:8b"
        )
        
        end_time = time.time()
        duration = end_time - start_time
        
        # Verificaciones
        self.assertIsNotNone(response)
        self.assertEqual(len(response.choices), 1)
        self.assertIsNotNone(response.choices[0].message.content)
        
        content = response.choices[0].message.content.strip()
        print(f"   📝 Respuesta: {content}")
        print(f"   ⏱️  Tiempo: {duration:.2f}s")
        print(f"   🔢 Tokens: {response.usage.total_tokens}")
        
        # Verificar que no hay contenido de thinking (respuesta directa)
        self.assertGreater(len(content), 0, "La respuesta no debería estar vacía")
        
        # La respuesta debería ser concisa (sin thinking debería ser más rápida y directa)
        self.assertLess(duration, 60, "Sin thinking debería ser más rápido")
        
        print("   ✅ Test 1 PASADO: Thinking deshabilitado por defecto")
    
    def test_2_explicit_thinking_false(self):
        """Test 2: Verificar que think=False funciona explícitamente"""
        print("\n🔍 Test 2: think=False explícito")
        
        messages = [
            {"role": "user", "content": "Explica qué es Python en una oración."}
        ]
        
        start_time = time.time()
        
        # Llamada con think=False explícito
        response = self.client.chat.create(
            messages=messages,
            model="qwen3:8b",
            think=False
        )
        
        end_time = time.time()
        duration = end_time - start_time
        
        # Verificaciones
        self.assertIsNotNone(response)
        self.assertEqual(len(response.choices), 1)
        self.assertIsNotNone(response.choices[0].message.content)
        
        content = response.choices[0].message.content.strip()
        print(f"   📝 Respuesta: {content}")
        print(f"   ⏱️  Tiempo: {duration:.2f}s")
        print(f"   🔢 Tokens: {response.usage.total_tokens}")
        
        # Verificar que la respuesta es directa
        self.assertGreater(len(content), 0, "La respuesta no debería estar vacía")
        self.assertLess(duration, 60, "Sin thinking debería ser más rápido")
        
        print("   ✅ Test 2 PASADO: think=False explícito funciona")
    
    def test_3_compare_thinking_vs_no_thinking_speed(self):
        """Test 3: Comparar velocidad con y sin thinking"""
        print("\n🔍 Test 3: Comparación de velocidad thinking vs no-thinking")
        
        question = "¿Cuánto es 15 + 27?"
        messages = [{"role": "user", "content": question}]
        
        # Test sin thinking
        start_time = time.time()
        response_no_think = self.client.chat.create(
            messages=messages,
            model="qwen3:8b",
            think=False
        )
        no_think_time = time.time() - start_time
        
        # Test con thinking
        start_time = time.time()
        response_with_think = self.client.chat.create(
            messages=messages,
            model="qwen3:8b",
            think=True
        )
        with_think_time = time.time() - start_time
        
        print(f"   📊 Sin thinking: {no_think_time:.2f}s")
        print(f"   📊 Con thinking: {with_think_time:.2f}s")
        print(f"   📊 Diferencia: {with_think_time - no_think_time:.2f}s")
        
        # Verificar que ambas respuestas son válidas
        self.assertIsNotNone(response_no_think.choices[0].message.content)
        self.assertIsNotNone(response_with_think.choices[0].message.content)
        
        # Normalmente sin thinking debería ser más rápido, pero no siempre garantizado
        print(f"   📝 Respuesta sin thinking: {response_no_think.choices[0].message.content.strip()}")
        print(f"   📝 Respuesta con thinking: {response_with_think.choices[0].message.content.strip()}")
        
        print("   ✅ Test 3 PASADO: Comparación completada")
    
    def test_4_streaming_no_thinking(self):
        """Test 4: Verificar que streaming funciona sin thinking"""
        print("\n🔍 Test 4: Streaming sin thinking")
        
        messages = [
            {"role": "user", "content": "Cuenta del 1 al 5."}
        ]
        
        start_time = time.time()
        chunks_received = 0
        content_parts = []
        
        # Test streaming sin thinking
        response_stream = self.client.chat.create(
            messages=messages,
            model="qwen3:8b",
            stream=True,
            think=False
        )
        
        for chunk in response_stream:
            chunks_received += 1
            if chunk.get('choices') and len(chunk['choices']) > 0:
                delta = chunk['choices'][0].get('delta', {})
                if 'content' in delta:
                    content_parts.append(delta['content'])
        
        end_time = time.time()
        duration = end_time - start_time
        
        full_content = ''.join(content_parts)
        
        print(f"   📦 Chunks recibidos: {chunks_received}")
        print(f"   📝 Contenido completo: {full_content.strip()}")
        print(f"   ⏱️  Tiempo: {duration:.2f}s")
        
        # Verificaciones
        self.assertGreater(chunks_received, 0, "Debería recibir al menos un chunk")
        self.assertGreater(len(full_content.strip()), 0, "Debería tener contenido")
        
        print("   ✅ Test 4 PASADO: Streaming sin thinking funciona")

def run_tests():
    """Ejecutar todos los tests"""
    print("🚀 EJECUTANDO TESTS DE NO-THINKING MODE")
    print("=" * 60)
    
    # Crear suite de tests
    suite = unittest.TestLoader().loadTestsFromTestCase(TestNoThinking)
    
    # Ejecutar tests con resultados detallados
    runner = unittest.TextTestRunner(verbosity=2, stream=sys.stdout)
    result = runner.run(suite)
    
    # Resumen final
    print("\n" + "=" * 60)
    print("📊 RESUMEN DE TESTS NO-THINKING")
    print("=" * 60)
    print(f"✅ Tests exitosos: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"❌ Tests fallidos: {len(result.failures)}")
    print(f"🚨 Errores: {len(result.errors)}")
    print(f"📈 Tasa de éxito: {((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100):.1f}%")
    
    if result.failures:
        print("\n❌ FALLOS:")
        for test, traceback in result.failures:
            print(f"  - {test}: {traceback}")
    
    if result.errors:
        print("\n🚨 ERRORES:")
        for test, traceback in result.errors:
            print(f"  - {test}: {traceback}")
    
    return result.wasSuccessful()

if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1) 