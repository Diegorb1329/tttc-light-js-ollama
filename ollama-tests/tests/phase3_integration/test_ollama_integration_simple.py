#!/usr/bin/env python3
"""
Test simple para verificar la integración de Ollama con el pyserver
"""

import sys
import os
sys.path.append('pyserver')

# Test de la función extract_json_from_response
def test_json_extraction():
    """Test de la función de extracción de JSON"""
    from main import extract_json_from_response
    
    print("🧪 Testing función extract_json_from_response...")
    
    # Test 1: JSON puro
    json_pure = '{"taxonomy": [{"topicName": "Test", "topicShortDescription": "Test desc"}]}'
    try:
        result = extract_json_from_response(json_pure)
        print("✅ Test 1 (JSON puro): PASSED")
        print(f"   Resultado: {result}")
    except Exception as e:
        print(f"❌ Test 1 (JSON puro): FAILED - {e}")
        return False
    
    # Test 2: JSON con <think> tags
    json_with_think = '''<think>
    El usuario quiere clasificar comentarios sobre mascotas.
    Voy a crear una taxonomía simple.
    </think>
    {"taxonomy": [{"topicName": "Pets", "topicShortDescription": "Comments about pets", "subtopics": [{"subtopicName": "Cats", "subtopicShortDescription": "Positive views on cats"}]}]}'''
    
    try:
        result = extract_json_from_response(json_with_think)
        print("✅ Test 2 (JSON con <think>): PASSED")
        print(f"   Resultado: {result}")
        
        # Verificar que la estructura sea correcta
        if 'taxonomy' in result and len(result['taxonomy']) > 0:
            print("✅ Test 2 (estructura correcta): PASSED")
        else:
            print("❌ Test 2 (estructura incorrecta): FAILED")
            return False
            
    except Exception as e:
        print(f"❌ Test 2 (JSON con <think>): FAILED - {e}")
        return False
    
    # Test 3: JSON inválido
    invalid_json = "esto no es json válido"
    try:
        result = extract_json_from_response(invalid_json)
        print("❌ Test 3 (JSON inválido): FAILED - debería haber fallado")
        return False
    except Exception as e:
        print("✅ Test 3 (JSON inválido): PASSED - falló como esperado")
    
    return True

def test_model_mapping():
    """Test del mapeo de modelos"""
    import ollama_config
    from main import get_model_name
    
    print("\n🧪 Testing mapeo de modelos...")
    
    if ollama_config.should_use_ollama():
        # Test mapeo OpenAI -> Ollama
        test_model = "gpt-4o-mini"
        mapped_model = get_model_name(test_model)
        expected_model = ollama_config.MODEL_MAPPING.get(test_model, test_model)
        
        if mapped_model == expected_model:
            print(f"✅ Mapeo de modelo: {test_model} -> {mapped_model}")
        else:
            print(f"❌ Mapeo de modelo fallido: {test_model} -> {mapped_model} (esperado: {expected_model})")
            return False
    else:
        print("ℹ️  Ollama deshabilitado, saltando test de mapeo")
    
    return True

def test_client_creation():
    """Test de creación de cliente"""
    from main import create_llm_client
    import ollama_config
    
    print("\n🧪 Testing creación de cliente...")
    
    try:
        client = create_llm_client("dummy-api-key")
        if ollama_config.should_use_ollama():
            print("✅ Cliente Ollama creado exitosamente")
        else:
            print("✅ Cliente OpenAI creado exitosamente")
        return True
    except Exception as e:
        print(f"❌ Error creando cliente: {e}")
        return False

def main():
    """Ejecutar todos los tests"""
    print("🚀 Iniciando tests de integración Ollama + T3C Pyserver")
    print("=" * 60)
    
    tests = [
        test_json_extraction,
        test_model_mapping,
        test_client_creation
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"❌ Test falló con excepción: {e}")
    
    print("\n" + "=" * 60)
    print(f"📊 Resultados: {passed}/{total} tests pasaron")
    
    if passed == total:
        print("🎉 ¡TODOS LOS TESTS PASARON! La integración básica funciona correctamente")
        return True
    else:
        print("💥 ALGUNOS TESTS FALLARON - Revisar configuración")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)