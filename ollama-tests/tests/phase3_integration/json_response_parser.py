#!/usr/bin/env python3
"""
JSON Response Parser for LLM outputs
Handles various edge cases and formats that different models might return
"""

import json
from json import JSONDecodeError
import re
from typing import Dict, Any


def clean_json_comments(json_str: str) -> str:
    """
    Remover comentarios de JavaScript/JSON
    
    Args:
        json_str: String JSON que puede contener comentarios
        
    Returns:
        String JSON limpio sin comentarios
    """
    lines = json_str.split('\n')
    cleaned_lines = []
    
    for line in lines:
        # Buscar comentarios fuera de strings
        in_string = False
        escape_next = False
        comment_pos = -1
        
        for i, char in enumerate(line):
            if escape_next:
                escape_next = False
                continue
                
            if char == '\\':
                escape_next = True
                continue
                
            if char == '"' and not escape_next:
                in_string = not in_string
                continue
                
            if not in_string and char == '/' and i + 1 < len(line) and line[i + 1] == '/':
                comment_pos = i
                break
        
        if comment_pos >= 0:
            line = line[:comment_pos].rstrip()
        
        if line.strip():
            cleaned_lines.append(line)
    
    return '\n'.join(cleaned_lines)


def extract_multiple_json_objects(content: str) -> Dict[str, Any]:
    """
    Manejar múltiples objetos JSON separados (caso problemático común)
    Buscar múltiples objetos con "claims" separados por comas
    
    Args:
        content: Contenido que puede tener múltiples objetos JSON
        
    Returns:
        Dict con claims combinados o None si no se encuentra
    """
    multiple_claims_pattern = r'(\{"claims":\s*\[.*?\]\s*\})'
    matches = re.findall(multiple_claims_pattern, content, re.DOTALL)
    
    if matches and len(matches) > 1:
        # Combinar múltiples objetos claims en uno solo
        all_claims = []
        for match in matches:
            try:
                obj = json.loads(match.strip())
                if "claims" in obj and isinstance(obj["claims"], list):
                    all_claims.extend(obj["claims"])
            except JSONDecodeError:
                continue
        
        if all_claims:
            return {"claims": all_claims}
    
    return None


def extract_json_from_markdown(content: str) -> Dict[str, Any]:
    """
    Buscar JSON en bloques de código markdown
    
    Args:
        content: Contenido que puede tener JSON en markdown
        
    Returns:
        Dict parseado o None si no se encuentra
    """
    markdown_pattern = r'```(?:json)?\s*(\{.*?\})\s*```'
    match = re.search(markdown_pattern, content, re.DOTALL)
    
    if match:
        json_content = match.group(1).strip()
        try:
            return json.loads(json_content)
        except JSONDecodeError:
            try:
                cleaned_content = clean_json_comments(json_content)
                return json.loads(cleaned_content)
            except JSONDecodeError:
                pass
    
    return None


def extract_json_after_think_tags(content: str) -> Dict[str, Any]:
    """
    Buscar JSON después de tags <think>
    
    Args:
        content: Contenido que puede tener JSON después de </think>
        
    Returns:
        Dict parseado o None si no se encuentra
    """
    think_pattern = r'</think>\s*(\{.*\})'
    match = re.search(think_pattern, content, re.DOTALL)
    
    if match:
        json_content = match.group(1).strip()
        try:
            return json.loads(json_content)
        except JSONDecodeError:
            try:
                cleaned_content = clean_json_comments(json_content)
                return json.loads(cleaned_content)
            except JSONDecodeError:
                pass
    
    return None


def extract_json_by_pattern(content: str, pattern: str, field_name: str = None) -> Dict[str, Any]:
    """
    Buscar JSON usando un patrón específico
    
    Args:
        content: Contenido a buscar
        pattern: Patrón regex para buscar
        field_name: Campo específico a buscar (opcional)
        
    Returns:
        Dict parseado o None si no se encuentra
    """
    match = re.search(pattern, content, re.DOTALL)
    if match:
        json_content = match.group(1).strip()
        
        # Asegurar que termine correctamente si es necesario
        if field_name == "taxonomy" and not json_content.endswith('}'):
            json_content += '}'
            
        try:
            return json.loads(json_content)
        except JSONDecodeError:
            try:
                cleaned_content = clean_json_comments(json_content)
                return json.loads(cleaned_content)
            except JSONDecodeError:
                pass
    
    return None


def extract_json_after_text(content: str) -> Dict[str, Any]:
    """
    Buscar JSON después de texto explicativo
    
    Args:
        content: Contenido que puede tener JSON después de texto
        
    Returns:
        Dict parseado o None si no se encuentra
    """
    text_json_pattern = r'(?:output|result|JSON|taxonomy|claims):\s*(\{.*?\})'
    match = re.search(text_json_pattern, content, re.DOTALL | re.IGNORECASE)
    
    if match:
        json_content = match.group(1).strip()
        try:
            return json.loads(json_content)
        except JSONDecodeError:
            pass
    
    return None


def repair_malformed_json(content: str) -> Dict[str, Any]:
    """
    Último recurso - intentar arreglar JSON malformado
    
    Args:
        content: Contenido con posible JSON malformado
        
    Returns:
        Dict parseado o None si no se puede reparar
    """
    if "{" not in content or "}" not in content:
        return None
        
    # Intentar extraer desde el primer { hasta el último }
    start_idx = content.find("{")
    end_idx = content.rfind("}")
    
    if start_idx == -1 or end_idx == -1 or end_idx <= start_idx:
        return None
        
    json_content = content[start_idx:end_idx + 1]
    
    # Si contiene múltiples objetos separados, intentar repararlos
    if json_content.count('{"claims"') > 1:
        objects = []
        temp_content = json_content
        
        while '{"claims"' in temp_content:
            start = temp_content.find('{"claims"')
            if start == -1:
                break
            
            # Encontrar el final de este objeto
            brace_count = 0
            end = start
            for i, char in enumerate(temp_content[start:]):
                if char == '{':
                    brace_count += 1
                elif char == '}':
                    brace_count -= 1
                    if brace_count == 0:
                        end = start + i
                        break
            
            if end > start:
                obj_str = temp_content[start:end + 1]
                try:
                    obj = json.loads(obj_str)
                    if "claims" in obj:
                        objects.append(obj)
                except JSONDecodeError:
                    pass
                temp_content = temp_content[end + 1:]
            else:
                break
        
        # Combinar todos los claims encontrados
        if objects:
            all_claims = []
            for obj in objects:
                if "claims" in obj and isinstance(obj["claims"], list):
                    all_claims.extend(obj["claims"])
            if all_claims:
                return {"claims": all_claims}
    
    # Intentar parsear como está
    try:
        return json.loads(json_content)
    except JSONDecodeError:
        try:
            cleaned_content = clean_json_comments(json_content)
            return json.loads(cleaned_content)
        except JSONDecodeError:
            pass
    
    return None


def extract_json_from_response(content: str) -> Dict[str, Any]:
    """
    Extraer JSON válido de la respuesta del modelo, manejando diferentes formatos
    
    Esta función implementa una cadena de estrategias para extraer JSON de respuestas
    de modelos LLM que pueden venir en diferentes formatos.
    
    Args:
        content: El contenido de la respuesta del modelo
        
    Returns:
        Dict: El JSON parseado
        
    Raises:
        ValueError: Si no se puede extraer JSON válido del contenido
    """
    if not content or not isinstance(content, str):
        raise ValueError("Contenido vacío o inválido")
    
    # Limpiar contenido
    content = content.strip()
    
    # Estrategia 1: Intentar parsear directamente como JSON
    try:
        return json.loads(content)
    except JSONDecodeError:
        # Intentar limpiando comentarios
        try:
            cleaned_content = clean_json_comments(content)
            return json.loads(cleaned_content)
        except JSONDecodeError:
            pass
    
    # Estrategia 2: Buscar JSON en bloques de código markdown
    result = extract_json_from_markdown(content)
    if result:
        return result
    
    # Estrategia 3: Buscar JSON después de tags <think>
    result = extract_json_after_think_tags(content)
    if result:
        return result
    
    # Estrategia 4: Manejar múltiples objetos JSON separados
    result = extract_multiple_json_objects(content)
    if result:
        return result
    
    # Estrategia 5: Buscar cualquier estructura JSON que contenga "taxonomy"
    result = extract_json_by_pattern(
        content, 
        r'(\{"taxonomy".*?\}\s*\]?\s*\})', 
        "taxonomy"
    )
    if result:
        return result
    
    # Estrategia 6: Buscar JSON más agresivamente, incluso parcial
    result = extract_json_by_pattern(
        content,
        r'(\{[^{}]*"taxonomy"[^{}]*\[.*?\]\s*\})'
    )
    if result:
        return result
    
    # Estrategia 7: Buscar cualquier objeto JSON válido con "claims"
    result = extract_json_by_pattern(
        content,
        r'(\{"claims":\s*\[.*?\]\s*\})'
    )
    if result:
        return result
    
    # Estrategia 8: Buscar JSON después de texto explicativo
    result = extract_json_after_text(content)
    if result:
        return result
    
    # Estrategia 9: Último recurso - intentar arreglar JSON malformado
    result = repair_malformed_json(content)
    if result:
        return result
    
    # Si todo falla, lanzar excepción con información útil
    raise ValueError(f"No se pudo extraer JSON válido del contenido. Contenido: {content[:200]}...")


# Función de conveniencia para testing
def test_parser(content: str, expected_fields: list = None) -> bool:
    """
    Función de testing para verificar que el parser funciona correctamente
    
    Args:
        content: Contenido a parsear
        expected_fields: Lista de campos que se esperan en el resultado
        
    Returns:
        bool: True si el parsing fue exitoso
    """
    try:
        result = extract_json_from_response(content)
        
        if expected_fields:
            for field in expected_fields:
                if field not in result:
                    print(f"Campo esperado '{field}' no encontrado en resultado")
                    return False
        
        print(f"✅ Parsing exitoso: {list(result.keys())}")
        return True
        
    except Exception as e:
        print(f"❌ Error en parsing: {e}")
        return False


if __name__ == "__main__":
    # Ejemplos de testing
    test_cases = [
        '{"taxonomy": [{"topicName": "Test"}]}',
        '```json\n{"claims": [{"claim": "test"}]}\n```',
        'Here is the result: {"taxonomy": []}',
        '{"claims": []} {"claims": [{"claim": "duplicate"}]}',  # Caso problemático
    ]
    
    for i, test_case in enumerate(test_cases):
        print(f"\n--- Test Case {i+1} ---")
        test_parser(test_case) 