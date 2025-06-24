#!/usr/bin/env python3
"""
Adaptador OpenAI-Compatible para Ollama
Simula la interfaz de OpenAI pero usa Ollama como backend
"""

import requests
import json
import time
import uuid
from typing import Dict, List, Optional, Union, Iterator
from dataclasses import dataclass
from datetime import datetime

@dataclass
class ChatMessage:
    role: str
    content: str

@dataclass
class ChatCompletionChoice:
    index: int
    message: ChatMessage
    finish_reason: str

@dataclass
class ChatCompletionUsage:
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    
    def model_dump(self):
        """Compatibilidad con Pydantic model_dump()"""
        return {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens
        }

@dataclass
class ChatCompletionResponse:
    id: str
    object: str
    created: int
    model: str
    choices: List[ChatCompletionChoice]
    usage: ChatCompletionUsage

class OllamaOpenAIAdapter:
    """
    Adaptador que simula la API de OpenAI usando Ollama como backend
    """
    
    def __init__(self, base_url: str = "http://localhost:11434", default_model: str = "qwen3:8b"):
        self.base_url = base_url.rstrip('/')
        self.default_model = default_model
        self.session = requests.Session()
    
    def _generate_id(self) -> str:
        """Generar ID único para la respuesta"""
        return f"chatcmpl-{uuid.uuid4().hex[:8]}"
    
    def _estimate_tokens(self, text: str) -> int:
        """Estimación simple de tokens (aproximadamente 4 caracteres por token)"""
        return max(1, len(text) // 4)
    
    def _ollama_to_openai_message(self, ollama_message: Dict) -> ChatMessage:
        """Convertir mensaje de Ollama a formato OpenAI"""
        return ChatMessage(
            role=ollama_message.get("role", "assistant"),
            content=ollama_message.get("content", "")
        )
    
    def _openai_to_ollama_messages(self, messages: List[Dict]) -> List[Dict]:
        """Convertir mensajes de OpenAI a formato Ollama"""
        ollama_messages = []
        for msg in messages:
            ollama_messages.append({
                "role": msg["role"],
                "content": msg["content"]
            })
        return ollama_messages
    
    def chat_completions_create(
        self,
        messages: List[Dict],
        model: Optional[str] = None,
        temperature: float = 0.0,  # Usar 0.0 por defecto para consistencia
        max_tokens: Optional[int] = None,
        stream: bool = False,
        response_format: Optional[Dict] = None,  # Agregar soporte para response_format
        think: bool = False,  # Por defecto, no thinking
        **kwargs
    ) -> Union[ChatCompletionResponse, Iterator[Dict]]:
        """
        Simular OpenAI chat.completions.create usando Ollama
        """
        model = model or self.default_model
        
        # Preparar payload para Ollama
        ollama_payload = {
            "model": model,
            "messages": self._openai_to_ollama_messages(messages),
            "stream": stream,
            "think": think,  # Usar el parámetro pasado
            "options": {
                "temperature": temperature,
            }
        }
        
        # Agregar max_tokens si está especificado
        if max_tokens:
            ollama_payload["options"]["num_predict"] = max_tokens
        
        try:
            if stream:
                return self._handle_streaming_response(ollama_payload, model)
            else:
                return self._handle_regular_response(ollama_payload, model, messages)
                
        except Exception as e:
            raise Exception(f"Error en Ollama adapter: {str(e)}")
    
    def _handle_regular_response(self, payload: Dict, model: str, original_messages: List[Dict]) -> ChatCompletionResponse:
        """Manejar respuesta no-streaming"""
        response = self.session.post(
            f"{self.base_url}/api/chat",
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=120  # Timeout más largo para modelos locales
        )
        
        if response.status_code != 200:
            raise Exception(f"Ollama API error: {response.status_code} - {response.text}")
        
        result = response.json()
        
        # Extraer mensaje de respuesta
        assistant_message = self._ollama_to_openai_message(result.get("message", {}))
        
        # Calcular tokens (estimación)
        prompt_text = " ".join([msg["content"] for msg in original_messages])
        prompt_tokens = self._estimate_tokens(prompt_text)
        completion_tokens = self._estimate_tokens(assistant_message.content)
        
        # Crear respuesta compatible con OpenAI
        return ChatCompletionResponse(
            id=self._generate_id(),
            object="chat.completion",
            created=int(time.time()),
            model=model,
            choices=[
                ChatCompletionChoice(
                    index=0,
                    message=assistant_message,
                    finish_reason="stop"
                )
            ],
            usage=ChatCompletionUsage(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=prompt_tokens + completion_tokens
            )
        )
    
    def _handle_streaming_response(self, payload: Dict, model: str) -> Iterator[Dict]:
        """Manejar respuesta streaming"""
        response = self.session.post(
            f"{self.base_url}/api/chat",
            json=payload,
            headers={"Content-Type": "application/json"},
            stream=True
        )
        
        if response.status_code != 200:
            raise Exception(f"Ollama API error: {response.status_code} - {response.text}")
        
        completion_id = self._generate_id()
        
        for line in response.iter_lines():
            if line:
                try:
                    data = json.loads(line.decode('utf-8'))
                    
                    # Crear chunk compatible con OpenAI
                    chunk = {
                        "id": completion_id,
                        "object": "chat.completion.chunk",
                        "created": int(time.time()),
                        "model": model,
                        "choices": []
                    }
                    
                    if 'message' in data and 'content' in data['message']:
                        content = data['message']['content']
                        chunk["choices"] = [{
                            "index": 0,
                            "delta": {
                                "role": "assistant",
                                "content": content
                            },
                            "finish_reason": None
                        }]
                    
                    # Chunk final
                    if data.get('done', False):
                        chunk["choices"] = [{
                            "index": 0,
                            "delta": {},
                            "finish_reason": "stop"
                        }]
                    
                    yield chunk
                    
                except json.JSONDecodeError:
                    continue


# Clase cliente compatible con OpenAI
class OpenAICompatibleClient:
    """
    Cliente que simula la interfaz de OpenAI
    """
    
    def __init__(self, base_url: str = "http://localhost:11434", default_model: str = "qwen3:8b", api_key: str = None):
        self.adapter = OllamaOpenAIAdapter(base_url, default_model)
        self.chat = ChatCompletions(self.adapter)
        self.api_key = api_key  # No se usa, pero mantenemos compatibilidad

class ChatCompletions:
    def __init__(self, adapter: OllamaOpenAIAdapter):
        self.adapter = adapter
        self.completions = self  # Para compatibilidad con OpenAI
    
    def create(self, **kwargs):
        return self.adapter.chat_completions_create(**kwargs)


# Función de conveniencia para crear cliente
def create_client(base_url: str = "http://localhost:11434", model: str = "qwen3:8b") -> OpenAICompatibleClient:
    """Crear cliente compatible con OpenAI"""
    return OpenAICompatibleClient(base_url, model)