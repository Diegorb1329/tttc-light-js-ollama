#!/usr/bin/env python3
"""
Servidor de debug mínimo para probar la integración Ollama
"""

from fastapi import FastAPI, Header
from pydantic import BaseModel
from typing import List
import json

# Importar adaptador Ollama
try:
    from ollama_openai_adapter import create_client
    import ollama_config
    print("✅ Imports exitosos")
except Exception as e:
    print(f"❌ Error en imports: {e}")
    raise

app = FastAPI()

class Comment(BaseModel):
    id: str
    text: str
    speaker: str

class LLMConfig(BaseModel):
    model_name: str
    system_prompt: str
    user_prompt: str

class CommentsLLMConfig(BaseModel):
    comments: List[Comment]
    llm: LLMConfig

def get_llm_client(api_key: str = None, model_name: str = None):
    """Obtener cliente LLM (OpenAI o Ollama) basado en configuración"""
    if ollama_config.should_use_ollama():
        # Usar Ollama con modelo mapeado
        ollama_model = ollama_config.get_ollama_model(model_name) if model_name else ollama_config.OLLAMA_DEFAULT_MODEL
        client = create_client(
            base_url=ollama_config.OLLAMA_BASE_URL,
            model=ollama_model
        )
        print(f"🦙 Usando Ollama: {ollama_model}")
        return client, ollama_model
    else:
        print(f"🤖 Usando OpenAI: {model_name}")
        return None, model_name

@app.get("/")
def read_root():
    return {"Hello": "World", "ollama_enabled": ollama_config.should_use_ollama()}

@app.post("/test_ollama")
def test_ollama(
    req: CommentsLLMConfig,
    x_openai_api_key: str = Header(..., alias="X-OpenAI-API-Key")
):
    """Test simple de Ollama"""
    try:
        print(f"📨 Received request: {len(req.comments)} comments")
        
        # Obtener cliente LLM
        client, actual_model = get_llm_client(x_openai_api_key, req.llm.model_name)
        
        # Crear prompt simple
        comments_text = "\n".join([c.text for c in req.comments])
        
        # Preparar argumentos para la llamada
        call_args = {
            "model": actual_model,
            "messages": [
                {"role": "system", "content": req.llm.system_prompt},
                {"role": "user", "content": req.llm.user_prompt + "\n" + comments_text},
            ],
            "temperature": 0.0,
            "think": False  # Asegurar que thinking esté deshabilitado
        }
        
        print(f"🚀 Calling Ollama with model: {actual_model}")
        response = client.chat.create(**call_args)
        
        print(f"✅ Response received, tokens: {response.usage.total_tokens}")
        
        return {
            "success": True,
            "model_used": actual_model,
            "response": response.choices[0].message.content,
            "usage": {
                "total_tokens": response.usage.total_tokens,
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens
            }
        }
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return {"success": False, "error": str(e)}

if __name__ == "__main__":
    import uvicorn
    print("🚀 Iniciando servidor debug...")
    uvicorn.run(app, host="0.0.0.0", port=8001) 