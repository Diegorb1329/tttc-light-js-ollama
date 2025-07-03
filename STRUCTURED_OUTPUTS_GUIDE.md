# Structured Outputs & OpenRouter Integration Guide

## Overview

Complete implementation of **OpenRouter support** with **Structured Outputs** to eliminate JSON parsing errors and provide flexible LLM provider switching.

## What Was Implemented

### 🔄 **OpenRouter Integration**
- **Dual provider support**: OpenAI + OpenRouter with automatic detection
- **Environment-based switching**: Simple URL change enables OpenRouter
- **API key management**: Automatic selection between OpenAI/OpenRouter keys

### 🎯 **Structured Outputs** 
- **JSON schema validation**: Eliminates markdown-wrapped responses (````json...````)
- **4 response schemas**: topic_tree, claims, dedup, crux
- **Graceful fallback**: Auto-fallback to `json_object` for unsupported models

### 📁 **File Size Limits**
- **Configurable limits**: Frontend (10MB) and backend (50MB) via environment variables
- **Consistent validation**: Unified error messages across frontend/backend

## File Changes Summary

### New Files
```
pyserver/schemas.py           # JSON schemas for structured outputs
express-server/src/config.ts  # Default LLM model configuration  
.example-env                  # Environment variables template
```

### Key Updates
```
pyserver/config.py           # OpenRouter + structured outputs config
pyserver/main.py            # 3 LLM calls updated with schemas
pyserver/utils.py           # OpenRouter client detection logic
express-server/src/utils.ts # API key selection logic
next-client/next.config.js  # File upload limits
```

## Quick Setup

### 1. OpenAI (Default)
```bash
OPENAI_API_KEY=your_openai_key
```

### 2. OpenRouter 
```bash
OPENAI_API_BASE_URL=https://openrouter.ai/api/v1
OPENROUTER_API_KEY=your_openrouter_key
OPENROUTER_MODELS=google/gemini-2.0-flash-exp:free
```

### 3. Optional Configuration
```bash
# Structured outputs (default: true)
USE_STRUCTURED_OUTPUTS=true

# File size limits  
NEXT_PUBLIC_MAX_FILE_SIZE_MB=10  # Frontend limit
MAX_REQUEST_SIZE_MB=50           # Backend limit
```

## Core Benefits

1. **🔄 Provider Flexibility**: Switch between OpenAI/OpenRouter instantly
2. **🛡️ Robust JSON Parsing**: No more markdown-wrapped response failures  
3. **📈 Scalable File Handling**: Configurable size limits (10MB→50MB)
4. **🔧 Zero Breaking Changes**: Full backward compatibility
5. **💰 Cost Optimization**: Use free OpenRouter models for testing

## Architecture

```
Frontend (Next.js)
├── File validation (configurable limits)
├── Form submission with provider detection
└── Error handling improvements

Express Server (Node.js)  
├── Automatic API key selection
├── Provider-agnostic pipeline calls
└── Configurable request size limits

PyServer (FastAPI/Python)
├── Structured output schemas
├── OpenRouter client detection  
└── Graceful fallback handling
```

## Testing

```bash
# Test structured outputs
cd pyserver && python -c "from schemas import get_structured_response_format; print('✅ Ready')"

# Test OpenRouter (if configured)
curl -X POST localhost:8000/topic_tree -H "Content-Type: application/json" -d '{...}'
```

## Production Ready

- ✅ **Backward compatible** with existing OpenAI setups
- ✅ **Environment-driven** configuration (no code changes needed)
- ✅ **Automatic fallbacks** for unsupported features
- ✅ **Comprehensive error handling** across all layers
- ✅ **Configurable limits** for different deployment environments

The implementation provides a **robust, flexible foundation** for LLM provider management with enhanced reliability. 