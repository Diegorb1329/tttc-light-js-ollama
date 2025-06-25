# Ollama Tests & Refactoring

## Overview
This directory contains tests and refactored components for the TTTC system with Ollama integration.

## What I Did

### 1. Edge Case Handling
The parser handles multiple LLM response formats:
- Direct JSON responses
- JSON in markdown code blocks
- JSON after `<think>` tags
- Multiple JSON objects (problematic duplicates)
- Malformed JSON repair
- JavaScript-style comments in JSON

## Running the System

### Prerequisites
```bash
# Ensure Ollama is running
ollama serve

# Install required models
ollama pull llama3.2:latest
```

### Start All Services
From the project root:
```bash
# Start all services (Redis, Express, Python, Next.js) ---> Read main README
npm run dev
```

### Individual Services
```bash
# Python FastAPI server only
cd pyserver && python -m uvicorn main:app --host 0.0.0.0 --port 8000

# Test JSON parser independently
cd ollama-tests/tests/phase3_integration && python json_response_parser.py
```

### Configuration
- **Model**: Uses `llama3.2:latest` by default
- **Ollama URL**: `http://localhost:11434`
- **Python Server**: Port 8000
- **Express Server**: Port 8080
- **Next.js Frontend**: Port 3000

## Testing
The JSON parser includes built-in test cases. Run independently:
```bash
python ollama-tests/tests/phase3_integration/json_response_parser.py
```