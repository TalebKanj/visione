# VisionAI Project Standards

This document defines the coding standards, conventions, and best practices for the VisionAI project.

## 1. Container Architecture

### 1.1 Service Structure
```
services/
├── analysis/           # Analysis services (feature extraction)
│   ├── features-*/    # CLIP, DinoV2, ALADIN, etc.
│   ├── objects-*/    # Object detection
│   └── scene-*/     # Scene detection
├── index/            # Index services
│   ├── faiss-index-manager/
│   ├── lucene-index-manager/
│   └── str-*-encoder/
└── core/            # Core Java/Tomcat service
```

### 1.2 Docker Image Build
- Each service MUST have a `Dockerfile`
- Base images must use specific versions (no `:latest`)
- Multi-stage builds preferred for production

### 1.3 Entrypoint
- Python services use `service.py` as entrypoint
- CMD in Dockerfile: `CMD ["python", "-u", "service.py"]`
- The `-u` flag ensures unbuffered output for logs

## 2. Python Code Standards

### 2.1 Logging
```python
import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(name)s: %(message)s'
)
```

- Use `logging.INFO` as default level
- Support `LOG_LEVEL` environment variable override
- Never use `print()` for production code

### 2.2 Imports
- Standard library first
- Third-party packages second
- Local imports last
- Sort alphabetically within groups

### 2.3 Flask App Pattern
```python
from flask import Flask, request, jsonify
import logging

app = Flask(__name__)

@app.route('/ping', methods=['GET'])
def ping():
    return "pong"

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--host', default='0.0.0.0')
    parser.add_argument('--port', default='8080')
    args = parser.parse_args()
    app.run(debug=False, host=args.host, port=args.port)
```

### 2.4 Model Loading
- Always use cache directories for models
- Support both eager and lazy loading
- Handle missing models gracefully

## 3. Environment Variables

### 3.1 Required Pattern
```yaml
environment:
  - MODEL_HANDLE=${MODEL_HANDLE}
  - FEATURES_NAME=${FEATURES_NAME}
```

### 3.2 Optional with Defaults
```python
default_model = os.environ.get('MODEL_HANDLE', 'openai/clip-vit-large-patch14')
```

### 3.3 Docker Compose Environment
- Use `${VARIABLE}` syntax for interpolation
- Define required variables in `.env` file
- Document all environment variables

## 4. Memory Management

### 4.1 Container Limits
```yaml
deploy:
  resources:
    limits:
      memory: 4G
    reservations:
      memory: 2G
      devices:
        - driver: nvidia
          count: 1
          capabilities: [gpu]
```

### 4.2 Model Memory
- Use `torch.float16` for large models
- Implement CPU offload when needed
- Use Accelerate library for memory-efficient loading

### 4.3 Cache Directories
```dockerfile
ENV TRANSFORMERS_CACHE /cache/huggingface
ENV TORCH_HUB_CACHE /cache/torch
```

## 5. Docker Compose Structure

### 5.1 Service Composition
```
docker-compose.yaml          # Main composition
├── analysis-services.yaml   # Analysis services
├── index-services.yaml   # Index services
└── devel-options.yaml    # Development overrides
```

### 5.2 Profiles
- `query`: Query services (core, faiss, str-encoder)
- `analysis`: Analysis services
- `router`: Router service

### 5.3 Named Volumes
```yaml
volumes:
  model_cache:
    driver: local
```

## 6. API Endpoints

### 6.1 Required Endpoints
| Endpoint | Method | Description |
|----------|--------|-------------|
| `/ping` | GET | Health check |

### 6.2 Feature Endpoints
| Endpoint | Method | Description |
|----------|--------|-------------|
| `/get-text-feature` | GET | Text to feature |
| `/get-image-feature` | GET/POST | Image to feature |

### 6.3 Deprecated Endpoints (Backward Compatible)
- `/text-to-image-search`
- `/internal-image-search`

## 7. Error Handling

### 7.1 HTTP Status Codes
- `200` - Success
- `400` - Bad Request (missing parameters)
- `500` - Internal Server Error

### 7.2 Error Messages
```python
if 'type' not in data:
    return "Missing 'type' key in request.", 400
```

## 8. Testing

### 8.1 Unit Tests
- Test model classes in isolation
- Mock external dependencies

### 8.2 Integration Tests
- Test API endpoints
- Test Docker composition

## 9. Documentation

### 9.1 Code Comments
- NO comments unless absolutely necessary
- Self-documenting code preferred

### 9.2 README Files
- Each service directory should have a README.md if non-trivial

## 10. Resource Targets

### 10.1 Desktop Configuration
- RAM: 16GB
- VRAM: 8GB (dGPU)
- Memory per container: 4GB (limits), 2GB (reservations)
- GPU: 1 per service

### 10.2 Server Configuration
- RAM: 64GB+
- VRAM: 24GB+ per GPU
- Multiple GPUs supported