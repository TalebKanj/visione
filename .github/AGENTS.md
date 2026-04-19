# OpenCode Agent Configuration

This file provides guidance for AI assistants (like OpenCode) when working with this project.

## Project Overview

- **Project Name**: VisionAI
- **Type**: Multi-container video analysis system
- **Language**: Python (services), Java (core)
- **Container Orchestration**: Docker Compose

## Key Files to Know

| File | Purpose |
|------|---------|
| `visione/services/docker-compose.yaml` | Main composition |
| `visione/services/analysis-services.yaml` | Analysis service definitions |
| `visione/services/index-services.yaml` | Index service definitions |
| `visione/services/*/Dockerfile` | Container definitions |
| `visione/services/*/service.py` | Service entrypoint |

## Common Tasks

### 1. Adding a New Service

1. Create service directory: `visione/services/{service-type}/{service-name}/`
2. Add `Dockerfile` based on existing patterns
3. Add `service.py` entrypoint
4. Add service definition to appropriate YAML file
5. Add to `docker-compose.yaml` or create include

### 2. Modifying Model Loading

1. Find service in `visione/services/analysis/` or `visione/services/index/`
2. Modify `service.py` - model class `__init__` method
3. Update memory limits in YAML if needed
4. Test with container rebuild

### 3. Adding Environment Variables

1. Add to `service.py`: `os.environ.get('VAR_NAME', 'default')`
2. Add to YAML service definition: `environment: - VAR_NAME=value`
3. Document in service README

### 4. Changing Resource Limits

Modify in service YAML:
```yaml
deploy:
  resources:
    limits:
      memory: 4G
    reservations:
      memory: 2G
```

## Important Patterns

### Model Loading Pattern
```python
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = AutoModel.from_pretrained(model_handle, cache_dir="/cache/huggingface")
model = model.to(device)
```

### Flask Endpoint Pattern
```python
@app.route('/endpoint', methods=['GET'])
def handler():
    param = request.args.get('param')
    result = model.predict(param)
    return jsonify(result)
```

### Health Check Pattern
```python
@app.route('/ping', methods=['GET'])
def ping():
    return "pong"
```

## Docker Commands

```bash
# Build a service
docker build -t service-name ./services/path

# Run with docker-compose
docker-compose -f docker-compose.yaml up service-name

# View logs
docker-compose logs -f service-name

# Check resource usage
docker stats
```

## Testing Approach

- No formal test framework detected
- Manual testing via curl/http requests
- Check `/ping` endpoint for health

## Code Style to Follow

- No comments unless critical
- logging.INFO level with format string
- argparse for CLI arguments
- Follow existing file structure exactly

## What NOT to Do

- Do NOT commit secrets or keys
- Do NOT use `:latest` in Docker base images
- Do NOT remove health check endpoints
- Do NOT break backward compatibility without version bump
- Do NOT run arbitrary bash commands without explanation