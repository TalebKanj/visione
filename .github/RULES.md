# Project Rules

This file defines hard rules and constraints that MUST be followed when making changes.

## 1. Security Rules

### 1.1 Secrets
- **NEVER** commit secrets, API keys, or credentials to the repository
- Use environment variables or secrets management
- Add sensitive files to `.gitignore`

### 1.2 Docker Base Images
- **ALWAYS** use specific version tags, never `:latest`
- Example: `pytorch/pytorch:2.9.0-cuda13.0-cudnn9-runtime` (OK)
- Example: `pytorch/pytorch:latest` (FORBIDDEN)

### 1.3 Network Security
- Do not expose unnecessary ports
- Use internal networking between containers
- Never bind to `0.0.0.0` in production without reason

## 2. Docker Rules

### 2.1 Container Entrypoint
- **MUST** use `["python", "-u", "service.py"]` for Python services
- The `-u` flag is required for unbuffered output

### 2.2 Volume Mounts
- **MUST** use named volumes for persistent data
- **MUST** mount cache volumes for model downloads
- Never mount host directories directly (use bind mounts only for dev)

### 2.3 Resource Limits
- **MUST** define memory limits in deploy section
- **SHOULD** define GPU reservations for ML services

```yaml
deploy:
  resources:
    limits:
      memory: 4G
    reservations:
      memory: 2G
```

## 3. Code Rules

### 3.1 Logging
- **MUST** use `logging.INFO` as default level
- **MUST** use format string: `'%(asctime)s %(levelname)s %(name)s: %(message)s'`
- **NEVER** use `print()` in production

### 3.2 Health Checks
- **MUST** implement `/ping` endpoint returning "pong"
- **MUST** return HTTP 200 for health check

### 3.3 Error Handling
- **MUST** return appropriate HTTP status codes
- **MUST** include error message in response

### 3.4 API Endpoints
- **MUST** use standard HTTP methods
- **MUST** return JSON for API responses

## 4. Memory Rules

### 4.1 Model Memory
- **MUST** use cache directories for model storage
- **MUST** handle out-of-memory gracefully
- **SHOULD** use FP16 for large models

### 4.2 Container Memory
- **MUST** set appropriate memory limits
- **SHOULD** use memory reservations for guaranteed allocation
- **NEVER** exceed container limits in normal operation

## 5. Dependency Rules

### 5.1 Python Packages
- **MUST** pin exact versions in `requirements.txt`
- **MUST** use `--no-cache-dir` in Dockerfile pip install
- Example: `transformers==4.27.4` (OK)
- Example: `transformers>=4.27` (FORBIDDEN for production)

### 5.2 Versions
- **MUST** document Python version requirement
- **MUST** document CUDA version if using GPU

## 6. Git Rules

### 6.1 Commits
- **MUST** be descriptive
- **MUST** follow conventional commits (optional)
- **NEVER** commit generated files accidentally

### 6.2 Branching
- Use feature branches for development
- **MUST** review before merging

## 7. Testing Rules

### 7.1 Before Deployment
- **MUST** verify container builds
- **MUST** verify service starts
- **MUST** verify health endpoint responds

### 7.2 Breaking Changes
- **MUST** increment version for breaking changes
- **MUST** document breaking changes

## 8. Documentation Rules

### 8.1 Code
- **NEVER** add unnecessary comments
- Use self-documenting variable names
- Keep functions small and focused

### 8.2 Files
- **SHOULD** document complex services
- **MUST** document environment variables

## 9. Compatibility Rules

### 9.1 Backward Compatibility
- **MUST** maintain deprecated endpoints
- **MUST** provide migration path
- Deprecate instead of remove

### 9.2 Platform Compatibility
- **SHOULD** work on both Linux and WSL2
- **MUST** work with NVIDIA Container Toolkit

## 10. Performance Rules

### 10.1 Startup Time
- **SHOULD** minimize startup time
- **MUST** handle slow model downloads

### 10.2 Memory Efficiency
- **SHOULD** use lazy loading where appropriate
- **MUST** clean up resources when done

## Enforcement

Violations of these rules may result in:
- Pull request rejection
- Build failures
- Runtime errors

When in doubt, ask before implementing.