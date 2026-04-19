# OpenCode Skills Configuration

This file defines the specialized skills and workflows available for the VisionAI project.

## Available Skills

### 1. docker-build
**Description**: Build and rebuild Docker containers

**Commands**:
- `docker build -t <name> .`
- `docker-compose build <service>`

**When to use**:
- After modifying Dockerfile
- After changing service configuration
- When adding new services

---

### 2. docker-debug
**Description**: Debug container issues

**Commands**:
- `docker logs <container>`
- `docker stats`
- `docker exec -it <container> bash`

**When to use**:
- Container crashes on startup
- Out of memory errors
- Health check failures

---

### 3. model-management
**Description**: Manage ML model loading and caching

**Knowledge Areas**:
- HuggingFace transformers model loading
- PyTorch Hub model downloads
- Accelerate library for memory-efficient loading

**Patterns**:
```python
# Standard loading
model = AutoModel.from_pretrained(handle, cache_dir="/cache/huggingface")

# Memory-efficient loading
from accelerate import init_empty_weights, load_checkpoint_and_dispatch
model = load_checkpoint_and_dispatch(model, device_map="auto")
```

**When to use**:
- OOM errors
- Model download issues
- Slow startup times

---

### 4. service-creation
**Description**: Create new analysis or index services

**Steps**:
1. Create directory structure
2. Create Dockerfile
3. Create service.py
4. Add to docker-compose YAML
5. Add environment variables
6. Test health endpoint

**Template locations**:
- `visione/services/analysis/features-clip/` (for feature services)
- `visione/services/index/faiss-index-manager/` (for index services)

---

### 5. resource-optimization
**Description**: Optimize memory and GPU resource usage

**Techniques**:
- FP16 quantization
- CPU offload with Accelerate
- Layer-by-layer loading
- Model chunking

**Commands to research**:
- Web search for latest Accelerate documentation
- Check PyTorch memory management

---

### 6. yaml-compose
**Description**: Work with Docker Compose configurations

**Key files**:
- `visione/services/docker-compose.yaml`
- `visione/services/analysis-services.yaml`
- `visione/services/index-services.yaml`

**Common tasks**:
- Add new service
- Modify resource limits
- Adjust profiles

---

### 7. code-review
**Description**: Review code changes before implementation

**Requirements**:
- Search web for documentation
- Check for breaking changes
- Verify compatibility with existing patterns

**For every change**:
1. Research latest documentation
2. Check existing patterns in codebase
3. Verify Docker best practices
4. Test locally if possible

---

### 8. container-orchestration
**Description**: Understand and modify container orchestration

**Architecture**:
```
router (nginx)
├── core (Tomcat)
├── analysis services (multiple)
│   ├── features-clip-*
│   ├── features-dinov2
│   ├── features-aladin
│   └── objects-*
└── index services
    ├── faiss-index-manager
    └── str-feature-encoder
```

**Flows**:
- Analysis: video → frames → features → index → search
- Query: text → feature extractor → faiss search → results

---

## Skill Usage Examples

### Adding a new feature service
```
Use service-creation skill to:
- Create new directory
- Copy template from features-clip
- Modify model loading
- Add to compose YAML
```

### Fixing OOM issue
```
Use resource-optimization skill to:
- Check model size
- Implement CPU offload
- Reduce memory limits
- Add quantization
```

### Debugging startup failure
```
Use docker-debug skill to:
- Check logs
- Inspect container state
- Verify environment
- Test model loading
```

---

## Workflows

### 1. Security Audit
1. Check for secrets in code
2. Verify base image versions
3. Check network exposure
4. Review resource limits

### 2. Performance Tuning
1. Identify bottleneck
2. Research solutions
3. Implement changes
4. Test with limits

### 3. New Feature
1. Understand requirements
2. Research implementation
3. Follow service-creation skill
4. Test thoroughly
5. Update documentation if needed

---

## Important Notes

- Always check web for latest docs before implementation
- Follow rules in RULES.md
- Use STANDARDS.md for code style
- Check AGENTS.md for common patterns

---

## Activation

These skills are automatically available to OpenCode when working in this repository. The agent will use these configurations to guide its decisions and workflows.