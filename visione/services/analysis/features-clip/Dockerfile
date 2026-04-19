FROM docker.io/pytorch/pytorch:2.9.0-cuda13.0-cudnn9-runtime

WORKDIR /usr/src/app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

ENV PYTHONPATH "${PYTHONPATH}:/usr/src/"
ENV TRANSFORMERS_CACHE /cache/huggingface
ENV TORCH_HUB_CACHE /cache/torch

RUN mkdir -p /cache/huggingface /cache/torch

COPY . .
CMD ["python", "-u", "service.py"]