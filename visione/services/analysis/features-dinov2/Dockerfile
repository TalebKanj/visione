FROM docker.io/pytorch/pytorch:2.9.0-cuda13.0-cudnn9-runtime

RUN echo "nameserver 8.8.8.8" > /etc/resolv.conf && \
    echo "nameserver 1.1.1.1" >> /etc/resolv.conf

RUN apt-get update && apt-get install -y --no-install-recommends \
    git build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN pip install xformers>=0.0.35

RUN cd /usr/src && \
    git clone https://github.com/facebookresearch/dinov2 && \
    cd /usr/src/dinov2 && \
    git checkout c3c2683a13cde94d4d99f523cf4170384b00c34c && \
    sed -i '/xformers/d' requirements.txt && \
    pip install -r requirements.txt --no-deps

WORKDIR /usr/src/app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

ENV PYTHONPATH "${PYTHONPATH}:/usr/src/"
ENV TORCH_HOME /cache/torch

COPY . .
CMD ["python", "-u", "service.py"]