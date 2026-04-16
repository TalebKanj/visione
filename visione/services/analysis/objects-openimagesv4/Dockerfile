FROM python:3.12

WORKDIR /usr/src/app

COPY requirements.txt ./
RUN pip install --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r requirements.txt

ENV PYTHONPATH "${PYTHONPATH}:/usr/src/"
ENV TF_CPP_MIN_LOG_LEVEL 2
ENV TFHUB_CACHE_DIR=/cache/tfhub_modules
