FROM python:3.12

RUN apt update && apt install -y mkvtoolnix && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir \
    av>=13.0 \
    moviepy>=2.0 \
    pandas>=2.0 \
    "scenedetect[opencv-headless]>=0.6.5"

ADD . /usr/src/app

CMD ["scenedetect"]
