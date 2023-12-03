FROM python:3.11.6-slim

ENV TZ=Asia/Tokyo
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

RUN apt-get update && apt-get install -y cmake make ffmpeg libsm6 libxext6 git curl \
    libglib2.0-0 libsm6 libxrender-dev libxext6 libgl1-mesa-dev libopenblas-dev libomp-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

ENV PYTHONPATH="${PYTHONPATH}:/workspace"
WORKDIR /workspace

# poetry
ENV POETRY_HOME=/opt/poetry
RUN curl -sSL https://install.python-poetry.org/ | python - --version 1.7.1 && \
    cd /usr/bin && \
    ln -s /opt/poetry/bin/poetry && \
    poetry config virtualenvs.create false
COPY pyproject.toml .
# COPY poetry.lock .
RUN poetry install
