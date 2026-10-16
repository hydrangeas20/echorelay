# CPU image for EchoRelay.
#   docker build -t echorelay .
#   docker run --rm -p 8000:8000 -v echorelay-models:/models echorelay
# Bake the model into the image (no download at startup):
#   docker build --build-arg PRELOAD_MODEL=1 -t echorelay .
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    ECHORELAY_MODEL_DIR=/models \
    ECHORELAY_DEVICE=cpu \
    ECHORELAY_COMPUTE_TYPE=int8

RUN useradd --create-home --uid 10001 app && mkdir -p /models && chown app /models
WORKDIR /app

COPY requirements.txt ./
RUN pip install -r requirements.txt

COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-deps .

USER app

ARG PRELOAD_MODEL=0
RUN if [ "$PRELOAD_MODEL" = "1" ]; then \
      python -c "from faster_whisper import WhisperModel; WhisperModel('base.en', device='cpu', compute_type='int8', download_root='/models')"; \
    fi

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=120s \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=4)" || exit 1

CMD ["python", "-m", "echorelay", "--host", "0.0.0.0", "--port", "8000"]
