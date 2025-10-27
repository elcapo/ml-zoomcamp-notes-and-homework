FROM agrigorev/zoomcamp-model:2025

WORKDIR /code

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/
COPY ["main.py", "pipeline_v1.bin", "pyproject.toml", "uv.lock", "./"]

RUN uv sync --frozen --no-cache

EXPOSE 8001
CMD ["uv", "run", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8001"]