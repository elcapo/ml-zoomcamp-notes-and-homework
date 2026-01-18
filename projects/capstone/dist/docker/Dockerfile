# Use Python with uv
FROM python:3.13-slim-trixie
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Copy the project into the image
COPY . /app

# Disable development dependencies
ENV UV_NO_DEV=1

# Sync the project into a new environment, asserting the lockfile is up to date
WORKDIR /app
RUN uv python install --compile-bytecode
RUN uv sync --compile-bytecode

# Start the API server
CMD ["uv", "run", "uvicorn", "serve:app", "--host", "0.0.0.0", "--port", "80", "--workers", "4"]
