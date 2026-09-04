FROM python:3.11-slim

RUN pip install --no-cache-dir uv

WORKDIR /app

COPY pyproject.toml uv.lock ./

RUN uv sync --frozen --no-dev

COPY deployment/ .

EXPOSE 9696

CMD ["uv","run","gunicorn","--bind=0.0.0.0:9696","predict:app"]