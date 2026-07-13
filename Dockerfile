# Lean runtime image: builds and serves the API, keyless by default.
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/app/src \
    LLM_PROVIDER=fake \
    SEARCH_PROVIDER=fake \
    FETCH_PROVIDER=fake

WORKDIR /app

# Runtime dependencies only (no ruff/pytest). streamlit is included so the
# optional UI service in docker-compose can share this image.
RUN pip install --no-cache-dir \
    "langgraph>=0.2" "langchain-core>=0.3" "pydantic>=2.5" "pydantic-settings>=2.1" \
    "fastapi>=0.110" "uvicorn>=0.27" "httpx>=0.27" "streamlit>=1.33"

COPY src ./src
COPY data ./data
COPY ui ./ui
COPY eval ./eval
COPY .streamlit ./.streamlit

EXPOSE 8000
CMD ["uvicorn", "agent.api:app", "--host", "0.0.0.0", "--port", "8000"]
