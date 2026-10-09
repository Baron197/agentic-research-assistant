# Lean runtime image: builds and serves the API, keyless by default.
# The same image also runs REAL mode — set LLM_PROVIDER / SEARCH_PROVIDER /
# FETCH_PROVIDER and the keys at RUNTIME (env vars, Secret Manager, or a
# gitignored .env via docker-compose). Keys are never baked into the image.
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/app/src \
    LLM_PROVIDER=fake \
    SEARCH_PROVIDER=fake \
    FETCH_PROVIDER=fake

WORKDIR /app

# Runtime dependencies only (no ruff/pytest). streamlit is included so the
# optional UI service in docker-compose can share this image. openai,
# tavily-python and trafilatura are the real-mode libraries: installed so a
# deployment can switch to real mode with env vars alone, but only IMPORTED when a
# real provider is selected (lazy imports in llm.py / search.py / fetch.py), so
# keyless runs never load them. trafilatura pulls the article text out of a
# fetched page; without it, evidence snippets are mostly the page's CSS and menus.
# Upper bounds stop a future breaking MAJOR release from reaching a deploy
# unnoticed; the calls the app makes are verified on openai 2.x and 3.x.
RUN pip install --no-cache-dir \
    "langgraph>=0.2" "langchain-core>=0.3" "pydantic>=2.5" "pydantic-settings>=2.1" \
    "fastapi>=0.110" "uvicorn>=0.27" "httpx>=0.27" "streamlit>=1.33" \
    "openai>=1.30,<4" "tavily-python>=0.3,<1" "trafilatura>=1.6,<3"

COPY src ./src
COPY data ./data
COPY ui ./ui
COPY eval ./eval
COPY .streamlit ./.streamlit

EXPOSE 8000
CMD ["uvicorn", "agent.api:app", "--host", "0.0.0.0", "--port", "8000"]
