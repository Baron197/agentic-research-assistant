# Running & Deploying in Real OpenAI Mode

The project is **keyless by default** (deterministic fakes, `$0`, offline). This guide turns on
**real mode** — a real OpenAI LLM, and optionally real web search — for genuine research, and shows
how to **run it locally** and **deploy it safely** with your own keys.

> **Windows note.** This project ships with no `make` and no system Python — it uses the repo's
> `.venv`. Every command below is the direct PowerShell form. Set `$env:PYTHONPATH="src"` once per
> shell (that's all the Makefile's `export PYTHONPATH := src` does; on Windows you set it yourself).

---

## What "real mode" means

Three independent switches (the Strategy pattern) — mix them to taste:

| Env var | Keyless (default) | Real |
|---|---|---|
| `LLM_PROVIDER` | `fake` (rule-based) | `openai` |
| `SEARCH_PROVIDER` | `fake` (your corpus) | `web` (Tavily) |
| `FETCH_PROVIDER` | `fake` (`local://`) | `http` (real pages) |

Two useful real-mode recipes:

| Recipe | Providers | What you get | Cost |
|---|---|---|---|
| **A — Real LLM over your own docs** *(recommended for work)* | `openai` + `fake` + `fake`, with `CORPUS_DIR` = your folder | A properly **synthesised, cited** report over **your own** `.md`/`.txt`/`.pdf` files — no web calls | OpenAI tokens only (cents) |
| **B — Full live web research** | `openai` + `web` + `http` | Plans, searches the **live web** (Tavily), fetches pages, writes a cited report | OpenAI + Tavily (both have free/cheap tiers) |

The config validator rejects impossible mixes (e.g. `web` search with `fake` fetch) at startup, so you
can't half-configure it.

---

## Part 1 — Run locally in real OpenAI mode

### 0. Get your keys
- **OpenAI** (both recipes): [platform.openai.com/api-keys](https://platform.openai.com/api-keys) → `sk-...`.
  Set a **spend limit** in Billing → Limits as a hard safety net.
- **Tavily** (Recipe B only): [app.tavily.com](https://app.tavily.com) → `tvly-...`. Free tier ~1,000 searches/mo.

### 1. Install the real-mode extras
The real SDKs are import-guarded (the keyless path never needs them), so install them once:

```powershell
.\.venv\Scripts\python.exe -m pip install openai tavily-python trafilatura
# optional — only if you'll research your own PDF files:
.\.venv\Scripts\python.exe -m pip install pypdfium2
```

### 2. Configure `.env` (gitignored — never commit it)
Copy `.env.example` to `.env` (or edit the one already in the repo root).

**Recipe A — real LLM over your own documents:**
```
LLM_PROVIDER=openai
OPENAI_API_KEY=sk-...
OPENAI_MODEL=gpt-4o-mini          # bump to gpt-4o for harder synthesis
SEARCH_PROVIDER=fake
FETCH_PROVIDER=fake
CORPUS_DIR=C:/Users/you/research-docs    # your own .md / .txt / .pdf files
TOKEN_BUDGET=60000                # hard per-run ceiling = your cost guardrail
```

**Recipe B — live web research:**
```
LLM_PROVIDER=openai      OPENAI_API_KEY=sk-...
SEARCH_PROVIDER=web      SEARCH_API_KEY=tvly-...
FETCH_PROVIDER=http
```

### 3. Run it (Windows — no `make`)
Set the path once, then run any entry point:

```powershell
$env:PYTHONPATH = "src"
$py = ".\.venv\Scripts\python.exe"

# CLI — a cited report straight to the terminal
& $py -m agent.runner "What are the trade-offs of hybrid retrieval?"

# API — Swagger UI at http://127.0.0.1:8000/docs
& $py -m uvicorn agent.api:app --port 8000

# UI — http://localhost:8501 (single self-contained app; no separate API needed)
& $py -m streamlit run ui/streamlit_app.py
```

*(The Mac/Linux shortcuts `make run Q="..."`, `make api`, `make ui` do the exact same thing — they
aren't available on a Windows laptop without `make`.)*

### 4. Verify real mode is actually on
- `GET /health` returns `keyless=false` (it's `true` only when **all** providers are fake).
- A completed run reports a **non-zero USD cost** and real token counts — keyless always shows `$0.0000`.
- Citations point at real sources (Recipe B: real URLs; Recipe A: your own filenames).

```powershell
& $py -c "import httpx; print(httpx.get('http://127.0.0.1:8000/health').json())"
# -> {'status': 'ok', 'version': '0.1.0', 'keyless': False}
```

### 5. Control the cost (real mode is paid — the guardrails are yours to set)
- **`TOKEN_BUDGET`** — the pipeline stops cleanly as `partial` once a run exceeds it. Lower it to cap per-run spend.
- **`OPENAI_MODEL`** — `gpt-4o-mini` is a fraction of a cent per run; reach for `gpt-4o` only on hard topics.
- **`EVIDENCE_PER_SUBQUESTION` / `TOP_SEARCH_RESULTS`** — depth knobs; more sources ⇒ more tokens.
- **`MAX_ITERATIONS`** — each critic revise re-runs the writer + critic (more tokens).
- **OpenAI dashboard spend limit** — the ultimate backstop. Set it.
- Ballpark: a `gpt-4o-mini` run is typically well under **$0.01**; Tavily searches are free within the tier.

---

## Part 2 — Deploy in real OpenAI mode

**The golden rule:** keys are **runtime secrets** — never committed, never baked into an image. The repo
itself stays keyless and safe to be public; your keys live only in the host's secret store. And note:
**a public real-mode URL spends _your_ money for anyone who opens it** — so protect it, or keep the public
demo keyless and run real mode privately.

### Option S — Streamlit Community Cloud (single app, easiest)
The UI runs the pipeline in-process (embedded backend), so real mode is just secrets:

1. Deploy the app as in [DEPLOYMENT.md](DEPLOYMENT.md) (main file `ui/streamlit_app.py`).
2. App → **Settings → Secrets**, and paste the keys as **top-level** keys (this matters — see below):
   ```toml
   LLM_PROVIDER = "openai"
   OPENAI_API_KEY = "sk-..."
   SEARCH_PROVIDER = "web"
   SEARCH_API_KEY = "tvly-..."
   FETCH_PROVIDER = "http"
   TOKEN_BUDGET = "40000"
   ```
   > **Why top-level?** Streamlit Community Cloud exposes **root-level** secrets as environment
   > variables (`os.environ`), which is exactly what the app's settings layer reads. Secrets nested
   > under a `[section]` are **only** reachable via `st.secrets`, so they would *not* switch the app to
   > real mode. Keep each key at the top level, no `[section]` header.
3. **Protect it / mind the cost.** A public Streamlit app has no auth — anyone with the link can run
   (paid) research. Prefer running real mode **locally**; if you deploy it, keep `TOKEN_BUDGET` low, set
   an OpenAI spend limit, and restrict viewers (private-app allow-list) rather than leaving it world-open.

### Option A — GCP Cloud Run (inject secrets, require auth)
Store the key in Secret Manager, inject it as an env var, and require IAM auth so strangers can't spend
your budget:

```bash
# store secrets once
printf 'sk-...'   | gcloud secrets create openai-key  --data-file=-
printf 'tvly-...' | gcloud secrets create tavily-key  --data-file=-

gcloud run deploy ara-ui \
  --image "$IMAGE" --region us-central1 --port 8080 --session-affinity \
  --no-allow-unauthenticated \                     # require IAM auth — protects your $
  --set-env-vars "LLM_PROVIDER=openai,SEARCH_PROVIDER=web,FETCH_PROVIDER=http,ARA_EMBEDDED=1,TOKEN_BUDGET=40000" \
  --set-secrets "OPENAI_API_KEY=openai-key:latest,SEARCH_API_KEY=tavily-key:latest" \
  --command streamlit \
  --args "run,ui/streamlit_app.py,--server.port=8080,--server.address=0.0.0.0,--server.headless=true,--server.enableCORS=false,--server.enableXsrfProtection=false"
```
- The Cloud Run runtime service account needs `roles/secretmanager.secretAccessor`.
- Grant yourself access with `gcloud run services add-iam-policy-binding ara-ui --region us-central1 --member="user:you@example.com" --role=roles/run.invoker`, or put it behind Identity-Aware Proxy.

### Option C — a VM (Oracle Ampere A1 / GCP e2-micro)
Put the keys in a `.env` **on the VM** (never in git), then run with docker-compose, and firewall the app
port to your own IP so it isn't world-open:

```bash
# on the VM, in the repo dir — .env is gitignored, stays on the box
cat > .env <<'EOF'
LLM_PROVIDER=openai
OPENAI_API_KEY=sk-...
SEARCH_PROVIDER=web
SEARCH_API_KEY=tvly-...
FETCH_PROVIDER=http
TOKEN_BUDGET=40000
EOF
docker compose up -d --build     # the api + ui services read .env
```

---

## Security & cost checklist (real mode)
- [ ] Keys only in the host secret store or a **gitignored `.env`** — never committed, never in the image.
- [ ] Access **restricted** (IAM / IAP / firewall / private-app allow-list) — a public real-mode URL spends your money for anyone.
- [ ] `TOKEN_BUDGET` set conservatively; `OPENAI_MODEL=gpt-4o-mini` unless you need more.
- [ ] OpenAI dashboard **spend limit** set as the hard backstop.
- [ ] Watch usage: the OpenAI Usage dashboard and the Tavily dashboard.

## Caveats
- **DSPy backend** (`AGENT_BACKEND=dspy`) configures its LM via process-global state — not recommended
  under concurrent API serving; the default `manual` backend is unaffected.
- **Fetching** — `HttpFetch` is bounded (byte cap, SSRF-guarded, per-redirect-hop checks) and runs in
  parallel across sub-questions; deeper runs still hit more pages, so mind `TOP_SEARCH_RESULTS`.
- Real mode is **not free** — keyless remains the right choice for the public portfolio demo. See
  [README.md](README.md) for the keyless quickstart and [DEPLOYMENT.md](DEPLOYMENT.md) for the free
  (keyless) hosting options.
