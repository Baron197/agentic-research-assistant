# Deployment Guide — free hosting (Streamlit Cloud, GCP & Oracle Cloud)

This guide deploys the **Agentic Research & Report Assistant** to the cloud for **$0**.
The app is ideal for a free tier: it's a small, keyless Python stack with **no database,
no API keys, and no GPU** — the whole thing runs offline on deterministic fake providers.

*A printable, illustrated version of this guide ships as [`DEPLOY_GUIDE.pdf`](DEPLOY_GUIDE.pdf)
in the repo root — it walks through the fastest free paths (Streamlit Community Cloud and GCP Cloud Run).*

There are two ways to run it, and the deploy target decides which you use:

> **One app, or two services.**
> - **Single self-contained app (easiest).** The Streamlit UI can run the agent pipeline
>   **in-process** — no separate API. When it can't reach an API at `API_URL`, it auto-falls
>   back to this *embedded* backend (force it with `ARA_EMBEDDED=1`). This is what makes the
>   **one-click Streamlit Community Cloud** deploy below possible. → **Option S**.
> - **Two services, one image.** The `Dockerfile` builds a single image; the API (`:8000`)
>   and the UI (`:8501`) run as two containers from it (see `docker-compose.yml`), and the UI
>   reaches the API over `API_URL`. → **Options A–C**.

---

## Which option should I pick?

| # | Option | Cloud | Free forever? | Persistent history? | Effort | Best for |
|---|--------|-------|---------------|---------------------|--------|----------|
| **S** | **Community Cloud** (single app) | Streamlit | ✅ (sleeps when idle) | ❌ ephemeral | **Lowest** | **The fastest free demo link — no Docker, no CLI** |
| **A** | **Cloud Run** (serverless) | GCP | ✅ scales to $0 when idle | ❌ ephemeral | Low | A shareable public demo link with a real API |
| **B** | **e2-micro VM** + compose | GCP | ✅ (1 GB RAM — tight) | ✅ | Medium | Always-on full stack on GCP |
| **C** | **Ampere A1 VM** + compose | Oracle | ✅ (2 OCPU / 12 GB) | ✅ | Medium | **Free-forever full stack (recommended)** |
| **T** | Anything | GCP **Free Trial** | 💳 $300 / 90 days | either | — | Your next 3 months — then migrate to A or C |

**Recommendations**
- **Just want a free demo link with the least effort?** → **Option S (Streamlit Community Cloud)** — point it at the repo and click Deploy; no Docker, no CLI, no card. Same as how you deployed your RAG app.
- **Want it free forever with a real separate API + UI?** → **Option C (Oracle Ampere A1)** — by far the most RAM headroom.
- **Want a scale-to-zero public link that costs nothing when nobody's using it?** → **Option A (GCP Cloud Run)**.
- **Have the GCP $300 trial for 3 months (you do)?** → Use **A or B freely now** (the credit covers any overage and unlocks any region/size), then **migrate to A or C before day 90** to stay at $0. See [§5](#5-gcp-free-trial-your-next-3-months).

Current free-tier facts used below (verified July 2026 — always re-check, they change):
- **Streamlit Community Cloud**: free public apps from a public GitHub repo, ~1 GB RAM, installs from `requirements.txt`; apps **sleep after inactivity** and wake on the next visit (a few seconds). No credit card.
- **GCP Cloud Run** always-free: 2M requests/mo, scales to zero. **e2-micro** always-free: 1 instance/mo in `us-central1` / `us-west1` / `us-east1`, 1 GB RAM, 30 GB disk.
- **Oracle Always Free**: Ampere A1 = **2 OCPU / 12 GB** (1,500 OCPU-hrs + 9,000 GB-hrs/mo), or 2× AMD E2.1.Micro (1 GB each); 200 GB block storage; 10 TB/mo egress. *(Oracle cut A1 from 4/24 to 2/12 in June 2026.)*
- **GCP Free Trial**: $300 credit, 90 days, **no automatic charges** — the account closes at 90 days or $300 and you're only billed if you *manually* upgrade.

---

## 0. Prerequisites

- This repository (the folder containing `Dockerfile` and `docker-compose.yml`).
- A Google account (Options A/B/T) and/or an Oracle Cloud account (Option C).
- For Options B/C you'll copy the code to a VM — either **push it to a Git repo and
  `git clone`**, or **`scp`** it up (commands are given). The app has no secrets, so a
  public repo is fine.
- Local Docker is **optional** — Cloud Run builds in the cloud, and the VMs install Docker themselves.

Two small deployment-support files ship with the repo:
- `.dockerignore` — keeps `.venv/`, `runs/`, etc. out of the build/upload.
- The `Dockerfile` copies `.streamlit/` so the deployed UI keeps its theme.

---

## Option S — Streamlit Community Cloud (single app, easiest, no card)

The fastest way to get a free public link — the **same flow you used for your RAG app**.
There's **no separate API and no Docker**: the UI detects that nothing is listening on
`API_URL` and runs the whole LangGraph pipeline **in-process** (the *embedded* backend).
Every page — Research, Critic A/B, History, Observability, Guide — works from that one process.

**1. Push this repo to GitHub** (public is fine — the app has no secrets):

```bash
git remote add origin https://github.com/Baron197/agentic-research-assistant.git
git push -u origin main
```

**2. Deploy on Streamlit Community Cloud**
- Go to **share.streamlit.io** → sign in with GitHub → **Create app** → **Deploy a public app from GitHub**.
- **Repository:** `Baron197/agentic-research-assistant`  ·  **Branch:** `main`
- **Main file path:** `ui/streamlit_app.py`
- (Optional) **Advanced settings → Python version 3.11**. No secrets are required.
- Click **Deploy**. Streamlit installs `requirements.txt` and boots the app; the first
  build takes a couple of minutes.

That's it — you get a `https://<your-app>.streamlit.app` link. The corpus (`data/`) ships
in the repo, so search works immediately; the sidebar health chip will read
`API up · vX.Y.Z · keyless=True` even though it's all one process.

**Notes**
- **No configuration needed.** With no `API_URL` reachable, the UI auto-selects the embedded
  backend. To make that explicit (and skip the one-time health probe), add an env var / secret
  `ARA_EMBEDDED=1` under *Advanced settings*.
- **Ephemeral history.** Runs are written to the container's temp disk, so the
  History/Observability pages show runs from the **current** app lifetime; a reboot or redeploy
  starts fresh. That's expected on a free single-container host (same as Cloud Run).
- **Sleeps when idle.** Free apps go to sleep after inactivity and wake on the next visit
  (a few seconds). Fine for a portfolio demo.
- **~1 GB RAM.** This keyless app fits comfortably; there's no model to load.
- **Real mode:** if you ever want a real LLM/web search, add `OPENAI_API_KEY` etc. as
  **Secrets** (never commit them) — see [§8](#8-optional-real-mode--not-free). Not needed for the demo.

---

## Option A — GCP Cloud Run (serverless, always-free, scales to zero)

Two Cloud Run services from the same source: `ara-api` and `ara-ui`. Idle = **$0**
(scales to zero). History is per-instance and not persisted (fine for a demo).

```bash
# 1. Install the gcloud CLI, then:
gcloud auth login
gcloud config set project YOUR_PROJECT_ID
gcloud services enable run.googleapis.com cloudbuild.googleapis.com artifactregistry.googleapis.com

# 2. Deploy the API (built from the Dockerfile by Cloud Build; no local Docker needed).
#    Run this from the repo root.
gcloud run deploy ara-api \
  --source . \
  --region us-central1 \
  --port 8000 \
  --memory 512Mi --cpu 1 \
  --min-instances 0 --max-instances 3 \
  --allow-unauthenticated

# 3. Grab the API URL it prints, e.g. https://ara-api-xxxx-uc.a.run.app
API_URL=$(gcloud run services describe ara-api --region us-central1 --format 'value(status.url)')
echo "$API_URL"

# 4. Reuse the image Cloud Build just produced — deploying from the SAME image (rather
#    than a second --source build) keeps you under the 0.5 GB Artifact Registry free
#    limit. Override the command to run Streamlit on Cloud Run's port (8080) and point
#    it at the API. Session affinity + disabling XSRF/CORS make Streamlit's websockets
#    work behind the proxy.
IMAGE=$(gcloud run services describe ara-api --region us-central1 \
  --format 'value(spec.template.spec.containers[0].image)')

gcloud run deploy ara-ui \
  --image "$IMAGE" \
  --region us-central1 \
  --port 8080 \
  --memory 512Mi --cpu 1 \
  --min-instances 0 --max-instances 3 \
  --session-affinity \
  --allow-unauthenticated \
  --set-env-vars "API_URL=${API_URL}" \
  --command streamlit \
  --args "run,ui/streamlit_app.py,--server.port=8080,--server.address=0.0.0.0,--server.headless=true,--server.enableCORS=false,--server.enableXsrfProtection=false"
```

Open the `ara-ui` URL it prints — that's your live app. Swagger for the API is at `${API_URL}/docs`.

**Notes**
- **Staying free:** 2M requests/month is far more than a demo needs, and `--min-instances 0`
  means you pay nothing while idle. Keep both services in one region.
- **Ephemeral history:** each cold start is a fresh container, so the History/Observability
  pages only show runs from the current instance's lifetime. That's expected on serverless.
  Want persistent history? Use a VM (Option B/C).
- **Streamlit blank / "connecting…"?** Make sure `--session-affinity` is set and the port is
  8080 — that's the usual fix behind Cloud Run.

---

## Option B — GCP Compute Engine e2-micro VM + docker-compose (always-free)

Runs the **full stack** (API + UI) persistently on the one always-free micro VM.

```bash
# 1. Create the always-free VM (region MUST be us-central1 / us-west1 / us-east1).
gcloud compute instances create ara-vm \
  --machine-type e2-micro \
  --zone us-central1-a \
  --image-family debian-12 --image-project debian-cloud \
  --boot-disk-size 30GB

# 2. Open the two app ports (tighten 0.0.0.0/0 to YOUR.IP/32 if you want it private).
gcloud compute firewall-rules create ara-ports \
  --allow tcp:8000,tcp:8501 --source-ranges 0.0.0.0/0 --target-tags http-server
gcloud compute instances add-tags ara-vm --zone us-central1-a --tags http-server

# 3. Get the code onto the VM — either clone your repo, or copy it up:
gcloud compute scp --recurse . ara-vm:~/ara --zone us-central1-a   # from the repo root

# 4. SSH in and run it.
gcloud compute ssh ara-vm --zone us-central1-a
```

Then **on the VM**:

```bash
# install Docker (engine + compose plugin) via the official convenience script
curl -fsSL https://get.docker.com | sudo sh
sudo usermod -aG docker $USER && newgrp docker

# e2-micro has only 1 GB RAM — add 1 GB swap so the build/UI don't OOM
sudo fallocate -l 1G /swapfile && sudo chmod 600 /swapfile && sudo mkswap /swapfile && sudo swapon /swapfile

cd ~/ara
docker compose up -d --build          # builds the image, starts api + ui
docker compose ps
```

Open **`http://EXTERNAL_IP:8501`** (find the IP with `gcloud compute instances list`).

**Notes**
- **Free:** the e2-micro is always-free in those three regions only; 30 GB disk and 1 GB
  egress/mo are also free.
- **RAM is tight (1 GB).** This light app fits with the swap above; if the build is slow,
  build the image once and `docker compose up -d` without `--build` afterwards.
- **Trial upgrade:** during your $300 trial you can use `e2-small` (2 GB) instead — just
  remember it's *not* always-free, so switch back to `e2-micro` before day 90.

---

## Option C — Oracle Cloud Ampere A1 VM + docker-compose (free forever, recommended)

The most generous free option: an **Arm Ampere A1 with 2 OCPU / 12 GB RAM**, free forever.
Runs the full stack comfortably.

**1. Create the instance (Console → Compute → Instances → Create)**
- **Shape:** *Ampere* → `VM.Standard.A1.Flex`, set **2 OCPUs / 12 GB** (the free cap).
- **Image:** Ubuntu 22.04 (or Oracle Linux). The image `python:3.11-slim` is multi-arch, so
  it builds natively on Arm — nothing special needed.
- **Networking:** assign a public IPv4; save the SSH key.
- If you see *"Out of host capacity"*, retry, or pick a different Availability Domain / region.

**2. Open the ports — Oracle needs BOTH the cloud firewall AND the OS firewall:**
- **Security List / NSG** (Console → VCN → Security Lists): add **Ingress** rules for
  TCP **8000** and **8501** from `0.0.0.0/0` (or your IP).
- **On the VM**, Oracle images also block ports at the OS level:

```bash
# Ubuntu image (iptables):
sudo iptables -I INPUT 6 -m state --state NEW -p tcp --dport 8000 -j ACCEPT
sudo iptables -I INPUT 6 -m state --state NEW -p tcp --dport 8501 -j ACCEPT
sudo netfilter-persistent save
# (Oracle Linux uses firewalld instead:)
# sudo firewall-cmd --permanent --add-port=8000/tcp --add-port=8501/tcp && sudo firewall-cmd --reload
```

**3. Copy the code up and run it** (from your machine, then on the VM):

```bash
scp -i your_key.pem -r . ubuntu@PUBLIC_IP:~/ara        # or: git clone your repo on the VM
ssh -i your_key.pem ubuntu@PUBLIC_IP
```

On the VM:

```bash
curl -fsSL https://get.docker.com | sudo sh          # engine + compose plugin
sudo usermod -aG docker $USER && newgrp docker
cd ~/ara
docker compose up -d --build
```

Open **`http://PUBLIC_IP:8501`**.

**Notes**
- **Free forever**, but Oracle may **reclaim Always-Free compute that stays idle ~7 days** —
  keeping the app running and occasionally used avoids that.
- 2 OCPU / 12 GB is ample; no swap needed. 10 TB/mo egress is effectively unlimited for a demo.

---

## 5. GCP Free Trial (your next 3 months)

You have the **$300 / 90-day** trial. What it changes:
- It **removes free-tier limits/regions** — deploy Option A or B anywhere, any size, and the
  credit absorbs anything beyond the always-free amounts (which, for this tiny app, is ~nothing).
- **You will not be auto-charged.** The trial account closes when the 90 days end *or* the $300
  is spent, and you're billed **only if you manually click "Upgrade"**. There's a 30-day grace
  period afterward to recover resources if you do upgrade.

**Recommended play:** run **Option A (Cloud Run)** now — it's essentially free even off-trial —
or spin up a comfortable **`e2-small`** VM (Option B) during the trial. **Before day 90**,
either (a) let the trial lapse and redeploy on the **always-free e2-micro** or **Oracle Ampere A1**,
or (b) if you upgrade, add a **$1 budget alert** (below) so nothing surprises you.

---

## 6. Staying at exactly $0 (cost guardrails)

- **Budget alert (GCP):** Billing → Budgets & alerts → create a budget of **$1** with email
  alerts at 50/90/100%. Cheap insurance.
- **Cloud Run:** keep `--min-instances 0` (already set) so idle = $0.
- **VMs:** `gcloud compute instances stop ara-vm` (GCP) or stop the instance in the OCI console
  when you don't need it; a *stopped* VM's disk is still within the free disk allowance.
- **Stay in free regions/shapes:** GCP always-free compute is only `us-central1/us-west1/us-east1`
  + `e2-micro`; Oracle Always-Free is the A1 (≤2 OCPU/12 GB) and E2.1.Micro shapes only.
- **Egress:** this app serves tiny JSON/HTML, so you'll never approach the 1 GB (GCP) / 10 TB (OCI) limits.

---

## 7. Security notes

The app is **keyless and safe by default** — it makes no outbound calls and stores no secrets —
but a public URL means **anyone can use it and rack up (free-tier) usage**. For a portfolio demo
that's usually fine. To lock it down:
- **Restrict the firewall** to your own IP (`YOUR.IP/32`) on the VM options.
- **Cloud Run:** drop `--allow-unauthenticated` and require IAM auth, or put it behind
  Identity-Aware Proxy.
- **Never bake keys into the image.** If you enable *real mode* (below), inject secrets at
  runtime via env vars / Secret Manager, not into the Dockerfile.

---

## 8. (Optional) Real mode — not free

To use a real LLM + web search instead of the keyless fakes, set these env vars on the
container/VM (this **costs money** — you pay OpenAI/Tavily, not GCP/Oracle):

```
LLM_PROVIDER=openai      OPENAI_API_KEY=sk-...
SEARCH_PROVIDER=web      SEARCH_API_KEY=...        # Tavily
FETCH_PROVIDER=http
```

Provide them via `--set-env-vars` (Cloud Run) or an `.env` on the VM (never commit it).
The settings layer validates provider combinations at startup, so a half-configured real
mode fails fast with a clear message rather than mid-run.

---

## 9. Teardown (remove everything)

```bash
# GCP Cloud Run
gcloud run services delete ara-api ara-ui --region us-central1

# GCP VM + firewall
gcloud compute instances delete ara-vm --zone us-central1-a
gcloud compute firewall-rules delete ara-ports

# Oracle: terminate the instance in the Console (Compute → Instances → … → Terminate),
# and delete the VCN if you created a dedicated one.
```

---

## 10. Troubleshooting

| Symptom | Fix |
|---|---|
| Streamlit Cloud: `ModuleNotFoundError` on boot | Confirm **Main file path** is `ui/streamlit_app.py` and the app deploys from the repo **root** — the UI adds `src/` to the path itself; a wrong root breaks the embedded import. |
| Streamlit Cloud: pages error with "connection refused" | It's trying to reach an API. Set secret/env `ARA_EMBEDDED=1` to force the in-process backend (it normally auto-detects). |
| Streamlit Cloud: app is slow on first open | It was asleep (free tier) — the first visit wakes it in a few seconds; subsequent loads are fast. |
| Cloud Run UI is blank / "connecting…" forever | Ensure `--session-affinity` is set and the UI listens on port **8080**; keep `--server.enableXsrfProtection=false`. |
| `gcloud run deploy --source` uploads for ages | Confirm `.dockerignore` exists (it excludes `.venv/`, `runs/`, `docs/`) so the context is small. |
| Oracle: page won't load though the app is running | You opened the **Security List** but not the **OS firewall** (iptables/firewalld) — do both (Option C step 2). |
| Oracle: "Out of host capacity" creating the A1 | Retry, or choose a different Availability Domain / home region. |
| GCP e2-micro build killed / OOM | Add the 1 GB swap (Option B step), or build the image once then `up -d` without `--build`. |
| UI can't reach the API | Check `API_URL` — `http://api:8000` inside docker-compose; the API's public URL on Cloud Run. |

---

*Fastest free demo link, zero setup: **Option S (Streamlit Community Cloud)** — one app,
no Docker, no card. Simplest path to free-forever with a separate API + UI:
**Option C (Oracle Ampere A1)**. Zero-idle-cost public link with a real API:
**Option A (GCP Cloud Run)**. Use the GCP trial to experiment freely for 3 months, then land
on one of those.*
