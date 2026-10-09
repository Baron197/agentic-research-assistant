# Deployment Guide — free hosting (Streamlit Cloud, GCP, Oracle Cloud & AWS)

This guide deploys the **Agentic Research & Report Assistant** to the cloud for **$0**.
The app is ideal for a free tier: it's a small, keyless Python stack with **no database,
no API keys, and no GPU** — the whole thing runs offline on deterministic fake providers.

It covers two different goals:
- **A public keyless demo** (Options S, A, B, C below) — anyone can open it; it costs nothing to run.
- **A private real-mode work tool** ([§8](#8-real-mode--privately-on-0-infrastructure)) — real
  OpenAI + live web search on GCP or Oracle **free-tier infrastructure** (or AWS on its 6-month
  Free plan), reachable only by you.
  The only bill is your OpenAI usage (about **$0.001 per research run** with `gpt-4o-mini`).

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
| **T** | Anything | GCP **Free Trial** | 💳 $300 / 90 days | either | — | Experimenting while a trial is active — then migrate to A or C |
| **R** | **Real mode, private** | GCP, Oracle or AWS | ✅ infra $0 · you pay OpenAI (AWS: 6 months) | depends | Medium | **Using it as your own work tool** → [§8](#8-real-mode--privately-on-0-infrastructure) |

**Recommendations**
- **Just want a free demo link with the least effort?** → **Option S (Streamlit Community Cloud)** — point it at the repo and click Deploy; no Docker, no CLI, no card. Same as how you deployed your RAG app.
- **Want it free forever with a real separate API + UI?** → **Option C (Oracle Ampere A1)** — by far the most RAM headroom.
- **Want a scale-to-zero public link that costs nothing when nobody's using it?** → **Option A (GCP Cloud Run)**.
- **Want to use real OpenAI + live web search for your own work, without paying for servers?** → **[§8](#8-real-mode--privately-on-0-infrastructure)** — private deployments on the same free tiers.
- **Want AWS experience (and on your CV)?** → **[§8 R4](#r4--aws-ec2--ssh-tunnel-free-plan-6-months)** — EC2 on AWS's Free plan: $0 for 6 months, not forever.
- **Have an active GCP $300 trial?** → Use **A or B freely** (the credit covers any overage and unlocks any region/size), then **migrate to A or C before it ends** to stay at $0. See [§5](#5-gcp-free-trial).

Current free-tier facts used below (**re-verified September 2026** against the providers' own
pages — always re-check, they change):
- **Streamlit Community Cloud**: free public apps from a public GitHub repo, ~1 GB RAM, installs from `requirements.txt`; apps **sleep after inactivity** and wake on the next visit (a few seconds). No credit card.
- **GCP Cloud Run** always-free (request-based billing): 2M requests, **180,000 vCPU-seconds** and 360,000 GiB-seconds per month; scales to zero. **Cloud Build**: 2,500 build-minutes/month. **Artifact Registry**: 0.5 GB storage. **Secret Manager**: 6 active secret versions + 10,000 access operations/month.
- **GCP e2-micro** always-free: 1 instance/month in `us-central1` / `us-west1` / `us-east1`, 1 GB RAM, **30 GB *standard* persistent disk** (other disk types are billed), 1 GB egress/month from North America.
- **Oracle Always Free**: Ampere A1 = **2 OCPU / 12 GB** (1,500 OCPU-hours + 9,000 GB-hours/month), or 2× AMD E2.1.Micro (1 GB each); 200 GB block storage; 10 TB/month egress. Idle Always-Free instances **may be reclaimed** (see [§8](#8-real-mode--privately-on-0-infrastructure)).
- **GCP Free Trial**: $300 credit, 90 days, **no automatic charges** — the account closes at 90 days or $300 and you're only billed if you *manually* upgrade.
- **Oracle Free Trial**: $300 credit for 30 days; afterwards paid resources are reclaimed, Always-Free resources keep running, and nothing is charged unless you upgrade.
- **AWS Free plan** (accounts opened since 15 July 2025; verified October 2026): $100 credit at sign-up + up to $100 more from *Explore AWS* activities, for **6 months**. Free-plan instance types are `t3.micro`, `t3.small`, `t4g.micro`, `t4g.small`, `c7i-flex.large` and `m7i-flex.large`, paid **from the credits** — no free hours. A public IPv4 address is $0.005/hour. The account can't be charged on the Free plan; it is suspended when the credits or the 6 months run out, and erased 90 days later unless upgraded.

---

## 0. Prerequisites

- This repository (the folder containing `Dockerfile` and `docker-compose.yml`).
- A Google account (Options A/B/T) and/or an Oracle Cloud account (Option C).
- For Options B/C you'll get the code onto a VM with **`git clone`** of the public repo.
  Don't `scp` your working folder instead — it would also upload your local `.env` (real
  keys) and `.venv/`.
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
in the repo, so search works immediately; the sidebar's mode box will read **Keyless demo** and
`In-process backend`, because it's all one process.

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
- **Real mode:** not needed for the demo, and a public Streamlit app has no login — anyone with
  the link could spend your OpenAI credit. For real mode use a private deployment from
  [§8](#8-real-mode--privately-on-0-infrastructure) instead.

---

## Option A — GCP Cloud Run (serverless, always-free, scales to zero)

Two Cloud Run services from the same source: `ara-api` and `ara-ui`. Idle = **$0**
(scales to zero). History is per-instance and not persisted (fine for a demo).

```bash
# 1. Install the gcloud CLI, then:
gcloud auth login
gcloud config set project YOUR_PROJECT_ID
gcloud services enable run.googleapis.com cloudbuild.googleapis.com \
  artifactregistry.googleapis.com compute.googleapis.com

# 1b. Let Cloud Build build from source. Cloud Build runs as the Compute Engine default
#     service account, and projects created after May 2024 no longer grant it this role
#     automatically — without it the first `--source` deploy fails with PERMISSION_DENIED.
PROJECT_NUMBER=$(gcloud projects describe "$(gcloud config get-value project)" --format='value(projectNumber)')
gcloud projects add-iam-policy-binding "$(gcloud config get-value project)" \
  --member="serviceAccount:${PROJECT_NUMBER}-compute@developer.gserviceaccount.com" \
  --role=roles/run.builder

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
#    --boot-disk-type pd-standard matters: only STANDARD persistent disk is in the free
#    tier (the Console usually defaults to "balanced", which is billed).
gcloud compute instances create ara-vm \
  --machine-type e2-micro \
  --zone us-central1-a \
  --image-family debian-12 --image-project debian-cloud \
  --boot-disk-size 30GB --boot-disk-type pd-standard

# 2. Open the two app ports (tighten 0.0.0.0/0 to YOUR.IP/32 if you want it private).
#    Keyless demo only — for REAL mode, skip this step entirely (see §8).
gcloud compute firewall-rules create ara-ports \
  --allow tcp:8000,tcp:8501 --source-ranges 0.0.0.0/0 --target-tags http-server
gcloud compute instances add-tags ara-vm --zone us-central1-a --tags http-server

# 3. SSH in.
gcloud compute ssh ara-vm --zone us-central1-a
```

Then **on the VM**:

```bash
# get the code with git — it copies only committed files. Do NOT `scp` your working
# folder: that would also upload your local .env (real keys) and the huge .venv/.
sudo apt-get update && sudo apt-get install -y git
git clone https://github.com/Baron197/agentic-research-assistant.git ~/ara

# install Docker (engine + compose plugin) via the official convenience script
curl -fsSL https://get.docker.com | sudo sh
sudo usermod -aG docker $USER

# e2-micro has only 1 GB RAM — add 1 GB swap so the build/UI don't OOM
sudo fallocate -l 1G /swapfile && sudo chmod 600 /swapfile && sudo mkswap /swapfile && sudo swapon /swapfile
exit      # log out so the docker group applies on your next login
```

SSH back in (`gcloud compute ssh ara-vm --zone us-central1-a`) and start it:

```bash
cd ~/ara
docker compose up -d --build          # builds the image, starts api + ui (keyless: no .env)
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

**0. Create your Oracle Cloud account** (once). Sign up for the **Free Tier** at
[oracle.com/cloud/free](https://www.oracle.com/cloud/free/). It asks for a card to verify your
identity; Always-Free resources aren't charged unless you upgrade. **Choose your home region
carefully — it can't be changed later**, and Always-Free compute runs only there, so pick the one
closest to you. Provisioning the account can take a few minutes.

**1. Create the instance (Console → Compute → Instances → Create instance)**
- **Name:** e.g. `ara-vm`.
- **Image and shape → Change shape → Ampere →** `VM.Standard.A1.Flex`, set **2 OCPUs / 12 GB**
  (the free cap). The shape should be marked *Always Free-eligible*.
- **Change image → Canonical Ubuntu 24.04** (the plain one, not *Minimal*; 22.04 leaves standard
  support in April 2027). With an Ampere shape you need the **aarch64 (Arm)**
  build; if the Console says the image isn't compatible, re-select Ubuntu *after* choosing the shape.
  (The app's base image `python:3.11-slim` is multi-arch and every dependency ships Arm wheels, so it
  builds natively — nothing special needed.)
- **Networking:** a new account has no network yet — choose **Create new virtual cloud network**
  and **Create new public subnet**, and turn on **Automatically assign public IPv4 address**. If
  that toggle won't switch on, create the instance anyway and add the IP afterwards: instance →
  Networking → primary VNIC → IPv4 addresses → ⋮ → Edit → **Ephemeral public IP**.
- **Add SSH keys → Generate a key pair for me → Save private key.** Keep the downloaded `.key` file
  safe (e.g. in `C:\Users\you\.ssh\`) — it can't be downloaded again. The login user is `ubuntu`.
- **Boot volume:** the default (~47 GB) fits inside the 200 GB free allowance.
- If you see *"Out of host capacity"*, retry later, pick a different Availability Domain (if your
  region has several), or try a smaller **1 OCPU / 6 GB** shape — free A1 capacity is often tight
  (see [§8 R2](#r2--oracle-cloud-ampere-a1-vm--ssh-tunnel) on Pay As You Go). Retry slowly: rapid
  attempts get *"Too many requests"*.

**2. Open the ports — Oracle needs BOTH the cloud firewall AND the OS firewall**
*(keyless demo only — for real mode, skip this step entirely; see [§8](#8-real-mode--privately-on-0-infrastructure))*:
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

**3. SSH in, get the code, and run it:**

```bash
ssh -i your_key.key ubuntu@PUBLIC_IP
```

On the VM:

```bash
# git copies only committed files — never scp your working folder (it would upload
# your local .env with real keys, plus .venv/).
sudo apt-get update && sudo apt-get install -y git
git clone https://github.com/Baron197/agentic-research-assistant.git ~/ara

curl -fsSL https://get.docker.com | sudo sh          # engine + compose plugin
sudo usermod -aG docker $USER
exit                                                 # log out so the docker group applies
```

SSH back in and start it:

```bash
cd ~/ara
docker compose up -d --build                         # keyless: no .env on the VM
```

Open **`http://PUBLIC_IP:8501`**.

**Notes**
- **Free forever**, but Oracle may **reclaim Always-Free compute that stays idle ~7 days** —
  keeping the app running and occasionally used avoids that.
- 2 OCPU / 12 GB is ample; no swap needed. 10 TB/mo egress is effectively unlimited for a demo.

---

## 5. GCP Free Trial

If you have an active **$300 / 90-day** trial, here's what it changes:
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
  + `e2-micro` + a **standard** boot disk; Oracle Always-Free is the A1 (≤2 OCPU/12 GB) and
  E2.1.Micro shapes only.
- **Artifact Registry (Cloud Run):** every `gcloud run deploy --source` stores another image, and
  storage above **0.5 GB** is billed. Keep only the newest two per service with a cleanup policy
  (Keep rules beat Delete rules; `--no-dry-run` makes it real — the default only reports):

  ```bash
  cat > cleanup.json <<'EOF'
  [
    {"name": "delete-old",    "action": {"type": "Delete"}, "condition": {"tagState": "any"}},
    {"name": "keep-latest-2", "action": {"type": "Keep"},   "mostRecentVersions": {"keepCount": 2}}
  ]
  EOF
  gcloud artifacts repositories set-cleanup-policies cloud-run-source-deploy \
    --location=us-central1 --policy=cleanup.json --no-dry-run
  ```
  Cleanup runs in the background, roughly once a day.
- **Egress:** this app serves tiny JSON/HTML, so you'll never approach the 1 GB (GCP) / 10 TB (OCI) limits.
- **Real mode:** the infrastructure stays free; see the cost table in [§8](#8-real-mode--privately-on-0-infrastructure)
  for what OpenAI and Tavily usage costs.

---

## 7. Security notes

The app is **keyless and safe by default** — it makes no outbound calls and stores no secrets —
but a public URL means **anyone can use it and rack up (free-tier) usage**. For a keyless
portfolio demo that's usually fine. To lock it down:
- **Restrict the firewall** to your own IP (`YOUR.IP/32`) on the VM options.
- **Cloud Run:** drop `--allow-unauthenticated` and require IAM auth (see [§8 R1](#r1--gcp-cloud-run-private-recommended)
  for how to then open it).
- **Never bake keys into the image.** Real mode injects secrets at runtime (Secret Manager or a
  gitignored `.env` on the VM) — see [§8](#8-real-mode--privately-on-0-infrastructure).

**The app has no login of its own.** That is fine for the keyless demo and **not fine for real
mode**, where every run spends your OpenAI credit. Never put a real-mode deployment on an open URL.

---

## 8. Real mode — privately, on $0 infrastructure

Everything above deploys the **keyless** demo. This section runs **real mode** (OpenAI + live web
search) as a **private work tool**: the servers stay inside the free tiers, and the only bill is your
OpenAI usage (plus Tavily beyond its free plan). Four options — pick one (R4 is free for six
months only).

> **Two rules for every option**
> 1. **Private, never public.** There is no app login, and every run spends *your* credit. Each
>    option below is reachable only by you: an authenticated proxy on Cloud Run, or **no open ports
>    at all** plus an SSH tunnel on a VM.
> 2. **Keys at runtime only.** Secret Manager on Cloud Run; a gitignored `.env` on the VM. The
>    public repo and the Docker image stay keyless.

### What it costs (measured on real runs, `gpt-4o-mini`)

| | Per research run | Free allowance | Runs/month for $0 |
|---|---|---|---|
| **OpenAI** | **≈ $0.001** ($0.0007–$0.0015 measured) | your prepaid credit | ~700–1,400 runs per **$1** |
| **Tavily** search | **5 searches** = 5 credits | 1,000 credits/month | **~200** |
| **Infrastructure** | — | the free tiers below | $0 |

### Before you start (all four options)
- **OpenAI → Settings → Billing: turn auto-recharge OFF.** OpenAI's monthly *budgets* only send
  alert emails — they don't stop requests. With prepaid credit and auto-recharge off, the most you
  can ever spend is the balance you already have. That is your real hard limit.
- **Tavily:** the free plan (1,000 credits/month) needs no card — key from app.tavily.com.
- **`TOKEN_BUDGET`** (e.g. `40000`) caps tokens per run inside the app; runs that hit it end
  `partial` instead of spending more. In real mode it is a ceiling, as is `MAX_ITERATIONS`: the
  UI's sidebar and API requests can lower them, never raise them. A typical real run uses about
  20k tokens, including ~11k cheap embedding tokens for picking evidence passages.
- **One OpenAI key per deployment.** Create a separate key for each place you deploy (e.g. named
  `ara-cloudrun`, `ara-oracle`, `ara-aws`) so you can revoke one without breaking the others, and see which
  deployment is spending.

| | **R1 — GCP Cloud Run** | **R2 — Oracle A1 VM** | **R3 — GCP e2-micro VM** | **R4 — AWS EC2** |
|---|---|---|---|---|
| How you open it | `gcloud run services proxy` → `localhost:8501` | SSH tunnel → `localhost:8501` | SSH tunnel → `localhost:8501` | SSH tunnel → `localhost:8501` |
| Infrastructure cost | $0 in the Cloud Run free tier (mind open tabs, below) | $0 (Always Free) | $0 (Always Free) | $0 for **6 months** (Free-plan credits) |
| Run history kept | ❌ resets when it scales to zero | ✅ | ✅ | ✅ |
| Memory | 1 GiB | 12 GB | 1 GB + swap | 2 GB (`t4g.small`) |
| Upkeep | none — nothing to patch | OS updates; **idle-reclaim risk** | OS updates | OS updates; **expires after 6 months** |
| Choose it when | **you want zero maintenance (recommended)** | you want always-on + history, lots of RAM | you want always-on on GCP | you want AWS experience |

---

### R1 — GCP Cloud Run, private (recommended)

One Cloud Run service runs the whole app — the Streamlit UI with the **embedded** backend
(`ARA_EMBEDDED=1`), so there's no separate API to secure. Keys come from **Secret Manager**, the
service **rejects unauthenticated requests**, and you open it through `gcloud run services proxy`,
which attaches your Google identity. Works from any machine where you're logged in to `gcloud`.

**Where to run the commands.** Steps 1–4 are **bash**. Run them in **Cloud Shell** — the
`>_` icon at the top of the Google Cloud Console opens a browser terminal with `gcloud` and `git`
preinstalled, already signed in as your Console account. On Windows this is the way to go: PowerShell
can't run bash syntax, and Git Bash often can't start `gcloud` (*"Python was not found"*). Only
Step 5 runs on your own laptop.

**0. A project with billing, and the right Google account.** Use the Google account that has the
open billing account (the trial, or a paid one). Console → project picker → **New project** → give
it a name and pick that **billing account** — a project without billing fails at Step 1. Note the
**project ID** it shows (e.g. `agentic-research-123456`); it's your `YOUR_PROJECT_ID` below.
Then, in Cloud Shell:
```bash
git clone https://github.com/Baron197/agentic-research-assistant.git
cd agentic-research-assistant
gcloud config set project YOUR_PROJECT_ID
```

**1. Enable the APIs and let Cloud Build build**
```bash
gcloud services enable run.googleapis.com cloudbuild.googleapis.com \
  artifactregistry.googleapis.com secretmanager.googleapis.com compute.googleapis.com

PROJECT_NUMBER=$(gcloud projects describe "$(gcloud config get-value project)" --format='value(projectNumber)')
SA="${PROJECT_NUMBER}-compute@developer.gserviceaccount.com"   # builds AND runs the service

gcloud projects add-iam-policy-binding "$(gcloud config get-value project)" \
  --member="serviceAccount:${SA}" --role=roles/run.builder
```

**2. Store the keys in Secret Manager.** Easiest: Console → **Security → Secret Manager → Create
secret**, name `openai-key`, paste the key; repeat for `tavily-key`. Or in the shell:
```bash
read -rs KEY && printf '%s' "$KEY" | gcloud secrets create openai-key --data-file=- && unset KEY
read -rs KEY && printf '%s' "$KEY" | gcloud secrets create tavily-key --data-file=- && unset KEY
```
`read -rs` hides what you paste and keeps it out of shell history. Avoid `echo` / PowerShell pipes
for this: they append a newline (the app strips surrounding whitespace from keys, but other tools
won't).

**3. Let the service read the secrets** (the first two lines recompute `SA`, so this still works
if Cloud Shell reconnected while you were adding the keys):
```bash
PROJECT_NUMBER=$(gcloud projects describe "$(gcloud config get-value project)" --format='value(projectNumber)')
SA="${PROJECT_NUMBER}-compute@developer.gserviceaccount.com"
for s in openai-key tavily-key; do
  gcloud secrets add-iam-policy-binding "$s" \
    --member="serviceAccount:${SA}" --role=roles/secretmanager.secretAccessor
done
```

**4. Deploy** (Cloud Build builds the repo's `Dockerfile`; answer **Y** if asked to create the
`cloud-run-source-deploy` repository):
```bash
gcloud run deploy ara-real \
  --source . \
  --region us-central1 \
  --no-allow-unauthenticated \
  --port 8080 --memory 1Gi --cpu 1 \
  --min-instances 0 --max-instances 1 \
  --timeout 3600 --session-affinity \
  --set-env-vars "LLM_PROVIDER=openai,SEARCH_PROVIDER=web,FETCH_PROVIDER=http,ARA_EMBEDDED=1,TOKEN_BUDGET=40000" \
  --set-secrets "OPENAI_API_KEY=openai-key:latest,SEARCH_API_KEY=tavily-key:latest" \
  --command streamlit \
  --args "run,ui/streamlit_app.py,--server.port=8080,--server.address=0.0.0.0,--server.headless=true,--server.enableCORS=false,--server.enableXsrfProtection=false"
```
Why these flags:
- `--no-allow-unauthenticated` — strangers get **403**; nobody can spend your credit.
- `--max-instances 1` — a hard ceiling on how much can ever run at once.
- `--timeout 3600` — Streamlit keeps a websocket open; the 300 s default would cut it every 5 minutes.
- `--memory 1Gi` — free-tier *vCPU*-seconds run out long before GiB-seconds do, so the extra memory costs nothing.

**5. Open it — from your laptop** (needs the [gcloud CLI](https://cloud.google.com/sdk/docs/install);
on Windows run this in **PowerShell**). Your laptop's `gcloud` has its own settings, separate from
Cloud Shell, so name the account and project explicitly:
```bash
gcloud auth login                  # once — sign in as the account that owns the project
gcloud run services proxy ara-real --region us-central1 --port 8501 --project YOUR_PROJECT_ID
```
Leave that running and open **http://localhost:8501**. The sidebar's mode box should read
**Real mode · gpt-4o-mini** and *live web search*. `Ctrl+C` closes it. As the project owner you already hold `run.invoker`; to let
someone else in, grant them `roles/run.invoker` on the service. (If you have several Google accounts
in `gcloud`, `gcloud auth list` shows which is active; add `--account you@gmail.com` to pick one.)

> **Open tabs cost free-tier time.** A Streamlit tab keeps a websocket open, and Cloud Run counts
> an open websocket as an active request. The free tier's 180,000 vCPU-seconds ≈ **50 hours** of an
> open tab per month; past that, roughly **$0.09 per hour**. Close the tab (or stop the proxy) when
> you're done — a research run itself takes only ~30 s.

> **Want to open it on your phone, with a Google sign-in instead of gcloud?** Enable
> **Identity-Aware Proxy** on the service (GA, no load balancer needed). Personal Gmail projects —
> those not inside a Google Workspace organization — must first do Google's one-time
> [custom OAuth client setup](https://docs.cloud.google.com/run/docs/securing/identity-aware-proxy-cloud-run).
> Then grant yourself `roles/iap.httpsResourceAccessor` and give the IAP service agent
> (`service-PROJECT_NUMBER@gcp-sa-iap.iam.gserviceaccount.com`) `roles/run.invoker`.

**Cost hygiene:** add the Artifact Registry cleanup policy from [§6](#6-staying-at-exactly-0-cost-guardrails)
— each redeploy stores another image. History is ephemeral (it resets when the service scales to
zero), so use the **Download** buttons for reports you want to keep.

---

### R2 — Oracle Cloud Ampere A1 VM + SSH tunnel

The most room for free (2 OCPU / 12 GB), always on, with **persistent history**. Nothing is exposed
to the internet: the app listens only on the VM's loopback, and you reach it through SSH.

**1. Create the VM** exactly as in [Option C](#option-c--oracle-cloud-ampere-a1-vm--docker-compose-free-forever-recommended)
step 1 (Ubuntu, `VM.Standard.A1.Flex`, 2 OCPU / 12 GB) — but **skip Option C step 2 entirely**:
don't add Security List rules for 8000/8501 and don't touch iptables. The default Security List
only allows SSH (port 22), which is all the tunnel needs.

**2. On the VM — install Docker.** Connect first (Windows PowerShell has `ssh` built in):
`ssh -i C:\path\to\your_oracle.key ubuntu@PUBLIC_IP`
```bash
sudo apt-get update && sudo apt-get install -y git
curl -fsSL https://get.docker.com | sudo sh
sudo usermod -aG docker $USER
exit
```
The `exit` is deliberate: **SSH back in** so your user picks up the `docker` group (more reliable
than `newgrp`, which opens a sub-shell that can swallow the rest of a paste). Then get the code:
```bash
git clone https://github.com/Baron197/agentic-research-assistant.git ~/ara && cd ~/ara
docker compose version        # must be v2.24 or newer (needed for env_file's "required:")
```

**Then create the `.env` with your keys** — run `nano .env`, paste this, and save with
`Ctrl+O`, `Enter`, `Ctrl+X`. (An editor rather than `cat <<EOF` keeps the keys out of your shell
history. The file lives only on the VM — it's gitignored and dockerignored.)
```
LLM_PROVIDER=openai
SEARCH_PROVIDER=web
FETCH_PROVIDER=http
OPENAI_API_KEY=sk-...
SEARCH_API_KEY=tvly-...
TOKEN_BUDGET=40000
ARA_BIND=127.0.0.1
```

**Then lock it down and start:**
```bash
chmod 600 .env                # readable by you only
docker compose up -d --build
```
- `docker-compose.yml` passes `.env` to the API container (`env_file`), which is what switches it
  to real mode. The UI container gets no keys — it only talks to the API.
- **`ARA_BIND=127.0.0.1` is the important line.** It publishes ports 8000/8501 on the VM's
  loopback only. Docker-published ports **bypass the host firewall** (iptables/ufw), so binding to
  loopback is what actually keeps the app private — even if a port gets opened by mistake later.
- The image includes `trafilatura`, which pulls the article text out of each fetched page.
  Without it, evidence snippets are mostly the page's CSS, scripts and menus, and a run costs
  several times more (see REAL_MODE.md §6).

**3. Check it's really in real mode** (still on the VM):
```bash
curl -s localhost:8000/health        # -> {"status":"ok", ..., "keyless":false}
```

**4. Open it from your laptop.** Windows PowerShell has `ssh` built in:
```powershell
ssh -i C:\path\to\your_oracle.key -N -L 8501:localhost:8501 ubuntu@PUBLIC_IP
```
Leave it running and open **http://localhost:8501**. `-N` means "tunnel only, no shell";
`Ctrl+C` closes it.

**Updating later** (new code from GitHub): run history lives in a Docker volume and both
containers restart on their own after a reboot, so an update is just:
```bash
cd ~/ara && git pull && docker compose up -d --build
```
The history survives it; only `docker compose down -v` deletes it. *Installed before October
2026?* Back then the history lived inside the API container, which this update replaces, so copy it
out first and back in afterwards:
```bash
docker compose cp api:/app/runs ~/runs-backup        # before git pull
docker compose cp ~/runs-backup/. api:/app/runs/     # after the update is up
```

> **Keeping the VM — read this.** Oracle may **reclaim idle Always-Free instances**: a VM counts
> as idle if, over 7 days, CPU (95th percentile), network **and** memory all stay under 20%. A
> private tool used a few times a day will look idle. Two honest options:
> - **Upgrade the tenancy to Pay As You Go** (Billing → Upgrade). Oracle doesn't apply idle
>   reclamation to PAYG accounts, and usage inside the Always-Free limits is still billed at
>   **$0**. It requires a card, so add a budget alert (Billing → Budgets) at $1 to catch mistakes.
>   PAYG also tends to help with *"Out of host capacity"* when creating A1 instances.
> - **Stay on Always Free** and accept that it may be reclaimed. Everything is reproducible:
>   the steps above take about ten minutes, and your keys are just the `.env`.

---

### R3 — GCP e2-micro VM + SSH tunnel (Always Free)

Always-on and persistent on GCP's always-free VM. Tighter than Oracle (1 GB RAM), but GCP doesn't
reclaim idle VMs.

**1. Create the VM** — as in [Option B](#option-b--gcp-compute-engine-e2-micro-vm--docker-compose-always-free),
but **do not create the `ara-ports` firewall rule**. The default network already allows SSH, which
is all you need:
```bash
gcloud compute instances create ara-vm \
  --machine-type e2-micro --zone us-central1-a \
  --image-family debian-12 --image-project debian-cloud \
  --boot-disk-size 30GB --boot-disk-type pd-standard
gcloud compute ssh ara-vm --zone us-central1-a
```

**2. On the VM** — add swap first (1 GB RAM), then follow R2 steps 2–3 unchanged (Docker, the
`.env`, start, health check):
```bash
sudo fallocate -l 1G /swapfile && sudo chmod 600 /swapfile && sudo mkswap /swapfile && sudo swapon /swapfile
```

**3. Open it from your laptop** (`gcloud` handles the SSH keys):
```bash
gcloud compute ssh ara-vm --zone us-central1-a --ssh-flag="-N" --ssh-flag="-L 8501:localhost:8501"
```
Then open **http://localhost:8501**.

- **Billing account:** the always-free e2-micro needs an active billing account; once a trial
  ends that means upgrading to a paid account. Usage inside the free limits is still **$0** — keep
  the $1 budget alert from [§6](#6-staying-at-exactly-0-cost-guardrails).
- **Egress:** your tunnel traffic and API calls count against the free 1 GB/month; OpenAI requests
  are a few KB each, so personal use stays well inside it.

---

### R4 — AWS EC2 + SSH tunnel (Free plan, 6 months)

Always-on with run history, on AWS — useful for learning AWS and for having it on your CV.
**It is not free forever.** A new AWS account starts on the **Free plan**: $100 in credits at
sign-up and up to $100 more from the activities in the *Explore AWS* widget on the Console Home,
for **6 months**. The instance, its disk and its public IP are paid from those credits, and a
Free-plan account can't be charged beyond them. For a permanent $0 home, use R1–R3. To let an
employer or another reviewer try it, see [the optional step at the end](#optional--let-a-reviewer-in-https--password).

**What it draws from the credits** (approximate; prices vary a little by region): a `t4g.small`
(2 vCPU Arm, 2 GB) about $12–16/month, a 20 GB gp3 disk about $2/month, and the public IPv4
address $0.005/hour (about $3.60/month) — **roughly $18–22/month**, so $200 covers the six months.
A `t4g.micro` (1 GB) halves the instance cost but needs the 1 GB swap from R3.

**0. Create the account** at [aws.amazon.com/free](https://aws.amazon.com/free/) and keep the
**Free plan** (the default for new accounts). It asks for a card to verify you; the Free plan
doesn't charge it. On the Console Home, open **Explore AWS → Earn AWS credits** for the extra $100.

**1. Launch the instance** (EC2 → Instances → **Launch instances**):
- **Region** (top right): the closest to you, e.g. Asia Pacific (Singapore) `ap-southeast-1`.
- **Name:** `ara-vm`.
- **AMI:** Ubuntu Server 24.04 LTS, architecture **64-bit (Arm)**.
- **Instance type:** `t4g.small` (marked *Free tier eligible*). If your region doesn't offer it,
  pick `t3.small` with the **64-bit (x86)** Ubuntu AMI instead — also 2 GB.
- **Key pair → Create new key pair:** any name (e.g. `ara-aws`), format **`.pem`**. It downloads
  once; move it to `C:\Users\you\.ssh\`.
- **Network settings:** default VPC, **Auto-assign public IP: Enable**, **Create security group**
  with **Allow SSH traffic from: My IP**. Leave the HTTP/HTTPS boxes **unticked** — SSH is the
  only port ever opened.
- **Configure storage:** raise the default 8 GiB to **20 GiB gp3** (Docker images need the room).
- **Launch instance**, then copy its **Public IPv4 address**.

Lock the key on Windows, or `ssh` refuses it as *unprotected* (PowerShell):
```powershell
icacls "C:\Users\you\.ssh\ara-aws.pem" /inheritance:r /grant:r "$($env:USERNAME):(R)"
```

**2. On the instance** — connect (the user is `ubuntu`), then follow
[R2 steps 2–3](#r2--oracle-cloud-ampere-a1-vm--ssh-tunnel) unchanged: Docker, the `.env` with
`ARA_BIND=127.0.0.1`, `docker compose up -d --build`, the health check.
```powershell
ssh -i C:\Users\you\.ssh\ara-aws.pem ubuntu@PUBLIC_IP
```

**3. Open it from your laptop:**
```powershell
ssh -i C:\Users\you\.ssh\ara-aws.pem -N -L 8501:localhost:8501 ubuntu@PUBLIC_IP
```
Then open **http://localhost:8501**.

- **SSH suddenly times out?** Your home or mobile IP changed, so the *My IP* rule no longer
  matches: EC2 → Security groups → edit the SSH rule → source **My IP** again.
- **Stop vs terminate.** Stopping pauses the instance hours, but the disk keeps drawing credits.
  On **Start**, the app and its run history come back by themselves (the containers restart on
  their own), but the public IP **changes**, so copy the new one for SSH. Terminate it when you're
  done.
- **Watch the credits:** the *Cost and usage* widget on the Console Home shows the balance and the
  days left.
- **When the six months end**, AWS suspends the account and keeps its data for 90 days; upgrading
  to the Paid plan in that window restores it, otherwise everything is erased. To keep running
  past it, upgrade first (it then costs the ~$18–22/month above) or move to R1–R3.

#### Optional — let a reviewer in (HTTPS + password)

For when someone, an employer for example, wants to try real mode themselves. Everything above
stays as it is; this adds an HTTPS address and a password screen in front of the UI. It's meant
for a review period, not as a permanent public site.

**Cap the spending first.** Every question the reviewer asks spends your OpenAI credit (about
$0.001 each). Keep OpenAI on prepaid credit with auto-recharge off and a small balance (e.g.
$5–10) while it's shared: that is the most a leaked link can ever cost. `TOKEN_BUDGET` already
caps each run, and Tavily's free plan stops at 1,000 searches a month without charging.

**1. Give the instance a fixed address (recommended).** EC2 → Elastic IPs → **Allocate Elastic IP
address**, then **Actions → Associate** it with `ara-vm`. It replaces the instance's public IP (use
the new one for SSH from now on) and costs the same $0.005/hour. Without it, the address changes
whenever the instance stops.

**2. Get a free domain name.** Sign in at [duckdns.org](https://www.duckdns.org/) with Google or
GitHub, add a subdomain such as `YOUR-NAME`, put the instance's **Elastic IP** in its *current ip*
box (it pre-fills your laptop's IP, so replace it) and click **update ip**.
`YOUR-NAME.duckdns.org` now points at the instance.

**3. Open the web ports.** EC2 → Security groups → the instance's group → **Edit inbound rules** →
add **HTTP (80)** and **HTTPS (443)** from **Anywhere-IPv4**. Port 80 is only used to get the
certificate and to redirect to HTTPS. SSH stays limited to *My IP*, and the app's own ports stay on
loopback (`ARA_BIND=127.0.0.1`).

**4. Turn on the password screen** (on the instance). It needs the current code, so pull and
rebuild:
```bash
cd ~/ara && git pull
echo "ARA_UI_PASSWORD=$(openssl rand -hex 16)" >> .env
echo "ARA_RUNS_VOLUME=runs-review" >> .env
grep ARA_UI_PASSWORD .env               # the password to give the reviewer
docker compose up -d --build
```
`ARA_RUNS_VOLUME=runs-review` gives the app a **separate, empty run history** while the reviewer
has access, so they can't see your own questions on the *History* page. (You'll see theirs.)
Your own history stays untouched in the `runs` volume.

**5. Put HTTPS in front with Caddy.** It gets and renews a free Let's Encrypt certificate by itself
(DuckDNS is on the Public Suffix List, so its rate limits are per subdomain):
```bash
sudo apt install -y debian-keyring debian-archive-keyring apt-transport-https curl
curl -1sLf 'https://dl.cloudsmith.io/public/caddy/stable/gpg.key' | sudo gpg --dearmor -o /usr/share/keyrings/caddy-stable-archive-keyring.gpg
curl -1sLf 'https://dl.cloudsmith.io/public/caddy/stable/debian.deb.txt' | sudo tee /etc/apt/sources.list.d/caddy-stable.list
sudo chmod o+r /usr/share/keyrings/caddy-stable-archive-keyring.gpg /etc/apt/sources.list.d/caddy-stable.list
sudo apt update && sudo apt install -y caddy
sudo nano /etc/caddy/Caddyfile
```
Replace the file's contents with these three lines (your own subdomain), save, and reload:
```
YOUR-NAME.duckdns.org {
    reverse_proxy localhost:8501
}
```
Then apply it (in the terminal, not in the file):
```bash
sudo systemctl reload caddy
```
Open **https://YOUR-NAME.duckdns.org**. After a few seconds for the certificate, the password
screen appears. Send the reviewer the address, and the password separately.

- **Why a password screen inside the app, and not the proxy's basic auth?** Streamlit runs over a
  WebSocket, and Safari doesn't send basic-auth credentials on WebSocket connections
  ([WebKit bug 80362](https://bugs.webkit.org/show_bug.cgi?id=80362), still open), so a reviewer
  on a Mac would sit on "connecting" forever. The app's own screen works in every browser.
- **The password is per browser session:** a new tab or browser asks again.

**6. When the review is over, close it:**
```bash
sudo systemctl disable --now caddy
```
Then remove the HTTP/HTTPS rules from the security group, and if you made one, **Disassociate** and
**Release** the Elastic IP: an unused Elastic IP still costs $0.005/hour, and the instance gets a
new public IP for SSH. If the link or password went further than intended, rotate the OpenAI key
as well. To get your own history back and drop the password screen, delete the `ARA_UI_PASSWORD`
and `ARA_RUNS_VOLUME` lines from `.env` and run `docker compose up -d`.

---

### Real mode on Streamlit Community Cloud?

Possible, but it's the least private option: a public Streamlit app has **no login**, so anyone
with the link can spend your credit. If you still want it, see
[REAL_MODE.md → Streamlit Community Cloud](REAL_MODE.md#option-s--streamlit-community-cloud-single-app-easiest)
— it needs two extra packages in `requirements.txt`, the keys as **top-level** secrets, and the app
restricted to your own account.

---

## 9. Teardown (remove everything)

```bash
# GCP Cloud Run (one service per command)
gcloud run services delete ara-api  --region us-central1
gcloud run services delete ara-ui   --region us-central1
gcloud run services delete ara-real --region us-central1     # real mode (§8 R1)
gcloud secrets delete openai-key
gcloud secrets delete tavily-key
gcloud artifacts repositories delete cloud-run-source-deploy --location us-central1   # stored images

# GCP VM + firewall (the firewall rule exists only if you made the keyless demo public)
gcloud compute instances delete ara-vm --zone us-central1-a
gcloud compute firewall-rules delete ara-ports

# Oracle: terminate the instance in the Console (Compute → Instances → … → Terminate),
# and delete the VCN if you created a dedicated one.

# AWS (§8 R4): EC2 → Instances → ara-vm → Instance state → Terminate (its disk goes with it),
# then delete its security group, and release the Elastic IP if you made one (an
# unattached one is still billed). On the Free plan, nothing else is billed.
```

If a real-mode host held your keys and you're done with it (or it was reclaimed), **revoke those
keys** in the OpenAI and Tavily dashboards and create new ones for the next deployment.

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
| Oracle: "Out of host capacity" creating the A1 | Retry later, try **1 OCPU / 6 GB**, or another Availability Domain if your region has several. Upgrading to Pay As You Go usually gets capacity. |
| Oracle: *Too many requests for the user* | Rate limit from rapid retries — wait 15–30 minutes, then retry at a relaxed pace. Check *Networking → Virtual cloud networks* for networks the failed attempts left behind, and reuse one (*Select existing virtual cloud network*). |
| GCP e2-micro build killed / OOM | Add the 1 GB swap (Option B step), or build the image once then `up -d` without `--build`. |
| UI can't reach the API | Check `API_URL` — `http://api:8000` inside docker-compose; the API's public URL on Cloud Run. |
| `gcloud run deploy --source` fails with `PERMISSION_DENIED` | Grant the build service account `roles/run.builder` (Option A step 1b / §8 R1 step 1). Projects created after May 2024 don't get it automatically. |
| Opening a **real-mode** Cloud Run URL shows *Error: Forbidden* (403) | Expected — the service is private (`--no-allow-unauthenticated`) and a browser sends no identity token. Open it with `gcloud run services proxy ara-real --region us-central1 --port 8501`. |
| Tunnel/proxy: *address already in use* on 8501 | Something local (e.g. your own Streamlit) already uses 8501. Pick another local port — `--port 8601` or `-L 8601:localhost:8501` — and open `localhost:8601`. |
| Windows `ssh`: *UNPROTECTED PRIVATE KEY FILE* / *bad permissions* | Restrict the key to your user: `icacls "C:\path\to\your_oracle.key" /inheritance:r /grant:r "$($env:USERNAME):(R)"` (PowerShell), then retry. |
| Real-mode VM reports `"keyless": true` | The `.env` must sit next to `docker-compose.yml` (`~/ara/.env`). Check what the API received: `docker compose exec api env \| grep PROVIDER`, then `docker compose up -d`. |
| Real-mode container: `No module named 'openai'` | The image predates the bundled real-mode SDKs — rebuild (`docker compose up -d --build`, or redeploy with `--source .`). |
| compose rejects `env_file` with `path:` / `required:` | Docker Compose older than v2.24. Update Docker (`curl -fsSL https://get.docker.com \| sudo sh`). |
| Oracle VM stopped or vanished after a quiet week | Idle reclamation of Always-Free compute — see [§8 R2](#r2--oracle-cloud-ampere-a1-vm--ssh-tunnel): upgrade to Pay As You Go, or recreate it from the repo. |
| `gcloud run services proxy`: *could not find* / *not found* service `ara-real` | Your laptop's `gcloud` is pointing at another project (its default). Add `--project YOUR_PROJECT_ID` (and `--account` if you have several Google accounts). |
| Console: *You need additional access* / *permission denied* on the project | Your browser is signed in with a different Google account. Switch accounts (avatar, top right) to the one that owns the project. |
| Windows Git Bash: `gcloud` → *Python was not found* | Git Bash picks a launcher that looks for a system Python. Run `gcloud` from **PowerShell** (or use Cloud Shell for the bash steps). |
| Oracle: *image is not compatible with the selected shape* | Choose the **Ampere** shape first, then re-select Ubuntu so the Console picks the **aarch64** build. |
| History page empty after an update | The install predates the history volume: the old runs were inside the replaced container. Copy them out *before* updating (see **Updating later** under R2). |
| Real-mode sidebar won't go above some budget | Intended: in real mode `TOKEN_BUDGET` and `MAX_ITERATIONS` in `.env` are the ceiling. Raise them there and run `docker compose up -d`. |
| AWS: SSH times out | Your IP changed (the *My IP* rule no longer matches — edit the security group), or the instance was stopped and restarted with a **new** public IP. |
| AWS reviewer link: the https:// address doesn't load | Check that DuckDNS points at the **Elastic IP**, that ports 80 and 443 are open in the security group, and Caddy's log: `sudo journalctl -u caddy --no-pager -n 50`. |
| AWS: *Permission denied (publickey)* | Log in as **`ubuntu`** (not `ec2-user`) with the `.pem` from that instance's key pair, locked with `icacls`. |

---

*Fastest free demo link, zero setup: **Option S (Streamlit Community Cloud)** — one app,
no Docker, no card. Simplest path to free-forever with a separate API + UI:
**Option C (Oracle Ampere A1)**. Zero-idle-cost public link with a real API:
**Option A (GCP Cloud Run)**. Real OpenAI + live web search as a private work tool on free
infrastructure: **[§8](#8-real-mode--privately-on-0-infrastructure)**, starting with **R1 (Cloud Run)**
(**R4** if you want it on AWS, free for six months).
With an active GCP trial, experiment freely, then land on one of those before it ends.*
