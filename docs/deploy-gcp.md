# Hosting on Google Cloud Platform (GCP)

This guide deploys the **web dashboard** (FastAPI + Angular frontend) as a public website on Cloud Run and runs the **daily pipeline + dashboard snapshot** automatically with Cloud Run Jobs and Cloud Scheduler. No manual updates and no need to run anything on your machine.

---

## Why GCP for this project

| Need | GCP solution |
|------|----------------|
| Dashboard always available at a URL | **Cloud Run** (FastAPI + static Angular frontend): scales to zero when idle, pay per request |
| Daily pipeline + snapshot without manual runs | **Cloud Run Jobs** + **Cloud Scheduler**: same Docker image, job runs pipeline then `write_dashboard_snapshot.py` on a schedule (e.g. weekdays 5:30 PM) |
| Shared database for dashboard and pipeline | **Cloud SQL (PostgreSQL)**: dashboard and job use the same `DATABASE_URL` |
| API keys and DB URL kept secret | **Secret Manager** (or env vars in Cloud Run) |

**Alternatives (if you prefer):**

- **Render**: Deploy the repo as a Web Service (Streamlit) + Cron Job for the pipeline. Simpler setup; fewer GCP concepts. [Render Python docs](https://render.com/docs/deploy-streamlit).
- **Railway / Fly.io**: Similar: one app + scheduled job. Good if you want a single dashboard and don’t need GCP integration.
- **AWS**: Use **App Runner** or **ECS** for the dashboard and **EventBridge + Lambda/Step Functions** or a scheduled **ECS task** for the pipeline. More moving parts than Cloud Run + Jobs.

GCP is recommended here because Cloud Run + Cloud Run Jobs use the **same image** and the same **Cloud Scheduler** pattern is straightforward and cost-effective.

---

## Architecture (GCP)

```
                    Cloud Scheduler (daily, e.g. 17:30 ET)
                              |
                              v
                    Cloud Run Job (pipeline + snapshot)
                    CMD: python scripts/run_pipeline_and_snapshot.py
                              |
                              v
    +------------------+     Cloud SQL (PostgreSQL)
    |  Cloud Run       |     DATABASE_URL
    |  (dashboard)     | <-----------------------+
    |  FastAPI + web   |                         |
    +--------+---------+                         |
             |                                   |
             v                                   +
    Users → https://your-service-xxx.run.app
```

---

## Prerequisites

- **Google Cloud account** and a project (e.g. `your-project-id`).
- **gcloud CLI** installed and logged in: `gcloud auth login` and `gcloud config set project YOUR_PROJECT_ID`.
- **Docker** (optional; the one-command deploy script uses Cloud Build, so you don’t need Docker locally).

### One-command deploy

From the project root, with your `.env` (and optionally `DB_PASSWORD` and `CREATE_CLOUD_SQL=1` if you want the script to create Cloud SQL):

```bash
./scripts/deploy-gcp.sh
```

The script reads `PROJECT_ID` and `PROJECT_NUMBER` from the environment (or `.env`). It enables APIs, creates Artifact Registry, optionally creates Cloud SQL, builds the image with Cloud Build, creates/updates secrets from `.env`, deploys the Cloud Run service and job, and **schedules the daily pipeline automatically** (see below).

### Pipeline job is automated

You do **not** need to run the pipeline by hand every day. The deploy script:

- **Step 7:** Creates or updates a **Cloud Run Job** (`options-agent-pipeline`) that runs `scripts/run_pipeline_and_snapshot.py` (full pipeline + dashboard snapshot).
- **Step 8:** Creates or updates a **Cloud Scheduler** job (`options-agent-daily`) that triggers that Cloud Run Job on a schedule: **weekdays at 22:30 UTC** (5:30 PM Eastern).

So after deploy, the pipeline runs automatically every weekday. The only time you run the job manually is for an **initial backfill** (e.g. right after first deploy) so the dashboard has data; after that, the scheduler runs it daily.

### Model for Performance and Trade Impact pages

The dashboard’s **Performance** and **Trade Impact** views use a trained RL model (`ablation_D_ppo_seed0.zip`) inside the **same Docker image** that the pipeline job runs. The snapshot script loads that model to compute equity curves and trade-impact bars. If the model is not in the image, those pages stay empty (the job still runs and fills other data).

**You have two ways to get the model into the image:**

1. **Put the file in `models/` before building**  
   After training (or from wherever you have the zip), copy it into the repo:
   ```bash
   cp /path/to/ablation_D_ppo_seed0.zip models/
   ./scripts/deploy-gcp.sh
   ```
   The build context will include `models/ablation_D_ppo_seed0.zip`, and the Dockerfile will copy it into the image.

2. **Download from GCS during deploy**  
   Upload the model once to Google Cloud Storage, then point the deploy script at it so it downloads into `models/` before building:
   ```bash
   # One-time upload (use your bucket)
   gcloud storage cp models/ablation_D_ppo_seed0.zip gs://YOUR_BUCKET/ablation_D_ppo_seed0.zip

   # Deploy; script will download into models/ then build
   MODEL_GCS_URI=gs://YOUR_BUCKET/ablation_D_ppo_seed0.zip ./scripts/deploy-gcp.sh
   ```
   The script will run `gcloud storage cp $MODEL_GCS_URI models/ablation_D_ppo_seed0.zip` before `gcloud builds submit`, so the image will contain the model even if it’s not in your local `models/` folder.

If you don’t set `MODEL_GCS_URI` and don’t have `models/ablation_D_ppo_seed0.zip`, the script will still build and deploy; it will print a short note that Performance/Trade Impact may be empty until you add the model and redeploy.

---

## Step-by-step (first-time deploy)

**Is `DATABASE_URL=postgresql://user@localhost:5432/ai_options_agent` OK?**  
No. That URL is for **local development only**. The dashboard and pipeline run **in GCP**, so they cannot reach `localhost`. You need a **cloud-accessible database** (e.g. Cloud SQL). Your `.env` can keep localhost for local runs; for deploy you either let the script create Cloud SQL or you use an existing cloud Postgres URL.

### Option A: Let the script create Cloud SQL (easiest)

1. **Log in and set project** (you already did this):
   ```bash
   gcloud auth login
   gcloud config set project $PROJECT_ID
   ```

2. **Run the deploy script and create the database in one go**  
   Pick a **strong password** for the new DB user (e.g. `app`). The script will create a small Cloud SQL instance, database, and user, then use that URL for the deployed app. Your `.env` is only used for API keys and (when not creating SQL) for `DATABASE_URL`; here we override with the new Cloud SQL URL.
   ```bash
   cd /path/to/Sentiment_Bot
   CREATE_CLOUD_SQL=1 DB_PASSWORD='YourStrongPasswordHere' ./scripts/deploy-gcp.sh
   ```
   This takes several minutes the first time (APIs, Artifact Registry, Cloud SQL creation, build, deploy).

3. **Open the dashboard**  
   At the end the script prints the service URL, e.g. `https://options-agent-dashboard-xxxxx.run.app`. Open it in your browser.

4. **Run the pipeline once for initial backfill (optional)**  
   Daily runs are already scheduled by Cloud Scheduler. To populate the dashboard right away instead of waiting for the first scheduled run:
   ```bash
   gcloud run jobs execute options-agent-pipeline --region=us-central1
   ```

5. **Keep using localhost locally**  
   Leave `DATABASE_URL=postgresql://user@localhost:5432/ai_options_agent` in `.env` for local development. The **deployed** app uses the Cloud SQL URL stored in Secret Manager, not your `.env`.

### Option B: You already have (or will create) Cloud SQL

1. **Log in and set project** (same as above).

2. **Create a Cloud SQL instance and database** (if you don’t have one), then get the **public IP** and set a **connection URL**:
   ```bash
   gcloud config set project $PROJECT_ID
   # Create instance (once), then database and user: see "2. Cloud SQL" below for full commands.
   gcloud sql instances describe options-agent-db --format='value(ipAddresses[0].ipAddress)'
   ```
   Set in your environment (or in a temporary `.env` for deploy only):
   ```bash
   export DATABASE_URL="postgresql://app:YOUR_DB_PASSWORD@THE_PUBLIC_IP/ai_options_agent"
   ```

3. **Run the deploy script** (it will push this `DATABASE_URL` to Secret Manager):
   ```bash
   cd /path/to/Sentiment_Bot
   ./scripts/deploy-gcp.sh
   ```
   If your `.env` still has `localhost`, the script sources `.env` and may overwrite `DATABASE_URL`. So either: put the Cloud SQL URL in `.env` only for this run, or run:
   ```bash
   DATABASE_URL="postgresql://app:PASSWORD@PUBLIC_IP/ai_options_agent" ./scripts/deploy-gcp.sh
   ```

4. **Open the dashboard** and, if you want data immediately, **run the job once** for initial backfill (as in step 4 of Option A). Daily runs are automatic.

### Dashboard stuck on "Loading..."

If the dashboard loads the layout but stays on "Loading..." and there is **no** `GET /api/overview` in Cloud Run logs, the backend is likely **hanging when connecting to Cloud SQL**. Cloud SQL with a public IP uses **authorized networks**; by default no external IP is allowed, so Cloud Run’s requests never complete.

Allow Cloud Run (and any client) to reach the instance by adding an authorized network. For a quick fix (allow any IP):

```bash
gcloud sql instances patch options-agent-db --authorized-networks=0.0.0.0/0
```

Then open the dashboard again. To restrict later, replace `0.0.0.0/0` with specific CIDR ranges or use a [VPC connector and private IP](https://cloud.google.com/sql/docs/postgres/connect-run).

### Empty data and 500 errors (e.g. "relation … does not exist")

If the Overview loads but shows all zeros and other pages error with "relation … does not exist", the **Cloud SQL database has no schema**: migrations were never applied. Apply them once from your machine (with `DATABASE_URL` pointing at Cloud SQL):

```bash
# From project root. Use your Cloud SQL URL (same as in Secret Manager).
export DATABASE_URL="postgresql://app:YOUR_DB_PASSWORD@YOUR_CLOUD_SQL_IP/ai_options_agent"
python -c "from src.db import apply_migrations; apply_migrations()"
```

Replace the password and IP if yours differ. That creates all tables (`feature_bars`, `dashboard_trade_impact`, `dashboard_ablation`, etc.). After this, the dashboard should load without 500s; metrics will stay at zero until the **pipeline job** has run at least once to backfill data and write dashboard snapshots.

The deploy script and Cloud Run Job have been updated so that the **next** job run will also apply migrations (step 1) before the rest of the pipeline, so future deploys or new DBs stay in sync.

### PM price points 0 / No Polymarket or Kalshi in Signal Feed

If Overview shows **PM price points: 0** and **PM: 0** in feature coverage, the pipeline’s step 4 (backfill PM) did not ingest any prediction-market prices. The Signal Feed will then show only News.

- **Check the job execution logs** for step 4:  
  Cloud Console → Cloud Run → Jobs → `options-agent-pipeline` → Executions → open the latest → Logs. Look for `Step 4 backfill_pm` and any `Polymarket backfill skipped` or `Kalshi skipped (auth required)`.
- **Kalshi:** The job must receive `KALSHI_API_KEY` and `KALSHI_PRIVATE_KEY` (PEM **contents**, not path). The deploy script pushes these from your `.env` when `KALSHI_PRIVATE_KEY_PATH` points at your local PEM file. If you see 401 in logs, the key in Secret Manager may be wrong or the job may not have the secret; re-run `./scripts/deploy-gcp.sh` so secrets are re-pushed, and ensure the PEM is the full key including `-----BEGIN/END-----` lines.
- **Polymarket:** Public CLOB/Gamma APIs are used; 400 errors often mean invalid or retired token IDs. The pipeline now skips Polymarket on failure and still runs Kalshi, so you can get PM data from Kalshi even if Polymarket fails.

### Performance and Trade Impact empty

If **Performance** says “No precomputed performance data” and **Trade Impact** shows “Found 0 bars”, the **dashboard snapshot** either did not run or could not write evaluation data.

- The snapshot runs **after** the pipeline (and now runs even if the pipeline had failures). It needs:
  - **Ablation model in the image:** `models/ablation_D_ppo_seed0.zip` must be present when the job runs. Put the file in `models/` before `./scripts/deploy-gcp.sh`, or set `MODEL_GCS_URI=gs://your-bucket/ablation_D_ppo_seed0.zip` so the script downloads it before building. Without the model, the snapshot skips evaluation and those pages stay empty.
  - **Step 12 (ablation) to have run successfully** in the same job run (or a previous run that left the model in the container). Step 12 trains a short run and saves the model; the snapshot then uses it. If the job **times out** (default 1 hour) before step 12 finishes, no model is saved and the snapshot has nothing to evaluate. Consider increasing the job’s task timeout or running a shorter ablation in step 12 for the first backfill.
- Check the latest job execution logs for `Step 12 run_ablation_smoke` and for `write_dashboard_snapshot` (e.g. “Warning: model not found” means the model path was missing).

---

## 1. Enable APIs and create Artifact Registry

```bash
export PROJECT_ID=your-project-id
export REGION=us-central1   # or us-east1, etc.

gcloud services enable run.googleapis.com \
  sqladmin.googleapis.com \
  secretmanager.googleapis.com \
  cloudscheduler.googleapis.com \
  artifactregistry.googleapis.com
```

Create a Docker repository:

```bash
gcloud artifacts repositories create options-agent \
  --repository-format=docker \
  --location=$REGION \
  --description="Options agent dashboard and pipeline"
```

---

## 2. Cloud SQL (PostgreSQL)

Create a small Postgres instance (or use an existing one):

```bash
# Optional: create a small instance (takes a few minutes)
gcloud sql instances create options-agent-db \
  --database-version=POSTGRES_15 \
  --tier=db-f1-micro \
  --region=$REGION

# Create database and user (replace YOUR_PASSWORD)
gcloud sql databases create ai_options_agent --instance=options-agent-db
gcloud sql users create app --instance=options-agent-db --password=YOUR_PASSWORD
```

Get the connection name for the next step:

```bash
gcloud sql instances describe options-agent-db --format='value(connectionName)'
# Example: your-project-id:us-central1:options-agent-db
```

**Connection from Cloud Run:** Use the Unix socket or the **Cloud SQL Auth Proxy**. Easiest is to connect via **Private IP** if VPC is set up, or use the **Public IP** and allow Cloud Run’s egress. For a quick start, use the instance’s public IP:

```bash
# Get public IP
gcloud sql instances describe options-agent-db --format='value(ipAddresses[0].ipAddress)'
```

Then set:

```text
DATABASE_URL=postgresql://app:YOUR_PASSWORD@PUBLIC_IP/ai_options_agent
```

For production, prefer **Secret Manager** (see below) and, if possible, **Private IP** + VPC connector for Cloud Run.

---

## 3. Build and push the Docker image

From the project root (where `Dockerfile` lives):

```bash
# Configure Docker for Artifact Registry
gcloud auth configure-docker ${REGION}-docker.pkg.dev

# Build and push (replace with your repo path)
IMAGE=${REGION}-docker.pkg.dev/${PROJECT_ID}/options-agent/dashboard:latest
docker build -t $IMAGE .
docker push $IMAGE
```

Or use **Cloud Build** (no local Docker needed):

```bash
IMAGE=${REGION}-docker.pkg.dev/${PROJECT_ID}/options-agent/dashboard:latest
gcloud builds submit --tag $IMAGE .
```

---

## 4. Store secrets (recommended)

Put `DATABASE_URL` and API keys in Secret Manager so Cloud Run and the Job can use them without hardcoding.

```bash
# Create secrets (replace values with your real ones)
echo -n "postgresql://app:PASSWORD@/ai_options_agent?host=/cloudsql/CONNECTION_NAME" | \
  gcloud secrets create DATABASE_URL --data-file=-

# If you use a public IP URL instead:
# echo -n "postgresql://app:PASSWORD@PUBLIC_IP/ai_options_agent" | gcloud secrets create DATABASE_URL --data-file=-

echo -n "your-polygon-key" | gcloud secrets create MASSIVE_API --data-file=-
echo -n "your-news-api-key" | gcloud secrets create NEWS_API --data-file=-
# Add KALSHI_API_KEY, etc., as needed
```

Grant Cloud Run access:

```bash
PROJECT_NUMBER=$(gcloud projects describe $PROJECT_ID --format='value(projectNumber)')
for SECRET in DATABASE_URL MASSIVE_API NEWS_API; do
  gcloud secrets add-iam-policy-binding $SECRET \
    --member="serviceAccount:${PROJECT_NUMBER}-compute@developer.gserviceaccount.com" \
    --role="roles/secretmanager.secretAccessor"
done
```

---

## 5. Deploy the dashboard (Cloud Run service)

Deploy the image as a **service** so the web dashboard is always reachable at a URL. The container runs FastAPI (uvicorn) on port 8080 and serves the built Angular frontend at `/`.

```bash
IMAGE=${REGION}-docker.pkg.dev/${PROJECT_ID}/options-agent/dashboard:latest

gcloud run deploy options-agent-dashboard \
  --image=$IMAGE \
  --region=$REGION \
  --platform=managed \
  --allow-unauthenticated \
  --set-secrets=DATABASE_URL=DATABASE_URL:latest,MASSIVE_API=MASSIVE_API:latest,NEWS_API=NEWS_API:latest \
  --port=8080
```

Cloud Run sets `PORT=8080`; the Dockerfile CMD runs `uvicorn src.api.main:app --host 0.0.0.0 --port ${PORT:-8080}`.

After deployment, the CLI prints the **service URL**, e.g. `https://options-agent-dashboard-xxx.run.app`. Open it in a browser to use the hosted dashboard.


---

## 6. Deploy the daily pipeline + snapshot (Cloud Run Job + Scheduler)

Use the **same image** but run the pipeline and dashboard snapshot script (so the web dashboard has fresh precomputed performance and trade-impact data).

**Create the Job:**

```bash
IMAGE=${REGION}-docker.pkg.dev/${PROJECT_ID}/options-agent/dashboard:latest

gcloud run jobs create options-agent-pipeline \
  --image=$IMAGE \
  --region=$REGION \
  --set-secrets=DATABASE_URL=DATABASE_URL:latest,MASSIVE_API=MASSIVE_API:latest,NEWS_API=NEWS_API:latest \
  --command="python" \
  --args="scripts/run_pipeline_and_snapshot.py" \
  --task-timeout=3600 \
  --max-retries=0
```

`run_pipeline_and_snapshot.py` runs `run_full_pipeline.py --steps 2,3,5,6,7,8,9,10` then `write_dashboard_snapshot.py`.

**Schedule it (e.g. weekdays at 5:30 PM Eastern):**

```bash
# 17:30 Eastern = 22:30 UTC (adjust for your timezone)
# PROJECT_NUMBER from: gcloud projects describe $PROJECT_ID --format='value(projectNumber)'
gcloud scheduler jobs create http options-agent-daily \
  --location=$REGION \
  --schedule="30 22 * * 1-5" \
  --uri="https://run.googleapis.com/v2/projects/${PROJECT_ID}/locations/${REGION}/jobs/options-agent-pipeline:run" \
  --http-method=POST \
  --oauth-service-account-email=${PROJECT_NUMBER}-compute@developer.gserviceaccount.com
```

Scheduler needs the **Cloud Run Invoker** role on the job. Grant it:

```bash
gcloud run jobs add-iam-policy-binding options-agent-pipeline \
  --region=$REGION \
  --member="serviceAccount:${PROJECT_NUMBER}-compute@developer.gserviceaccount.com" \
  --role="roles/run.invoker"
```

To trigger the job manually once:

```bash
gcloud run jobs execute options-agent-pipeline --region=$REGION
```

---

## 7. Port and healthcheck (optional)

The Dockerfile runs `uvicorn src.api.main:app --host 0.0.0.0 --port ${PORT:-8080}`, so the app listens on Cloud Run’s `PORT` (8080). The `/health` route runs a quick DB check; Cloud Run can use it for health checks if needed.

---

## 8. Custom domain (optional)

1. In **Cloud Run** → your service → **Manage custom domains**, add your domain and follow the verification steps.
2. Map the domain to the Cloud Run service (GCP will show the DNS records).

---

## 9. Cost (ballpark)

- **Cloud Run (dashboard):** Free tier is generous; low traffic usually stays within free tier (e.g. 2M requests/month).
- **Cloud Run Jobs:** Billed per run time; a daily 10-20 minute pipeline is a few dollars per month.
- **Cloud Scheduler:** First 3 jobs free; then about $0.10/job/month.
- **Cloud SQL:** `db-f1-micro` is low cost but not free; check current pricing.
- **Secret Manager:** Small number of secrets is negligible.

Overall, expect on the order of **tens of dollars per month** for light use, mostly from Cloud SQL and job runtime.

---

## Summary

1. **Dashboard:** One Cloud Run **service** from your Docker image → always-on website.  
2. **Daily updates:** One Cloud Run **Job** (same image, different command) triggered by **Cloud Scheduler** so the pipeline runs without you.  
3. **Database:** Cloud SQL (or any Postgres) with `DATABASE_URL` in Secret Manager.  
4. No more manual pipeline or dashboard runs: both are automated and hosted.

For step-by-step issues (e.g. IAM, Private IP, or Cloud SQL proxy), see [Cloud Run docs](https://cloud.google.com/run/docs) and [Cloud Run Jobs](https://cloud.google.com/run/docs/create-jobs).
