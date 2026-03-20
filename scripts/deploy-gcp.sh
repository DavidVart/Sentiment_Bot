#!/usr/bin/env bash
# Deploy Sentiment Bot dashboard and pipeline to GCP in one run.
#
# Usage:   ./scripts/deploy-gcp.sh
# Optional: CREATE_CLOUD_SQL=1 DB_PASSWORD=yourpass ./scripts/deploy-gcp.sh
#           (creates a new Cloud SQL instance and uses it for DATABASE_URL)
#
# Requires: gcloud CLI logged in (gcloud auth login)
# Reads:    .env in project root (DATABASE_URL, MASSIVE_API, NEWS_API, KALSHI_*)
#           If KALSHI_PRIVATE_KEY_PATH is set and the file exists, its content
#           is stored as secret KALSHI_PRIVATE_KEY for Cloud Run.

set -e

# --- Configuration (edit or override with env vars) ---
PROJECT_ID="${PROJECT_ID:-sentiment-bot-487918}"
PROJECT_NUMBER="${PROJECT_NUMBER:-722852346958}"
REGION="${REGION:-us-central1}"
SERVICE_NAME="${SERVICE_NAME:-options-agent-dashboard}"
JOB_NAME="${JOB_NAME:-options-agent-pipeline}"
REPO_NAME="${REPO_NAME:-options-agent}"
IMAGE_NAME="${REPO_NAME}/dashboard:latest"
FULL_IMAGE="${REGION}-docker.pkg.dev/${PROJECT_ID}/${IMAGE_NAME}"

# Set to 1 to create a new Cloud SQL instance (takes a few minutes). You must set DB_PASSWORD.
CREATE_CLOUD_SQL="${CREATE_CLOUD_SQL:-0}"
DB_PASSWORD="${DB_PASSWORD:-}"
SQL_INSTANCE="${SQL_INSTANCE:-options-agent-db}"
SQL_DATABASE="${SQL_DATABASE:-ai_options_agent}"
SQL_USER="${SQL_USER:-app}"

# Load .env if present (for secrets)
if [[ -f .env ]]; then
  set -a
  # shellcheck source=/dev/null
  source .env
  set +a
else
  echo "WARNING: No .env file. Secrets (DATABASE_URL, MASSIVE_API, NEWS_API) will not be created."
  echo "         Create .env or add secrets manually in Secret Manager."
  if [[ -t 0 ]]; then
    read -r -p "Continue anyway? [y/N] " cont
    [[ "${cont,,}" != "y" && "${cont,,}" != "yes" ]] && exit 1
  fi
fi

echo "=============================================="
echo " Deploy Sentiment Bot to GCP"
echo " Project: $PROJECT_ID ($PROJECT_NUMBER)"
echo " Region:  $REGION"
echo "=============================================="

# --- 1. Enable APIs ---
echo "[1/8] Enabling required APIs..."
gcloud config set project "$PROJECT_ID"
gcloud services enable run.googleapis.com \
  sqladmin.googleapis.com \
  secretmanager.googleapis.com \
  cloudscheduler.googleapis.com \
  artifactregistry.googleapis.com

# --- 2. Artifact Registry ---
echo "[2/8] Ensuring Artifact Registry repository..."
if ! gcloud artifacts repositories describe "$REPO_NAME" --location="$REGION" &>/dev/null; then
  gcloud artifacts repositories create "$REPO_NAME" \
    --repository-format=docker \
    --location="$REGION" \
    --description="Options agent dashboard and pipeline"
else
  echo "  Repository $REPO_NAME already exists."
fi

# --- 3. Optional: Cloud SQL ---
CONNECTION_NAME=""
if [[ "$CREATE_CLOUD_SQL" == "1" ]]; then
  if [[ -z "$DB_PASSWORD" ]]; then
    echo "ERROR: CREATE_CLOUD_SQL=1 requires DB_PASSWORD to be set."
    echo "  Example: DB_PASSWORD=yourpassword ./scripts/deploy-gcp.sh"
    exit 1
  fi
  echo "[3/8] Creating Cloud SQL instance (this may take a few minutes)..."
  if ! gcloud sql instances describe "$SQL_INSTANCE" &>/dev/null; then
    gcloud sql instances create "$SQL_INSTANCE" \
      --database-version=POSTGRES_15 \
      --tier=db-f1-micro \
      --region="$REGION"
  fi
  gcloud sql databases create "$SQL_DATABASE" --instance="$SQL_INSTANCE" 2>/dev/null || true
  gcloud sql users create "$SQL_USER" --instance="$SQL_INSTANCE" --password="$DB_PASSWORD" 2>/dev/null || \
    gcloud sql users set-password "$SQL_USER" --instance="$SQL_INSTANCE" --password="$DB_PASSWORD"
  CONNECTION_NAME=$(gcloud sql instances describe "$SQL_INSTANCE" --format='value(connectionName)')
  PUBLIC_IP=$(gcloud sql instances describe "$SQL_INSTANCE" --format='value(ipAddresses[0].ipAddress)')
  export DATABASE_URL="postgresql://${SQL_USER}:${DB_PASSWORD}@${PUBLIC_IP}/${SQL_DATABASE}"
  echo "  DATABASE_URL set from Cloud SQL (public IP)."
else
  echo "[3/8] Skipping Cloud SQL (CREATE_CLOUD_SQL not set). Using DATABASE_URL from .env or existing secrets."
  if [[ -z "$DATABASE_URL" ]]; then
    echo "  WARNING: DATABASE_URL is not set. You must create secret DATABASE_URL in Secret Manager before the dashboard will work."
  fi
fi

# --- 4. Build and push image (Cloud Build) ---
echo "[4/8] Building and pushing Docker image (Cloud Build)..."
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$ROOT_DIR"

# Optional: download trained model into models/ so the image has it (Performance/Trade Impact pages).
# Set MODEL_GCS_URI=gs://YOUR_BUCKET/ablation_D_ppo_seed0.zip to use this.
if [[ -n "${MODEL_GCS_URI:-}" ]]; then
  echo "  Downloading model from $MODEL_GCS_URI into models/..."
  mkdir -p models
  if gcloud storage cp "$MODEL_GCS_URI" models/ablation_D_ppo_seed0.zip 2>/dev/null; then
    echo "  Model downloaded."
  elif gsutil -q cp "$MODEL_GCS_URI" models/ablation_D_ppo_seed0.zip 2>/dev/null; then
    echo "  Model downloaded (via gsutil)."
  else
    echo "  WARNING: Failed to download model. Image will build without it; Performance/Trade Impact may be empty."
  fi
elif [[ ! -f models/ablation_D_ppo_seed0.zip ]]; then
  echo "  NOTE: models/ablation_D_ppo_seed0.zip not found (and MODEL_GCS_URI not set)."
  echo "        Dashboard will work but Performance/Trade Impact need the pipeline job to run with a model."
  echo "        To include the model: put the zip in models/ before deploy, or set MODEL_GCS_URI=gs://bucket/path.zip"
fi

if [[ -f cloudbuild.yaml ]]; then
  echo '  Using cloudbuild.yaml (E2_HIGHCPU_8, 40 min timeout).'
  gcloud builds submit --config=cloudbuild.yaml --substitutions=_IMAGE_TAG="$FULL_IMAGE" .
else
  gcloud builds submit --tag "$FULL_IMAGE" .
fi

# --- 5. Secrets ---
echo "[5/8] Creating or updating secrets from .env..."
COMPUTE_SA="${PROJECT_NUMBER}-compute@developer.gserviceaccount.com"

create_or_update_secret() {
  local name=$1
  local value=$2
  if [[ -z "$value" ]]; then return; fi
  if gcloud secrets describe "$name" --project="$PROJECT_ID" &>/dev/null; then
    echo -n "$value" | gcloud secrets versions add "$name" --data-file=-
    echo "  Updated secret: $name"
  else
    echo -n "$value" | gcloud secrets create "$name" --data-file=-
    echo "  Created secret: $name"
  fi
  gcloud secrets add-iam-policy-binding "$name" \
    --member="serviceAccount:${COMPUTE_SA}" \
    --role="roles/secretmanager.secretAccessor" \
    --project="$PROJECT_ID" \
    --quiet
}

create_or_update_secret "DATABASE_URL" "$DATABASE_URL"
create_or_update_secret "MASSIVE_API" "$MASSIVE_API"
create_or_update_secret "NEWS_API" "$NEWS_API"
create_or_update_secret "KALSHI_API_KEY" "$KALSHI_API_KEY"
if [[ -n "${KALSHI_PRIVATE_KEY_PATH:-}" && -f "${KALSHI_PRIVATE_KEY_PATH}" ]]; then
  KALSHI_PEM=$(cat "${KALSHI_PRIVATE_KEY_PATH}")
  create_or_update_secret "KALSHI_PRIVATE_KEY" "$KALSHI_PEM"
fi

# Build --set-secrets flag for run deploy and job (only for secrets that exist)
SECRETS_LIST=""
for s in DATABASE_URL MASSIVE_API NEWS_API KALSHI_API_KEY KALSHI_PRIVATE_KEY; do
  if gcloud secrets describe "$s" --project="$PROJECT_ID" &>/dev/null; then
    SECRETS_LIST="${SECRETS_LIST}${s}=${s}:latest,"
  fi
done
if [[ -n "$SECRETS_LIST" ]]; then
  SET_SECRETS="--set-secrets=${SECRETS_LIST%,}"
else
  SET_SECRETS=""
fi

# --- 6. Deploy Cloud Run service ---
echo "[6/8] Deploying Cloud Run service..."
DEPLOY_CMD=(gcloud run deploy "$SERVICE_NAME" \
  --image="$FULL_IMAGE" \
  --region="$REGION" \
  --platform=managed \
  --allow-unauthenticated \
  --port=8080)
[[ -n "$SET_SECRETS" ]] && DEPLOY_CMD+=( $SET_SECRETS )
"${DEPLOY_CMD[@]}"

SERVICE_URL=$(gcloud run services describe "$SERVICE_NAME" --region="$REGION" --format='value(status.url)')
echo "  Dashboard URL: $SERVICE_URL"

# --- 7. Cloud Run Job (pipeline + snapshot) ---
echo "[7/8] Creating or updating Cloud Run Job..."
JOB_EXTRA=()
[[ -n "$SET_SECRETS" ]] && JOB_EXTRA=( $SET_SECRETS )
if gcloud run jobs describe "$JOB_NAME" --region="$REGION" &>/dev/null; then
  gcloud run jobs update "$JOB_NAME" \
    --image="$FULL_IMAGE" \
    --region="$REGION" \
    "${JOB_EXTRA[@]}" \
    --command="python" \
    --args="scripts/run_pipeline_and_snapshot.py" \
    --task-timeout=3600 \
    --max-retries=0
else
  gcloud run jobs create "$JOB_NAME" \
    --image="$FULL_IMAGE" \
    --region="$REGION" \
    "${JOB_EXTRA[@]}" \
    --command="python" \
    --args="scripts/run_pipeline_and_snapshot.py" \
    --task-timeout=3600 \
    --max-retries=0
fi

gcloud run jobs add-iam-policy-binding "$JOB_NAME" \
  --region="$REGION" \
  --member="serviceAccount:${COMPUTE_SA}" \
  --role="roles/run.invoker" \
  --quiet

# --- 8. Cloud Scheduler (daily run, weekdays 5:30 PM Eastern = 22:30 UTC) ---
echo "[8/8] Scheduling daily pipeline..."
SCHEDULER_JOB="options-agent-daily"
RUN_URI="https://run.googleapis.com/v2/projects/${PROJECT_ID}/locations/${REGION}/jobs/${JOB_NAME}:run"
if gcloud scheduler jobs describe "$SCHEDULER_JOB" --location="$REGION" &>/dev/null; then
  gcloud scheduler jobs update http "$SCHEDULER_JOB" \
    --location="$REGION" \
    --schedule="30 22 * * 1-5" \
    --uri="$RUN_URI" \
    --http-method=POST \
    --oauth-service-account-email="$COMPUTE_SA"
else
  gcloud scheduler jobs create http "$SCHEDULER_JOB" \
    --location="$REGION" \
    --schedule="30 22 * * 1-5" \
    --uri="$RUN_URI" \
    --http-method=POST \
    --oauth-service-account-email="$COMPUTE_SA"
fi

echo ""
echo "=============================================="
echo " Deployment complete"
echo "=============================================="
echo " Dashboard:  $SERVICE_URL"
echo " Job:        $JOB_NAME (runs weekdays 22:30 UTC / 5:30 PM ET)"
echo " Manual run: gcloud run jobs execute $JOB_NAME --region=$REGION"
echo ""
