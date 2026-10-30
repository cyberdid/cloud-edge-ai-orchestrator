#!/usr/bin/env bash
# Colab scale study storage (docs/colab_scale/INFRA.md). Run by the owner; idempotent;
# never prints the key.
set -euo pipefail
PROJECT=gen-lang-client-0903767759
BUCKET=gs://oe-colab-scale-0903767759
SA_NAME=oe-colab-sync
SA="$SA_NAME@$PROJECT.iam.gserviceaccount.com"
KEY="$HOME/.config/oe-colab/sa-key.json"

gcloud services enable storage.googleapis.com iam.googleapis.com --project "$PROJECT"
if gcloud storage buckets describe "$BUCKET" --project "$PROJECT" >/dev/null 2>&1; then
  echo "bucket exists: $BUCKET"
else
  gcloud storage buckets create "$BUCKET" --project "$PROJECT" --location=us-west4 --uniform-bucket-level-access
fi
if gcloud iam service-accounts describe "$SA" --project "$PROJECT" >/dev/null 2>&1; then
  echo "service account exists: $SA"
else
  gcloud iam service-accounts create "$SA_NAME" --project "$PROJECT" --display-name "OE Colab sync (bucket-only)"
fi
gcloud storage buckets add-iam-policy-binding "$BUCKET" --member "serviceAccount:$SA" \
  --role roles/storage.objectAdmin --format none
if [ -f "$KEY" ]; then
  echo "key exists: $KEY"
else
  mkdir -p "$(dirname "$KEY")"
  chmod 700 "$(dirname "$KEY")"
  (umask 077; gcloud iam service-accounts keys create "$KEY" --iam-account "$SA" --project "$PROJECT")
fi
chmod 600 "$KEY"
echo "ok: $BUCKET, $SA, key at $KEY"
