# Cloud Run Deployment Guide

This guide explains how to deploy the DIPG Safety Gym evaluation server to Google Cloud Run using the direct source deployment method.

## Prerequisites

1. **Google Cloud Project**: You need a GCP project with billing enabled
2. **gcloud CLI**: Install from https://cloud.google.com/sdk/docs/install
3. **Docker**: Install from https://docs.docker.com/get-docker/ (Optional, as Cloud Run builds remotely)

## Quick Deployment

### Option 1: Using the Helper Script (Recommended)

We have provided a `deploy.sh` script that handles the deployment process automatically.

```bash
# Make the script executable
chmod +x deploy.sh

# Run the deployment script
./deploy.sh
```

This script will:
1. Check for your GCP project
2. Create a service account if needed
3. Deploy the server directly from source
4. Output the service URL

### Option 2: Manual Deployment

If you prefer to run the commands manually, here are the steps:

```bash
# 1. Set your GCP project
export PROJECT_ID=$(gcloud config get-value project)

# 2. Create Service Account (if not exists)
gcloud iam service-accounts create dipg-server-sa \
    --display-name="DIPG Server Service Account"

# 3. Deploy to Cloud Run from source
gcloud run deploy dipg-server \
    --service-account=dipg-server-sa@$PROJECT_ID.iam.gserviceaccount.com \
    --allow-unauthenticated \
    --region=us-central1 \
    --source=. \
    --memory=2Gi \
    --cpu=2 \
    --timeout=300 \
    --max-instances=10 \
    --set-env-vars=DIPG_DATASET_PATH=surfiniaburger/dipg-eval-dataset \
    --labels=project=dipg-safety-gym

# 4. Get the service URL
gcloud run services describe dipg-server --region us-central1 --format 'value(status.url)'
```

## Configuration

### Environment Variables

- `DIPG_DATASET_PATH`: Hugging Face dataset path (default: `surfiniaburger/dipg-eval-dataset`)
- `PORT`: Server port (default: 8080, set by Cloud Run)

### Resource Limits

- **Memory**: 2Gi (adjustable based on dataset size)
- **CPU**: 2 cores (adjustable for performance)
- **Timeout**: 300 seconds (5 minutes for long evaluations)
- **Max Instances**: 10 (adjustable for scaling)

## Testing the Deployment

```bash
# Get your service URL
export SERVICE_URL=$(gcloud run services describe dipg-server --region us-central1 --format 'value(status.url)')

# Test health endpoint
curl $SERVICE_URL/health

# Test evaluation endpoint
curl -X POST $SERVICE_URL/evaluate \
  -H "Content-Type: application/json" \
  -d '{
    "evaluations": [{
      "response": "{\"analysis\": \"test\", \"proof\": \"test\", \"final\": \"test\"}",
      "ground_truth": {
        "context": "test context",
        "question": "test question",
        "expected_answer": "test answer"
      }
    }],
    "format": "json"
  }'
```

## Using in Colab

Once deployed, update your Colab notebook to use the Cloud Run URL:

```python
from examples.run_eval_colab import run_evaluation_http

# Get your Cloud Run URL from the deployment
SERVICE_URL = "https://dipg-server-xxxxx-uc.a.run.app"

metrics = run_evaluation_http(
    model=model,
    tokenizer=tokenizer,
    num_samples=100,
    server_url=SERVICE_URL
)
```

## Cost Estimation

Cloud Run pricing (as of 2024):
- **Free tier**: 2 million requests/month
- **CPU**: $0.00002400 per vCPU-second
- **Memory**: $0.00000250 per GiB-second
- **Requests**: $0.40 per million requests

**Estimated cost for 1000 evaluations**:
- ~5 minutes total compute time
- ~$0.05 per 1000 evaluations
- Well within free tier for development

## Monitoring

View logs and metrics:
```bash
# View logs
gcloud run services logs read dipg-server --region us-central1

# View metrics in Cloud Console
gcloud run services describe dipg-server --region us-central1
```

## Updating the Deployment

To update the deployment after code changes, simply run the deployment command again:

```bash
./deploy.sh
```

## Cleanup

```bash
# Delete the Cloud Run service
gcloud run services delete dipg-server --region us-central1
```
