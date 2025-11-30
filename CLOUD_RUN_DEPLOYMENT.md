# Cloud Run Deployment Guide

This guide explains how to deploy the DIPG Safety Gym evaluation server to Google Cloud Run.

## Prerequisites

1. **Google Cloud Project**: You need a GCP project with billing enabled
2. **gcloud CLI**: Install from https://cloud.google.com/sdk/docs/install
3. **Docker**: Install from https://docs.docker.com/get-docker/

## Quick Deployment

### Option 1: Automated Deployment (Recommended)

```bash
# 1. Set your GCP project
export PROJECT_ID="your-gcp-project-id"
gcloud config set project $PROJECT_ID

# 2. Enable required APIs
gcloud services enable cloudbuild.googleapis.com run.googleapis.com containerregistry.googleapis.com

# 3. Deploy using Cloud Build
gcloud builds submit --config cloudbuild.yaml

# 4. Get the service URL
gcloud run services describe dipg-server --region us-central1 --format 'value(status.url)'
```

### Option 2: Manual Deployment

```bash
# 1. Set your GCP project
export PROJECT_ID="your-gcp-project-id"
gcloud config set project $PROJECT_ID

# 2. Build the Docker image
docker build -t gcr.io/$PROJECT_ID/dipg-server:latest .

# 3. Push to Container Registry
docker push gcr.io/$PROJECT_ID/dipg-server:latest

# 4. Deploy to Cloud Run
gcloud run deploy dipg-server \
  --image gcr.io/$PROJECT_ID/dipg-server:latest \
  --region us-central1 \
  --platform managed \
  --allow-unauthenticated \
  --memory 2Gi \
  --cpu 2 \
  --timeout 300 \
  --max-instances 10 \
  --set-env-vars DIPG_DATASET_PATH=surfiniaburger/dipg-eval-dataset

# 5. Get the service URL
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

```bash
# Rebuild and redeploy
gcloud builds submit --config cloudbuild.yaml

# Or manually
docker build -t gcr.io/$PROJECT_ID/dipg-server:latest .
docker push gcr.io/$PROJECT_ID/dipg-server:latest
gcloud run deploy dipg-server --image gcr.io/$PROJECT_ID/dipg-server:latest --region us-central1
```

## Troubleshooting

### Build Fails

- Check Docker is running: `docker ps`
- Verify project ID: `gcloud config get-value project`
- Check APIs are enabled: `gcloud services list --enabled`

### Deployment Fails

- Check logs: `gcloud run services logs read dipg-server --region us-central1`
- Verify memory/CPU limits are sufficient
- Check dataset is accessible from Cloud Run

### Evaluation Errors

- Test health endpoint: `curl $SERVICE_URL/health`
- Check dataset path is correct
- Verify request format matches API expectations

## Security

### Authentication (Optional)

To require authentication:

```bash
# Deploy with authentication required
gcloud run deploy dipg-server \
  --image gcr.io/$PROJECT_ID/dipg-server:latest \
  --region us-central1 \
  --no-allow-unauthenticated

# Get auth token
gcloud auth print-identity-token

# Use in requests
curl -H "Authorization: Bearer $(gcloud auth print-identity-token)" $SERVICE_URL/health
```

## Cleanup

```bash
# Delete the Cloud Run service
gcloud run services delete dipg-server --region us-central1

# Delete container images
gcloud container images delete gcr.io/$PROJECT_ID/dipg-server:latest
```

## Next Steps

1. Deploy the server using one of the methods above
2. Update your Colab notebooks with the Cloud Run URL
3. Run evaluations from anywhere!
4. Monitor usage and costs in GCP Console
