#!/usr/bin/env bash
#
# Simple deployment script for DIPG Server to Google Cloud Run
# Following Google Cloud Run best practices
#

set -e

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${GREEN}🚀 Deploying DIPG Server to Cloud Run${NC}"
echo "======================================"

# Get project ID
PROJECT_ID=$(gcloud config get-value project 2>/dev/null)
if [ -z "$PROJECT_ID" ]; then
    echo "Error: No GCP project set"
    exit 1
fi

echo -e "${GREEN}📦 Project: $PROJECT_ID${NC}"
echo -e "${GREEN}🌍 Region: us-central1${NC}"

# Create service account if it doesn't exist
echo -e "\n${YELLOW}🔐 Setting up service account...${NC}"
if ! gcloud iam service-accounts describe dipg-server-sa@$PROJECT_ID.iam.gserviceaccount.com &>/dev/null; then
    gcloud iam service-accounts create dipg-server-sa \
        --display-name="DIPG Server Service Account"
    echo -e "${GREEN}✅ Service account created${NC}"
else
    echo -e "${GREEN}✅ Service account already exists${NC}"
fi

# Deploy to Cloud Run from source
echo -e "\n${YELLOW}🚀 Deploying to Cloud Run...${NC}"
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

# Get the service URL
echo -e "\n${GREEN}✅ Deployment complete!${NC}"
SERVICE_URL=$(gcloud run services describe dipg-server --region us-central1 --format 'value(status.url)')

echo -e "\n${GREEN}🌐 Service URL:${NC}"
echo "$SERVICE_URL"

# Test the deployment
echo -e "\n${YELLOW}🧪 Testing deployment...${NC}"
if curl -s -f "$SERVICE_URL/health" > /dev/null; then
    echo -e "${GREEN}✅ Health check passed!${NC}"
else
    echo -e "${YELLOW}⚠️  Health check pending (server may still be starting)${NC}"
fi

# Print usage instructions
echo -e "\n${GREEN}📝 Next Steps:${NC}"
echo "1. Set the server URL in your environment:"
echo "   export DIPG_SERVER_URL=\"$SERVICE_URL\""
echo ""
echo "2. Use in Colab:"
echo "   import os"
echo "   os.environ['DIPG_SERVER_URL'] = '$SERVICE_URL'"
echo "   from examples.run_eval_colab import run_evaluation_http"
echo "   metrics = run_evaluation_http(model, tokenizer)"
echo ""
echo "3. View logs:"
echo "   gcloud run services logs read dipg-server --region us-central1"

echo -e "\n${GREEN}🎉 Deployment successful!${NC}"
