# DIPG Server - Cloud Run Deployment

## 🎉 Deployment Successful!

The DIPG Safety Gym evaluation server is now live on Google Cloud Run!

### Service Information

- **Service URL**: https://dipg-server-5hurbdoigq-uc.a.run.app
- **Region**: us-central1
- **Project**: gem-creator
- **Status**: ✅ Running

### Quick Start

#### For Colab Notebooks

```python
import os

# Set the Cloud Run URL
os.environ['DIPG_SERVER_URL'] = 'https://dipg-server-5hurbdoigq-uc.a.run.app'

# Import and run evaluation
from examples.run_eval_colab import run_evaluation_http

metrics = run_evaluation_http(
    model=model,
    tokenizer=tokenizer,
    num_samples=100
)
```

#### For Local Testing

```bash
# Set environment variable
export DIPG_SERVER_URL="https://dipg-server-5hurbdoigq-uc.a.run.app"

# Test health endpoint
curl https://dipg-server-5hurbdoigq-uc.a.run.app/health

# Run evaluation
python examples/run_eval_colab.py
```

### Deployment Details

- **Deployment Method**: `gcloud run deploy --source`
- **Build Tool**: `uv` (fast Python package manager)
- **Memory**: 2Gi
- **CPU**: 2 cores
- **Timeout**: 300 seconds
- **Max Instances**: 10
- **Authentication**: Public (allow-unauthenticated)

### Monitoring

```bash
# View logs
gcloud run services logs read dipg-server --region us-central1

# View service details
gcloud run services describe dipg-server --region us-central1

# View in Cloud Console
https://console.cloud.google.com/run/detail/us-central1/dipg-server?project=gem-creator
```

### Redeployment

To redeploy after making changes:

```bash
./deploy.sh
```

### Cost Estimation

- **Free tier**: 2 million requests/month
- **Estimated cost**: ~$0.05 per 1000 evaluations
- **Cold start**: ~2-3 seconds
- **Warm request**: ~100-500ms

### Design Notes

**Current Design**: Dataset loaded from environment variable (`DIPG_DATASET_PATH=surfiniaburger/dipg-eval-dataset`)

**Future Improvements** (tracked in task.md):
1. Make dataset path optional in request body
2. Support multiple datasets per request
3. Implement lazy loading + caching
4. Add dataset management endpoints

### Troubleshooting

If you encounter issues:

1. **Check logs**: `gcloud run services logs read dipg-server --region us-central1`
2. **Verify health**: `curl https://dipg-server-5hurbdoigq-uc.a.run.app/health`
3. **Check service status**: `gcloud run services describe dipg-server --region us-central1`

### Next Steps

1. ✅ Server deployed and running
2. ✅ Health check passing
3. 📝 Update Colab notebooks with Cloud Run URL
4. 📝 Re-train SFT model with JSON dataset
5. 📝 Run GRPO training with proper format

---

**Deployment Date**: 2025-11-30  
**Deployed By**: Automated deployment script  
**Version**: Initial release
