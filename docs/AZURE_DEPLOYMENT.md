# 🚀 Azure Deployment Guide - DeepBreast AI

## 📋 Prerequisites

1. [Azure CLI](https://docs.microsoft.com/en-us/cli/azure/install-azure-cli) installed
2. [Docker](https://docs.docker.com/get-docker/) installed
3. Active Azure subscription

---

## 🛠️ Step 1: Azure Setup (One-time)

### Login to Azure
```bash
az login
```

### Create Resource Group
```bash
az group create --name deepbreast-rg --location westeurope
```

### Create Azure Container Registry (ACR)
```bash
az acr create --resource-group deepbreast-rg \
    --name deepbreastai \
    --sku Basic \
    --admin-enabled true
```

### Create Container Apps Environment
```bash
az containerapp env create \
    --name deepbreast-env \
    --resource-group deepbreast-rg \
    --location westeurope
```

---

## 🏗️ Step 2: Build & Push Docker Images

### Login to ACR
```bash
az acr login --name deepbreastai
```

### Build and Push Backend
```bash
docker build -t deepbreastai.azurecr.io/deepbreast-backend:latest .
docker push deepbreastai.azurecr.io/deepbreast-backend:latest
```

### Build and Push Frontend
```bash
docker build -t deepbreastai.azurecr.io/deepbreast-frontend:latest ./frontend
docker push deepbreastai.azurecr.io/deepbreast-frontend:latest
```

---

## 🚀 Step 3: Deploy to Azure Container Apps

### Get ACR Credentials
```bash
ACR_PASSWORD=$(az acr credential show --name deepbreastai --query "passwords[0].value" -o tsv)
```

### Deploy Backend
```bash
az containerapp create \
    --name deepbreast-backend \
    --resource-group deepbreast-rg \
    --environment deepbreast-env \
    --image deepbreastai.azurecr.io/deepbreast-backend:latest \
    --target-port 8000 \
    --ingress external \
    --registry-server deepbreastai.azurecr.io \
    --registry-username deepbreastai \
    --registry-password $ACR_PASSWORD \
    --cpu 2 \
    --memory 4Gi \
    --min-replicas 1 \
    --max-replicas 3
```

### Get Backend URL
```bash
BACKEND_URL=$(az containerapp show --name deepbreast-backend --resource-group deepbreast-rg --query "properties.configuration.ingress.fqdn" -o tsv)
echo "Backend URL: https://$BACKEND_URL"
```

### Deploy Frontend (with Backend URL)
```bash
az containerapp create \
    --name deepbreast-frontend \
    --resource-group deepbreast-rg \
    --environment deepbreast-env \
    --image deepbreastai.azurecr.io/deepbreast-frontend:latest \
    --target-port 80 \
    --ingress external \
    --registry-server deepbreastai.azurecr.io \
    --registry-username deepbreastai \
    --registry-password $ACR_PASSWORD \
    --env-vars VITE_API_URL=https://$BACKEND_URL \
    --cpu 0.5 \
    --memory 1Gi \
    --min-replicas 1 \
    --max-replicas 5
```

### Get Frontend URL
```bash
FRONTEND_URL=$(az containerapp show --name deepbreast-frontend --resource-group deepbreast-rg --query "properties.configuration.ingress.fqdn" -o tsv)
echo "🎉 Application deployed! Visit: https://$FRONTEND_URL"
```

---

## 🔄 Step 4: Update Deployment

### Update Backend
```bash
docker build -t deepbreastai.azurecr.io/deepbreast-backend:latest .
docker push deepbreastai.azurecr.io/deepbreast-backend:latest

az containerapp update \
    --name deepbreast-backend \
    --resource-group deepbreast-rg \
    --image deepbreastai.azurecr.io/deepbreast-backend:latest
```

### Update Frontend
```bash
docker build -t deepbreastai.azurecr.io/deepbreast-frontend:latest ./frontend
docker push deepbreastai.azurecr.io/deepbreast-frontend:latest

az containerapp update \
    --name deepbreast-frontend \
    --resource-group deepbreast-rg \
    --image deepbreastai.azurecr.io/deepbreast-frontend:latest
```

---

## 🖥️ Local Development with Docker

### Run Locally (Mac/Windows/Linux)
```bash
# Build and run both services
docker-compose up -d

# View logs
docker-compose logs -f

# Stop
docker-compose down
```

### Access Local App
- Frontend: http://localhost
- Backend API: http://localhost:8000
- API Docs: http://localhost:8000/docs

---

## 💰 Cost Estimation

| Resource | Configuration | Est. Monthly Cost |
|----------|--------------|-------------------|
| Container Apps (Backend) | 2 vCPU, 4GB RAM | ~$50-80 |
| Container Apps (Frontend) | 0.5 vCPU, 1GB RAM | ~$15-25 |
| Container Registry | Basic tier | ~$5 |
| **Total** | | **~$70-110/month** |

> 💡 Tip: Use Azure Free Tier for testing (includes free credits)

---

## 🔐 GitHub Actions (CI/CD)

### Setup Secrets
Add these secrets to your GitHub repository:

1. Go to Repository **Settings** → **Secrets and variables** → **Actions**
2. Add `AZURE_CREDENTIALS`:
   ```bash
   # Generate service principal
   az ad sp create-for-rbac --name "deepbreast-github" \
       --role contributor \
       --scopes /subscriptions/<subscription-id>/resourceGroups/deepbreast-rg \
       --sdk-auth
   ```
   Copy the JSON output and paste as secret value.

### Auto-Deploy on Push
Once configured, every push to `main` will:
1. Build Docker images
2. Push to Azure Container Registry
3. Deploy to Azure Container Apps

---

## 🛡️ Production Checklist

- [ ] Enable HTTPS (Azure provides free SSL)
- [ ] Configure custom domain
- [ ] Set up Azure Monitor for logging
- [ ] Configure autoscaling rules
- [ ] Enable authentication (Azure AD B2C)
- [ ] Set up backup for data directory
- [ ] Configure CORS for production domain

---

## 🆘 Troubleshooting

### View Container Logs
```bash
az containerapp logs show \
    --name deepbreast-backend \
    --resource-group deepbreast-rg \
    --follow
```

### Restart Container
```bash
az containerapp revision restart \
    --name deepbreast-backend \
    --resource-group deepbreast-rg \
    --revision <revision-name>
```

### Check Health
```bash
curl https://$BACKEND_URL/health
```
