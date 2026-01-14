# Render.com Deployment Guide

## Prerequisites
- GitHub account with aegis-INTEL repository
- Render.com account (free signup)

## Deployment Steps

### Option 1: Deploy via Render Dashboard (Recommended)

1. **Sign Up / Login to Render**
   - Go to https://render.com
   - Sign up with GitHub (easier integration)

2. **Create New Web Service**
   - Click "New +" button (top right)
   - Select "Web Service"

3. **Connect Your Repository**
   - If first time: Click "Connect GitHub" → Authorize Render
   - Select your GitHub account
   - Find and select `lucasWAFULA/aegis-INTEL` repository
   - Click "Connect"

4. **Configure the Service**
   Fill in the following settings:

   **Name:** `aegis-intel-dashboard` (or your preferred name)
   
   **Region:** Oregon (US West) or Frankfurt (EU)
   
   **Branch:** `main`
   
   **Root Directory:** Leave blank
   
   **Runtime:** `Python 3`
   
   **Build Command:**
   ```
   pip install -r requirements.txt
   ```
   
   **Start Command:**
   ```
   streamlit run dashboard.py --server.port=$PORT --server.address=0.0.0.0 --server.headless=true
   ```

5. **Choose Your Plan**
   - **Free Plan:** 512MB RAM (may struggle with full app)
   - **Starter Plan:** $7/month - 512MB RAM (recommended minimum)
   - **Standard Plan:** $25/month - 2GB RAM (best performance)
   
   **Recommendation:** Start with Starter ($7/month)

6. **Advanced Settings** (Click "Advanced")
   
   Add these Environment Variables:
   ```
   PYTHON_VERSION = 3.10.0
   STREAMLIT_SERVER_HEADLESS = true
   STREAMLIT_SERVER_ENABLE_CORS = false
   STREAMLIT_SERVER_ENABLE_XSRF_PROTECTION = false
   ```

7. **Deploy**
   - Click "Create Web Service"
   - Render will start building (5-10 minutes first time)
   - Watch the build logs in real-time

8. **Access Your App**
   - Once deployed, you'll get a URL like: `https://aegis-intel-dashboard.onrender.com`
   - Click the URL to access your dashboard

---

### Option 2: Deploy via Blueprint (Using render.yaml)

1. **Push render.yaml to GitHub**
   ```bash
   cd "d:\FINAL HUMINT DASH"
   git add render.yaml Aptfile RENDER_DEPLOYMENT.md
   git commit -m "feat: add Render deployment configuration"
   git push origin main
   ```

2. **Deploy via Blueprint**
   - Go to https://render.com/dashboard
   - Click "New +" → "Blueprint"
   - Connect your GitHub repo
   - Render automatically detects `render.yaml`
   - Click "Apply" → Service deploys automatically

---

## Monitoring & Management

### Check Deployment Status
- Dashboard → Your service → "Logs" tab
- Watch for errors during build/deployment

### Common Build Issues

**Issue: Memory exceeded during pip install**
- Solution: Upgrade to Standard plan (2GB RAM)

**Issue: Module not found errors**
- Solution: Check requirements.txt is in root directory
- Verify all packages are spelled correctly

**Issue: Port binding error**
- Solution: Ensure start command uses `--server.port=$PORT`

### Updating Your App

**Automatic Deploys:**
- Every push to `main` branch triggers auto-deployment
- Disable in: Settings → "Auto-Deploy" toggle

**Manual Deploy:**
- Dashboard → Your service → "Manual Deploy" → "Deploy latest commit"

### View Logs
```
Dashboard → Your service → Logs
```

### Restart Service
```
Dashboard → Your service → Manual Deploy → "Clear build cache & deploy"
```

---

## Cost Estimate

| Plan | RAM | CPU | Monthly Cost | Suitable For |
|------|-----|-----|--------------|--------------|
| Free | 512MB | 0.1 CPU | $0 | Testing only (may fail) |
| Starter | 512MB | 0.5 CPU | $7 | Development, light use |
| Standard | 2GB | 1 CPU | $25 | **Production (recommended)** |
| Pro | 4GB | 2 CPU | $85 | High traffic |

**Recommendation for your app:** Standard ($25/month) for reliable performance with all ML features.

---

## Custom Domain (Optional)

1. Go to service Settings → "Custom Domain"
2. Add your domain (e.g., `dashboard.yourdomain.com`)
3. Update DNS records as instructed
4. Render provides free SSL certificate

---

## Environment Variables for Production

Add these in Render Dashboard → Settings → Environment:

```
STREAMLIT_SERVER_HEADLESS=true
STREAMLIT_SERVER_ENABLE_CORS=false
STREAMLIT_BROWSER_GATHER_USAGE_STATS=false
```

---

## Troubleshooting

### App shows "Application Error"
1. Check Logs for Python errors
2. Verify all dependencies installed
3. Check memory usage (upgrade if needed)

### Build takes too long / times out
1. Render free tier has 15-minute build limit
2. Upgrade to paid plan for longer builds
3. Your app may take 8-12 minutes to build (TensorFlow + SHAP)

### App crashes after deployment
1. Check if memory exceeded (Logs will show "OOMKilled")
2. Upgrade to Standard plan (2GB RAM)
3. Verify all model files are in repository

---

## Next Steps After Deployment

1. ✅ Test all features (login, optimization, charts)
2. ✅ Set up custom domain (optional)
3. ✅ Configure auto-deploy settings
4. ✅ Set up health checks (Settings → "Health Check Path": `/`)
5. ✅ Add secrets/credentials via Environment Variables

---

## Support

- Render Docs: https://render.com/docs
- Community: https://community.render.com
- Status: https://status.render.com

