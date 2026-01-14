# Streamlit Cloud Deployment Troubleshooting

## Current Error: "Error running app. If you need help..."

This error typically occurs due to:

### 1. Missing Dependencies
**Solution:** Use `requirements-streamlit.txt` instead of `requirements.txt`

In Streamlit Cloud settings:
- Go to **Advanced settings**
- Set **Python version**: 3.9 or 3.10
- Set **Requirements file**: `requirements-streamlit.txt`

### 2. Memory Issues
The full `requirements.txt` includes heavy packages that exceed 1GB limit:
- `pyomo==6.7.0` (requires GLPK solver - 200MB+)
- `cvxpy==1.4.1` (requires additional solvers)
- `shap==0.44.1` (memory intensive - 150MB+)

**These are EXCLUDED from `requirements-streamlit.txt`**

### 3. System Dependencies
The `packages.txt` file installs system libraries:
```
libgomp1
glpk-utils
libglpk-dev
```

However, for Streamlit Cloud, **DELETE packages.txt** if optimization features aren't needed.

### 4. Import Errors
The dashboard now handles missing dependencies gracefully:
- Pyomo/CVXPY: Optimization features disabled
- SHAP: Explanation features disabled
- App continues to run with core features

## Deployment Steps

### Option A: Full Features (May fail on free tier)
1. Use `requirements.txt`
2. Keep `packages.txt`
3. Requires Streamlit Cloud Pro (4GB memory)

### Option B: Streamlit Cloud Free Tier (RECOMMENDED)
1. **Delete or rename** `requirements.txt` to `requirements-full.txt`
2. **Rename** `requirements-streamlit.txt` to `requirements.txt`
3. **Delete** `packages.txt` (or comment out glpk lines)
4. Deploy to Streamlit Cloud
5. Expected behavior: Optimization runs with `api.py` stub (demo mode)

### Option C: Use Different Requirements File
In Streamlit Cloud:
1. Click **"⚙️ Settings"**
2. Go to **"Advanced settings"**
3. Under **"Requirements file"**, enter: `requirements-streamlit.txt`
4. Click **"Save"** and **"Reboot"**

## Files Status

| File | Size | Purpose | Cloud Compatibility |
|------|------|---------|-------------------|
| `requirements.txt` | Full | Production (all features) | ❌ Too heavy for free tier |
| `requirements-streamlit.txt` | Lightweight | Cloud deployment | ✅ Works on free tier |
| `packages.txt` | System libs | GLPK solver for Pyomo | ⚠️ Optional (delete if not needed) |
| `api.py` | Stub | Demo optimization | ✅ Always works |
| `dashboard.py` | Main app | Streamlit interface | ✅ Works with graceful degradation |

## Quick Fix

**To fix the current error immediately:**

```bash
cd "d:\FINAL HUMINT DASH"

# Backup current requirements
copy requirements.txt requirements-full.txt

# Use lightweight requirements
copy requirements-streamlit.txt requirements.txt

# Optional: Remove system dependencies
del packages.txt

# Commit and push
git add requirements.txt requirements-streamlit.txt requirements-full.txt
git commit -m "fix: use lightweight requirements for Streamlit Cloud"
git push origin main
```

Then in Streamlit Cloud:
1. Click **"Reboot app"** or **"Manage app" → "Reboot"**
2. Wait 2-3 minutes for redeployment
3. Check logs for any remaining errors

## Expected Behavior After Fix

✅ **Dashboard loads successfully**  
✅ **Login page appears (click logo to access)**  
✅ **All visualization features work**  
✅ **Demo optimization runs via api.py stub**  
⚠️ **SHAP explanations unavailable** (shows message)  
⚠️ **Pyomo optimization unavailable** (uses fallback)

## Verification

After deployment, check:
1. Dashboard loads without errors
2. Can login (demo credentials work)
3. All tabs visible
4. Charts render properly
5. "Run Optimization" shows demo results

## Support

If error persists:
1. Check Streamlit Cloud logs (click "Manage app" → "Logs")
2. Look for specific import errors
3. Verify Python version (3.9 or 3.10)
4. Ensure all files pushed to GitHub (`git ls-files`)

