#  Health Check & Troubleshooting

## System Health Check

The dashboard now includes a comprehensive health check system visible at the top of the app.

### What It Shows:

1. **Core Packages Status** (10 packages monitored)
   - Streamlit, NumPy, Pandas, Plotly, Matplotlib
   - XGBoost, Scikit-learn, TensorFlow, SHAP, Pyomo
   - Shows version numbers and availability

2. **API Status**
   -  Available: Using local api.py module
   -  Fallback: Using stub functions (limited features)

3. **Model Files**
   - Checks for 4 critical model files
   - Shows which files are present/missing

4. **Detailed Diagnostics**
   - Python version
   - Package versions
   - Model file paths
   - API configuration

## If You See Loading Screen:

1. **Wait 2-3 minutes** - Streamlit Cloud builds on first deploy
2. **Check Build Logs** in Streamlit Cloud dashboard
3. **Look for the startup banner** - Should say \"Loading ML-TSSP Dashboard\"
4. **Expand Health Check** - See exactly what's working/missing

## Common Issues:

### Issue: Stuck on loading screen
**Cause:** Build process or startup error  
**Fix:** Check Streamlit Cloud logs, reboot app

### Issue: \"Package not available\" warnings
**Cause:** Missing dependencies in requirements.txt  
**Fix:** Already included in requirements.txt, wait for rebuild

### Issue: \"Model files missing\"
**Cause:** Model files not in repository or wrong path  
**Fix:** Model files are committed and pushed

### Issue: \"API Fallback\" status
**Cause:** api.py not found or import error  
**Fix:** Already fixed with graceful fallback

## Health Check Location:

- **In App:** Top of page, expandable \" System Health Check\"
- **Startup Banner:** Shows \" Loading ML-TSSP Dashboard...\"
- **After Load:** Health check collapses, full app appears

## What Should Work:

 Dashboard loads within 1-2 minutes  
 Health check shows all green statuses  
 Main navigation appears  
 Optimization controls visible  
 All tabs functional

## If Still Stuck:

1. Go to Streamlit Cloud dashboard
2. Click \"Reboot app\"
3. Or delete and redeploy from scratch
4. Check that runtime.txt = python-3.11
5. Verify requirements.txt is present

## Live URL:
Your app should be at: https://share.streamlit.io/lucaswafula/aegis-intel

Wait ~2 minutes after push for automatic rebuild! 
