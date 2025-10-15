# Streamlit Cloud Deployment Guide

## ✅ Changes Made for Cloud Deployment

Your app is now **ready for Streamlit Cloud**! Here are the changes that were made:

### 1. **Fixed File Paths** ✓
- Replaced all hardcoded absolute paths (`/mnt/Code/demo_data/...`) with relative paths
- Added `BASE_DIR` and `DATA_DIR` variables at the top of `streamlit_demo.py`
- All file operations now use `Path` objects for cross-platform compatibility

### 2. **Updated runtime.txt** ✓
- Changed from `3.11` to `python-3.11` (correct format for Streamlit Cloud)

### 3. **Requirements.txt** ✓
- Already properly formatted with all dependencies

## 🚀 How to Deploy to Streamlit Cloud

### Step 1: Push to GitHub
Make sure all your changes are committed and pushed to GitHub:

```bash
cd demo1_open
git add .
git commit -m "Fix paths for Streamlit Cloud deployment"
git push origin main
```

### Step 2: Deploy on Streamlit Cloud

1. Go to [share.streamlit.io](https://share.streamlit.io)
2. Sign in with your GitHub account
3. Click "New app"
4. Select:
   - **Repository**: `darren236/demo1_open`
   - **Branch**: `main`
   - **Main file path**: `streamlit_demo.py`
5. Click "Deploy!"

### Step 3: Wait for Deployment
- Streamlit Cloud will automatically install dependencies from `requirements.txt`
- It will use Python 3.11 as specified in `runtime.txt`
- The app will be live at: `https://darren236-demo1-open.streamlit.app` (or similar)

## 📝 Important Notes

### ✅ What Works
- All file paths are now relative and will work on Streamlit Cloud
- All demo data (CSV, images, PDB files) is included in the repository
- 3D visualization with py3Dmol should work fine
- All dependencies are properly specified

### ⚠️ Potential Considerations

1. **Repository Visibility**: The GitHub repo shows "Internal Use Only" but is currently public. If you need it to be private:
   - You can deploy from private repos on Streamlit Cloud (Community or Teams plan)
   - Consider adding a password protection or authentication

2. **File Size**: The repository includes:
   - Multiple PDB structure files
   - Images and demo data
   - Total size should be fine for Streamlit Cloud (< 1GB limit)

3. **Performance**: 
   - 3D structure loading may be slower on first load
   - Consider adding caching if needed (already uses `@st.cache_data` in some places)

## 🧪 Testing Locally

To test the updated app locally:

```bash
cd demo1_open
pip install -r requirements.txt
streamlit run streamlit_demo.py
```

The app will open at `http://localhost:8501`

## 🔧 Configuration (Optional)

If you want to customize the app appearance on Streamlit Cloud, create `.streamlit/config.toml`:

```toml
[theme]
primaryColor = "#1f77b4"
backgroundColor = "#ffffff"
secondaryBackgroundColor = "#f0f2f6"
textColor = "#262730"
font = "sans serif"

[server]
maxUploadSize = 200
enableXsrfProtection = true
```

## 📊 Features Available in Cloud

All features should work on Streamlit Cloud:
- ✅ Interactive protein sequence generation
- ✅ 3D structure visualization (py3Dmol)
- ✅ GO term-based targeting
- ✅ Performance evaluation
- ✅ Image and data visualization

## 🐛 Troubleshooting

If you encounter issues:

1. **Check Streamlit Cloud logs**: Available in the app management console
2. **Verify paths**: All paths should be relative to the repo root
3. **Browser compatibility**: Use Chrome/Edge for best 3D visualization
4. **Clear cache**: Sometimes needed after updates

## 📚 Resources

- [Streamlit Cloud Documentation](https://docs.streamlit.io/streamlit-community-cloud)
- [Deploying Apps](https://docs.streamlit.io/streamlit-community-cloud/deploy-your-app)
- [Managing Apps](https://docs.streamlit.io/streamlit-community-cloud/manage-your-app)

---

**Ready to deploy!** 🎉

All necessary changes have been made. Your app should work perfectly on Streamlit Cloud.

