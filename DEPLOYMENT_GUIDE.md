# FRAUDGUARD: Complete Deployment Guide (Frontend + Backend)

> **Deploying Your College Machine Learning Full-Stack Project to the Cloud**

---

## 1. How Deployment Works in this Project

Because this application uses **Flask** as a unified full-stack architecture, **your Frontend and Backend are packaged together into a single web service**:

```
                              ┌──────────────────────────────────────────────┐
                              │           CLOUD SERVER (Render / PythonAnywhere)        │
                              │                                              │
User Browser                  │  Flask Backend (app.py)                       │
  │                           │   ├── Serves HTML/CSS/JS (Frontend)          │
  ├── GET / ─────────────────►│   │    (index.html, predict.html, etc.)      │
  │                           │   │                                          │
  ├── POST /predict ─────────►│   └── Runs ML Pipeline (Backend)             │
  │   (Form / JSON payload)   │        (StandardScaler + RandomForest)       │
  │                           │                                              │
  └── Receives Result HTML ◄──│                                              │
                              └──────────────────────────────────────────────┘
```

**Key Advantage for College Submission:**  
You do **NOT** need two separate servers or CORS configurations. You deploy **one service**, and both your frontend web interface and backend ML engine go live under a single public URL (e.g., `https://fraudguard.onrender.com`).

---

## 2. Option 1: Deploy on Render.com (Recommended — 100% Free)

[Render](https://render.com) is the most popular, modern cloud hosting platform for Flask college projects.

### Step 1: Push Your Project to GitHub
1. Open PowerShell in `d:\credit-card-fraud-detection`:
   ```powershell
   git init
   git add .
   git commit -m "Deploy FraudGuard Full-Stack Project"
   ```
   *(Note: The 150MB CSV is already in `.gitignore`. Your serialized models in `models/` are lightweight and self-contained, so the entire repo is under 10MB!)*

2. Create a new repository on [GitHub.com](https://github.com) named `credit-card-fraud-detection`.
3. Link and push your repository:
   ```powershell
   git branch -M main
   git remote add origin https://github.com/<your-username>/credit-card-fraud-detection.git
   git push -u origin main
   ```

### Step 2: Deploy on Render
1. Go to [https://render.com](https://render.com) and sign up / log in with your GitHub account.
2. In the Render Dashboard, click **New +** and select **Web Service**.
3. Select your `credit-card-fraud-detection` repository.
4. Fill in the deployment details:
   - **Name:** `fraudguard` (or any custom name)
   - **Region:** Any (e.g., *Singapore* or *Frankfurt*)
   - **Branch:** `main`
   - **Runtime:** `Python 3`
   - **Build Command:** `pip install -r requirements.txt`
   - **Start Command:** `gunicorn app:app`
   - **Instance Type:** `Free`
5. Click **Create Web Service**.

Render will automatically install dependencies, load your pre-trained models, and generate a live URL:
👉 `https://fraudguard.onrender.com`

---

## 3. Option 2: Deploy on PythonAnywhere (No GitHub Required)

If you prefer not to use GitHub, [PythonAnywhere](https://www.pythonanywhere.com) lets you host Flask apps for free.

### Step 1: Sign Up
Create a free beginner account at [https://www.pythonanywhere.com](https://www.pythonanywhere.com).

### Step 2: Upload Files
1. Go to the **Files** tab.
2. Upload your project folder or clone it via the Bash console:
   ```bash
   git clone https://github.com/<your-username>/credit-card-fraud-detection.git
   ```

### Step 3: Configure Virtualenv & Web App
1. Go to the **Web** tab and click **Add a new web app**.
2. Select **Flask** and **Python 3.10 / 3.11**.
3. In the virtualenv path, create and activate a virtual environment:
   ```bash
   mkvirtualenv --python=/usr/bin/python3.10 fraudguard-env
   pip install -r requirements.txt
   ```
4. In the **WSGI configuration file** link, set:
   ```python
   import sys
   path = '/home/<your-username>/credit-card-fraud-detection'
   if path not in sys.path:
       sys.path.append(path)
   from app import app as application
   ```
5. Click **Reload <your-username>.pythonanywhere.com**. Your project is now live!

---

## 4. Option 3: Local Network College Presentation (No Cloud Needed)

If your college exam room has poor internet connectivity or firewalls, you can run the full application locally on your laptop and allow examiners/classmates to connect over the same Wi-Fi network:

1. Connect your laptop to the college Wi-Fi or mobile hotspot.
2. Find your laptop's local IP address:
   ```powershell
   ipconfig
   ```
   *(Look for `IPv4 Address`, e.g., `192.168.1.45`)*
3. Start the application:
   ```powershell
   python app.py
   ```
4. Any device (examiner's laptop, tablet, or phone) connected to the same Wi-Fi can now open:
   ```
   http://192.168.1.45:5000
   ```

---

## 5. Deployment Verification Checklist

Before presenting your deployed application:
- [x] `requirements.txt` includes `gunicorn` for cloud WSGI serving.
- [x] `Procfile` is created with `web: gunicorn app:app`.
- [x] `app.py` binds to `0.0.0.0` and listens to the environment `$PORT`.
- [x] Large raw CSV is excluded from Git via `.gitignore` (prevents GitHub upload error).
- [x] Pre-trained models (`models/fraud_model.pkl`, `models/scaler.pkl`) and sample test transactions (`models/sample_transactions.json`) are committed and ready.
