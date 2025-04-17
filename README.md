# PhishNet - Phishing Detection System

## Introduction

PhishNet is a system designed to detect phishing URLs in real-time. It utilizes a machine learning model (URLSet Ensemble) trained on URL characteristics, combined with a backend API and a browser extension for seamless integration. When you browse the web, the extension sends the current URL to the backend API, which uses the trained model to predict whether the URL is likely malicious (phishing) or legitimate.

This project consists of three main components:
1.  **Backend:** A FastAPI application that serves the ML model predictions via an API endpoint.
2.  **Machine Learning (URLSet Ensemble):** A model trained using features extracted from the `urlset.csv` dataset. The training scripts and related assets are included.
3.  **Browser Extension:** A simple browser extension that communicates with the backend API to check URLs as you visit them.

## Features

*   Real-time URL analysis via browser extension.
*   Phishing detection powered by a URLSet Ensemble machine learning model.
*   FastAPI backend for efficient API request handling.
*   Modular structure with separate components for the backend, ML training, and extension.
*   Includes scripts for data preprocessing and model retraining.

## Project Structure

```
PhishNet/
├── backend/                  # FastAPI backend application
│   ├── ml_assets/            # (Contains assets for other models - not used by default)
│   ├── traditional_ml_assets/ # (Contains assets for other models - not used by default)
│   ├── urlset_ml_assets/     # Model assets (scaler, columns, model) - NOT in repo, see below
│   ├── .env.example          # Example environment file for MongoDB URI
│   ├── feature_extraction.py # Feature extraction logic for prediction
│   ├── main.py               # FastAPI application entry point
│   └── requirements.txt      # Backend Python dependencies
├── backend_env/              # (Ignored) Virtual environment for backend
├── data/                     # (Ignored) Datasets
├── extension/                # Browser extension files
│   ├── icons/                # Extension icons
│   ├── background.js         # Extension logic
│   └── manifest.json         # Extension configuration
├── ml_training/              # Scripts for ML model training
│   ├── feature_extraction.py # Feature extraction logic for training
│   ├── preprocess_urlset.py  # Preprocessing script for urlset.csv
│   ├── train_urlset.py       # Training script for the URLSet model
│   └── requirements.txt      # ML training Python dependencies
├── ml_training_env/          # (Ignored) Virtual environment for ML training
├── README.md                 # This file
└── ...                       # Other configuration/installer files
```

**Note:** Virtual environments, large binary/model/data files, and sensitive assets are not stored in the repository. See below for how to obtain required model files.

## Setup Instructions

### Prerequisites

*   **Python:** Version 3.10 or newer recommended.
*   **Git:** For cloning the repository.
*   **MongoDB:** A MongoDB instance is required for the backend to potentially store data (though the current core prediction logic might not heavily rely on it, it's often used in such systems). We strongly recommend using MongoDB Atlas (a cloud-based service) for ease of setup.

### Important: Model Files and Data

- **Model and data files (`.pkl`, etc.) are NOT included in the repository** due to GitHub file size and LFS bandwidth limits.
- You must **download the required model files** (e.g. `urlset_ensemble_model.pkl`, `scaler.pkl`, `feature_columns.pkl`, `processed_data.pkl`) from cloud storage and place them in `backend/urlset_ml_assets/`.
- A script (`download_models.py`) is provided to automate this step. Update the script with your actual download links.

#### Example: Downloading Model Files

1. Edit `backend/download_models.py` to include the direct download links for your model files.
2. Run:
    ```bash
    cd backend
    python download_models.py
    ```
   This will fetch all required model assets into the correct directory.

### Installation

1.  **Clone the Repository:**
    ```bash
    git clone <repository-url> # Replace <repository-url> with the actual URL
    cd PhishNet
    ```

2.  **MongoDB Atlas Setup (Recommended):**
    *   Go to [MongoDB Atlas](https://www.mongodb.com/cloud/atlas) and sign up for a free account.
    *   Create a new **Free Tier Cluster** (M0). Choose a cloud provider and region close to you.
    *   **Create a Database User:** Under "Database Access", create a new user with a secure username and password. Grant this user "Read and write to any database" privileges for simplicity in this project context. **Remember this username and password.**
    *   **Configure Network Access:** Under "Network Access", click "Add IP Address". For ease of use (especially if your IP address changes), select "ALLOW ACCESS FROM ANYWHERE". This will add the entry `0.0.0.0/0`.
        *   **Security Note:** Allowing access from anywhere (`0.0.0.0/0`) is less secure than whitelisting specific IP addresses. While convenient for development/testing, be aware that anyone on the internet could *attempt* to connect if they guess your username/password. For production systems, restricting access to specific IPs is highly recommended.
    *   **Get Connection String:** Go back to your cluster's "Overview" tab and click "Connect". Choose "Connect your application". Select "Python" as the driver and the appropriate version. Copy the provided **connection string (URI)**. It will look something like `mongodb+srv://<username>:<password>@<cluster-address>/...`.

3.  **Backend Setup:**
    *   Navigate to the backend directory:
        ```bash
        cd backend
        ```
    *   Create and activate a Python virtual environment:
        ```bash
        # Windows (cmd)
        python -m venv ../backend_env
        ..\backend_env\Scripts\activate

        # Linux / macOS / Git Bash
        # python3 -m venv ../backend_env
        # source ../backend_env/bin/activate
        ```
    *   Install dependencies:
        ```bash
        pip install -r requirements.txt
        ```
    *   Create the environment configuration file:
        *   Copy the example file: `cp .env.example .env` (or copy manually).
        *   Edit the `.env` file with a text editor.
        *   Replace `your_mongodb_connection_string_here` with the actual **MongoDB Atlas connection string** you copied earlier. Make sure to replace `<username>` and `<password>` in the string with the database user credentials you created. You can also specify a database name (e.g., `phishnet`) in the string if it's not already there (e.g., `...mongodb.net/phishnet?retryWrites=true...`).

4.  **ML Training Setup (Optional - Only if you want to retrain the model):**
    *   Navigate to the ML training directory:
        ```bash
        cd ../ml_training
        ```
    *   Create and activate a Python virtual environment:
        ```bash
        # Windows (cmd)
        python -m venv ../ml_training_env
        ..\ml_training_env\Scripts\activate

        # Linux / macOS / Git Bash
        # python3 -m venv ../ml_training_env
        # source ../ml_training_env/bin/activate
        ```
    *   Install dependencies:
        ```bash
        pip install -r requirements.txt
        ```

5.  **Browser Extension Setup:**
    *   Open your Chromium-based browser (Chrome, Edge, Brave, etc.).
    *   Go to the extensions management page (e.g., `chrome://extensions` or `edge://extensions`).
    *   Enable **"Developer mode"** (usually a toggle switch in the top right corner).
    *   Click **"Load unpacked"**.
    *   Navigate to and select the `PhishNet/extension` directory from this project.
    *   The PhishNet extension icon should appear in your browser toolbar.

## Running the Application

### Deploying on Render (Recommended)

1. **Push your code to GitHub (without large model/data files).**
2. **Deploy on [Render.com](https://dashboard.render.com/):**
    - Create a new Web Service, connect your repo, set root directory to `backend`.
    - Set environment variables from your `.env` file.
    - Set port to `8000`.
    - In your Dockerfile, ensure the model download script runs before starting the server (see `CMD` example below).
3. **Model Download on Deploy:**
    - The `download_models.py` script will automatically fetch required models from your cloud storage.
4. **Update the browser extension:**
    - In `extension/background.js`, change the backend URL from `http://localhost:8000` to your Render public URL (e.g., `https://your-app.onrender.com`).
    - Reload the extension in your browser.

### Local Development

1.  **Start the Backend Server:**
    *   Ensure you have downloaded the required model files to `backend/urlset_ml_assets/` (see above).
    *   Open a terminal or command prompt.
    *   Navigate to the backend directory: `cd path/to/PhishNet/backend`
    *   Activate the backend virtual environment:
        ```bash
        # Windows (cmd)
        ..\backend_env\Scripts\activate

        # Linux / macOS / Git Bash
        # source ../backend_env/bin/activate
        ```
    *   Run the Uvicorn server:
        ```bash
        uvicorn main:app --reload --host 0.0.0.0 --port 8000
        ```

    *   You should see output indicating the server is running, typically on `http://0.0.0.0:8000`.

2.  **Using the Extension:**
    *   With the backend server running, the browser extension should automatically communicate with it (`http://localhost:8000` by default, as configured in `extension/background.js`).
    *   As you navigate to different websites, the extension icon may change (e.g., to a warning icon) based on the prediction received from the backend.

## Training the Model (Optional)

If you want to retrain the URLSet ensemble model using the provided data or your own data:

1.  Ensure you have completed the **ML Training Setup** steps (virtual environment activated, dependencies installed).
2.  Navigate to the ML training directory: `cd path/to/PhishNet/ml_training`
3.  Run the training script:
    ```bash
    python train_urlset.py
    ```
    *   This script will typically perform preprocessing (using `preprocess_urlset.py` and `feature_extraction.py`) on the `data/urlset.csv` file and then train the ensemble model.
4.  **Copy Assets:** After successful training, new model assets will likely be generated within the `ml_training` directory (or a subdirectory). You need to manually copy the updated assets (e.g., `urlset_ensemble_model.pkl`, `scaler.pkl`, `feature_columns.pkl`, `processed_data.pkl`) to the `backend/urlset_ml_assets/` directory, overwriting the existing files.
5.  Restart the backend server for the changes to take effect.

## License

This project is licensed under the MIT License.

---

## Notes on Large Files & LFS

- This repository does **not** store model/data files or virtual environments. These are ignored via `.gitignore`.
- **GitHub hard-limits files to 100 MB, even with Git LFS.**
- Model files are distributed via cloud storage and must be downloaded as described above.
