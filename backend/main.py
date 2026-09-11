import os
import pathlib
import pickle
from datetime import datetime
from urllib.parse import urlparse  # Added for domain extraction

import pandas as pd  # Added for traditional model preprocessing
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware

# Import the feature extraction function from the copied file
from feature_extraction import comprehensive_phishing_features
from motor.motor_asyncio import AsyncIOMotorClient
from pydantic import BaseModel, HttpUrl

# --- Configuration & Setup ---
# Get the directory where this script (main.py) is located
BACKEND_DIR = pathlib.Path(__file__).parent.resolve()
# Construct the path to the .env file relative to this script
dotenv_path = BACKEND_DIR / ".env"
if dotenv_path.is_file():
    print(f"Loading .env file from: {dotenv_path}")
    load_dotenv(dotenv_path=dotenv_path)
else:
    print(
        f".env file not found at {dotenv_path}. "
        "Using default settings or environment variables."
    )
    # load_dotenv() # Optionally load from CWD as fallback if needed,
    # but might be confusing


# --- Environment Variables ---
MONGO_DETAILS = os.getenv(
    "MONGO_URI", "mongodb://localhost:27017"
)  # Default if not set
DATABASE_NAME = os.getenv("MONGO_DB_NAME", "phishnet_db")
PREDICTIONS_COLLECTION = "predictions"
FEEDBACK_COLLECTION = "feedback"

# --- Paths ---
# Construct absolute paths relative to this script's location
# Urlset Model Assets (Updated)
URLSET_ML_ASSETS_DIR = BACKEND_DIR / "urlset_ml_assets"  # Changed directory
URLSET_MODEL_PATH = (
    URLSET_ML_ASSETS_DIR / "urlset_ensemble_model.pkl"
)  # Changed model filename
URLSET_SCALER_PATH = URLSET_ML_ASSETS_DIR / "scaler.pkl"
URLSET_FEATURE_COLUMNS_PATH = URLSET_ML_ASSETS_DIR / "feature_columns.pkl"

# --- Whitelist ---
# Define a set of known safe domains to bypass model prediction
# Use domain names (netloc) without 'www.' if applicable
WHITELISTED_DOMAINS = {
    "github.com",
    "google.com",
    "wikipedia.org",
    "facebook.com",
    "twitter.com",
    "linkedin.com",
    "youtube.com",
    "amazon.com",
    "apple.com",
    "microsoft.com",
    "bbc.com",
    "nytimes.com",
    "cnn.com",
    "instagram.com",
    "reddit.com",
    "pinterest.com",
    "dropbox.com",
    "adobe.com",
    "reuters.com",
    "bloomberg.com",
    "theguardian.com",
    "aljazeera.com",
    "stackoverflow.com",
    "leetcode.com",
    "codeforces.com",
    "codechef.com",
    # Add other trusted domains here
}

# --- Global Variables ---
app = FastAPI(title="PhishNet Prediction API")
db_client: AsyncIOMotorClient = None
# Urlset Model Assets
urlset_model = None
urlset_scaler = None
urlset_feature_columns = None


# --- Pydantic Models ---
class URLRequest(BaseModel):
    # Keep HttpUrl for initial validation, but convert to str for feature extraction
    url: HttpUrl


class ReportRequest(BaseModel):
    # Keep as string; user corrections might not be valid HttpUrl.
    url: str
    reported_label: int  # 0 for legit, 1 for phishing


# --- Helper Functions ---
def preprocess_single_url_traditional(url: str, scaler_obj, feature_columns_list: list):
    """Extract features and scale a single URL for the traditional model."""
    try:
        # 1. Extract Features
        features_dict = comprehensive_phishing_features(url)
        if not features_dict:
            raise ValueError("Feature extraction returned empty dictionary.")

        # 2. Convert to DataFrame (single row)
        features_df = pd.DataFrame([features_dict])

        # 3. Handle TLD column (drop as done in training)
        if "tld" in features_df.columns:
            features_df = features_df.drop(columns=["tld"])

        # 4. Ensure numeric and fill NaNs (ideally handled in extraction,
        # but kept here as fallback)
        features_df = features_df.apply(pd.to_numeric, errors="coerce").fillna(0)

        # 5. Reorder columns to match training order
        # Ensure all expected columns are present, adding missing ones filled with 0
        for col in feature_columns_list:
            if col not in features_df.columns:
                features_df[col] = 0
        # Correct indentation: This should be outside the for loop
        features_df = features_df[feature_columns_list]  # Select and reorder

        # --- DEBUGGING: Print features before scaling ---
        # --- END DEBUGGING ---

        # 6. Convert DataFrame to NumPy array before scaling to match training
        features_array = features_df.values

        # 7. Scale features using the NumPy array
        # Correct indentation: Align with step 6
        scaled_features = scaler_obj.transform(features_array)

        # --- END DEBUGGING ---

        # Correct indentation: Align with the try block
        return scaled_features  # Returns a 2D numpy array (1 row)

    except Exception as e:
        print(f"Error preprocessing URL for traditional model '{url}': {e}")
        # Raise HTTPException so the API returns a proper error response
        raise HTTPException(
            status_code=500,
            detail=f"Error preprocessing URL for traditional model: {e}",
        ) from e


# --- FastAPI Events ---
@app.on_event("startup")
async def startup_event():
    """Load ML models, assets and connect to MongoDB on startup."""
    # Urlset Assets
    global db_client, urlset_model, urlset_scaler, urlset_feature_columns

    print("API Starting up...")

    # Connect to MongoDB
    print(f"Connecting to MongoDB at {MONGO_DETAILS}...")
    try:
        db_client = AsyncIOMotorClient(MONGO_DETAILS)
        # Ping server to check connection
        await db_client.admin.command("ping")
        print("MongoDB connection successful.")
        app.mongodb = db_client[DATABASE_NAME]  # Attach db object to app state
    except Exception as e:
        print(f"ERROR: Could not connect to MongoDB: {e}")
        db_client = None
        app.mongodb = None

    # --- Load Urlset Model Assets ---
    print("--- Loading Urlset Model Assets ---")
    # Load Model
    print(f"Loading urlset model from {URLSET_MODEL_PATH}...")
    if not os.path.exists(URLSET_MODEL_PATH):
        print(f"ERROR: Urlset model file not found at {URLSET_MODEL_PATH}.")
        urlset_model = None
    else:
        try:
            with open(URLSET_MODEL_PATH, "rb") as f:
                urlset_model = pickle.load(f)
            print("Urlset model loaded successfully.")
        except Exception as e:
            print(f"ERROR: Error loading urlset model: {e}")
            urlset_model = None

    # Load Scaler
    print(f"Loading scaler from {URLSET_SCALER_PATH}...")
    if not os.path.exists(URLSET_SCALER_PATH):
        print(f"ERROR: Scaler file not found at {URLSET_SCALER_PATH}.")
        urlset_scaler = None
    else:
        try:
            with open(URLSET_SCALER_PATH, "rb") as f:
                urlset_scaler = pickle.load(f)
            print("Scaler loaded successfully.")
        except Exception as e:
            print(f"ERROR: Error loading scaler: {e}")
            urlset_scaler = None

    # Load Feature Columns
    print(f"Loading feature columns from {URLSET_FEATURE_COLUMNS_PATH}...")
    if not os.path.exists(URLSET_FEATURE_COLUMNS_PATH):
        print(
            f"ERROR: Feature columns file not found at {URLSET_FEATURE_COLUMNS_PATH}."
        )
        urlset_feature_columns = None
    else:
        try:
            with open(URLSET_FEATURE_COLUMNS_PATH, "rb") as f:
                urlset_feature_columns = pickle.load(f)
            print(
                f"Feature columns loaded successfully "
                f"({len(urlset_feature_columns)} columns)."
            )
        except Exception as e:
            print(f"ERROR: Error loading feature columns: {e}")
            urlset_feature_columns = None

    # --- Final Checks ---
    if not all([urlset_model, urlset_scaler, urlset_feature_columns]):
        print(
            "CRITICAL WARNING: One or more URLSET ML assets failed to load. "
            "Prediction endpoint WILL NOT function correctly."
        )
    if app.mongodb is None:
        print(
            "WARNING: MongoDB connection failed. Logging endpoints will not function."
        )


@app.on_event("shutdown")
async def shutdown_event():
    """Close MongoDB connection on shutdown."""
    global db_client
    print("API Shutting down...")
    if db_client:
        print("Closing MongoDB connection.")
        db_client.close()


# --- CORS Middleware ---
# Allow requests from the Chrome extension (adjust origins if needed)
origins = [
    "chrome-extension://*",  # Be cautious with wildcard in production
    # Add specific extension ID in production:
    # "chrome-extension://<YOUR_EXTENSION_ID>"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for now, restrict in production
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods (GET, POST, etc.)
    allow_headers=["*"],  # Allow all headers
)


# --- API Endpoints ---
@app.post("/predict")
async def predict_url(request: URLRequest, http_request: Request):
    """Predict phishing (1) vs legitimate (0) with the urlset ensemble model."""
    # Check if urlset model assets are loaded
    if not all([urlset_model, urlset_scaler, urlset_feature_columns]):
        raise HTTPException(
            status_code=503,
            detail="Urlset ML Model or assets not loaded. Prediction unavailable.",
        )

    url_to_predict = str(request.url)  # Get URL string from Pydantic model
    print(f"Received prediction request for URL: {url_to_predict}")

    # --- Whitelist Check ---
    try:
        parsed_url = urlparse(url_to_predict)
        domain = parsed_url.netloc
        # Optional: Remove 'www.' prefix for broader matching
        if domain.startswith("www."):
            domain = domain[4:]

        if domain in WHITELISTED_DOMAINS:
            print(
                f"URL domain '{domain}' found in whitelist. Returning safe prediction."
            )
            # Return a safe prediction immediately, bypassing the model
            # Add a flag indicating it was whitelisted
            return {
                "url": url_to_predict,
                "prediction": 0,  # 0 = Legitimate
                "probability": 1.0,  # Assign high confidence for whitelisted
                "whitelisted": True,
            }
    except Exception as parse_e:
        print(f"Warning: Could not parse URL for whitelist check: {parse_e}")
        # Continue to model prediction if parsing fails

    # --- If not whitelisted, proceed with model prediction ---
    print("URL domain not in whitelist. Proceeding with model prediction...")
    # Preprocess the URL using the urlset pipeline
    try:
        # Use the urlset assets for preprocessing
        processed_features = preprocess_single_url_traditional(
            url_to_predict, urlset_scaler, urlset_feature_columns
        )
    except HTTPException as e:
        # If preprocessing itself raised an HTTPException, re-raise it
        raise e
    except Exception as e:
        # Catch any other unexpected errors during preprocessing
        print(f"Unexpected error during preprocessing for urlset model: {e}")
        raise HTTPException(
            status_code=500, detail=f"Unexpected preprocessing error: {e}"
        ) from e

    # Make prediction using the urlset model
    try:
        # Predict directly - VotingClassifier (hard voting) outputs class labels
        prediction = urlset_model.predict(processed_features)[
            0
        ]  # Get the single prediction
        prediction = int(prediction)  # Ensure it's a standard Python int

        # Get probability if the model supports it
        # (requires 'soft' voting and predict_proba)
        prediction_prob = -1.0  # Default if probability is not available
        if hasattr(urlset_model, "predict_proba"):
            try:
                # predict_proba returns probabilities for [class_0, class_1]
                probabilities = urlset_model.predict_proba(processed_features)[0]
                # Use the probability of the predicted class (or class 1 if you prefer)
                prediction_prob = probabilities[prediction]
            except Exception as proba_e:
                print(
                    f"Warning: Could not get probability from urlset model: {proba_e}"
                )

        # REMEMBER: urlset uses 0=Legit, 1=Phishing
        print(
            f"Prediction (Urlset Model): {prediction} "
            f"(Probability: {prediction_prob:.4f})"
        )

    except Exception as e:
        print(f"Error during urlset model prediction: {e}")
        raise HTTPException(
            status_code=500, detail=f"Urlset model prediction error: {e}"
        ) from e

    # Log prediction to MongoDB (if connected)
    if hasattr(http_request.app, "mongodb") and http_request.app.mongodb is not None:
        try:
            db = http_request.app.mongodb
            log_entry = {
                "url": url_to_predict,
                "prediction": prediction,
                "probability": float(prediction_prob),  # Log probability if available
                "model_type": "urlset_ensemble",  # Update model type identifier
                "timestamp": datetime.utcnow(),
            }
            await db[PREDICTIONS_COLLECTION].insert_one(log_entry)
            print("Prediction logged to MongoDB.")
        except Exception as e:
            print(f"Error logging prediction to MongoDB: {e}")
    else:
        print("Skipping MongoDB logging (connection unavailable).")

    # Add the whitelisted flag (False if model prediction was used)
    return {
        "url": url_to_predict,
        "prediction": prediction,
        "probability": float(prediction_prob),
        "whitelisted": False,
    }


@app.post("/report")
async def report_prediction(report: ReportRequest, http_request: Request):
    """Logs user feedback about an incorrect prediction."""
    print(
        f"Received feedback report for URL: {report.url}, "
        f"Reported Label: {report.reported_label}"
    )

    # Check if mongodb attribute exists and is None (connection failed at startup)
    if not hasattr(http_request.app, "mongodb") or http_request.app.mongodb is None:
        raise HTTPException(
            status_code=503,
            detail="Database connection unavailable. Cannot log feedback.",
        )

    if report.reported_label not in [0, 1]:
        raise HTTPException(
            status_code=400, detail="Invalid reported_label. Must be 0 or 1."
        )

    try:
        db = http_request.app.mongodb
        feedback_entry = {
            "url": report.url,
            "reported_label": report.reported_label,
            "timestamp": datetime.utcnow(),
        }
        await db[FEEDBACK_COLLECTION].insert_one(feedback_entry)
        print("Feedback logged to MongoDB.")
        return {"message": "Feedback received successfully."}
    except Exception as e:
        print(f"Error logging feedback to MongoDB: {e}")
        raise HTTPException(
            status_code=500, detail=f"Error saving feedback: {e}"
        ) from e


@app.get("/")
async def root():
    """Root endpoint for basic API check."""
    return {"message": "PhishNet Prediction API is running."}


# --- Run with Uvicorn (for local testing) ---
# You would typically run this using: uvicorn backend.main:app --reload --port 8000
# The following block is usually commented out or removed in production setups.
# if __name__ == "__main__":
#     import uvicorn
#     print("Starting Uvicorn server directly...")
#     uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
