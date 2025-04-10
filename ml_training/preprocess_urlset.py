import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pickle
import os
# Assuming feature_extraction.py is in the same directory or accessible
try:
    from feature_extraction import comprehensive_phishing_features
except ImportError:
    print("Error: feature_extraction.py not found. Make sure it's in the ml_training directory.")
    exit()

# --- Constants ---
DATA_PATH = 'data/urlset.csv' # Relative path from project root
OUTPUT_DIR = 'backend/urlset_ml_assets' # NEW directory for these assets
PROCESSED_DATA_FILE = os.path.join(OUTPUT_DIR, 'processed_data.pkl')
SCALER_FILE = os.path.join(OUTPUT_DIR, 'scaler.pkl')
FEATURE_COLUMNS_FILE = os.path.join(OUTPUT_DIR, 'feature_columns.pkl')

# Label Convention for urlset.csv
LABEL_LEGIT = 0
LABEL_PHISHING = 1

# --- Functions ---

def load_data_urlset(file_path):
    """Loads data from the specified CSV file (urlset.csv format)."""
    try:
        # Specify comma delimiter, try latin-1 encoding, and warn on bad lines
        df = pd.read_csv(file_path, delimiter=',', encoding='latin-1', on_bad_lines='warn')
        # Check for required columns
        if 'domain' not in df.columns or 'label' not in df.columns:
            raise ValueError("CSV file must contain 'domain' and 'label' columns.")
        print(f"Data loaded successfully from {file_path}. Shape: {df.shape}")
        # Drop rows where URL or label is missing
        df.dropna(subset=['domain', 'label'], inplace=True)
        # Ensure label is integer
        df['label'] = df['label'].astype(int)
        # Verify label distribution based on THIS dataset's convention
        print(f"Label distribution ({LABEL_LEGIT}=Legit, {LABEL_PHISHING}=Phishing):\n", df['label'].value_counts())
        return df
    except FileNotFoundError:
        print(f"Error: Data file not found at {file_path}")
        return None
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

def extract_features_from_df_urlset(df):
    """Applies feature extraction to each URL in the 'domain' column."""
    print("Extracting features from URLs (using 'domain' column)...")
    # Apply the feature extraction function to the 'domain' column
    features_list = df['domain'].apply(comprehensive_phishing_features).tolist()

    # Convert list of dicts to DataFrame
    features_df = pd.DataFrame(features_list)
    print(f"Feature extraction complete. Shape: {features_df.shape}")
    return features_df

def preprocess_and_save_urlset(data_path=DATA_PATH, test_size=0.2, random_state=42):
    """Loads urlset data, extracts features, preprocesses, splits, and saves."""

    # 1. Load Data
    df_raw = load_data_urlset(data_path)
    if df_raw is None:
        return False

    # Get labels (0=Legit, 1=Phishing)
    labels = df_raw['label'].values

    # 2. Extract Features using the 'domain' column
    df_features = extract_features_from_df_urlset(df_raw)

    if df_features.empty:
         print("Error: Feature extraction resulted in an empty DataFrame.")
         return False

    # 3. Handle Categorical Features (TLD) - Drop
    if 'tld' in df_features.columns:
        print("Dropping 'tld' column.")
        df_features = df_features.drop(columns=['tld'])

    # 4. Ensure all columns are numeric, fill NaNs
    df_features = df_features.apply(pd.to_numeric, errors='coerce').fillna(0)

    # 5. Save feature column order
    feature_columns = df_features.columns.tolist()
    print(f"Using {len(feature_columns)} numerical features.")

    X = df_features.values

    # 6. Split Data
    print("Splitting data...")
    # Stratify based on the labels to maintain distribution
    X_train, X_test, y_train, y_test = train_test_split(
        X, labels, test_size=test_size, random_state=random_state, stratify=labels
    )
    print(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")
    print(f"Train label distribution: Legit={np.sum(y_train == LABEL_LEGIT)}, Phishing={np.sum(y_train == LABEL_PHISHING)}")
    print(f"Test label distribution: Legit={np.sum(y_test == LABEL_LEGIT)}, Phishing={np.sum(y_test == LABEL_PHISHING)}")


    # 7. Scale Numerical Features
    print("Scaling features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 8. Save Processed Data and Assets to the NEW directory
    print(f"Saving processed data and assets to {OUTPUT_DIR}...")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    try:
        # Save data splits
        with open(PROCESSED_DATA_FILE, 'wb') as f:
            pickle.dump({'X_train': X_train_scaled, 'X_test': X_test_scaled,
                         'y_train': y_train, 'y_test': y_test}, f)
        print(f"Processed data saved to {PROCESSED_DATA_FILE}")

        # Save scaler
        with open(SCALER_FILE, 'wb') as f:
            pickle.dump(scaler, f)
        print(f"Scaler saved to {SCALER_FILE}")

        # Save feature column order
        with open(FEATURE_COLUMNS_FILE, 'wb') as f:
            pickle.dump(feature_columns, f)
        print(f"Feature columns saved to {FEATURE_COLUMNS_FILE}")

        return True

    except Exception as e:
        print(f"Error saving processed data/assets: {e}")
        return False

if __name__ == '__main__':
    print("Running preprocessing script for urlset.csv...")
    success = preprocess_and_save_urlset()
    if success:
        print("Preprocessing and saving complete for urlset.csv.")
    else:
        print("Preprocessing failed for urlset.csv.")
