import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pickle
import os
from feature_extraction import comprehensive_phishing_features # Import the function

# --- Constants ---
DATA_PATH = 'data/dtset.xlsx' # Relative path from project root
OUTPUT_DIR = 'backend/traditional_ml_assets' # New directory for these assets
PROCESSED_DATA_FILE = os.path.join(OUTPUT_DIR, 'processed_data.pkl')
SCALER_FILE = os.path.join(OUTPUT_DIR, 'scaler.pkl')
FEATURE_COLUMNS_FILE = os.path.join(OUTPUT_DIR, 'feature_columns.pkl')

# Label Convention (as per dtset.xlsx)
LABEL_LEGIT = 1
LABEL_PHISHING = 0

# --- Functions ---

def load_data(file_path):
    """Loads data from the specified Excel file."""
    try:
        df = pd.read_excel(file_path)
        if 'url' not in df.columns or 'type' not in df.columns:
            raise ValueError("Excel file must contain 'url' and 'type' columns.")
        print(f"Data loaded successfully from {file_path}. Shape: {df.shape}")
        df.dropna(subset=['url', 'type'], inplace=True)
        df['type'] = df['type'].astype(int)
        # --- KEEP ORIGINAL LABELS: 0=Legit, 1=Phishing ---
        print(f"Label distribution ({LABEL_LEGIT}=Legit, {LABEL_PHISHING}=Phishing):\n", df['type'].value_counts())
        # -------------------------------------------------
        return df
    except FileNotFoundError:
        print(f"Error: Data file not found at {file_path}")
        return None
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

def extract_features_from_df(df):
    """Applies feature extraction to each URL in the DataFrame."""
    print("Extracting features from URLs...")
    # Use .progress_apply for progress bar if tqdm is installed (optional)
    # try:
    #     from tqdm.auto import tqdm
    #     tqdm.pandas()
    #     features_list = df['url'].progress_apply(comprehensive_phishing_features).tolist()
    # except ImportError:
    #     print("tqdm not found, processing without progress bar...")
    features_list = df['url'].apply(comprehensive_phishing_features).tolist()
    
    # Convert list of dicts to DataFrame
    features_df = pd.DataFrame(features_list)
    print(f"Feature extraction complete. Shape: {features_df.shape}")
    return features_df

def preprocess_and_save(data_path=DATA_PATH, test_size=0.2, random_state=42):
    """Loads data, extracts features, preprocesses, splits, and saves."""
    
    # 1. Load Data
    df_raw = load_data(data_path)
    if df_raw is None:
        return False
        
    labels = df_raw['type'].values
    
    # 2. Extract Features
    df_features = extract_features_from_df(df_raw)
    
    # Handle potential errors during feature extraction (empty dicts)
    # For now, we assume it worked, but robust code would handle this
    if df_features.empty:
         print("Error: Feature extraction resulted in an empty DataFrame.")
         return False

    # 3. Handle Categorical Features (TLD) - Drop for now
    if 'tld' in df_features.columns:
        print("Dropping 'tld' column for now.")
        df_features = df_features.drop(columns=['tld'])
    
    # Ensure all columns are numeric, fill NaNs if any resulted from extraction errors
    # (More robust error handling in comprehensive_phishing_features is better)
    df_features = df_features.apply(pd.to_numeric, errors='coerce').fillna(0) 
    
    feature_columns = df_features.columns.tolist()
    print(f"Using {len(feature_columns)} numerical features.")

    X = df_features.values

    # 4. Split Data
    print("Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, labels, test_size=test_size, random_state=random_state, stratify=labels
    )
    print(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")

    # 5. Scale Numerical Features
    print("Scaling features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test) # Use the same scaler fitted on training data

    # 6. Save Processed Data and Assets
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
    print("Running traditional ML preprocessing script...")
    success = preprocess_and_save()
    if success:
        print("Preprocessing and saving complete.")
    else:
        print("Preprocessing failed.")
