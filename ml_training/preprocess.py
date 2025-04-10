import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
import os

# --- Constants ---
# Using character-level tokenization, so vocab size is based on possible characters
VOCAB_SIZE = 128 # Approximate size for ASCII + common chars, can be adjusted
MAX_URL_LENGTH = 200 # Max length of URL sequence after padding, adjust based on dataset analysis
DATA_PATH = 'data/dtset.xlsx' # Relative path from project root
OUTPUT_DIR = 'backend/ml_assets' # Relative path from project root

# --- Functions ---

def load_data(file_path):
    """Loads data from the specified Excel file."""
    try:
        df = pd.read_excel(file_path)
        # Assuming columns are named 'url' and 'type' (0=legit, 1=phishing)
        if 'url' not in df.columns or 'type' not in df.columns:
            raise ValueError("Excel file must contain 'url' and 'type' columns.")
        print(f"Data loaded successfully from {file_path}. Shape: {df.shape}")
        # Basic check for missing values
        print(f"Missing values:\n{df.isnull().sum()}")
        df.dropna(subset=['url', 'type'], inplace=True) # Drop rows with missing url or type
        df['type'] = df['type'].astype(int) # Ensure type is integer
        return df
    except FileNotFoundError:
        print(f"Error: Data file not found at {file_path}")
        return None
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

def preprocess_urls(urls, max_len=MAX_URL_LENGTH):
    """
    Tokenizes URLs at the character level and pads sequences.
    Returns padded sequences and the tokenizer object.
    """
    # Using char_level=True for character-level tokenization
    tokenizer = Tokenizer(num_words=VOCAB_SIZE, char_level=True, oov_token='<OOV>')
    tokenizer.fit_on_texts(urls)

    sequences = tokenizer.texts_to_sequences(urls)
    padded_sequences = pad_sequences(sequences, maxlen=max_len, padding='post', truncating='post')

    print(f"URLs tokenized and padded. Example sequence shape: {padded_sequences.shape}")
    return padded_sequences, tokenizer

def save_preprocessing_assets(tokenizer, max_len, output_dir):
    """Saves the tokenizer and max_len to the specified directory."""
    os.makedirs(output_dir, exist_ok=True)
    tokenizer_path = os.path.join(output_dir, 'tokenizer.pkl')
    max_len_path = os.path.join(output_dir, 'max_len.pkl')

    try:
        with open(tokenizer_path, 'wb') as f:
            pickle.dump(tokenizer, f)
        with open(max_len_path, 'wb') as f:
            pickle.dump(max_len, f)
        print(f"Tokenizer saved to {tokenizer_path}")
        print(f"Max length ({max_len}) saved to {max_len_path}")
    except Exception as e:
        print(f"Error saving preprocessing assets: {e}")


def load_and_preprocess_data(data_path=DATA_PATH, test_size=0.2, random_state=42):
    """Loads, preprocesses, and splits the data."""
    df = load_data(data_path)
    if df is None:
        return None

    # --- SWAP LABELS: 0=Phishing, 1=Legit ---
    # Original: 0=Legit, 1=Phishing. New: 1-Original
    print("Original label distribution:\n", df['type'].value_counts())
    df['type'] = 1 - df['type']
    print("Swapped label distribution (0=Phishing, 1=Legit):\n", df['type'].value_counts())
    # -----------------------------------------

    urls = df['url'].astype(str).tolist() # Ensure URLs are strings
    labels = df['type'].values

    padded_sequences, tokenizer = preprocess_urls(urls, MAX_URL_LENGTH)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        padded_sequences, labels, test_size=test_size, random_state=random_state, stratify=labels
    )

    print("Data split into training and testing sets:")
    print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
    print(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")

    # Save tokenizer and max_len for later use in the backend
    save_preprocessing_assets(tokenizer, MAX_URL_LENGTH, OUTPUT_DIR)

    return X_train, X_test, y_train, y_test, tokenizer, MAX_URL_LENGTH

if __name__ == '__main__':
    # Example usage when running this script directly
    print("Running preprocessing script...")
    result = load_and_preprocess_data()
    if result:
        print("Preprocessing complete.")
    else:
        print("Preprocessing failed.")
