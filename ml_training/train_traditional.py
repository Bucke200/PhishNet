import pickle
import os
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import time
import numpy as np

# --- Constants ---
ASSET_DIR = 'backend/traditional_ml_assets'
PROCESSED_DATA_FILE = os.path.join(ASSET_DIR, 'processed_data.pkl')
SCALER_FILE = os.path.join(ASSET_DIR, 'scaler.pkl') # Not needed for training, but good to know it exists
FEATURE_COLUMNS_FILE = os.path.join(ASSET_DIR, 'feature_columns.pkl')
MODEL_OUTPUT_FILE = os.path.join(ASSET_DIR, 'traditional_ensemble_model.pkl')

# Label Convention (as per preprocessing)
LABEL_LEGIT = 0
LABEL_PHISHING = 1

# --- Load Data ---
print(f"Loading processed data from {PROCESSED_DATA_FILE}...")
try:
    with open(PROCESSED_DATA_FILE, 'rb') as f:
        data = pickle.load(f)
    X_train_scaled = data['X_train']
    X_test_scaled = data['X_test']
    y_train = data['y_train']
    y_test = data['y_test']
    print("Data loaded successfully.")
    print(f"Training data shape: {X_train_scaled.shape}")
    print(f"Test data shape: {X_test_scaled.shape}")
except FileNotFoundError:
    print(f"Error: Processed data file not found at {PROCESSED_DATA_FILE}. Please run preprocess_traditional.py first.")
    exit()
except Exception as e:
    print(f"Error loading data: {e}")
    exit()

# --- Load Feature Columns (Optional but good practice) ---
try:
    with open(FEATURE_COLUMNS_FILE, 'rb') as f:
        feature_columns = pickle.load(f)
    print(f"Loaded {len(feature_columns)} feature column names.")
except FileNotFoundError:
    print(f"Warning: Feature columns file not found at {FEATURE_COLUMNS_FILE}.")
    feature_columns = None # Continue without them if not found
except Exception as e:
    print(f"Warning: Error loading feature columns: {e}")
    feature_columns = None

# --- Define Individual Models ---
print("Defining individual models for the ensemble...")
# Using balanced class weight where applicable due to potential imbalance
# Setting random_state for reproducibility
# Using n_jobs=-1 for parallelism where possible
rf_clf = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced', n_jobs=-1, verbose=0) # Reduce verbosity
lr_clf = LogisticRegression(random_state=42, class_weight='balanced', solver='liblinear', max_iter=1000) # Use liblinear for smaller datasets
dt_clf = DecisionTreeClassifier(random_state=42, class_weight='balanced')
gb_clf = GradientBoostingClassifier(n_estimators=100, random_state=42, verbose=0) # Reduce verbosity

# --- Create Ensemble Model (Voting Classifier) ---
# 'hard' voting: predicts the class label based on the majority vote of classifiers
# 'soft' voting: predicts the class label based on the argmax of the sums of the predicted probabilities (requires classifiers to have predict_proba)
print("Creating VotingClassifier ensemble (hard voting)...")
ensemble_clf = VotingClassifier(
    estimators=[
        ('rf', rf_clf),
        ('lr', lr_clf),
        ('dt', dt_clf),
        ('gb', gb_clf)
    ],
    voting='hard', # Can change to 'soft' if all classifiers support predict_proba and it performs better
    n_jobs=-1 # Parallelize voting if possible
)

# --- Model Training ---
print("Training the ensemble model...")
start_time = time.time()
ensemble_clf.fit(X_train_scaled, y_train)
end_time = time.time()
print(f"Ensemble training completed in {end_time - start_time:.2f} seconds.")

# --- Model Evaluation ---
print("\nEvaluating model on the test set...")
y_pred = ensemble_clf.predict(X_test_scaled)

accuracy = accuracy_score(y_test, y_pred)
print(f"\nAccuracy: {accuracy:.4f}")

print("\nConfusion Matrix:")
# Use numpy for better formatting if available
try:
    cm = confusion_matrix(y_test, y_pred)
    print(cm)
except NameError: # If numpy wasn't imported or failed
    print(confusion_matrix(y_test, y_pred))


print("\nClassification Report:")
# Define target names based on our label convention
target_names = [f'Legit ({LABEL_LEGIT})', f'Phishing ({LABEL_PHISHING})']
print(classification_report(y_test, y_pred, target_names=target_names))

# --- Save the Trained Model ---
print(f"\nSaving the trained ensemble model to {MODEL_OUTPUT_FILE}...")
try:
    with open(MODEL_OUTPUT_FILE, 'wb') as f:
        pickle.dump(ensemble_clf, f)
    print("Model saved successfully.")
except Exception as e:
    print(f"Error saving model: {e}")

print("\nTraining script finished.")
