import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Conv1D, GlobalMaxPooling1D, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import numpy as np
import os

# Import preprocessing function and constants
from preprocess import load_and_preprocess_data, VOCAB_SIZE, MAX_URL_LENGTH

# --- Constants ---
EMBEDDING_DIM = 64  # Dimension of the embedding vectors
NUM_FILTERS = 128   # Number of filters in the Conv1D layer
KERNEL_SIZE = 5     # Kernel size for the Conv1D layer
HIDDEN_UNITS = 128  # Number of units in the dense hidden layer
DROPOUT_RATE = 0.5  # Dropout rate for regularization
EPOCHS = 10         # Number of training epochs (can be adjusted)
BATCH_SIZE = 128    # Batch size for training
MODEL_SAVE_DIR = 'backend/ml_assets' # Relative path from project root
MODEL_NAME = 'phishnet_cnn_model.h5' # Name for the saved model file

# --- Build Model ---

def build_cnn_model(vocab_size=VOCAB_SIZE, embedding_dim=EMBEDDING_DIM, max_length=MAX_URL_LENGTH):
    """Builds the CNN model architecture."""
    print("Building CNN model...")
    model = Sequential([
        # 1. Embedding Layer: Turns positive integers (indexes) into dense vectors of fixed size.
        Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=max_length),

        # 2. Convolutional Layer: Learns local patterns in the sequence.
        Conv1D(filters=NUM_FILTERS, kernel_size=KERNEL_SIZE, activation='relu'),

        # 3. Pooling Layer: Reduces dimensionality, makes model robust to variations.
        GlobalMaxPooling1D(),

        # 4. Dense Hidden Layer: Learns higher-level combinations of features.
        Dense(units=HIDDEN_UNITS, activation='relu'),

        # 5. Dropout Layer: Regularization to prevent overfitting.
        Dropout(DROPOUT_RATE),

        # 6. Output Layer: Single neuron with sigmoid activation for binary classification (0 or 1).
        Dense(units=1, activation='sigmoid')
    ])

    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    print("Model built successfully.")
    model.summary() # Print model summary
    return model

# --- Main Training Function ---

def train():
    """Loads data, builds model, trains, evaluates, and saves the model."""
    print("Starting training process...")

    # 1. Load and preprocess data
    print("Loading and preprocessing data...")
    data = load_and_preprocess_data()
    if data is None:
        print("Failed to load or preprocess data. Exiting training.")
        return
    X_train, X_test, y_train, y_test, _, _ = data # We don't need tokenizer/max_len here

    # Check if GPU is available
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print(f"GPUs available: {gpus}")
        try:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)
    else:
        print("No GPU detected. Training will use CPU.")


    # 2. Build the model
    model = build_cnn_model()

    # 3. Define Callbacks
    # Early stopping: Stop training if validation loss doesn't improve
    early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
    # Model checkpoint: Save the best model based on validation accuracy
    os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
    model_checkpoint_path = os.path.join(MODEL_SAVE_DIR, MODEL_NAME)
    model_checkpoint = ModelCheckpoint(filepath=model_checkpoint_path,
                                       monitor='val_accuracy',
                                       save_best_only=True,
                                       verbose=1)

    # 4. Train the model
    print("Starting model training...")
    history = model.fit(X_train, y_train,
                        epochs=EPOCHS,
                        batch_size=BATCH_SIZE,
                        validation_split=0.1, # Use 10% of training data for validation
                        callbacks=[early_stopping, model_checkpoint],
                        verbose=1) # Set verbose=1 or 2 to see progress

    print("Training finished.")

    # 5. Evaluate the best model (restored by EarlyStopping or saved by ModelCheckpoint)
    print("Evaluating model on the test set...")
    # If EarlyStopping restored weights, model is already the best one.
    # If not, load the best one saved by ModelCheckpoint.
    if not early_stopping.best_weights: # Check if weights were restored
         try:
             print(f"Loading best model from {model_checkpoint_path}")
             model.load_weights(model_checkpoint_path)
         except Exception as e:
             print(f"Could not load best model weights: {e}. Evaluating current model state.")

    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"Test Loss: {loss:.4f}")
    print(f"Test Accuracy: {accuracy:.4f}")

    # 6. Ensure the final best model is saved (ModelCheckpoint already does this, but belt-and-suspenders)
    try:
        # Check if the best model was saved by ModelCheckpoint, otherwise save current state
        if not os.path.exists(model_checkpoint_path) or early_stopping.best_weights:
             model.save(model_checkpoint_path)
             print(f"Final model explicitly saved to {model_checkpoint_path}")
    except Exception as e:
        print(f"Error saving final model: {e}")


if __name__ == '__main__':
    train()
