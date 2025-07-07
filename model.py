import pandas as pd
import numpy as np
import os
import pickle
import re

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Bidirectional, LSTM, Dense, Dropout
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

# 🔧 Hyperparameters
VOCAB_SIZE = 10000
MAX_LEN = 200
EMBEDDING_DIM = 128
EPOCHS = 10
BATCH_SIZE = 32

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

print("🚀 Script started...")

# ✅ Cleaner: remove only URLs and special characters — keep stopwords and context
def clean_text(text):
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)  # Remove URLs
    text = re.sub(r"[^a-zA-Z\s']", "", text)             # Keep letters and apostrophes
    return text.lower().strip()

# ✅ Model architecture: stronger & regularized
def build_model(vocab_size, input_len):
    model = Sequential([
        Embedding(vocab_size, EMBEDDING_DIM, input_length=input_len),
        Bidirectional(LSTM(64, return_sequences=False)),
        Dropout(0.5),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def train_model(df):
    df['comment_text'] = df['comment_text'].astype(str).apply(clean_text)

    tokenizer = Tokenizer(num_words=VOCAB_SIZE, oov_token="<OOV>")
    tokenizer.fit_on_texts(df['comment_text'])

    sequences = tokenizer.texts_to_sequences(df['comment_text'])
    padded = pad_sequences(sequences, maxlen=MAX_LEN)

    X_train, X_val, y_train, y_val = train_test_split(
        padded, df['toxic'], test_size=0.2, stratify=df['toxic'], random_state=42
    )

    # ⚖️ Handle imbalance with class weights
    class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    class_weights = dict(enumerate(class_weights))

    model = build_model(VOCAB_SIZE, MAX_LEN)
    early_stop = EarlyStopping(monitor='val_loss', patience=2, restore_best_weights=True)

    print("🔁 Training model...")
    model.fit(X_train, y_train,
              epochs=EPOCHS,
              batch_size=BATCH_SIZE,
              validation_data=(X_val, y_val),
              class_weight=class_weights,
              callbacks=[early_stop],
              verbose=1)

    os.makedirs("models", exist_ok=True)
    model.save("models/toxicity_model.h5")
    with open("models/tokenizer.pkl", "wb") as f:
        pickle.dump(tokenizer, f)

    print("✅ Training complete. Model and tokenizer saved.")

if __name__ == "__main__":
    train_path = "data/train.csv"
    train_df = pd.read_csv(train_path)

    print("📊 Train data loaded:", train_df.shape)
    if 'comment_text' not in train_df.columns or 'toxic' not in train_df.columns:
        raise ValueError("❌ 'comment_text' and 'toxic' columns must be in train.csv")

    train_model(train_df)

print("✅ Script finished running.")