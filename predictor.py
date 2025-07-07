from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle

# Load model and tokenizer once
model = load_model('models/toxicity_model.h5')
with open('models/tokenizer.pkl', 'rb') as f:
    tokenizer = pickle.load(f)

def predict(text):
    sequence = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(sequence, maxlen=200)
    prob = model.predict(padded)[0][0]
    return ('Toxic' if prob >= 0.5 else 'Non-toxic'), float(prob)