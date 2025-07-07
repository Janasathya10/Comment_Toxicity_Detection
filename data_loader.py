import pandas as pd
import re
import nltk
from nltk.corpus import stopwords

nltk.download('stopwords')
stop_words = set(stopwords.words('english'))  # ✅ Moved this line outside the function

def clean_text(text):
    text = re.sub(r"[^a-zA-Z\s]", "", str(text).lower())
    tokens = text.split()
    # 🔥 Comment this out:
    # tokens = [word for word in tokens if word not in stop_words]
    return " ".join(tokens)

def load_data(filepath):
    df = pd.read_csv(filepath)
    df['comment_text'] = df['comment_text'].apply(clean_text)
    return df