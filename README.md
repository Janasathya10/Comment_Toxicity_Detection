# Comment_Toxicity_Detection

This project is a deep learning-based **Streamlit application** that classifies user comments as **toxic** or **non-toxic**. It uses a Bidirectional LSTM model trained on labeled comment data.

---

## 📁 Project Structure

```
Comment_Toxicity_project/
├── app.py                  # Streamlit web app UI
├── data/
│   ├── train.csv           # Training dataset (required)
│   └── test.csv            # Test dataset (optional for evaluation)
├── models/
│   ├── toxicity_model.h5   # Saved Keras model
│   └── tokenizer.pkl       # Tokenizer used during training
├── src/
│   ├── data_loader.py      # Text cleaning and data loading functions
│   ├── model.py            # Model training script
│   └── predictor.py        # Inference logic for prediction
├── tf-venv/                # Virtual environment folder (not version-controlled)
├── requirements.txt        # List of exact package versions
└── README.md               # Project overview and usage guide
```

---

## 🚀 How to Run

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/Comment_Toxicity_project.git
cd Comment_Toxicity_project
```

---

### 2. Create and Activate Virtual Environment

```bash
python3 -m venv tf-venv
source tf-venv/bin/activate
```

---

### 3. Install Requirements

```bash
pip install -r requirements.txt
```

If NLTK gives a stopwords error, download them manually:

```bash
python -m nltk.downloader stopwords
```

---

### 4. Train the Model

Ensure that `data/train.csv` exists with the following columns:

* `comment_text` – the user comment text
* `toxic` – binary label (0 = non-toxic, 1 = toxic)

Then run:

```bash
python src/model.py
```

This will:

* Clean and tokenize the text
* Handle class imbalance
* Train and save the model to `models/`
* Save the tokenizer for future predictions

> If `data/test.csv` exists, the model will be evaluated after training.

---

### 5. Run the Streamlit App

```bash
streamlit run app.py
```

Visit the link in your browser (usually [http://localhost:8501](http://localhost:8501)) to use the web UI.

---

## 💡 Features

* LSTM-based text classification model
* Real-time web UI for toxicity prediction
* Handles class imbalance during training
* Modular structure (training, prediction, UI, data loading)
* Interactive UI for real-time single comment prediction
* Option to upload a CSV file for bulk predictions

---

---

## 📝 Example Comments

Here are some example inputs and how the model classifies them:

### 🔴 Toxic Comments

1. "You're such a loser, no one likes you."
2. "Shut up, idiot. No one cares about your opinion."
3. "You’re a disgrace to humanity."

### 🟢 Non-Toxic Comments

1. "I appreciate your perspective, thanks for sharing!"
2. "Let’s work together to solve this."
3. "Have a great day!"

---

## 🛆 Requirements

Content of `requirements.txt`:

```txt
tensorflow==2.17.1
numpy==1.24.4
pandas==1.5.3
matplotlib==3.7.1
scikit-learn==1.2.2
opencv-python==4.8.0.76
streamlit==1.33.0
seaborn==0.12.2
nltk==3.8.1
```
