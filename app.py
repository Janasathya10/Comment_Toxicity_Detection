import streamlit as st
import pandas as pd
from src.predictor import predict

# ✅ MUST be first Streamlit call
st.set_page_config(
    page_title="Toxic Comment Detector",
    page_icon="🧠",
    layout="centered"
)

st.title("🧠 Toxic Comment Detector")
st.markdown("Enter a comment below to analyze whether it's **toxic** or **non-toxic**...")

# 🔹 Single Comment Input
user_input = st.text_area("💬 Comment Text", height=150)

if st.button("🔍 Analyze"):
    if user_input.strip():
        label, prob = predict(user_input)
        st.success(f"**Prediction:** {label}")
        st.markdown(f"**Confidence Score:** `{prob:.2f}`")
    else:
        st.warning("Please enter a comment before analyzing.")

st.markdown("---")

# 🔸 CSV Upload Section
st.subheader("📁 Upload CSV for Bulk Predictions")
uploaded_file = st.file_uploader("Upload a CSV file with a column named `comment_text`", type="csv")

if uploaded_file:
    try:
        df = pd.read_csv(uploaded_file)

        if "comment_text" in df.columns:
            df["prediction"], df["confidence"] = zip(*df["comment_text"].apply(predict))
            st.success("✅ Predictions completed!")
            st.dataframe(df[["comment_text", "prediction", "confidence"]])

            # Download option
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="⬇️ Download Results as CSV",
                data=csv,
                file_name='toxic_comment_predictions.csv',
                mime='text/csv'
            )
        else:
            st.error("The CSV must contain a column named `comment_text`.")
    except Exception as e:
        st.error(f"⚠️ Error processing file: {e}")

st.markdown("🧪 Powered by LSTM + Keras")