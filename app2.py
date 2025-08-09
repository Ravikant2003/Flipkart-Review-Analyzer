import streamlit as st
import pandas as pd
from transformers import pipeline
from keybert import KeyBERT
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, classification_report


# 1. Load Sentiment Model (cached)

@st.cache_resource
def load_sentiment_model():
    return pipeline("sentiment-analysis", model="cardiffnlp/twitter-roberta-base-sentiment-latest")

sentiment_pipeline = load_sentiment_model()


# 2. Load KeyBERT Model (cached)

@st.cache_resource
def load_keybert():
    return KeyBERT()

kw_model = load_keybert()


# 3. Sentiment Prediction Function

def predict_batch(texts):
    return [res['label'] for res in sentiment_pipeline(texts, truncation=True)]

# 4. WordCloud Plotter


def plot_wordcloud(text, title):
    wc = WordCloud(width=800, height=400, background_color="white").generate(text)
    plt.figure(figsize=(8, 4))
    plt.imshow(wc, interpolation="bilinear")
    plt.axis("off")
    plt.title(title)
    st.pyplot(plt)


# 5. Streamlit UI




st.title("Product Review Sentiment & Keyword Analysis")
st.write("Upload your CSV file to analyze sentiment and extract keywords.")

uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    # Check required columns
    if "Review" not in df.columns or "Summary" not in df.columns:
        st.error("CSV must have 'Review' and 'Summary' columns.")
    elif "sentiment" not in df.columns:
        st.error("CSV must have 'sentiment' column for evaluation.")
    else:
        # Combine text columns
        df["combined_text"] = df["Summary"].astype(str) + " " + df["Review"].astype(str)

        # Run Sentiment Prediction
        with st.spinner("Analyzing sentiments..."):
            df["predicted_sentiment"] = predict_batch(df["combined_text"].tolist())

        # Calculate and display accuracy & classification report
        accuracy = accuracy_score(df['sentiment'], df['predicted_sentiment'])
        st.write(f"### Model Accuracy: {accuracy:.2%}")
        report = classification_report(df['sentiment'], df['predicted_sentiment'])
        st.text("Classification Report:\n" + report)

        # Keyword Extraction
        with st.spinner("Extracting keywords..."):
            df["keywords"] = df["combined_text"].apply(
                lambda x: ", ".join([kw[0] for kw in kw_model.extract_keywords(x, top_n=3)])
            )

        # Display Results
        st.subheader("Preview of Analysis")
        st.dataframe(df[["Summary", "Review", "predicted_sentiment", "keywords"]].head())

        # WordClouds by Sentiment
        st.subheader("☁ Word Clouds by Sentiment")
        for sentiment in df["predicted_sentiment"].unique():
            sentiment_text = " ".join(df[df["predicted_sentiment"] == sentiment]["combined_text"])
            plot_wordcloud(sentiment_text, f"{sentiment} Reviews")

        # Download Option
        csv_download = df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label=" Download Results as CSV",
            data=csv_download,
            file_name="sentiment_keywords_results.csv",
            mime="text/csv"
        )
