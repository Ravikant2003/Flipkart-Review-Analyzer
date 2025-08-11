import streamlit as st
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline
import pandas as pd

# Define your label mapping (must match what you used during training)
LABEL_MAP = {
    "LABEL_0": "negative",
    "LABEL_1": "neutral",
    "LABEL_2": "positive"
}

# Load model and tokenizer with label mapping
@st.cache_resource
@st.cache_resource
def load_model():
    model = AutoModelForSequenceClassification.from_pretrained("./sentiment_model")
    tokenizer = AutoTokenizer.from_pretrained("./sentiment_model")

    # Set label mapping inside model config
    model.config.id2label = {0: "negative", 1: "neutral", 2: "positive"}
    model.config.label2id = {"negative": 0, "neutral": 1, "positive": 2}

    # Create pipeline (no id2label argument here)
    return pipeline(
        "text-classification", 
        model=model, 
        tokenizer=tokenizer,
        function_to_apply="softmax",  # Ensures probabilities sum to 1
        top_k=None  # Return all results
    )

# Function to format sentiment prediction
def format_prediction(prediction):
    # Get the top prediction
    top_pred = prediction[0]
    sentiment = top_pred['label']
    score = top_pred['score']
    
    emoji = "😠" if sentiment == "negative" else "😐" if sentiment == "neutral" else "😊"
    return f"{emoji} {sentiment.capitalize()} ({score:.2%} confidence)"

# Static performance data
performance_data = {
    "Classification Report": {
        "negative": {"precision": 0.92, "recall": 0.91, "f1-score": 0.91, "support": 200},
        "neutral": {"precision": 0.95, "recall": 0.97, "f1-score": 0.96, "support": 200},
        "positive": {"precision": 0.97, "recall": 0.97, "f1-score": 0.97, "support": 518},
        "accuracy": 0.95,
        "macro_avg": {"precision": 0.95, "recall": 0.95, "f1-score": 0.95, "support": 918},
        "weighted_avg": {"precision": 0.95, "recall": 0.95, "f1-score": 0.95, "support": 918}
    },
    "Class Distribution": {
        "positive": 518,
        "neutral": 200,
        "negative": 200
    }
}

# Streamlit UI
st.set_page_config(page_title="Sentiment Analyzer", page_icon="😊", layout="wide")

# Sidebar for input
st.sidebar.header("Sentiment Analysis Demo")
user_input = st.sidebar.text_area("Enter your text here:", "The product was amazing! Loved the quality and fast delivery.")
analyze_button = st.sidebar.button("Analyze Sentiment")

# Load model (only once)
classifier = load_model()

# Main content
st.title("📝 Customer Sentiment Analysis")
st.markdown("This tool analyzes product review sentiment using a fine-tuned RoBERTa model.")

# Create columns layout
col1, col2 = st.columns([1, 1])

# Results column
with col1:
    st.header("Analysis Result")
    
    if analyze_button and user_input:
        with st.spinner("Analyzing sentiment..."):
            result = classifier(user_input)[0]
            formatted = format_prediction(result)
            
            # Display with appropriate color
            if "negative" in formatted:
                st.error(formatted)
            elif "neutral" in formatted:
                st.warning(formatted)
            else:
                st.success(formatted)
                
            st.subheader("Review Text:")
            st.write(user_input)
            
            # Show confidence scores for all classes
            st.subheader("Confidence Breakdown:")
            conf_df = pd.DataFrame({
                "Sentiment": [LABEL_MAP.get(p['label'], p['label']) for p in result],
                "Confidence": [p['score'] for p in result]
            })
            st.bar_chart(conf_df.set_index("Sentiment"))
    else:
        st.info(" Enter text and click 'Analyze Sentiment' to get started")

# Performance metrics column
with col2:
    st.header("Model Performance")
    st.caption("Based on test dataset of 918 samples")
    
    # Classification report table
    st.subheader("Detailed Classification Report")
    report_df = pd.DataFrame(performance_data["Classification Report"]).T
    st.dataframe(report_df.style.format("{:.2f}"), height=350)
    
    # Class distribution chart
    st.subheader("Class Distribution in Test Set")
    dist_df = pd.DataFrame(
        list(performance_data["Class Distribution"].items()),
        columns=["Sentiment", "Count"]
    )
    st.bar_chart(dist_df.set_index("Sentiment"))

# Footer
st.markdown("---")
st.caption("Model: RoBERTa-base fine-tuned on product reviews | "
           "Supports Negative/Neutral/Positive sentiment detection")

# How to run instructions
st.sidebar.markdown("""
**How to Run Locally:**
1. Save model in `./sentiment_model`
2. Install requirements: `pip install streamlit transformers torch pandas`
3. Run: `streamlit run app.py`
""")