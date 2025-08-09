Here's the full README.md content formatted as requested.

````markdown
# Flipkart Product Review Analyzer — Transformer Powered 🚀

This repository contains a **Flipkart Product Review Analyzer** that processes product reviews and classifies them into **Positive**, **Negative**, and **Neutral** sentiments using **state-of-the-art Transformer models**.
It also extracts **key phrases** from reviews using **KeyBERT** and visualizes them with **WordClouds**, making the analysis richer and more insightful.

The interactive app is built with **Streamlit** and allows users to upload a dataset, analyze sentiments, extract keywords, and download the results.

**Dataset:** [Flipkart Product Customer Reviews Dataset](https://www.kaggle.com/datasets/niraliivaghani/flipkart-product-customer-reviews-dataset)

---
## Table of Contents
- [Project Overview](#project-overview)
- [Features](#features)
- [Technologies Used](#technologies-used)
- [Setup Instructions](#setup-instructions)
- [Usage](#usage)
- [Evaluation Metrics](#evaluation-metrics)
- [Example Output](#example-output)
- [Notes & Troubleshooting](#notes--troubleshooting)
- [Contributing](#contributing)
- [License & Contact](#license--contact)

---
## Project Overview

The **Flipkart Product Review Analyzer** processes review text (summary + detailed review) and performs two main tasks:

### Sentiment Classification
- Uses **CardiffNLP Twitter-RoBERTa** model (`cardiffnlp/twitter-roberta-base-sentiment-latest`) for high-accuracy sentiment analysis.

### Keyword Extraction
- Uses **KeyBERT** to extract the top keywords per review for better interpretability.

The app displays **sentiment-based WordClouds** for visual insight and allows CSV download of results.

---
## Features
- **Upload CSV** with `Summary`, `Review`, and optionally `sentiment` columns.
- **Automatic Text Combination** (summary + review).
- **Sentiment Prediction** (Positive, Neutral, Negative) using a Transformer.
- **Keyword Extraction** (top keywords per review).
- **Model Performance Report** (Accuracy, Precision, Recall, F1-score) if `sentiment` labels are provided.
- **Sentiment-wise WordClouds** for visualization.
- **Download Results as CSV**.

---
## Technologies Used
- **Python 3.x**
- **Pandas** — Data processing
- **Transformers (Hugging Face)** — RoBERTa sentiment model
- **KeyBERT** — Keyword extraction
- **WordCloud** — Visualization
- **Matplotlib** — Plot rendering
- **scikit-learn** — Accuracy & classification reports
- **Streamlit** — Web app interface

---
## Setup Instructions
### 1. Clone the Repository
```bash
git clone [https://github.com/username/sentiment-analysis.git](https://github.com/username/sentiment-analysis.git)
cd sentiment-analysis
````

### 2\. (Recommended) Create a virtual environment

```bash
python -m venv .venv
# macOS / Linux
source .venv/bin/activate
# Windows
.venv\Scripts\activate
```

### 3\. Install Dependencies

```bash
pip install -r requirements.txt
```

If you do not have a requirements.txt, you can install the core dependencies:

```bash
pip install pandas transformers keybert sentence-transformers wordcloud matplotlib scikit-learn streamlit
```

**Note (Apple Silicon / M1 / M2):**
If you run into issues installing torch on macOS, follow the official PyTorch instructions for macOS / MPS support: https://pytorch.org/get-started/locally/ — choose the proper wheel for your Python and platform. On many M1/M2 setups you can use:

```bash
pip install torch --index-url [https://download.pytorch.org/whl/cpu](https://download.pytorch.org/whl/cpu)
```

or use the command recommended by the PyTorch website for MPS.

### 4\. Run the Streamlit App

```bash
streamlit run app.py
```

The app will open in your browser (`http://localhost:8501` by default). The transformer's weights will be downloaded the first time you run the app.

-----

## Usage

1.  **Prepare CSV:** Ensure your CSV has at least:
      - `Summary` (short title)
      - `Review` (full review text)
      - `sentiment` (optional; used for evaluation if present — values such as positive, neutral, negative)
2.  **Upload CSV** via the Streamlit interface.
3.  **Inspect Output:**
      - Predicted sentiment for each review
      - Keywords for each review
      - Sentiment-wise word clouds
      - If `sentiment` exists in CSV, you will see accuracy and a classification report
4.  **Download** the enriched CSV with predictions and keywords using the download button.

-----

## Evaluation Metrics

If the uploaded dataset includes a `sentiment` column, the app computes:

  - **Accuracy** — overall fraction of correct predictions
  - **Classification Report** — Precision, Recall, F1-score per class

These metrics help you evaluate model performance on your labeled data.

-----

## Example Output

  - **Sentiment Accuracy:** 84.3% (example — depends on dataset)
  - **Top Keywords Example:** "fast delivery, great quality, value for money"
  - **WordClouds:** Separate visual clouds for Positive, Neutral, Negative review text

-----


## License & Contact

This project is licensed under the MIT License.


```
```
