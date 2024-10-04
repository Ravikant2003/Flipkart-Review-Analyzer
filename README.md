# Flipkart Product Review Analyzer

This repository contains a **Flipkart Product Review Analyzer** that processes product reviews and classifies them into **Positive**, **Negative**, and **Neutral** sentiments using machine learning models. The application allows users to upload a dataset of product reviews, evaluate different models, and analyze the sentiment of individual reviews via a **Streamlit** interface.

Dataset taken from : https://www.kaggle.com/datasets/niraliivaghani/flipkart-product-customer-reviews-dataset

## Table of Contents
- [Project Overview](#project-overview)
- [Features](#features)
- [Technologies Used](#technologies-used)
- [Setup Instructions](#setup-instructions)
- [Usage](#usage)
- [Model Evaluation](#model-evaluation)
- [Review Sentiment Analysis](#review-sentiment-analysis)

## Project Overview

The **Flipkart Product Review Analyzer** processes product reviews, removes stopwords, and uses **sentiment analysis** to classify reviews into three categories:
- **Positive**
- **Negative**
- **Neutral**

It utilizes various machine learning models, including **Logistic Regression**, **Random Forest**, **MLP Classifier**, and **Support Vector Classifier (SVC)**, for classification and **TextBlob** for sentiment analysis. The app offers a visual analysis of the model's performance and provides functionality for analyzing individual reviews.

## Features
- **Data Preprocessing**: 
  - Handles missing values and duplicate entries.
  - Cleans product price data.
  - Combines review summary and detailed review text for analysis.
  - Removes stopwords from the combined review text.
  
- **Sentiment Analysis**: 
  - Uses **TextBlob** to analyze the sentiment polarity of the reviews with adjustable thresholds.

- **Model Training & Evaluation**:
  - Trains multiple machine learning models: **Logistic Regression**, **Random Forest**, **MLP Classifier**, and **SVM**.
  - Provides **hyperparameter tuning** for the best Logistic Regression model.
  - Visualizes model performance using classification reports and confusion matrices.

- **Review Analysis**:
  - Allows users to input a product review and predicts its sentiment using the best-performing model.

## Technologies Used
- **Python 3.x**
- **Pandas**
- **nltk (Natural Language Toolkit)**: For stopword removal.
- **TextBlob**: For sentiment analysis.
- **scikit-learn**: For machine learning models, TF-IDF vectorization, and hyperparameter tuning.
- **Streamlit**: For building the interactive web app interface.
- **Seaborn & Matplotlib**: For visualizing model performance with confusion matrices.

## Setup Instructions

1. **Clone the Repository**:
 
2. **Install Dependencies**:
   Install the required Python packages:
   ```bash
   pip install -r requirements.txt
   ```

   Required packages:
   - `pandas`
   - `nltk`
   - `textblob`
   - `scikit-learn`
   - `streamlit`
   - `seaborn`
   - `matplotlib`

3. **Download NLTK Stopwords**:
   ```bash
   python -m nltk.downloader stopwords
   ```

4. **Run the Streamlit App**:
   To launch the web app, use the following command:
   ```bash
   streamlit run app.py
   ```

## Usage

1. **Upload Dataset**: Start by uploading a CSV file containing product reviews with columns like `Summary`, `Review`, and `product_price`. The app will preprocess the data, including handling missing values and duplicates, cleaning product prices, and preparing text for analysis.

2. **Model Evaluation**: The app will train various machine learning models and display evaluation metrics including accuracy, classification reports, and confusion matrices.

3. **Review Analysis**: Analyze the sentiment of individual product reviews by typing or pasting a review in the app, and the best-trained model will predict the sentiment.

## Model Evaluation

The following models are trained on the dataset:
- **Logistic Regression**: Trains using default parameters and fine-tuned through **GridSearchCV** for hyperparameter optimization.
- **Random Forest Classifier**: A robust ensemble method for classification.
- **MLP Classifier**: A feed-forward neural network.
- **Support Vector Classifier (SVC)**: Uses a linear kernel to separate the data.

For each model, key metrics are displayed:
- **Accuracy**: Overall performance of the model.
- **Classification Report**: Precision, recall, and F1 scores for each class (Positive, Negative, Neutral).
- **Confusion Matrix**: Visualized as a heatmap to show how well the model distinguishes between different sentiment categories.

## Review Sentiment Analysis

In the **Review Analysis** section, users can input a single product review, and the app will classify its sentiment as **Positive**, **Negative**, or **Neutral** based on the best-performing model from the evaluation.
