# SentimentLens: Product Review Analyzer (Fine Tuned Roberta)

Working Project Video Url: https://youtu.be/BSC7Rf5NheA

## Overview
This project implements a customer product review sentiment classification system using a fine-tuned RoBERTa transformer model. The system classifies reviews into three categories:
- Negative
- Neutral
- Positive

**Key Features**:
- Data augmentation to balance class distribution and improve robustness
- Weighted loss training to handle class imbalance
- Fine-tuned RoBERTa-base model using Hugging Face Transformers
- Streamlit web app for interactive sentiment analysis
- Confidence scores and performance metrics visualization

## Project Structure

### Model Training & Augmentation
1. Loads Flipkart reviews dataset (`flip2.csv`)
2. Cleans and standardizes sentiment labels
3. Combines summary and review text as input
4. Uses advanced text augmentation techniques:
   - Synonym replacement
   - Random insertion/deletion/swapping
5. Encodes labels and creates stratified train-test split
6. Tokenizes text using RoBERTa tokenizer
7. Trains model with custom weighted cross-entropy loss
8. Implements early stopping based on neutral class F1-score
9. Saves fine-tuned model to `./sentiment_model`

### Streamlit Web App (`app.py`)
- Loads saved model and tokenizer
- User interface for text input and analysis
- Displays:
  - Predicted sentiment with emoji
  - Confidence score
  - Class confidence breakdown (bar chart)
  - Static performance metrics
  - Test set class distribution

## Installation & Setup

### Requirements
- Python 3.7+
- Packages: `transformers datasets torch scikit-learn nlpaug textattack pandas streamlit`

### Installation
pip install transformers datasets torch scikit-learn nlpaug textattack pandas streamlit



###Additional Setup
Download NLTK data:

import nltk
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')


###Usage
1. Train the Model
Place dataset CSV (flip2.csv) in project root with columns:
ProductName, Price, Rating, Summary, Review, sentiment


2. Run Streamlit App
After training and downloading your model , Please keep the app.py and the model in same directory .

streamlit run app.py

## Model Performance
Evaluation on test set (918 samples):

| Sentiment | Precision | Recall | F1-score | Support |
|-----------|-----------|--------|----------|---------|
| Negative  | 0.92      | 0.91   | 0.91     | 200     |
| Neutral   | 0.95      | 0.97   | 0.96     | 200     |
| Positive  | 0.97      | 0.97   | 0.97     | 518     |
| **Accuracy** | | | **0.95** | **918** |

## Code Highlights
- **Advanced text augmentation**: Synonym replacement and random perturbations for minority classes
- **Custom weighted loss function**: Addresses class imbalance during training
- **Hugging Face Trainer**: With early stopping based on F1-score
- **Interactive Streamlit visualization**: Real-time sentiment analysis with confidence scores
- **Confidence breakdown charts**: Visual class probability distribution

## Folder Structure

project-root/
├── sentiment_model/ # Fine-tuned model & tokenizer
├── app.py # Streamlit application
├── train_sentiment_model.py # Training + augmentation script
├── flip2.csv # Example dataset (ProductName, Price, Rating, etc.)
├── requirements.txt # Dependencies (optional)
└── README.md # Documentation

###Key improvements:

Model Performance Table:

Proper Markdown table formatting

Added alignment headers

Bolded accuracy metrics for emphasis

Fixed inconsistent spacing in original data

###Code Highlights:

Converted to bullet points with clear explanations

Added bold headers for key techniques

Expanded descriptions while keeping concise

Logical ordering from data processing to deployment

###Folder Structure:

Maintained as code block for better readability

Added brief comments explaining each file's purpose

Fixed directory tree formatting

Clarified dataset columns in comment


```bash
