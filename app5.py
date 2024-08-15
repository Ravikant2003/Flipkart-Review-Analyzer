import pandas as pd
import nltk
from nltk.corpus import stopwords
from textblob import TextBlob
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt

# Download stopwords if not already available
nltk.download('stopwords')

# Load and preprocess the dataset
def preprocess_data(file_path):
    df = pd.read_csv(file_path)
    
    # Handle Missing Values
    df.dropna(inplace=True)

    # Clean 'product_price' column
    df['product_price'] = df['product_price'].replace('[\₹,]', '', regex=True)
    df = df[pd.to_numeric(df['product_price'], errors='coerce').notnull()]
    df['product_price'] = df['product_price'].astype(int)

    # Handle Duplicates
    df.drop_duplicates(inplace=True)

    # Combine 'Summary' and 'Review' columns into a new column
    df['combined_text'] = df['Summary'] + " " + df['Review']

    # Remove Stopwords from the combined text
    stop_words = set(stopwords.words('english'))
    df['preprocessed_text'] = df['combined_text'].apply(lambda x: ' '.join([word for word in x.split() if word.lower() not in stop_words]))

    # Perform Sentiment Analysis
    df['sentiment'] = df['preprocessed_text'].apply(get_sentiment_label)
    
    return df

# Define the sentiment analysis function with adjusted threshold
def get_sentiment_label(text):
    analysis = TextBlob(text)
    if analysis.sentiment.polarity > 0.2:
        return 'Positive'
    elif analysis.sentiment.polarity < -0.2:
        return 'Negative'
    else:
        return 'Neutral'

# Train and evaluate models
def train_and_evaluate_models(df):
    # Initialize the TF-IDF vectorizer
    tfidf_vectorizer = TfidfVectorizer(max_df=0.8, min_df=5, stop_words='english')
    
    # Fit and transform the preprocessed text data
    tfidf_matrix = tfidf_vectorizer.fit_transform(df['preprocessed_text'])
    
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(tfidf_matrix, df['sentiment'], test_size=0.2, random_state=42)
    
    # Initialize and train multiple classifiers
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000),
        'Random Forest': RandomForestClassifier(random_state=42),
        'MLP Classifier': MLPClassifier(random_state=42),
        'SVM': SVC(kernel='linear', random_state=42)
    }

    # Train and evaluate each model
    model_reports = {}
    for model_name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        report = {
            'Accuracy': accuracy_score(y_test, y_pred),
            'Classification Report': classification_report(y_test, y_pred, target_names=['Positive', 'Negative', 'Neutral']),
            'Confusion Matrix': confusion_matrix(y_test, y_pred, labels=['Positive', 'Negative', 'Neutral'])
        }
        model_reports[model_name] = report

    # Hyperparameter tuning for Logistic Regression
    param_grid_logreg = {
        'C': [0.001, 0.01, 0.1, 1, 10, 100],  # Regularization parameter
        'max_iter': [200, 300, 400, 500, 600]  # Increase the maximum number of iterations
    }

    grid_search_logreg = GridSearchCV(estimator=LogisticRegression(max_iter=1000), param_grid=param_grid_logreg, cv=5, scoring='accuracy', n_jobs=-1)
    grid_search_logreg.fit(X_train, y_train)
    
    best_params_logreg = grid_search_logreg.best_params_
    best_model_logreg = grid_search_logreg.best_estimator_
    best_model_logreg_accuracy = best_model_logreg.score(X_test, y_test)
    
    model_reports['Best Logistic Regression Model'] = {
        'Best Hyperparameters': best_params_logreg,
        'Accuracy': best_model_logreg_accuracy
    }
    
    return tfidf_vectorizer, model_reports, best_model_logreg

# Function to analyze a single review
def analyze_review(review_text, tfidf_vectorizer, best_model):
    # Preprocess user input
    stop_words = set(stopwords.words('english'))
    user_review_cleaned = ' '.join([word for word in review_text.split() if word.lower() not in stop_words])
    user_review_tfidf = tfidf_vectorizer.transform([user_review_cleaned])
    
    # Predict sentiment using the best model
    sentiment = best_model.predict(user_review_tfidf)
    return sentiment[0]

# Main function for the Streamlit app
def main():
    st.title('Flipkart Product Review Analyzer')

    # Ask for CSV upload at the start
    file = st.file_uploader("Upload your CSV file", type=["csv"])

    if file is not None:
        # Load and preprocess data
        df = preprocess_data(file)
        tfidf_vectorizer, model_reports, best_model_logreg = train_and_evaluate_models(df)
        
        # Sidebar for navigation
        st.sidebar.title('Navigation')
        option = st.sidebar.selectbox('Select a tab:', ['Home', 'Model Evaluation', 'Review Analysis'])
        
        if option == 'Home':
            st.write("""
            ## Project Description
            This application analyzes product reviews from Flipkart to determine the sentiment of each review. 
            It uses multiple machine learning models to classify reviews into three sentiment categories: Positive, Negative, and Neutral.
            The app provides options to view model evaluation results and analyze individual product reviews.
            """)
        
        elif option == 'Model Evaluation':
            # Display model evaluation results
            st.subheader('Model Evaluation Results')
            for model_name, report in model_reports.items():
                st.write(f'### {model_name}')
                st.write(f"Accuracy: {report['Accuracy']:.2f}")
                
                # Display Classification Report if it exists
                if 'Classification Report' in report:
                    st.write("Classification Report:")
                    st.text(report['Classification Report'])
                
                # Display Confusion Matrix if it exists
                if 'Confusion Matrix' in report:
                    st.write("Confusion Matrix:")
                    cm = report['Confusion Matrix']
                    plt.figure(figsize=(10, 7))
                    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Positive', 'Negative', 'Neutral'], yticklabels=['Positive', 'Negative', 'Neutral'])
                    plt.title(f'Confusion Matrix for {model_name}')
                    st.pyplot(plt)
                    plt.clf()  # Clear the plot to avoid overlapping in subsequent plots
        
        elif option == 'Review Analysis':
            # User input for review analysis
            st.subheader('Analyze a Product Review')
            user_review = st.text_area("Enter the product review here:")

            if st.button('Analyze Review'):
                if user_review:
                    sentiment = analyze_review(user_review, tfidf_vectorizer, best_model_logreg)
                    st.write(f"The sentiment of the review is: {sentiment}")
                else:
                    st.write("Please enter a review to analyze.")

if __name__ == "__main__":
    main()
