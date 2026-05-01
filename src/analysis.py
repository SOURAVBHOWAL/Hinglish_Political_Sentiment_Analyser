# File: src/analysis.py

from transformers import pipeline
from bertopic import BERTopic
import pandas as pd

def load_sentiment_model():
    """
    Loads the fine-tuned sentiment model from the local directory.
    """
    print("Loading fine-tuned sentiment model...")
    MODEL_PATH = "./models/my_political_sentiment_model"
    
    try:
        sentiment_task = pipeline("sentiment-analysis", model=MODEL_PATH)
        print("Sentiment model loaded successfully.")
        return sentiment_task
    except Exception as e:
        print(f"Error loading model from {MODEL_PATH}.")
        print("Make sure your model is downloaded and in the correct folder.")
        print(f"Error: {e}")
        return None

def apply_sentiment_analysis(df, sentiment_task):
    """
    Applies sentiment analysis to the 'text' column of the DataFrame.
    """
    if sentiment_task is None:
        return df

    if 'text' not in df.columns:
        return df
        
    print("Applying sentiment analysis...")
    
    # Get just the text for the pipeline
    texts = df['text'].tolist()
    
    # Run predictions
    # We set truncation=True to handle long comments
    results = sentiment_task(texts, truncation=True, max_length=128)
    
    # Process results back into the DataFrame
    df['sentiment_label'] = [r['label'] for r in results]
    df['sentiment_score'] = [r['score'] for r in results]
    
    
    print("Sentiment analysis complete.")
    return df

def perform_topic_modeling(df, sentiment_label_to_cluster='Negative'):
    """
    Performs BERTopic on tweets matching a specific sentiment.
    """
    
    docs_to_cluster = df[
        df['sentiment_label'] == sentiment_label_to_cluster
    ]['text'].tolist()

    if len(docs_to_cluster) < 10:
        print("Not enough documents for topic modeling.")
        return None, None # Return None if no model can be trained

    print(f"Running topic modeling on {len(docs_to_cluster)} documents...")
    
    # We set min_topic_size=3 to find smaller topics
    topic_model = BERTopic(language="english", min_topic_size=3)
    
    try:
        topics, probabilities = topic_model.fit_transform(docs_to_cluster)
        
        print("Topic modeling complete.")
        # Return the model and the interactive figure
        return topic_model, topic_model.visualize_topics()
        
    except Exception as e:
        print(f"Error during topic modeling: {e}")
        return None, None