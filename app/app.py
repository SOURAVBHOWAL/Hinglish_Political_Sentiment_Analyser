# File: app/app.py

import streamlit as st
import pandas as pd
import plotly.express as px
import sys
import os

# --- Path Setup ---
# This is a critical step to make sure the app can find your 'src' folder
# It adds the parent directory (your project root) to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# --- Import Your Custom Functions ---
from src.reddit_collection import fetch_reddit_comments
from src.analysis import load_sentiment_model, apply_sentiment_analysis, perform_topic_modeling

# --- Page Configuration ---
st.set_page_config(
    page_title="Reddit Political Sentiment Analyzer",
    layout="wide"
)
st.title("Reddit Political Sentiment Analyzer")
st.markdown("Using a fine-tuned `bertweet` model and `BERTopic` to analyze live Reddit data.")

# --- Load Model (Cached) ---
# @st.cache_resource ensures the model is loaded only ONCE.
@st.cache_resource
def get_sentiment_model():
    return load_sentiment_model()

sentiment_task = get_sentiment_model()

if sentiment_task is None:
    st.error("Fatal Error: Could not load sentiment model. Check logs.")
    st.stop() # Stop the app if the model isn't loaded

# --- Sidebar Inputs ---
st.sidebar.header("Analyzer Controls")
subreddit = st.sidebar.text_input("1. Subreddit", value="politics")
post_limit = st.sidebar.slider("2. Posts to Fetch", 1, 25, 5)
topic_sentiment = st.sidebar.selectbox(
    "3. Topic Model Sentiment", 
    ('Negative', 'Positive', 'Neutral'), 
    index=0 # Default to 'Negative'
)
run_button = st.sidebar.button("Run Analysis")

# --- Main Dashboard ---
if run_button:
    
    # --- 1. Data Collection (Reddit) ---
    with st.spinner(f"Fetching comments from r/{subreddit}..."):
        df_raw = fetch_reddit_comments(subreddit, limit=post_limit)
    
    if df_raw.empty:
        st.error("No comments found. Check credentials or subreddit name.")
    else:
        st.success(f"Fetched {len(df_raw)} comments.")
        
        # --- 2. Sentiment Analysis ---
        with st.spinner("Applying sentiment analysis..."):
            df_sent = apply_sentiment_analysis(df_raw, sentiment_task)

        st.header("Sentiment Analysis Results")
        
        # --- 3. Visualization (Sentiment) ---
        fig_total = px.pie(df_sent, names='sentiment_label', 
                           title=f"Sentiment in r/{subreddit}", hole=0.3)
        st.plotly_chart(fig_total, use_container_width=True)

        st.divider()
        
        # --- 4. Topic Modeling ---
        st.header(f"🔬 Deep Dive: What are people '{topic_sentiment}' about?")
        
        with st.spinner(f"Running topic modeling on '{topic_sentiment}' comments..."):
            model, fig_topics = perform_topic_modeling(
                df_sent, 
                sentiment_label_to_cluster=topic_sentiment
            )
        
        if model and fig_topics:
            st.subheader(f"Top '{topic_sentiment}' Topics")
            # Show the interactive plotly chart from BERTopic
            st.plotly_chart(fig_topics, use_container_width=True)
            
            # Show the topic keywords as a table
            st.dataframe(model.get_topic_info())
        else:
            st.warning(f"Could not perform topic modeling. Not enough '{topic_sentiment}' comments.")

        st.divider()
        
        # --- 5. Raw Data ---
        st.header("Analyzed Data")
        st.dataframe(df_sent[['text', 'sentiment_label', 'score', 'post_title']])

else:
    st.info("Click 'Run Analysis' in the sidebar to begin.")