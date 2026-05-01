# File: src/reddit_collection.py

import praw
import pandas as pd
import os

# --- PASTE YOUR CREDENTIALS HERE ---

# Your "personal use script" (a 14-character string)
CLIENT_ID = "Llyy4ekJMDYDjp37vMQ9Ww" 

# Your "secret" (a 27-character string)
CLIENT_SECRET = "mpjMYK9m72B-Zow53T1hLsyKARJ8lw"

# Your "developer's name" (or just your app name + username)
USER_AGENT = "Political Sentiment Analyzer by /u/Smart-Formal6667" # <-- CHANGE THIS

# --- END CREDENTIALS ---


def fetch_reddit_comments(subreddit_name="politics", limit=10):
    """
    Fetches comments from the top 'limit' hot posts in a subreddit.
    """
    try:
        reddit = praw.Reddit(
            client_id=CLIENT_ID,
            client_secret=CLIENT_SECRET,
            user_agent=USER_AGENT,
        )
        
        # Check if authentication is successful (optional but good)
        reddit.user.me() 
        print("Successfully authenticated with Reddit.")
        
    except Exception as e:
        print(f"Error: Could not authenticate with Reddit. Check credentials. {e}")
        return pd.DataFrame() # Return an empty DataFrame on failure

    subreddit = reddit.subreddit(subreddit_name)
    
    comments_list = []
    
    print(f"Fetching comments from r/{subreddit_name}...")
    
    try:
        # Get the top 'limit' hot posts
        for submission in subreddit.hot(limit=limit):
            
            # Get all comments from this post (up to 50 for speed)
            submission.comments.replace_more(limit=0) # Expands "load more comments"
            
            for comment in submission.comments.list()[:50]:
                comments_list.append({
                    "text": comment.body,
                    "score": comment.score,
                    "post_title": submission.title
                })

    except Exception as e:
        print(f"Error fetching posts/comments: {e}")
        return pd.DataFrame()

    print(f"Fetched {len(comments_list)} comments.")
    return pd.DataFrame(comments_list)