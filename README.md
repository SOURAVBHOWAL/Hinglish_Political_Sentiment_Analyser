# Hinglish Political Sentiment Analyzer

> A real-time, two-layer political sentiment analysis pipeline for Indian social media discourse — combining a fine-tuned **BERTweet** classifier with **BERTopic** topic modelling, served via an interactive **Streamlit** dashboard.

![Python](https://img.shields.io/badge/Python-3.9+-blue?logo=python&logoColor=white)
![HuggingFace](https://img.shields.io/badge/HuggingFace-BERTweet-orange?logo=huggingface&logoColor=white)
![Accuracy](https://img.shields.io/badge/Accuracy-98.51%25-brightgreen)
![License](https://img.shields.io/badge/License-MIT-lightgrey)
![Platform](https://img.shields.io/badge/Platform-Reddit%20%7C%20Streamlit-red)

---

## 📌 Overview

Social media platforms like Reddit are a goldmine of unfiltered political opinion. But analysing Indian political discourse is uniquely challenging — it mixes Hindi and English (Hinglish), carries heavy cultural sarcasm, and uses India-specific political vocabulary that general NLP models struggle to interpret.

This project addresses that gap with a **multi-stage NLP pipeline** that:

1. **Classifies** the sentiment of live Reddit comments using a fine-tuned BERTweet model (98.51% accuracy)
2. **Discovers topics** driving each sentiment using BERTopic — answering not just *what* people feel, but *why*
3. **Visualises** everything in a real-time, interactive Streamlit dashboard

---

## ✨ Features

- 🎯 **98.51% accuracy** on Indian political tweet classification (Positive / Neutral / Negative)
- 🔴 **Live Reddit data** fetching via PRAW from any subreddit
- 🧠 **BERTopic integration** for automatic topic discovery within each sentiment group
- 🌐 **Hinglish support** — handles code-mixed Hindi-English text
- 📊 **Interactive dashboard** with sentiment pie charts and intertopic distance maps
- ⚖️ Outperforms baseline models: SVM (94.35%) and Bi-LSTM (97.06%)

---

## 🏗️ Architecture

```
User Input (Subreddit)
        │
        ▼
┌───────────────────┐
│   PRAW (Reddit)   │  ← Fetches live comments via .hot()
└────────┬──────────┘
         │
         ▼
┌───────────────────────────┐
│  Fine-tuned BERTweet      │  ← vinai/bertweet-base, fine-tuned on
│  Sentiment Classifier     │    162K Indian Political Tweets
└────────┬──────────────────┘
         │  Labels: Positive / Neutral / Negative
         ▼
┌───────────────────────────┐
│  BERTopic Model           │  ← UMAP + HDBSCAN + c-TF-IDF
│  Topic Discovery          │    Discovers themes within sentiment groups
└────────┬──────────────────┘
         │
         ▼
┌───────────────────────────┐
│  Streamlit Dashboard      │  ← Pie charts, topic maps, raw data table
└───────────────────────────┘
```

---

## 📊 Model Performance

### Accuracy Comparison

| Model    | Accuracy |
|----------|----------|
| SVM (TF-IDF) | 94.35% |
| Bi-LSTM (Keras) | 97.06% |
| **BERTweet (Ours)** | **98.51%** |

### Class-wise Performance — Fine-tuned BERTweet

| Class    | Precision (%) | Recall (%) | F1-Score (%) |
|----------|--------------|------------|--------------|
| Positive | 98.41 | 98.43 | 98.42 |
| Neutral  | 99.08 | 98.61 | 98.85 |
| Negative | 98.03 | 98.49 | 98.26 |

---

## 🗂️ Dataset

- **Source:** [Indian Political Sentiment on Twitter — Kaggle]
- **Size:** ~162,000 labelled tweets
- **Languages:** English & Hinglish (Roman-script Hindi-English code-mixing)
- **Split:** 80% train / 20% test (stratified)

| Class    | Records |
|----------|---------|
| Positive | 56,233  |
| Neutral  | 55,209  |
| Negative | 54,189  |

---

## 🚀 Getting Started

### Prerequisites

- Python 3.9+
- A Reddit account with API credentials ([create here](https://www.reddit.com/prefs/apps))
- GPU recommended for inference (Google Colab T4 used for training)

### Installation

```bash
# Clone the repository
git clone https://github.com/SOURAVBHOWAL/Hinglish_Political_Sentiment_Analyser.git
cd Hinglish_Political_Sentiment_Analyser

# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Reddit API Setup

Create a `.env` file in the project root:

```env
REDDIT_CLIENT_ID=your_client_id
REDDIT_CLIENT_SECRET=your_client_secret
REDDIT_USER_AGENT=your_app_name/1.0
```

### Model Setup

The fine-tuned BERTweet model weights are required to run inference.

**Option A — Download pre-trained weights:**
```bash
# Download from HuggingFace Hub (once uploaded)
python scripts/download_model.py
```

**Option B — Fine-tune from scratch:**
```bash
python train/finetune_bertweet.py \
  --dataset_path data/indian_political_tweets.csv \
  --output_dir models/bertweet-finetuned \
  --epochs 5 \
  --batch_size 32 \
  --learning_rate 2e-5
```

> **Note:** Fine-tuning requires a GPU. Training was performed on Google Colab with a T4 GPU (~2–3 hours).

### Run the App

```bash
streamlit run app.py
```

Open your browser at `http://localhost:8501`

---

## 🖥️ Usage

1. Enter a subreddit name (e.g., `india`, `IndiaSpeaks`, `politics`)
2. Set the number of posts to fetch
3. Select a sentiment to analyse topics for (Positive / Neutral / Negative)
4. Click **Run Analysis**
5. View the sentiment distribution pie chart, topic map, and labelled comment table

---

## 📁 Project Structure

```
indian-political-sentiment-analyzer/
│
├── app.py                        # Streamlit frontend
├── requirements.txt
├── .env.example
│
├── train/
│   ├── finetune_bertweet.py      # BERTweet fine-tuning script
│   ├── train_svm.py              # SVM baseline
│   └── train_bilstm.py           # Bi-LSTM baseline
│
├── src/
│   ├── reddit_fetcher.py         # PRAW data collection
│   ├── sentiment_classifier.py   # HuggingFace inference pipeline
│   ├── topic_modeller.py         # BERTopic pipeline
│   └── preprocessor.py           # Tweet normalization utilities
│
├── models/
│   └── bertweet-finetuned/       # Fine-tuned model weights (gitignored)
│
├── data/
│   └── README.md                 # Dataset download instructions
│
├── notebooks/
│   ├── 01_EDA.ipynb              # Exploratory data analysis
│   ├── 02_Training.ipynb         # Model training (Colab)
│   └── 03_Evaluation.ipynb       # Confusion matrices & metrics
│
└── assets/
    └── screenshots/              # Dashboard screenshots
```

---

## 🛠️ Tech Stack

| Component | Technology |
|-----------|------------|
| Sentiment Model | `vinai/bertweet-base` (HuggingFace Transformers) |
| Topic Modelling | BERTopic (UMAP + HDBSCAN + c-TF-IDF) |
| Live Data | PRAW (Python Reddit API Wrapper) |
| Frontend | Streamlit + Plotly |
| Training | PyTorch + HuggingFace Trainer API |
| Hardware | Google Colab (T4 GPU) |

---

## 📖 How It Works

### Stage 1 — Model Fine-tuning

`vinai/bertweet-base` is a RoBERTa-based model pre-trained on 850 million English tweets. We fine-tune it on the Indian Political Sentiment dataset using the following configuration:

| Hyperparameter | Value |
|----------------|-------|
| Learning Rate | 2e-5 |
| Optimizer | AdamW (PyTorch) |
| Batch Size (Train) | 32 |
| Epochs | 5 |
| Max Sequence Length | 64 tokens |
| Weight Decay | 0.01 |
| Warmup Steps | 10% of total steps |
| Gradient Clipping | 1.0 |

Tweet normalization is applied before training — user mentions are replaced with `@USER` and URLs with `HTTPURL`, matching BERTweet's pre-training format.

### Stage 2 — Live Analysis Pipeline

1. **Fetch** — PRAW connects to a user-specified subreddit and retrieves the top hot posts and their comment threads
2. **Classify** — Each comment is passed through the fine-tuned BERTweet pipeline, receiving a sentiment label and confidence score
3. **Filter** — Comments are grouped by the selected sentiment
4. **Topic Model** — BERTopic embeds the filtered comments, applies UMAP for dimensionality reduction, HDBSCAN for clustering, and c-TF-IDF to extract representative topic keywords
5. **Visualise** — Results are rendered in the Streamlit dashboard

---

## 🔬 Baselines Compared

### SVM Configuration

| Parameter | Value |
|-----------|-------|
| Vectorizer | TF-IDF |
| Max Features | 50,000 |
| N-gram Range | Unigram + Bigram |
| Classifier | LinearSVC |

### Bi-LSTM Configuration

| Parameter | Value |
|-----------|-------|
| Embedding | Keras Embedding (64-dim) |
| LSTM Units | 64 |
| Max Seq Length | 64 |
| Optimizer | Adam (lr=1e-3) |
| Epochs | 5 (early stopping, patience=2) |

---

## 🔮 Future Work

- [ ] **Twitter/X integration** via Tweepy for cross-platform sentiment comparison
- [ ] **Bot detection** to filter inauthentic comments before analysis
- [ ] **Multilingual support** — fine-tune XLM-RoBERTa on Hindi (Devanagari), Hinglish, and regional languages
- [ ] **LLM-assisted topic naming** — use GPT/LLaMA to label BERTopic clusters (following Pluin et al., 2023)
- [ ] **Formal error analysis** — categorise misclassified samples (sarcasm, boundary cases, Hinglish ambiguity)
- [ ] **Temporal analysis** — track sentiment trends over time for the same political topic

---

## 🌍 SDG Alignment

This project contributes to **UN Sustainable Development Goal 16 — Peace, Justice and Strong Institutions** by providing transparent, data-driven insights into public political opinion, enabling citizens and researchers to engage with governance issues like corruption, healthcare, and economic policy more objectively.

---

## 📚 Key References

- Nguyen et al. (2020) — [BERTweet: A pre-trained language model for English Tweets](https://arxiv.org/abs/2005.10200)
- Grootendorst (2022) — [BERTopic: Neural topic modeling with a class-based TF-IDF procedure](https://arxiv.org/abs/2203.05794)
- Devlin et al. (2018) — [BERT: Pre-training of Deep Bidirectional Transformers](https://arxiv.org/abs/1810.04805)

Full bibliography available in the [project report](docs/report.pdf).

---

## 👥 Author

**[Sourav Bhowal]** 

*Final Year Project — [Your Institution Name], [Year]*

---

## 📄 License

This project is licensed under the MIT License.

---

> *"Sentiment analysis tells you what people feel. Topic modelling tells you why."*
