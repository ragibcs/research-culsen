---
annotations_creators:
  - machine-generated
language_creators:
  - crowdsourced
language:
  - bn
  - en
license: apache-2.0
multilinguality:
  - multilingual
size_categories:
  - 10K<n<100K
source_datasets:
  - original
task_categories:
  - text-classification
task_ids:
  - sentiment-analysis
pretty_name: Coke Studio Bangla YouTube Comments Sentiment Dataset
tags:
  - music
  - youtube
  - coke-studio
  - coke-studio-bangla
  - bangla
  - bengali
  - banglish
  - sentiment-analysis
  - social-media
  - cross-lingual
configs:
  - config_name: default
    data_files:
      - split: train
        path: data/full/train-*
    default: true
  - config_name: curated
    data_files:
      - split: train
        path: data/curated/train-*
dataset_info:
  - config_name: default
    features:
      - name: comment_id
        dtype: string
      - name: video_id
        dtype: string
      - name: song
        dtype: string
      - name: season
        dtype: int64
      - name: topic
        dtype: string
      - name: subtopic
        dtype: string
      - name: text_clean
        dtype: string
      - name: text_raw
        dtype: string
      - name: emojis
        dtype: string
      - name: script
        dtype: string
      - name: n_chars
        dtype: int64
      - name: n_words
        dtype: int64
      - name: n_emoji
        dtype: int64
      - name: votes
        dtype: int64
      - name: is_reply
        dtype: bool
      - name: parent_cid
        dtype: string
      - name: has_heart
        dtype: bool
      - name: published_at
        dtype: timestamp[ns, tz=UTC]
      - name: author_id
        dtype: string
      - name: p_negative
        dtype: float64
      - name: p_neutral
        dtype: float64
      - name: p_positive
        dtype: float64
      - name: sentiment
        dtype: string
      - name: confidence
        dtype: float64
      - name: sentiment_model
        dtype: string
      - name: needs_review
        dtype: bool
    splits:
      - name: train
        num_bytes: 12966351
        num_examples: 57592
  - config_name: curated
    features:
      - name: comment_id
        dtype: string
      - name: video_id
        dtype: string
      - name: song
        dtype: string
      - name: season
        dtype: int64
      - name: text_clean
        dtype: string
      - name: script
        dtype: string
      - name: emojis
        dtype: string
      - name: votes
        dtype: int64
      - name: is_reply
        dtype: bool
      - name: published_at
        dtype: timestamp[ns, tz=UTC]
      - name: sentiment
        dtype: string
      - name: confidence
        dtype: float64
      - name: p_negative
        dtype: float64
      - name: p_neutral
        dtype: float64
      - name: p_positive
        dtype: float64
      - name: needs_review
        dtype: bool
    splits:
      - name: train
        num_bytes: 7261754
        num_examples: 57592
---

# Coke Studio Bangla YouTube Comments Sentiment Dataset

A rich collection of **57,592 YouTube comments** across **33 songs** and **4 seasons** of **Coke Studio Bangla**, labeled with sentiment classes (`positive`, `neutral`, `negative`), model confidence scores, language scripts (Bengali, Latin/Banglish, mixed), topics, and engagement metrics.

---

## Dataset Summary

- **Total Comments:** 57,592
- **Unique Songs:** 33 tracks across Coke Studio Bangla Seasons 1 to 4
- **Language / Scripts:**
  - **Latin / Banglish / English:** 32,098 (55.7%)
  - **Bengali script (বাংলা):** 23,877 (41.5%)
  - **Mixed script:** 1,518 (2.6%)
  - **Emoji or other:** 99 (0.2%)
- **Sentiment Distribution:**
  - **Positive:** 36,167 (62.8%)
  - **Neutral:** 10,945 (19.0%)
  - **Negative:** 10,480 (18.2%)
- **Sentiment Model:** `cardiffnlp/twitter-xlm-roberta-base-sentiment`
- **Configurations:**
  - `default`: Complete 26-column schema with raw text, clean text, metadata, topic classification, and sentiment scores.
  - `curated`: Streamlined 16-column schema focused on clean text and sentiment classification.

---

## Quickstart & Usage

Install the `datasets` library:

```bash
pip install datasets pyarrow
```

### Loading the Default (Full) Dataset

```python
from datasets import load_dataset

# Load default configuration (26 features)
ds = load_dataset("ragibhasan/coke_studio_bangla_sentiment")

print(ds["train"][0])
```

### Loading the Curated Dataset

```python
# Load curated configuration (16 features)
ds_curated = load_dataset("ragibhasan/coke_studio_bangla_sentiment", "curated")

print(ds_curated["train"][0])
```

---

## Dataset Structure

### Configurations

| Configuration | Features | Description |
| :--- | :--- | :--- |
| `default` | 26 | Full metadata: `comment_id`, `video_id`, `song`, `season`, `topic`, `subtopic`, `text_clean`, `text_raw`, `emojis`, `script`, `n_chars`, `n_words`, `n_emoji`, `votes`, `is_reply`, `parent_cid`, `has_heart`, `published_at`, `author_id`, `p_negative`, `p_neutral`, `p_positive`, `sentiment`, `confidence`, `sentiment_model`, `needs_review` |
| `curated` | 16 | Focused on modeling: `comment_id`, `video_id`, `song`, `season`, `text_clean`, `script`, `emojis`, `votes`, `is_reply`, `published_at`, `sentiment`, `confidence`, `p_negative`, `p_neutral`, `p_positive`, `needs_review` |

### Field Descriptions

- `comment_id` *(string)*: Unique identifier for the YouTube comment.
- `video_id` *(string)*: YouTube video ID.
- `song` *(string)*: Name of the Coke Studio Bangla song (e.g., *Bulbuli*, *Kotha Koiyo Na*, *Murir Tin*, *Long Distance Love*).
- `season` *(int64)*: Coke Studio Bangla season number (`1`, `2`, `3`, `4`).
- `topic` *(string)*: Broad comment theme/topic (e.g., `music`, `artist`, `production`).
- `subtopic` *(string)*: Granular subtopic classification.
- `text_clean` *(string)*: Preprocessed comment text (normalized, stripped of URLs/excess whitespace).
- `text_raw` *(string)*: Original unfiltered comment text as retrieved from YouTube.
- `emojis` *(string)*: Emojis extracted from the comment text.
- `script` *(string)*: Detected writing script (`bangla`, `latin`, `mixed`, `emoji_or_other`).
- `n_chars` *(int64)*: Character length of the comment.
- `n_words` *(int64)*: Word count of the comment.
- `n_emoji` *(int64)*: Number of emoji characters present.
- `votes` *(int64)*: YouTube like count on the comment.
- `is_reply` *(bool)*: Whether the comment is a reply (`True`) or a top-level comment (`False`).
- `parent_cid` *(string)*: Parent comment ID (if this is a reply).
- `has_heart` *(bool)*: Whether the comment was hearted by the creator channel.
- `published_at` *(timestamp)*: UTC publication timestamp.
- `author_id` *(string)*: Anonymized hash of the comment author ID.
- `p_negative` *(float64)*: Probability of negative sentiment predicted by XLM-RoBERTa.
- `p_neutral` *(float64)*: Probability of neutral sentiment predicted by XLM-RoBERTa.
- `p_positive` *(float64)*: Probability of positive sentiment predicted by XLM-RoBERTa.
- `sentiment` *(string)*: Predicted sentiment label (`positive`, `neutral`, `negative`).
- `confidence` *(float64)*: Model prediction confidence (max probability).
- `sentiment_model` *(string)*: Sentiment analysis model checkpoint used.
- `needs_review` *(bool)*: Flag indicating low confidence or ambiguous predictions for human review.

---

## Dataset Breakdown & Insights

### Sentiment Distribution
- **Positive:** 36,167 (~62.8%)
- **Neutral:** 10,945 (~19.0%)
- **Negative:** 10,480 (~18.2%)

### Season Breakdown
| Season | Comments |
| :--- | :--- |
| Season 2 | 20,561 |
| Season 1 | 15,537 |
| Season 3 | 14,553 |
| Season 4 | 6,941 |

### Top Songs Represented
1. **Bulbuli** (2,598 comments)
2. **Long Distance Love** (2,551 comments)
3. **Kotha Koiyo Na** (2,423 comments)
4. **Murir Tin** (2,403 comments)
5. **Moha Jadu** (2,235 comments)
6. *...and 28 other popular tracks.*

### Visualizations & Insights

#### Sentiment Distribution by Song
![Sentiment Distribution by Song](https://huggingface.co/datasets/ragibhasan/coke_studio_bangla_sentiment/resolve/main/insights/sentiment_by_song.png)

#### Net Sentiment Trend
![Net Sentiment Trend](https://huggingface.co/datasets/ragibhasan/coke_studio_bangla_sentiment/resolve/main/insights/net_sentiment_trend.png)

#### Net Sentiment by Season
![Net Sentiment by Season](https://huggingface.co/datasets/ragibhasan/coke_studio_bangla_sentiment/resolve/main/insights/net_sentiment_by_season.png)

---

## Use Cases

1. **Code-Mixed & Multilingual Sentiment Analysis:** Benchmark models on Bengali text written in Bengali script vs. Banglish (Bengali in Latin script).
2. **Music & Entertainment Domain NLP:** Analyze reception, fan sentiment, and audience engagement across music releases.
3. **Low-Resource & Regional NLP:** Fine-tune multilingual transformers (XLM-RoBERTa, BanglaBERT, IndicBERT, mT5) for social media text processing.
4. **Topic & Engagement Modeling:** Study the relationship between comment likes, reply depth, emojis, and overall sentiment.

---

## Citation & Attribution

If you use this dataset in your research or project, please cite:

```bibtex
@dataset{coke_studio_bangla_sentiment,
  author       = {Hasan, Ragib and Sen Toma, Indrani and Galib, Asadulla Hel},
  title        = {Coke Studio Bangla YouTube Comments Sentiment Dataset},
  year         = {2026},
  publisher    = {Hugging Face},
  version      = {226b36f},
  doi          = {10.57967/hf/10724},
  url          = {https://huggingface.co/datasets/ragibhasan/coke_studio_bangla_sentiment}
}
```

## License
This dataset is distributed under the [Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0). YouTube comments remain subject to YouTube's Terms of Service.
