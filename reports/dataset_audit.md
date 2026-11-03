# CulSen-BN: Dataset Audit & Quality Report

**Dataset Name:** Coke Studio Bangla Sentiment Dataset
**Total Records:** 57,592
**Total Columns:** 26
**Unique Videos:** 33
**Seasons Covered:** 1, 2, 3, 4

## 1. Sentiment Class Distribution (Pseudo-Labels)
| Sentiment Class | Count | Percentage (%) |
| :--- | :--- | :--- |
| `positive` | 36,167 | 62.80% |
| `neutral` | 10,945 | 19.00% |
| `negative` | 10,480 | 18.20% |

## 2. Season-wise Breakdown
| Season | Total Comments | Unique Videos | Positive | Neutral | Negative |
| :--- | :--- | :--- | :--- | :--- | :--- |
| Season 1 | 15,537 | 8 | 9,896 | 2,895 | 2,746 |
| Season 2 | 20,561 | 12 | 13,088 | 3,812 | 3,661 |
| Season 3 | 14,553 | 8 | 9,174 | 2,850 | 2,529 |
| Season 4 | 6,941 | 5 | 4,009 | 1,388 | 1,544 |

## 3. Duplication and Leakage Hazards
- **Exact Duplicates:** 1,344 comments
- **Normalized Duplicates:** 1,344 comments
- **Texts Spanning Multiple Videos:** 649 texts (must be isolated in splits)

## 4. Script and Linguistic Characteristics
| Script / Category | Count | Percentage (%) |
| :--- | :--- | :--- |
| `latin` | 32,098 | 55.73% |
| `bangla` | 23,877 | 41.46% |
| `mixed` | 1,518 | 2.64% |
| `emoji_or_other` | 99 | 0.17% |

## 5. Salient Linguistic Features
- **Comments with Emojis:** 26,154 (45.41%)
- **Comments with Negation Markers:** 3,515 (6.10%)
- **Machine-Flagged for Review (`needs_review`):** 16,840 (29.24%)

## 6. Length Distribution Summary
- **Character Length:** Mean = 73.6, Median = 41.0, Max = 1945
- **Word Length:** Mean = 13.0, Median = 7.0, Max = 388

---
*Note: Original sentiment labels are machine-generated pseudo-labels (`cardiffnlp/twitter-xlm-roberta-base-sentiment`). Ground truth validation must be conducted via manual human adjudication (Stage 2).*