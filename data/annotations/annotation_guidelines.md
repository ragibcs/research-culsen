# CulSen-BN: Human Annotation & Verification Guidelines

## 1. Objective
This protocol establishes standard guidelines for the researcher to manually verify and adjudicate sentiment labels for Coke Studio Bangla YouTube comments.

**Important Note on Scientific Rigor:**
- The original labels in the dataset are **machine-generated pseudo-labels** produced by `cardiffnlp/twitter-xlm-roberta-base-sentiment`.
- Under no circumstances should pseudo-labels be treated as ground truth.
- Human verification must be conducted manually by the researcher without automated pre-filling.
- If only one researcher conducts the verification, inter-annotator agreement (Cohen's kappa) cannot be reported, and the limitation of single-annotator subjectivity must be explicitly documented in the manuscript.

---

## 2. Label Taxonomy

Annotators must assign one of the following labels to `annotator_label`:

| Label | Definition | Examples |
| :--- | :--- | :--- |
| `positive` | Expression of praise, enjoyment, aesthetic appreciation, cultural pride, or emotional connection to the performance, singers, music, or production. | *"গানটা শুনে কলিজা ঠান্ডা হয়ে গেল"*, *"অসাধারণ পারফরম্যান্স ❤️"*, *"Best song of the season"* |
| `negative` | Expression of criticism, disappointment, boredom, technical flaws, or disapproval regarding the song, arrangement, or artists. | *"আগের গানটা ভালো ছিল, এটা একদম নষ্ট করেছে"*, *"সুর ভালো লাগেনি"*, *"Total disaster"* |
| `neutral` | Factual inquiry, objective statement without sentiment, timestamps, lyrics transcription, or informational remarks. | *"গানটির মূল শিল্পী কে?"*, *"৩:১৫ মিনিটে ড্রামস কার?"*, *"Song released on Friday"* |
| `ambiguous` | Sarcasm, conflicting polarities without a dominant sentiment, incomprehensible typos, or statements where intent cannot be reliably deduced. | *"বাহ! কী সুন্দর করে গানটা নষ্ট করল!"* (if intent is unclear), *"গানটা ভালো কিন্তু খারাপ"* |

---

## 3. Cultural & Domain Nuances in Coke Studio Bangla

### A. Words of Sadness / Crying as Intense Positive Appreciation
In Bengali music commentary, weeping or goosebumps often represent supreme appreciation:
- **Example:** *"গানটা শুনে চোখে পানি চলে আসলো ❤️"* $\rightarrow$ **Label: `positive`**
- **Guideline:** Do not classify literal mentions of tears, crying, or heartache as `negative` if they signify emotional movingness.

### B. Contrasting Vocabulary (Sarcasm and Disappointment)
Comments often juxtapose complimentary words with negative outcomes:
- **Example:** *"এত সুন্দর গানটা অটোর টিউন দিয়ে শেষ করে দিল"* $\rightarrow$ **Label: `negative`**
- **Guideline:** Focus on the net evaluative sentiment toward the artwork.

### C. Banglish and Code-Mixed Comments
Many comments are written in English letters (Banglish) or mix English with Bengali script:
- **Example:** *"Ei gaan ta shune mon bhalo hoye gelo"* $\rightarrow$ **Label: `positive`**
- **Example:** *"Vocal quality ekdom baje"* $\rightarrow$ **Label: `negative`**
- Record `language_type` as `bengali`, `banglish_latin`, or `mixed`.

---

## 4. Annotation Template Schema

The exported file `data/annotations/annotation_template.csv` contains the following fields:

1. `sample_id`: Unique YouTube comment identifier (do not modify).
2. `batch_type`: `natural_evaluation` (unbiased test stratum) or `challenge_evaluation` (oversampled negation/emojis).
3. `comment_text`: Cleaned comment text.
4. `pseudo_label`: The original model prediction (retained for comparison; do not rely upon).
5. `season_id`: Season number (1, 2, 3, or 4).
6. `video_id`: YouTube video identifier.
7. `annotator_label`: Enter `positive`, `negative`, `neutral`, or `ambiguous`.
8. `ambiguity_flag`: Enter `TRUE` if the comment contains heavy sarcasm, mixed polarity, or uncertainty; otherwise `FALSE`.
9. `language_type`: Enter `bengali`, `banglish_latin`, `mixed`, or `emoji_or_other`.
10. `annotator_notes`: Optional notes explaining tricky cases.
11. `adjudicated_label`: Final resolved label after adjudication (if multiple annotators or second pass).
12. `verification_status`: Change from `unreviewed` to `verified` once completed.

---

## 5. Verification Workflow

1. Open `data/annotations/annotation_template.csv` in Excel, LibreOffice Calc, or Google Sheets.
2. For each row, read `comment_text` carefully.
3. Fill in `annotator_label`, `ambiguity_flag`, and change `verification_status` to `verified`.
4. Save the completed file as `data/annotations/human_verified_test.csv`.
5. Run the verification import and validation check:
   ```bash
   python scripts/evaluate.py --human-annotations data/annotations/human_verified_test.csv
   ```
