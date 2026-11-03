# CulSen-BN: An Uncertainty-Aware, Culture-Sensitive Contrastive Learning Framework for Robust Bengali Sentiment Analysis Across Music Seasons

**Ragib Hasan**  
*Department of Computer Science and Engineering*  

---

## Abstract
Informal social media commentary in low-resource and culturally nuanced contexts presents significant hurdles for conventional sentiment analysis systems. In this work, we investigate audience reactions to *Coke Studio Bangla* across Seasons 1 to 4 (comprising over 57,000 YouTube comments). Bengali sentiment classification in this domain is challenged by: (1) machine-generated pseudo-label noise, (2) pervasive code-mixing and Romanized Bengali (Banglish), (3) culturally rooted aesthetic expressions where tears and sadness denote intense artistic appreciation, and (4) domain shift across distinct musical seasons. To address these challenges, we introduce **CulSen-BN**, an uncertainty-aware, culture-sensitive contrastive learning framework. CulSen-BN integrates: (i) sample-level reliability weighting to down-weight noisy pseudo-labels, (ii) confidence-filtered supervised contrastive representation learning to separate ambiguous positive and negative commentary, and (iii) domain-adversarial regularization using a Gradient Reversal Layer to mitigate season-specific overfitting. We design rigorous, leakage-free evaluation protocols (video-grouped in-domain and leave-one-season-out cross-season testing) and establish a CPU-reproducible classical machine learning baseline (Char TF-IDF + Linear SVM) alongside contextual transformer architectures (BanglaBERT and XLM-RoBERTa). Our framework establishes an auditable foundation for culturally situated, noise-tolerant sentiment analysis in South Asian low-resource languages.

---

## 1. Introduction
Sentiment analysis has achieved substantial maturity in high-resource languages such as English. However, applying these techniques to informal social media text in low-resource Indic languages—particularly Bengali—remains fraught with systemic vulnerabilities. Bengali is spoken by over 230 million native speakers worldwide, yet digital text processing is hampered by dialectal variation, non-standard orthography, rampant transliteration into the Latin alphabet (Banglish), and intra-sentential English–Bengali code-mixing.

These linguistic challenges are amplified in creative and musical domains, exemplified by *Coke Studio Bangla*. Audience expressions in music discourse often invert conventional lexicon-based sentiment associations. For example, comments expressing crying or heartache (*"গানটা শুনে চোখে পানি চলে আসলো"* — "Tears came to my eyes listening to this song") convey profound aesthetic enjoyment and high artistic appreciation, rather than negative distress. Conversely, comments juxtaposing praise with sarcastic disappointment (*"এত সুন্দর গানটা অটোর টিউন দিয়ে শেষ করে দিল"* — "They ruined such a beautiful song with autotune") frequently mislead models that rely on superficial keyword cues.

Furthermore, large-scale web-scraped datasets typically rely on automated pseudo-labeling by pretrained multilingual encoders (e.g., `cardiffnlp/twitter-xlm-roberta-base-sentiment`), which inherit systematic calibration errors, cultural blindness, and label ambiguity. Finally, models evaluated on comments from specific musical releases risk memorizing artist names, song titles, or viral trends rather than learning transferable sentiment semantics.

To resolve these interconnected problems, this paper presents **CulSen-BN**. The key contributions of this study are:
1. **Dataset Characterization & Leakage-Free Benchmarks:** We perform an exhaustive audit of the Coke Studio Bangla Sentiment dataset (57,592 comments across 33 songs and 4 seasons) and formalize two leakage-free evaluation protocols: Protocol A (Video-Grouped In-Domain) and Protocol B (Leave-One-Season-Out Generalization), with automated assertions barring ID, video, and normalized text overlap.
2. **Human Verification Workflow:** We isolate human adjudication from automated predictions, deploying stratified sampling across linguistic categories (Bengali script, Banglish, mixed code, emojis, and negation) for researcher-driven ground-truth verification.
3. **CulSen-BN Framework:** We propose an integrated learning architecture that combines uncertainty-weighted classification loss, confidence-filtered supervised contrastive learning (SupCon), and adversarial season-invariance learning.
4. **Reproducible Empirical Baselines:** We deliver an executable, open-source experimental testbed including character n-gram SVM and pretrained transformer baselines.

---

## 2. Related Work

### 2.1 Bengali Sentiment Analysis and Code-Mixing
Early Bengali sentiment analysis relied predominantly on hand-crafted lexicons and classical feature engineering. The introduction of transformer language models catalyzed substantial improvements. Bhattacharjee et al. (2022) developed BanglaBERT, demonstrating state-of-the-art performance across classification and token-level tasks. In the domain of informal social media, Karim et al. and Hossain et al. explored Bengali–English code-mixed datasets (such as BnSentMix). While these works demonstrated the feasibility of code-mixed classification, they largely operated on in-distribution splits without addressing pseudo-label noise or out-of-season temporal generalization.

### 2.2 Learning from Noisy Labels and Uncertainty
When datasets are compiled at web scale, human annotation is cost-prohibitive, leading practitioners to rely on machine-generated pseudo-labels. However, models trained naively on noisy labels exhibit memorization and degraded generalization. Confidence-weighted supervision and label smoothing offer principled regularizations. Guo et al. (2017) demonstrated that modern deep networks are prone to overconfidence, necessitating calibration techniques such as temperature scaling and Expected Calibration Error (ECE) monitoring.

### 2.3 Contrastive Learning and Domain Adaptation
Supervised contrastive learning (Khosla et al., 2020) pulls same-class representations together while repelling distinct classes, yielding robust decision boundaries. In domain generalization, Ganin et al. (2016) proposed Domain-Adversarial Neural Networks (DANN), using a Gradient Reversal Layer (GRL) to encourage features to be discriminative for the primary task while invariant to domain shifts. CulSen-BN synergistically combines these paradigms for South Asian sentiment processing.

---

## 3. Dataset and Annotation Quality

### 3.1 Corpus Source and Schema
The primary dataset is derived from `ragibhasan/coke_studio_bangla_sentiment`, comprising 57,592 YouTube comments across 33 songs spanning Seasons 1 through 4 of Coke Studio Bangla. The original schema includes raw comment texts, cleaned texts, engagement metrics, detected scripts, video identifiers, season numbers, and machine-generated sentiment labels (`cardiffnlp/twitter-xlm-roberta-base-sentiment`).

### 3.2 Linguistic and Distributional Characteristics
Our audit reveals the following empirical distributions:
- **Sentiment Labels (Pseudo-labels):** Positive = 36,167 (62.8%), Neutral = 10,945 (19.0%), Negative = 10,480 (18.2%).
- **Linguistic Scripts:** Latin / Banglish / English = 55.7%, Bengali Script = 41.5%, Mixed Code = 2.6%, Emoji / Other = 0.2%.
- **Season Breakdown:** Season 1 (15,537 comments, 10 videos), Season 2 (20,561 comments, 12 videos), Season 3 (14,553 comments, 8 videos), Season 4 (6,941 comments, 3 videos).
- **Linguistic Markers:** 39.4% of comments contain Unicode emojis; 12.8% contain explicit Bengali or Banglish negation markers.

### 3.3 Leakage Hazards and Protocol Definition
We identify cross-video duplicate comments (identical text appearing under multiple songs) as a critical leakage hazard. Under **Protocol A**, all videos are grouped strictly into train (70%), validation (15%), or test (15%) partitions, and cross-video duplicate texts are restricted to training, ensuring zero text or group leakage into evaluation sets. Under **Protocol B**, individual seasons are held out sequentially for zero-shot seasonal transfer evaluation.

### 3.4 Human Verification Protocol
To prevent circular validation, machine-generated pseudo-labels are strictly segregated from ground truth. A stratified evaluation sample of 800 comments (covering both classes, all seasons, and linguistic subgroups) alongside 200 challenge comments (oversampling negation, low model confidence, and heavy emoji usage) is isolated for manual adjudication by the researcher.

---

## 4. Proposed CulSen-BN Methodology

The CulSen-BN architecture consists of four interconnected components:

```text
Input Comment X
      │
Contextual Encoder (BanglaBERT / XLM-R) ──> Pooled Embedding h
      │                                             ├──> Contrastive Projection Head ──> z (SupCon Loss)
      │                                             └──> Gradient Reversal Layer ──────> Season Discriminator
      ▼
Sentiment Classifier Head ──> Logits ──> Uncertainty-Weighted Loss
```

### 4.1 Contextual Representation
Given an input sequence $X = (x_1, \dots, x_n)$, a pretrained transformer encoder computes contextual representations $H = f_\theta(X)$. Mean pooling over unmasked token positions yields comment representation $h = \text{Pool}(H) \in \mathbb{R}^d$.

### 4.2 Uncertainty-Aware Supervision
To penalize noisy pseudo-labels, we assign each training instance a reliability weight $r_i = \text{Clip}(c_i, r_{\min}, 1.0)$, where $c_i$ is the model confidence and $r_{\min} = 0.10$. The uncertainty-weighted loss is:
$$\mathcal{L}_{cls} = -\frac{1}{N} \sum_{i=1}^N r_i \left[ y_i \log(p_i) + (1 - y_i) \log(1 - p_i) \right]$$

### 4.3 Confidence-Filtered Supervised Contrastive Learning
Embeddings $h_i$ are projected to $z_i = \text{Normalize}(g_\phi(h_i)) \in \mathbb{R}^p$. To ensure noisy labels do not corrupt positive pairs, pairs are filtered by confidence threshold $\tau_{conf} = 0.65$:
$$\mathcal{L}_{con} = \sum_{i \in I} \frac{-1}{|P(i)|} \sum_{p \in P(i)} \log \frac{\exp(z_i^\top z_p / \tau)}{\sum_{a \in A(i)} \exp(z_i^\top z_a / \tau)}$$

### 4.4 Season-Invariance Adversarial Regularization
To prevent overfitting to season-specific musical arrangements or artist names, pooled embeddings pass through a Gradient Reversal Layer (GRL) into a season discriminator $D_\psi(h)$ trained with cross-entropy $\mathcal{L}_{domain}$.

### 4.5 Integrated Multi-Task Objective
$$\mathcal{L}_{CulSen} = \mathcal{L}_{cls} + \alpha \mathcal{L}_{con} + \beta \mathcal{L}_{domain} + \gamma \mathcal{L}_{cons}$$
where $\alpha=0.20, \beta=0.10, \gamma=0.05$.

---

## 5. Experimental Setup

- **Baselines:** 
  1. Classical ML: Character TF-IDF (2–5 n-grams, 300,000 features) + Platt-calibrated Linear SVM.
  2. Monolingual Pretrained Transformer: BanglaBERT (`sagorsarker/banglabert`).
  3. Multilingual Pretrained Transformer: XLM-RoBERTa (`xlm-roberta-base`).
- **Evaluation Metrics:** Macro-F1 (primary), Positive F1, Negative F1, Balanced Accuracy, Matthews Correlation Coefficient (MCC), and Expected Calibration Error (ECE).
- **Statistical Testing:** Paired bootstrap resampling (1,000 resamples, 95% CI) and McNemar's test with continuity correction.

---

## 6. Empirical Results

### 6.1 Baseline Performance (Protocol A In-Domain)
*(Empirical metrics for SVM and planned transformer baselines)*

| Model | Supervision | Macro-F1 | Accuracy | Pos F1 | Neg F1 | MCC | ECE (%) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Char TF-IDF + Linear SVM** | Squared Hinge (Platt Calibrated) | **0.8092** | **0.8736** | **0.9201** | **0.6983** | **0.6214** | **9.47%** |
| **BanglaBERT Baseline** | Standard Cross-Entropy | *[Pending GPU training]* | — | — | — | — | — |
| **XLM-RoBERTa Baseline** | Standard Cross-Entropy | *[Pending GPU training]* | — | — | — | — | — |
| **CulSen-BN (Proposed)** | Integrated Multi-Task | *[Pending GPU training]* | — | — | — | — | — |

---

## 7. Cross-Season Generalization (Protocol B)
*(Leave-One-Season-Out Evaluation across Seasons 1 to 4)*

Under Protocol B, models are evaluated in a zero-shot cross-season transfer configuration where an entire season is held out strictly for testing, with zero video or normalized text overlap from training and validation partitions. Table 3 reports empirical generalization metrics for the classical Char TF-IDF + Platt-calibrated Linear SVM baseline alongside pending contextual transformer models.

| Target Evaluation Season | Training Source Seasons | Test Samples | Baseline SVM Macro-F1 | Baseline SVM Accuracy | CulSen-BN Macro-F1 | Transfer Gap |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Season 1** | Seasons 2, 3, 4 | 12,642 | **0.8093** | **0.8644** | *[Pending GPU]* | — |
| **Season 2** | Seasons 1, 3, 4 | 16,749 | **0.8213** | **0.8756** | *[Pending GPU]* | — |
| **Season 3** | Seasons 1, 2, 4 | 11,703 | **0.8126** | **0.8678** | *[Pending GPU]* | — |
| **Season 4** | Seasons 1, 2, 3 | 5,553 | **0.8244** | **0.8550** | *[Pending GPU]* | — |
| **Cross-Season Mean** | — | — | **0.8169** | **0.8657** | *[Pending GPU]* | — |

---

## 8. Ablation Study
We systematically ablate CulSen-BN components across six variants:
- **B0:** Base Transformer Encoder (Standard Cross-Entropy)
- **B1:** Base + Uncertainty Weighting ($\mathcal{L}_{cls}$)
- **B2:** Base + Supervised Contrastive Head ($\mathcal{L}_{cls} + \alpha \mathcal{L}_{con}$)
- **B3:** Base + Uncertainty + Contrastive Learning
- **B4:** Base + Uncertainty + Contrastive + Domain Invariance ($\mathcal{L}_{domain}$)
- **B5:** Full CulSen-BN Architecture (+ Consistency Regularization $\mathcal{L}_{cons}$)

---

## 9. Error Analysis and Explainability
We isolate failure modes across distinct linguistic strata:
1. **Sarcasm and Ironic Commendation:** Sentences containing glowing vocabulary used mockingly.
2. **Emotional Ambiguity:** Expressions of crying (*"চোখে পানি"*) misattributed to negative sadness.
3. **Phonetic Banglish Variations:** Colloquial abbreviations (*"fatafati"*, *"agun"*, *"baje"*).

---

## 10. Limitations, Ethics, and Data Governance
- **Data Governance:** YouTube comments are subject to platform terms of service. Raw comment texts are preserved locally for auditability; only reproducible manifests and derived feature representations should be redistributed.
- **Single-Annotator Subjectivity:** If human adjudication is conducted by a single researcher, individual cognitive bias is present, precluding inter-annotator agreement metrics.
- **Hardware Footprint:** While classical SVM baselines execute in minutes on CPU, transformer training requires modern GPU acceleration.

---

## 11. Conclusion
CulSen-BN introduces an integrated, culturally sensitive, and uncertainty-aware framework for Bengali sentiment analysis in creative social media domains. By systematically addressing pseudo-label noise, subtle cultural expressions, and cross-season domain shifts, this research establishes a principled and reproducible blueprint for low-resource Indic sentiment classification.
