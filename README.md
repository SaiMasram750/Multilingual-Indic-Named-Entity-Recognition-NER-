
# Multilingual Indic Named Entity Recognition (NER) using MuRIL

An end-to-end NLP project fine-tuning Google's **MuRIL (Multilingual Representations for Indian Languages)** model on sequence labeling to extract Named Entities across three major Indic languages: **Bengali (`bn`)**, **Marathi (`mr`)**, and **Telugu (`te`)**.

The final fine-tuned model and tokenizer are published on the Hugging Face Hub: [`Learner-sai/muril-ner-multilingual-v3`](https://huggingface.co/Learner-sai/muril-ner-multilingual-v3).

---

## 📌 Project Overview

- **Model Architecture:** `google/muril-base-cased` with a token classification head (`AutoModelForTokenClassification`).
- **Target Entities (IOB2 Format):**
  - `PER` (Person)
  - `ORG` (Organization)
  - `LOC` (Location)
  - `O` (Outside / Non-entity token)
- **Languages Supported:** Bengali (`bn`), Marathi (`mr`), Telugu (`te`).
- **Dataset:** [AI4Bharat Naamapadam (`ai4bharat/naamapadam`)](https://huggingface.co/datasets/ai4bharat/naamapadam).

---

## 🛠️ Key Technical Features

1. **Subword Label Alignment:** Implemented custom word-level to subword token alignment handling WordPiece tokenization (assigning entity labels to the primary subword and `-100` to subsequent subwords to ignore loss during backpropagation).
2. **Schema & Label Indexing:** Resolved schema shift issues (V2) to strictly align outside tokens (`O`) to index `0` and mapped IOB2 tags across 7 standard categories (`O`, `B-PER`, `I-PER`, `B-ORG`, `I-ORG`, `B-LOC`, `I-LOC`).
3. **Training Strategy & Optimization (V3):**
   - **Sequence Length:** Expanded to 256 tokens to eliminate entity truncation.
   - **Optimization:** Utilized gradient accumulation and learning rate warmup (`warmup_ratio=0.1`) for robust convergence.
   - **Inference Metrics:** Achieved **85%–92%+ extraction confidence** across complex multilingual test cases.

---

## 🚀 Quickstart & Usage

### 1. Installation

```bash
pip install transformers torch datasets seqeval

```

### 2. Inference via Hugging Face Pipeline

You can directly load and test the published fine-tuned model using the Hugging Face `pipeline` API:

```python
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline

# Load model and tokenizer from Hugging Face Hub
model_name = "Learner-sai/muril-ner-multilingual-v3"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForTokenClassification.from_pretrained(model_name)

# Create NER pipeline
ner_pipeline = pipeline("ner", model=model, tokenizer=tokenizer, aggregation_strategy="simple")

# Sample Inference (Marathi Example)
sample_text_mr = "टाटा कन्सल्टन्सी सर्व्हिसेस ही मुंबईत स्थित आहे."
results = ner_pipeline(sample_text_mr)

print(results)

```

---

## 📂 Notebook Structure

The included Jupyter Notebook (`.ipynb`) covers the full end-to-end execution pipeline:

1. **Environment Setup & Dependencies:** GPU check and library installations.
2. **Dataset Loading & Preprocessing:** Filtering Naamapadam dataset for Bengali, Marathi, and Telugu.
3. **Custom Subword Alignment:** Tokenization and label alignment functions.
4. **Model Initialization:** Fine-tuning setup using Hugging Face `Trainer` and `TrainingArguments`.
5. **Evaluation:** Performance evaluation using `seqeval` metrics (Precision, Recall, F1).
6. **Model Publishing:** Automated deployment to the Hugging Face Hub.

---

## 📊 Label Mapping Reference

| Index | Label | Description |
| --- | --- | --- |
| **0** | `O` | Outside of any named entity |
| **1** | `B-PER` | Beginning of Person name |
| **2** | `I-PER` | Inside of Person name |
| **3** | `B-ORG` | Beginning of Organization name |
| **4** | `I-ORG` | Inside of Organization name |
| **5** | `B-LOC` | Beginning of Location name |
| **6** | `I-LOC` | Inside of Location name |

---

## 🔗 Model Links & Repository

* **Hugging Face Model:** [Learner-sai/muril-ner-multilingual-v3](https://www.google.com/url?sa=E&source=gmail&q=https://huggingface.co/Learner-sai/muril-ner-multilingual-v3)
* **Author:** Sai Masram

