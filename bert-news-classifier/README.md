# 📰 News Topic Classifier Using BERT

## 🎯 Objective
Fine-tune a pre-trained BERT transformer model to automatically classify 
news headlines into 4 categories: **World, Sports, Business, and Sci/Tech**, 
using the AG News dataset from Hugging Face.

---

## 🧠 Methodology / Approach

### 1. Dataset
- **AG News Dataset** (Hugging Face) — 120,000 train / 7,600 test samples
- Balanced across 4 categories: World, Sports, Business, Sci/Tech

### 2. Preprocessing
- Tokenized using `bert-base-uncased` tokenizer
- Max sequence length: 128 tokens
- Padding & truncation applied

### 3. Model
- **bert-base-uncased** (110M parameters) from Hugging Face Transformers
- Added 4-class classification head
- Fine-tuned for 3 epochs using Hugging Face `Trainer` API
- Training subset: 8,000 samples

### 4. Evaluation
- Metrics: Accuracy, Weighted F1-Score, Per-class Precision/Recall
- Visualizations: Confusion Matrix, Training Curves, Per-class F1 Bar Chart

### 5. Deployment
- Deployed on **Hugging Face Spaces** using Gradio
- Live demo: [🔗 Click here](https://huggingface.co/spaces/zoha12/bert-ag-news-classifier)

---

## 📊 Key Results

| Metric | Score |
|--------|-------|
| Accuracy | **92.15%** |
| Weighted F1 | **92.15%** |
| Best Class | Sports (F1: 0.977) |
| Weakest Class | Business (F1: 0.889) |

### Per-Class Performance
| Category | Precision | Recall | F1 |
|----------|-----------|--------|----|
| 🌍 World | 0.93 | 0.93 | 0.93 |
| ⚽ Sports | 0.98 | 0.98 | 0.98 |
| 💼 Business | 0.92 | 0.86 | 0.89 |
| 🔬 Sci/Tech | 0.86 | 0.92 | 0.89 |

---

## 🔍 Key Observations
- Sports headlines are easiest to classify (F1: 0.977) due to unique vocabulary
- Biggest confusion: Business → Sci/Tech (54 cases) due to tech company news overlap
- 92% accuracy achieved with only 8,000 training samples, demonstrating BERT's powerful transfer learning
- Full dataset training (120k samples) would push accuracy to ~94%

---

## 🛠️ Tech Stack
- Python, PyTorch, Hugging Face Transformers & Datasets
- Gradio, Scikit-learn, Matplotlib, Seaborn

---

## 🚀 How to Run Locally
```bash
git clone https://github.com/zoha12/bert-news-classifier
cd bert-news-classifier
pip install -r requirements.txt
jupyter notebook notebook.ipynb
```