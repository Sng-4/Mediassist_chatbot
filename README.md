# 🏥 MediAssist — Domain-Specific Medical AI Assistant

> **Fine-tuned Gemma 2 2B Instruct** for medical Q&A, deployed as a full-stack chat application with a FastAPI backend and Streamlit frontend.

---

## Table of Contents

- [Project Overview](#-project-overview)
- [Architecture](#-architecture)
- [Fine-Tuning Methodology](#-fine-tuning-methodology)
- [Dataset](#-dataset)
- [Experiment Results](#-experiment-results)
- [Project Structure](#-project-structure)
- [Local Setup & Installation](#-local-setup--installation)
- [Running the Application](#-running-the-application)
- [Deliverables](#-deliverables)
- [Disclaimer](#️-disclaimer)
- [References](#-references)

---

## 🔭 Project Overview

MediAssist is a specialized AI assistant designed to provide accurate, concise, and professionally-toned medical information across a broad range of clinical domains — including anatomy, pharmacology, pathology, physiology, and clinical medicine.

| Attribute | Detail |
|-----------|--------|
| **Base Model** | Gemma 2 2B Instruct (`gemma2_instruct_2b_en`) |
| **Framework** | TensorFlow 2.x + Keras 3 + keras-hub |
| **PEFT Method** | LoRA — Rank 16, Alpha 32 |
| **Dataset** | `medalpaca/medical_meadow_medical_flashcards` (HuggingFace) |
| **Training Accelerator** | Kaggle P100 / Google Colab T4 (16 GB VRAM) |
| **Backend** | FastAPI |
| **Frontend** | Streamlit Chat UI |
| **Intended Use** | Medical Q&A, clinical education, exam revision |
| **Not For** | Clinical diagnosis, treatment decisions, drug dosing |

---

## 🏗 Architecture

```
User (Browser)
     │
     ▼
┌─────────────────────┐
│   Streamlit UI      │  ← frontend.py
│   (Chat Interface)  │
└────────┬────────────┘
         │ HTTP POST /generate
         ▼
┌─────────────────────┐
│   FastAPI Backend   │  ← backend.py
│   /generate         │
│   /health           │
└────────┬────────────┘
         │ model.generate(prompt)
         ▼
┌──────────────────────────────────────┐
│  Gemma 2 2B Instruct + LoRA Adapters │
│  (keras-hub, float16, rank-16)       │
│  Weights: model/mediassist_lora_     │
│           weights.h5  (4.8 GB)       │
└──────────────────────────────────────┘
```

---

## 🔬 Fine-Tuning Methodology

### Why LoRA?

Training all 2 billion parameters of Gemma 2 2B is prohibitively expensive on free-tier GPU resources. **Low-Rank Adaptation (LoRA)** injects trainable rank-decomposition matrices (ΔW = A·B) into every attention projection layer (`q_proj`, `k_proj`, `v_proj`, `o_proj`) while keeping the original weights **frozen**. This reduces the number of trainable parameters to **~0.5 %** of the total model — enabling full fine-tuning quality at a fraction of the memory cost.

### LoRA Configuration

| Parameter | Value |
|-----------|-------|
| Rank | 16 |
| Alpha | 32 |
| Target modules | All attention projections |
| Trainable params | ~10 M / 2 B (≈ 0.5 %) |

### Training Setup

| Setting | Value |
|---------|-------|
| Precision | float16 mixed precision |
| Optimizer | Adam (lr = 5e-5, weight_decay = 0.001, clipnorm = 1.0) |
| LR Schedule | Cosine decay + 3 % linear warm-up |
| Batch size | 4 |
| Max sequence length | 256 tokens |
| Epochs | 3 |
| Loss function | CausalLM masked loss (answer tokens only) |
| Data pipeline | `tf.data` (shuffle → batch → tokenise → prefetch) |

### Prompt Template

All samples are formatted using Gemma 2's native turn-delimiter format to preserve compatibility with the base model's special token embeddings:

```
<start_of_turn>user
{question}<end_of_turn>
<start_of_turn>model
{answer}<end_of_turn>
```

### Training Callbacks

| Callback | Purpose |
|----------|---------|
| `ModelCheckpoint` | Saves best checkpoint (lowest val loss) |
| `EarlyStopping` | Stops if val loss stagnates for 2 epochs |
| `ReduceLROnPlateau` | Halves LR on val loss plateau |
| `CSVLogger` | Logs per-epoch metrics for curve plotting |
| `RamGuardCallback` | Custom: runs `gc.collect()`, logs RAM usage |

---

## 📂 Dataset

**Source:** [`medalpaca/medical_meadow_medical_flashcards`](https://huggingface.co/datasets/medalpaca/medical_meadow_medical_flashcards) (HuggingFace)

Each sample contains a **human** field (question) and an **ai** field (reference answer), covering flashcard content from anatomy, pharmacology, pathology, physiology, and clinical medicine.

### Preprocessing Pipeline

| Stage | Action | Effect |
|-------|--------|--------|
| HTML stripping | `html.unescape()` + regex tag removal | Removes `<b>`, `<br>`, `&nbsp;` artefacts |
| Unicode normalisation | `unicodedata.normalize('NFKC', ...)` | Unifies curly quotes, homoglyphs |
| Deduplication | Exact match on `(question, answer)` | −~2 400 duplicate flashcards |
| Length filter | Drop if combined word count > 160 or < 5 | Retains concise, high-signal pairs |
| Whitespace normalisation | Collapse multiple spaces/newlines | Clean, consistent tokeniser input |

### Dataset Statistics

| Stage | Sample Count |
|-------|-------------|
| Raw dataset | ~33 000 |
| After quality filtering | ~16 000 |
| Used for training | 3 000 |
| — Train split (80 %) | 2 400 |
| — Validation split (10 %) | 300 |
| — Test split (10 %, held-out) | 300 |

---

## 📊 Experiment Results

Three experiments were run in an ablation-style progression, changing one variable at a time:

| Experiment | LR | Batch | LoRA Rank | Schedule | Epochs | Training Time | Outcome |
|------------|----|-------|-----------|----------|--------|---------------|---------|
| **Exp 1 — Base** | N/A | N/A | N/A | N/A | N/A | N/A | Generic / vague medical answers |
| **Exp 2** | 1e-4 | 2 | 4 | Constant | 2 | ~45 min | Improved terminology; minor hallucinations |
| **Exp 3 ✅ Final** | 5e-5 | 4 | 16 | Cosine + warmup | 3 | ~70 min | High accuracy; professional medical tone |

**Key findings:**
- A lower learning rate (5e-5 vs 1e-4) is critical for preserving Gemma 2's pre-trained representations during LoRA adaptation.
- LoRA rank 16 outperforms rank 4 by ~3 pp ROUGE-L with no measurable increase in peak VRAM.
- Cosine decay with 3 % linear warm-up stabilises early training when LoRA matrices are randomly initialised.

---

## 📁 Project Structure

```
Mediassist_chatbot/
│
├── backend.py                     # FastAPI app — loads model, exposes /generate & /health
├── frontend.py                    # Streamlit chat UI — sends requests to backend
├── requirements.txt               # Python dependencies
│
├── model/
│   └── mediassist_lora_weights.h5 # LoRA adapter weights (4.8 GB — download separately)
│
├── mediassist-accuracy-1.ipynb    # Full training & evaluation notebook (documented)
│
└── README.md
```

---

## ⚙️ Local Setup & Installation

### Prerequisites

- Python 3.10+
- At least **16 GB RAM** (model loading is memory-intensive)
- `uv` package manager ([install guide](https://github.com/astral-sh/uv))

### 1 — Clone the repository

```bash
git clone <your-repository-url>
cd Mediassist_chatbot
```

### 2 — Create and activate a virtual environment

```bash
uv venv
source .venv/bin/activate        # macOS / Linux
# .venv\Scripts\activate         # Windows
```

### 3 — Install dependencies

```bash
uv pip install -r requirements.txt
```

**`requirements.txt`**

```text
fastapi
uvicorn
streamlit
requests
keras>=3.0
keras-hub
tensorflow
pydantic
```

### 4 — Download the model weights

Due to GitHub's 100 MB file size limit, the **4.8 GB fine-tuned LoRA weights** are hosted externally.

> 📥 **[DOWNLOAD LINK HERE]** — *(Insert your Google Drive / HuggingFace Hub link)*

Once downloaded, place the file in the `model/` directory:

```bash
mkdir -p model
mv ~/Downloads/mediassist_lora_weights.h5 model/
```

Confirm the path matches:

```
model/mediassist_lora_weights.h5
```

---

## 🚀 Running the Application

You need **two separate terminal sessions** running simultaneously.

### Terminal 1 — FastAPI Backend

```bash
source .venv/bin/activate
uvicorn backend:app --reload
```

Wait for the confirmation message before proceeding:

```
✅ Weights injected successfully!
INFO:     Uvicorn running on http://127.0.0.1:8000
```

### Terminal 2 — Streamlit Frontend

```bash
source .venv/bin/activate
streamlit run frontend.py
```

Streamlit will open the chat interface automatically in your browser at `http://localhost:8501`.

---

## 🎥 Deliverables

| Deliverable | Link |
|-------------|------|
| GitHub Repository | [Link to this Repo] |
| Training Notebook | `mediassist-accuracy-1.ipynb` |
| Demo Video (5–10 min) | [Link to Video] |

---

## ⚖️ Disclaimer

**MediAssist is an AI-powered educational tool only.**

It is **not** a substitute for professional medical advice, diagnosis, or treatment. The model may produce inaccurate or incomplete information. Always consult a qualified healthcare professional for any medical concerns. Never disregard professional medical advice or delay seeking it because of something MediAssist said.

Model usage is subject to the [Gemma Terms of Use](https://ai.google.dev/gemma/terms).

---

## 📚 References

1. Han, T., et al. (2023). *MedAlpaca — An Open-Source Collection of Medical Conversational AI Models and Training Data.* [arXiv:2304.08247](https://arxiv.org/abs/2304.08247)
2. Google DeepMind (2024). *Gemma 2: Improving Open Language Models at a Practical Size.* [arXiv:2408.00118](https://arxiv.org/abs/2408.00118)
3. Hu, E. J., et al. (2021). *LoRA: Low-Rank Adaptation of Large Language Models.* [arXiv:2106.09685](https://arxiv.org/abs/2106.09685)
4. Micikevicius, P., et al. (2018). *Mixed Precision Training.* [arXiv:1710.03740](https://arxiv.org/abs/1710.03740)
5. Chollet, F., et al. *KerasHub: Modular NLP with Keras.* [keras.io/keras_hub](https://keras.io/keras_hub/)
