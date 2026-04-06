# MileLLM: LLM-based Feature Engineering for Tabular Data

## 📌 Overview

This project implements a Large Language Model (LLM)-driven feature engineering framework for tabular data.  
Given a raw dataset $\mathcal{D}$, the system automatically generates meaningful derived features using prompt-based reasoning, and evaluates them on downstream machine learning tasks.

---

## 📂 Project Structure

```
.
├── config.py                 # global configuration (dataset, task type, paths, API key)
├── main.py                   # entry point of the whole pipeline
├── requirements.txt          # dependencies
├── Readme.md                 # project documentation

├── data/                     # datasets and metadata
│   └── *.csv

├── prompt/                   # prompt templates for LLM
│   └── pc1/
│       ├── island1.txt
│       ├── island2.txt
│       └── island3.txt

├── llm/                      # LLM interaction module
│   ├── llm_client.py         # API wrapper for LLM calls
│   ├── prompt_builder.py     # construct prompts from dataset
│   └── feature_generator.py  # generate new features via LLM

├── pipeline/                 # core pipeline logic
│   ├── feature_pipeline.py   # feature generation + execution pipeline
│   └── experiment.py         # experiment orchestration

├── evaluation/               # downstream model evaluation
│   ├── classifier.py         # classification models & metrics
│   └── regressor.py          # regression models & metrics

├── utils/                    # utility functions
│   ├── data_utils.py         # data loading / preprocessing
│   └── feature_utils.py      # feature execution & transformation helpers
```

---

## ⚙️ Configuration

Edit `config.py`:

```python
DATASET = "dataset"
TASK_TYPE = "classification"

DATA_PATH = f"data/{DATASET}.csv"
PROMPT_DIR = f"prompt/{DATASET}"

API_KEY = "your_api_key"
```

---

## 🚀 How to Run

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the pipeline

```bash
python main.py
```

---

## 🔄 Workflow

$$
\mathcal{D} \rightarrow \text{Prompt} \rightarrow \text{LLM} \rightarrow \phi(\mathbf{x}) \rightarrow \text{Model} \rightarrow \text{Evaluation}
$$

1. Load dataset  
2. Build prompts from metadata  
3. Generate candidate features via LLM  
4. Execute feature transformation  
5. Train downstream model  
6. Evaluate performance  
