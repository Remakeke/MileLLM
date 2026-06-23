# MileLLM

## Project Structure

```
.
├── config.py                  # dataset, task type, evolution params
├── main.py                    # entry point
├── requirements.txt

├── data/                      # CSV datasets
├── prompt/                    # prompt templates per dataset (island1.txt, island2.txt, ...)

├── llm/
│   ├── llm_client.py          # OpenAI API wrapper
│   ├── prompt_builder.py      # prompt construction (prior + exemplar injection)
│   └── feature_generator.py   # two-stage LLM feature generation

├── pipeline/
│   ├── island.py              # island state management (best program, stagnation, migration)
│   ├── feature_pipeline.py    # execute LLM code, extract used features
│   └── experiment.py          # evolution loop orchestration

├── evaluation/
│   ├── classifier.py          # XGBoost classifier (K-fold CV + final eval)
│   ├── regressor.py           # XGBoost regressor (K-fold CV + final eval)
│   └── __init__.py            # task-type routing

└── utils/
    ├── data_utils.py          # data loading, train/test split, statistics
    └── feature_utils.py       # code safety fixes (division, log, sqrt)
```

## Configuration

Edit `config.py`:

```python
DATASET = "vehicle"              # dataset name
TASK_TYPE = "classification"     # "classification" or "regression"

API_KEY = "your_api_key"
BASE_URL = "https://api.openai.com/v1"

SHOT_SIZE = 64                   # training set size (few-shot setting)
N_FOLDS = 5                      # cross-validation folds
MAX_GENERATIONS = 5              # evolution generations (T)
PROGRAMS_PER_GEN = 5             # programs per island per generation (m)
STAGNATION_THRESHOLD = 2        # generations without improvement before migration (k)
```

## How to Run

```bash
pip install -r requirements.txt
python main.py
```

## Algorithm

```
Input:  Training set D_train, n islands, max generations T, m programs per generation
Output: Best feature program x*

1. Compute statistical priors from D_train (variance, Pearson correlation, mutual information)
2. Initialize n heterogeneous islands, each with a unique functional identity
3. For generation = 1 to T:
     For each island i:
       For j = 1 to m:
         Generate feature program via LLM (with exemplar feedback)
         Evaluate fitness via K-fold CV on D_train
       Update island best program
       If stagnant for k generations: trigger cross-island migration
4. Return global best program
5. Report final performance on unseen test set
```

### Key Mechanisms

- **Functional Identity**: Each island has a unique role (e.g., "geometric analyst", "mass composition analyst"), guiding LLM to explore different feature subspaces
- **Statistical Priors**: Training-set-only statistics injected into prompts to ground LLM reasoning
- **In-context Exemplars**: Best programs from previous generations fed back to LLM for iterative refinement
- **Cross-island Migration**: When an island stagnates, best programs from other islands are injected as new exemplars

## Evaluation Protocol

- **Training set**: 64 samples (few-shot), **Test set**: remaining samples (isolated)
- **Feature search**: K-fold CV on training set only
- **Baseline**: All original features
- **Enhanced**: Used original features (referenced by LLM code) + newly generated features
- **Final report**: One-time evaluation on unseen test set
