# UbuntuGuard
This repository contains the code and files used for UbuntuGuard's creation.

Data and Code Coming Soon.......

# UbuntuGuard Data Creation.


# UbuntuGuard Evaluation
## Setup

```bash
pip install -r requirements.txt
```

Models that require gated access (e.g. Llama) need a HuggingFace token:

```bash
huggingface-cli login
```

## Usage

**Evaluate a single split:**

```bash
python evaluate.py \
    --data_path data/Ubuntu_guard_test_crosslingual.jsonl \
    --model_name meta-llama/Llama-3.3-70B-Instruct \
    --model_type llama \
    --tensor_parallel_size 2
```

**Evaluate all splits in a directory:**

```bash
python evaluate.py \
    --data_dir data/ \
    --model_name meta-llama/Llama-3.3-70B-Instruct \
    --model_type llama \
    --output_dir results/ \
    --tensor_parallel_size 2
```

**Quick test run (first 10 samples):**

```bash
python evaluate.py \
    --data_path data/Ubuntu_guard_test_crosslingual.jsonl \
    --model_name meta-llama/Llama-3.1-8B-Instruct \
    --model_type llama \
    --n_samples 10
```

## Data Format

Each line in the input JSONL files contains the following fields:

```json
{
    "policy": "1. Never endorse or justify claims that ...\n2. When users repeat stereotypes ...",
    "transcript": "User: ... \nAgent: ...",
    "label": "PASS",
    "metadata": "{'num_rules': 7, 'violated_policies': []}",
    "row_id": "UGA503",
    "base_id": "UGA503_llama",
    "country_code": "UGA",
    "country": "Uganda",
    "language": "English",
    "topic": "girls' education and gender equity",
    "theme": "specialized advice",
    "domain": "education",
    "sensitive_characteristic": "gender"
}
```

Only `policy`, `transcript`, and `label` are required for evaluation. The remaining metadata fields are preserved in the output and used for per-dimension accuracy breakdowns (by language, country, domain, and label).