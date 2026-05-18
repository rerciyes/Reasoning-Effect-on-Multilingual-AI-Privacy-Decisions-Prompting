# Privacy Decisions Prompting

This repository evaluates open-source Hugging Face language models on
multilingual privacy-decision tasks using the Gemini Apps Privacy Notice.

The runner prompts models with policy files in multiple languages, asks them
to make structured privacy decisions for fixed scenarios, and writes one JSON
result file per `(model, language, scenario, run)` combination.

This version is intended for Hugging Face models on the University of Maryland (UMD) cluster for the course project CMSC848T Multilingual AI. Results with explanations and gpt5.5 are obtained by Wellington Esposito Barbosa from George Washington University via API of the models.

## What This Does

For every model, language, scenario, and run, the runner:

1. Loads `policies/<lang>.txt`, where each policy line has a stable
   `[L0001]`-style line ID.
2. Builds a closed-book prompt using the selected privacy policy and scenario.
3. Runs a Hugging Face model on a UMD GPU cluster node.
4. Extracts structured JSON from the model output.
5. Validates the decision schema.
6. Optionally verifies cited policy excerpts against the referenced line IDs.
7. Saves the full result to `results/`.

Result files are named like:

```text
<model>__<lang>__<scenario>__run<NN>.json
```

Rerunning the script is safe: completed result files are skipped.

## Connect To UMD Cluster

Open a terminal and SSH into Nexus:

```bash
ssh <your_username>@nexusclass.umiacs.umd.edu
```

## First-Time Setup

Install Miniconda in your home directory:

```bash
cd ~

curl -L https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -o miniconda.sh

bash miniconda.sh -b -p ~/miniconda3

source ~/miniconda3/bin/activate
```

Install Python dependencies:

```bash
pip install --upgrade pip

pip install torch transformers accelerate tqdm requests sentencepiece protobuf safetensors huggingface_hub
```

## Running On A GPU Node

Before running models, request an interactive GPU node.

Small/default GPU allocation:

```bash
srun --partition=class --account=class --qos=default --cpus-per-task=4 --mem=2gb --gres=gpu:1 --pty bash
```

Larger GPU allocation:

```bash
srun --partition=class --account=class --qos=medium \
  --cpus-per-task=8 --mem=32gb --gres=gpu:rtxa5000:1 --pty bash
```

After the GPU node starts, activate Conda:

```bash
source ~/miniconda3/bin/activate
```

Check GPU availability:

```bash
nvidia-smi
```

## Hugging Face Login

Some models are gated and require Hugging Face authentication.

```bash
hf auth login
```

Paste your Hugging Face token when prompted.

## Run The Experiment

Go to the repository directory:

```bash
cd path/to/privacy-decisions-prompting
```

Run the configured full experiment:

```bash
python runner.py
```

Run a small test first:

```bash
python runner.py --langs en --scenarios S1 --runs 1
```

Depending on your `runner.py`, you may also be able to filter models:

```bash
python runner.py --models qwen3-8b --langs en --scenarios S1 --runs 1
```

## Hugging Face Cache Management

Large models are downloaded into the Hugging Face cache. If you run into
storage issues, inspect the cache size:

```bash
du -sh ~/.cache/huggingface
```

To inspect individual cached models:

```bash
du -sh ~/.cache/huggingface/hub/*
```

Remove unused cached model files only if you are sure you no longer need them.

## Output Schema

Each run produces a JSON file in `results/`, for example:

```json
{
  "ts": "2026-05-09T01:26:50+00:00",
  "model": "qwen3-8b",
  "language": "en",
  "scenario_id": "S1",
  "run": 1,
  "latency_ms": [8421.0],
  "raw": "<raw model text>",
  "json_extract_reason": "direct",
  "schema_problems": [],
  "decision": "ALLOW",
  "confidence": "High",
  "justification": "English explanation...",
  "source_excerpts": [
    {
      "line_id": "L0042",
      "quote": "..."
    }
  ],
  "quote_verification": [
    {
      "line_id": "L0042",
      "ok": true,
      "match": "exact"
    }
  ],
  "all_quotes_verified": true
}
```

## Decisions

The expected decision labels are:

```text
ALLOW
DENY
ALLOW AFTER VERIFICATION
ESCALATE
```

## Scenarios

`scenarios.json` contains the test scenarios.

Each scenario has:

```json
{
  "id": "S1",
  "title": "...",
  "description": "The full English scenario text sent to the model."
}
```

Use `--scenarios` to run only selected scenario IDs.

## Policies

Policy files live in `policies/`.

Each language has one file:

```text
en.txt
es.txt
pt-BR.txt
fr.txt
de.txt
tr.txt
ko.txt
zh-CN.txt
hi.txt
ar.txt
ur.txt
```

The language code passed to `--langs` must match the policy filename without
`.txt`.

## Inspecting Results

Pretty-print a JSON result file:

```bash
python -m json.tool results/<result-file>.json
```

Example:

```bash
python -m json.tool results/qwen3-8b__en__S1__run01.json
```

If analysis notebooks or scripts are included, run them after the experiment
completes.

## Ending A Session

Deactivate Conda:

```bash
conda deactivate
```

Exit the GPU node:

```bash
exit
```

## Troubleshooting

### CUDA Or GPU Not Found

Check whether you are actually on a GPU node:

```bash
nvidia-smi
```

If no GPU is shown, request a GPU node again with `srun`.

### Hugging Face Gated Model Error

Run:

```bash
hf auth login
```

Make sure your Hugging Face account has accepted the model license page.

### Disk Or Cache Problem

Check Hugging Face cache size:

```bash
du -sh ~/.cache/huggingface
du -sh ~/.cache/huggingface/hub/*
```

### Out Of Memory

Try one of the following:

- request a larger GPU node
- reduce the number of models run at once
- use a smaller model
- reduce context length or generation length in the runner/config
- clear unused Hugging Face cache files
