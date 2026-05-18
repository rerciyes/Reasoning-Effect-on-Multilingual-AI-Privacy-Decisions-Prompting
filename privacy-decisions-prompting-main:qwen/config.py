from pathlib import Path

# --- Paths ---------------------------------------------------------------
ROOT          = Path(__file__).resolve().parent
REPO_ROOT     = ROOT.parent
POLICIES_DIR  = ROOT / "policies"
SCENARIOS_PATH = ROOT / "scenarios.json"
RESULTS_DIR   = ROOT / "results"  # one JSON file per (model, lang, scenario, run)

# --- Model backend -------------------------------------------------------
# "ollama" uses the local Ollama HTTP API.
# "hf" runs Hugging Face Transformers locally in this Python process.
MODEL_BACKEND = "hf"

# --- Ollama endpoint -----------------------------------------------------
OLLAMA_URL = "http://localhost:11434/api/chat"

# --- Hugging Face local inference ----------------------------------------
# Use a Hub repo id if the cluster can download models, or a local directory
# containing config/tokenizer/weights if network access is blocked.
HF_DEVICE_MAP = "auto"
HF_TORCH_DTYPE = "auto"
HF_LOCAL_FILES_ONLY = False
HF_TRUST_REMOTE_CODE = False

# Qwen3-specific: disable hidden thinking so the model returns only the JSON
# decision text. Leave as None for models/templates that do not use it.
HF_ENABLE_THINKING = False

# --- Models to evaluate --------------------------------------------------
# Each entry: (model_id, num_ctx). For Hugging Face, model_id is usually a
# Hub repo id such as "Qwen/Qwen3-8B", or a local model directory.
# See README.md for a context-window reference table.
MODELS = [
    # ("llama3.1:8b",         65536),
    # ("qwen3.5:0.8b",        32768),
    ("Qwen/Qwen3-8B",       32768),
    # ("CohereLabs/aya-expanse-8b", 32768),
    # ("mistral-nemo:12b",    32768),
    # ("gemma3:4b",           32768),
    #("qwen3:8b",              32768),   # best multilingual coverage (11/11)
    # ("qwen3:4b",            32768),   # smaller, same coverage
]

# --- Languages to evaluate ----------------------------------------------
# Must match filenames in POLICIES_DIR (e.g. "en" -> "en.txt").
LANGUAGES = ["en", "es", "pt-BR", "fr", "de", "tr", "ko", "zh-CN", "hi", "ar", "ur"]

# --- Run parameters ------------------------------------------------------
RUNS_PER_CELL = 1           # repetitions per (model, language, scenario)
TEMPERATURE   = 0
TOP_P         = 1
NUM_PREDICT   = 800         # cap output tokens so the model can't run away
REQUEST_TIMEOUT_S = 600     # 10 min; bump if you need larger contexts on CPU
MAX_JSON_RETRIES = 2        # retry the whole call if JSON extraction fails
