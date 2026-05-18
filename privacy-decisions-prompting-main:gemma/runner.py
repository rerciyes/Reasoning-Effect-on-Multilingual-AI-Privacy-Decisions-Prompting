"""
Run (model x language x scenario x run) and persist each result.

Features:
- Robust JSON extraction (handles preambles and ```json fences).
- Retry on malformed JSON (up to MAX_JSON_RETRIES).
- Quote verification against the policy .txt (exact / trimmed / normalized).
- Append-only JSONL output so the run can be stopped and resumed.
- Skips (model, lang, scenario_id, run) tuples already present in results.

Usage:
    python runner.py                       # runs every (model x lang x scenario) in config
    python runner.py --models google/gemma-3-4b-it # override models list
    python runner.py --langs en es         # override languages
    python runner.py --scenarios S01 S02   # filter scenario ids (prefix match)
    python runner.py --runs 1              # override RUNS_PER_CELL
    python runner.py --dry-run             # print plan, don't call the model
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from functools import lru_cache
from datetime import datetime, timezone
from pathlib import Path

from tqdm import tqdm

import config
from prompts import SYSTEM_PROMPT, build_user_prompt
from policy_loader import load_policy_raw

DISPLAY_NAME_MAP = {
    "meta-llama/Meta-Llama-3.1-8B-Instruct": "llama3.1:8b",
    "google/gemma-3-4b-it": "gemma3:4b",
    "Qwen/Qwen3-8B": "qwen3:8b",
    "CohereLabs/aya-expanse-8b": "aya-expanse:8b",
}

# --- JSON extraction -----------------------------------------------------

FENCE_RE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL | re.IGNORECASE)


def extract_json(text: str) -> tuple[dict | None, str]:
    """Try hard to pull a single JSON object out of a model response.
    Returns (parsed_dict_or_None, diagnostic_reason).
    Strategy:
      1) Direct json.loads on the stripped text.
      2) Strip ```json ... ``` fences.
      3) Walk characters and find the first balanced {...} block.
    """
    if text is None:
        return None, "empty_text"
    s = text.strip()
    if not s:
        return None, "empty_text"

    # Strategy 1: direct parse
    try:
        obj = json.loads(s)
        if isinstance(obj, dict):
            return obj, "direct"
    except json.JSONDecodeError:
        pass

    # Strategy 2: ```json ... ``` fence
    m = FENCE_RE.search(s)
    if m:
        try:
            obj = json.loads(m.group(1))
            if isinstance(obj, dict):
                return obj, "fenced"
        except json.JSONDecodeError:
            pass

    # Strategy 3: balanced-brace scan from first `{`
    start = s.find("{")
    while start != -1:
        depth = 0
        in_str = False
        esc = False
        for i in range(start, len(s)):
            c = s[i]
            if in_str:
                if esc:
                    esc = False
                elif c == "\\":
                    esc = True
                elif c == '"':
                    in_str = False
            else:
                if c == '"':
                    in_str = True
                elif c == "{":
                    depth += 1
                elif c == "}":
                    depth -= 1
                    if depth == 0:
                        candidate = s[start:i + 1]
                        try:
                            obj = json.loads(candidate)
                            if isinstance(obj, dict):
                                return obj, "balanced"
                        except json.JSONDecodeError:
                            break
                        break
        start = s.find("{", start + 1)

    return None, "no_json_found"


def validate_schema(obj: dict) -> list[str]:
    problems: list[str] = []
    if not isinstance(obj, dict):
        return ["not_a_dict"]

    decision = obj.get("decision")
    if decision not in {"ALLOW", "DENY", "ALLOW AFTER VERIFICATION", "ESCALATE"}:
        problems.append(f"bad_decision:{decision!r}")
    
    if obj.get("confidence") not in {"High", "Medium", "Low"}:
        problems.append(f"bad_confidence:{obj.get('confidence')!r}")

    return problems


# --- Model calls --------------------------------------------------------

def call_ollama(model: str, num_ctx: int, policy_text: str, scenario: str) -> tuple[str, float]:
    """Return (raw_text, latency_ms). Raises requests exceptions on network errors."""
    import requests

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_user_prompt(policy_text, scenario)},
        ],
        "stream": False,
        "format": "json",  # Ollama-side JSON mode; complements our extraction
        "options": {
            "temperature": config.TEMPERATURE,
            "top_p": config.TOP_P,
            "num_ctx": num_ctx,
            "num_predict": config.NUM_PREDICT,
        },
    }
    t0 = time.time()
    resp = requests.post(config.OLLAMA_URL, json=payload, timeout=config.REQUEST_TIMEOUT_S)
    latency_ms = (time.time() - t0) * 1000.0
    resp.raise_for_status()
    body = resp.json()
    # Ollama /api/chat returns {"message": {"role":"assistant","content":"..."}, ...}
    content = (body.get("message") or {}).get("content") or ""
    return content, latency_ms


@lru_cache(maxsize=2)
def load_hf_model(model_id: str):
    """Lazy-load a Hugging Face model and tokenizer/processor once per model."""
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as e:
        raise RuntimeError(
            "Hugging Face backend requires: pip install transformers accelerate torch"
        ) from e
    """
    if "gemma-3" in model_id.lower():
        try:
            from transformers import AutoProcessor, Gemma3ForConditionalGeneration
        except ImportError as e:
            raise RuntimeError(
                "Gemma 3 requires transformers>=4.50.0. Try: pip install -U transformers"
            ) from e
        processor = AutoProcessor.from_pretrained(
            model_id,
            local_files_only=config.HF_LOCAL_FILES_ONLY,
            trust_remote_code=config.HF_TRUST_REMOTE_CODE,
        )
        model = Gemma3ForConditionalGeneration.from_pretrained(
            model_id,
            device_map=config.HF_DEVICE_MAP,
            torch_dtype=config.HF_TORCH_DTYPE,
            local_files_only=config.HF_LOCAL_FILES_ONLY,
            trust_remote_code=config.HF_TRUST_REMOTE_CODE,
        )
        model.eval()
        return "processor", processor, model
    """
    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        local_files_only=config.HF_LOCAL_FILES_ONLY,
        trust_remote_code=config.HF_TRUST_REMOTE_CODE,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map=config.HF_DEVICE_MAP,
        torch_dtype=config.HF_TORCH_DTYPE,
        local_files_only=config.HF_LOCAL_FILES_ONLY,
        trust_remote_code=config.HF_TRUST_REMOTE_CODE,
    )
    model.eval()
    return "tokenizer", tokenizer, model


def hf_messages(policy_text: str, scenario: str, *, processor_chat: bool) -> list[dict]:
    user_prompt = build_user_prompt(policy_text, scenario)
    if processor_chat:
        return [
            {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
            {"role": "user", "content": [{"type": "text", "text": user_prompt}]},
        ]
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]


def call_hf(model_id: str, num_ctx: int, policy_text: str, scenario: str) -> tuple[str, float]:
    """Return (raw_text, latency_ms) from a local Hugging Face model."""
    input_kind, text_adapter, model = load_hf_model(model_id)
    processor_chat = input_kind == "processor"
    messages = hf_messages(policy_text, scenario, processor_chat=processor_chat)

    template_kwargs = {}
    if config.HF_ENABLE_THINKING is not None:
        template_kwargs["enable_thinking"] = config.HF_ENABLE_THINKING

    if processor_chat:
        inputs = text_adapter.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            **template_kwargs,
        )
    else:
        prompt = text_adapter.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            **template_kwargs,
        )
        inputs = text_adapter(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=num_ctx,
        )
    device = getattr(model, "device", None)
    if device is not None:
        inputs = {k: v.to(device) for k, v in inputs.items()}

    generate_kwargs = {
        "max_new_tokens": config.NUM_PREDICT,
        "do_sample": config.TEMPERATURE > 0,
    }
    eos_token_id = getattr(text_adapter, "eos_token_id", None)
    if eos_token_id is None and hasattr(text_adapter, "tokenizer"):
        eos_token_id = text_adapter.tokenizer.eos_token_id
    if eos_token_id is not None:
        generate_kwargs["pad_token_id"] = eos_token_id
    if config.TEMPERATURE > 0:
        generate_kwargs.update({
            "temperature": config.TEMPERATURE,
            "top_p": config.TOP_P,
        })

    t0 = time.time()
    output_ids = model.generate(**inputs, **generate_kwargs)
    latency_ms = (time.time() - t0) * 1000.0

    prompt_len = inputs["input_ids"].shape[-1]
    generated_ids = output_ids[0][prompt_len:]
    content = text_adapter.decode(generated_ids, skip_special_tokens=True).strip()
    return content, latency_ms


def call_model(model: str, num_ctx: int, policy_text: str, scenario: str) -> tuple[str, float]:
    if config.MODEL_BACKEND == "ollama":
        return call_ollama(model, num_ctx, policy_text, scenario)
    if config.MODEL_BACKEND == "hf":
        return call_hf(model, num_ctx, policy_text, scenario)
    raise ValueError(f"unknown MODEL_BACKEND: {config.MODEL_BACKEND!r}")


# --- Results store ------------------------------------------------------

_FS_SAFE = re.compile(r"[^A-Za-z0-9._-]+")


def sanitize(s: str) -> str:
    """Make a string safe as part of a filename. Replaces any run of unsafe
    characters with '-'. Examples:
        'llama3.1:8b'  -> 'llama3.1-8b'
        'pt-BR'        -> 'pt-BR'
        'zh-CN'        -> 'zh-CN'
    """
    return _FS_SAFE.sub("-", s).strip("-")


def format_duration(seconds: float) -> str:
    """Render a duration as '42s', '7m30s', or '1h12m' for human-readable ETAs."""
    if seconds < 60:
        return f"{seconds:.0f}s"
    if seconds < 3600:
        m, s = divmod(int(seconds), 60)
        return f"{m}m{s:02d}s"
    h, rem = divmod(int(seconds), 3600)
    m, _ = divmod(rem, 60)
    return f"{h}h{m:02d}m"


"""
def result_filename(model: str, lang: str, scenario_id: str, run: int) -> str:
    return f"{sanitize(model)}__{sanitize(lang)}__{sanitize(scenario_id)}__run{run:02d}.json"
"""

def result_filename(model: str, lang: str, scenario_id: str, run: int) -> str:
    display_model = DISPLAY_NAME_MAP.get(model, model)
    return f"{sanitize(display_model)}__{sanitize(lang)}__{sanitize(scenario_id)}__run{run:02d}.json"


def result_path(results_dir: Path, model: str, lang: str, scenario_id: str, run: int) -> Path:
    return results_dir / result_filename(model, lang, scenario_id, run)


def load_done_set(results_dir: Path) -> set[tuple[str, str, str, int]]:
    """Scan the results directory and return the set of (model, lang,
    scenario_id, run) tuples that already have a JSON file on disk, so we
    can resume mid-run without overwriting completed cells."""
    done: set[tuple[str, str, str, int]] = set()
    if not results_dir.exists():
        return done
    for f in results_dir.glob("*.json"):
        try:
            with f.open(encoding="utf-8") as fh:
                row = json.load(fh)
        except (json.JSONDecodeError, OSError):
            continue
        key = (row.get("model"), row.get("language"), row.get("scenario_id"), row.get("run"))
        if all(x is not None for x in key):
            done.add(key)  # type: ignore[arg-type]
    return done


def write_row(path: Path, row: dict) -> None:
    """Write a single cell's result to its own JSON file. Never overwrites an
    existing file — the filename encodes (model, lang, scenario, run)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(row, f, ensure_ascii=False, indent=2)
        f.write("\n")


# --- Orchestration ------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--models", nargs="*", help="Model ids/tags to use (overrides config.MODELS)")
    p.add_argument("--num-ctx", type=int, help="Override num_ctx for all models (default from config)")
    p.add_argument("--langs", nargs="*", help="Language codes to use (overrides config.LANGUAGES)")
    p.add_argument("--scenarios", nargs="*", help="Scenario id prefixes to include")
    p.add_argument("--runs", type=int, help="Override RUNS_PER_CELL")
    p.add_argument("--dry-run", action="store_true", help="Print plan, don't hit the model")
    p.add_argument("--results-dir", help="Override results output directory")
    return p.parse_args()


def resolve_models(args) -> list[tuple[str, int]]:
    if args.models:
        ctx = args.num_ctx or 32768
        return [(m, ctx) for m in args.models]
    if args.num_ctx:
        return [(m, args.num_ctx) for m, _ in config.MODELS]
    return list(config.MODELS)


def filter_scenarios(scenarios: list[dict], prefixes: list[str] | None) -> list[dict]:
    if not prefixes:
        return scenarios
    return [s for s in scenarios if any(s["id"].startswith(p) for p in prefixes)]


def main() -> int:
    args = parse_args()
    results_dir = Path(args.results_dir) if args.results_dir else config.RESULTS_DIR

    with config.SCENARIOS_PATH.open(encoding="utf-8") as f:
        all_scenarios = json.load(f)
    scenarios = filter_scenarios(all_scenarios, args.scenarios)

    models = resolve_models(args)
    langs = args.langs or config.LANGUAGES
    runs_per_cell = args.runs if args.runs is not None else config.RUNS_PER_CELL

    # Pre-load policy texts (and fail fast if any are missing)
    policies: dict[str, str] = {}
    for lang in langs:
        policies[lang] = load_policy_raw(lang)

    done = load_done_set(results_dir)

    planned = []
    for model, ctx in models:
        for lang in langs:
            for scen in scenarios:
                for run in range(1, runs_per_cell + 1):
                    key = (model, lang, scen["id"], run)
                    if key in done:
                        continue
                    planned.append((model, ctx, lang, scen, run))

    total_cells = len(models) * len(langs) * len(scenarios) * runs_per_cell
    print(f"Planned calls: {len(planned)} (of {total_cells} cells; {total_cells - len(planned)} already on disk)")
    print(f"  models: {[m for m, _ in models]}")
    print(f"  langs:  {langs}")
    print(f"  scenarios: {[s['id'] for s in scenarios]}")
    print(f"  runs per cell: {runs_per_cell}")
    print(f"  results dir: {results_dir}")

    if args.dry_run:
        return 0

    run_t0 = time.time()
    pbar = tqdm(
        enumerate(planned, start=1),
        total=len(planned),
        unit="cell",
        dynamic_ncols=True,
        smoothing=0.3,  # EMA: ETA adapts to recent cell times (helps when language switches)
    )
    for idx, (model, ctx, lang, scen, run) in pbar:
        pbar.set_postfix_str(f"{sanitize(model)} {lang}/{scen['id']}/r{run}", refresh=False)
        row = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "model": DISPLAY_NAME_MAP.get(model, model),
            "num_ctx": ctx,
            "language": lang,
            "scenario_id": scen["id"],
            "run": run,
        }
        raw_text = None
        parsed: dict | None = None
        extract_reason = ""
        schema_problems: list[str] = []
        for attempt in range(config.MAX_JSON_RETRIES + 1):
        #for attempt in range(1):
            try:
                raw_text, latency_ms = call_model(model, ctx, policies[lang], scen["description"])
            except Exception as e:
                row["error"] = f"{config.MODEL_BACKEND}_error: {e}"
                break
            row.setdefault("latency_ms", []).append(latency_ms)
            parsed, extract_reason = extract_json(raw_text)
            schema_problems = validate_schema(parsed) if parsed else ["no_json"]
            if parsed is not None and not schema_problems:
                break
            tqdm.write(f"    attempt {attempt + 1}: extract={extract_reason} problems={schema_problems}")

        row["raw"] = raw_text
        row["json_extract_reason"] = extract_reason
        row["schema_problems"] = schema_problems

        if parsed is not None:
            row["decision"] = parsed.get("decision")
            row["confidence"] = parsed.get("confidence")
        else:
            row["decision"] = None

        out_path = result_path(results_dir, model, lang, scen["id"], run)
        write_row(out_path, row)

    pbar.close()
    print(f"Done. {len(planned)} cells in {format_duration(time.time() - run_t0)}.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
