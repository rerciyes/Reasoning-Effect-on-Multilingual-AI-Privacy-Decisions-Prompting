"""
Load per-language policy .txt files and verify model-cited quotes against them.

Each .txt has lines in the form:
    [L0001] <content>

Conventions used across the runner:
- `line_id` on the output is the token inside the brackets, without the prefix,
  e.g. "L0001" (not "[L0001]" and not "1").
"""
from pathlib import Path
import re
import unicodedata

from config import POLICIES_DIR

LINE_RE = re.compile(r"^\[(L\d+)\]\s?(.*)$")


def policy_path(lang: str) -> Path:
    return POLICIES_DIR / f"{lang}.txt"


def load_policy_raw(lang: str) -> str:
    """Return the full file contents, with line tags intact — this is what we
    send to the model."""
    return policy_path(lang).read_text(encoding="utf-8")


def load_policy_index(lang: str) -> dict[str, str]:
    """Return {line_id -> content-without-tag}, used for quote verification."""
    idx: dict[str, str] = {}
    with policy_path(lang).open(encoding="utf-8") as f:
        for raw in f:
            raw = raw.rstrip("\n")
            m = LINE_RE.match(raw)
            if m:
                idx[m.group(1)] = m.group(2)
    return idx


def _normalize(s: str) -> str:
    """Unicode-NFC + collapse whitespace. Used only for loose fallback matching
    — we also try exact and trimmed matches first."""
    s = unicodedata.normalize("NFC", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def verify_quote(quote: str, line_id: str, index: dict[str, str]) -> dict:
    """Return a dict describing whether the model's quote really appears on the
    line it claims. Three tiers of match:
        exact     - byte-for-byte substring of the line's content
        trimmed   - substring after stripping leading/trailing whitespace
        normalized - substring after NFC + whitespace collapsing on both sides
    """
    if line_id not in index:
        return {"ok": False, "match": "no_such_line_id"}
    content = index[line_id]
    if not quote:
        return {"ok": False, "match": "empty_quote"}
    if quote in content:
        return {"ok": True, "match": "exact"}
    if quote.strip() in content.strip():
        return {"ok": True, "match": "trimmed"}
    if _normalize(quote) in _normalize(content):
        return {"ok": True, "match": "normalized"}
    return {"ok": False, "match": "mismatch"}


def verify_excerpts(excerpts: list[dict], lang: str) -> list[dict]:
    """Verify each excerpt and return a list of {line_id, ok, match, quote}."""
    index = load_policy_index(lang)
    out = []
    for ex in excerpts or []:
        line_id = (ex.get("line_id") or "").strip()
        # Tolerate "[L0001]" or "L0001" or "0001" — normalize to "L####".
        line_id = line_id.strip("[]")
        if line_id and not line_id.startswith("L") and line_id.isdigit():
            line_id = "L" + line_id.zfill(4)
        quote = ex.get("quote") or ""
        result = verify_quote(quote, line_id, index)
        out.append({"line_id": line_id, "quote": quote, **result})
    return out
