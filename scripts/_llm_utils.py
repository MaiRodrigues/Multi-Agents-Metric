"""Shared helpers for LLM-based QE metric scripts (GEMBA, TASER)."""
from pathlib import Path
import time
import openai

ROOT = Path(__file__).resolve().parent.parent


def load_pairs():
    """Load aligned (src, hyp) lines from src.en and hyp.pt at repo root."""
    src = (ROOT / "src.en").read_text(encoding="utf-8").splitlines()
    hyp = (ROOT / "hyp.pt").read_text(encoding="utf-8").splitlines()
    assert len(src) == len(hyp), f"line mismatch: {len(src)} vs {len(hyp)}"
    return src, hyp


def write_scores(name, scores):
    """Write one score per line to results/<name>.score."""
    out_dir = ROOT / "results"
    out_dir.mkdir(exist_ok=True)
    path = out_dir / f"{name}.score"
    path.write_text("\n".join(f"{s:.6f}" for s in scores) + "\n", encoding="utf-8")
    print(f"wrote {len(scores)} scores -> {path.relative_to(ROOT)}")


def score_all(client, pairs, score_pair, progress=25):
    """Apply score_pair(client, src, hyp) across pairs with periodic progress."""
    scores = []
    total = len(pairs)
    for i, (source, hypothesis) in enumerate(pairs):
        scores.append(score_pair(client, source, hypothesis))
        if (i + 1) % progress == 0:
            print(f"  {i + 1}/{total}")
    return scores


def call_with_retry(client, retries=2, **kwargs):
    """Call chat.completions.create with exponential backoff on OpenAI errors."""
    for attempt in range(retries + 1):
        try:
            return client.chat.completions.create(**kwargs)
        except openai.OpenAIError as exc:
            print(f"[{attempt}] openai error: {exc}")
            if attempt == retries:
                return None
            time.sleep(2 ** attempt)
    return None
