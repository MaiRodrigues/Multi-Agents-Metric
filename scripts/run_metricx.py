"""Run MetricX-25 QE on src.en / hyp.pt -> results/metricx.score.

MetricX outputs a 0-25 error score (LOWER IS BETTER). We write the raw score;
invert (25 - x) at analysis time if you need higher-is-better.
"""
from pathlib import Path

import torch
from transformers import AutoTokenizer
from metricx24 import models  # pip install git+https://github.com/google-research/metricx

ROOT = Path(__file__).resolve().parent.parent
MODEL = "google/metricx-24-hybrid-xl-v2p6"  # swap to metricx-25-qe-* when on HF
DEVICE = "cuda:0"


def main():
    """Load MetricX, score each (src, hyp) pair, write results/metricx.score."""
    src = (ROOT / "src.en").read_text(encoding="utf-8").splitlines()
    hyp = (ROOT / "hyp.pt").read_text(encoding="utf-8").splitlines()
    assert len(src) == len(hyp), f"line mismatch: {len(src)} vs {len(hyp)}"

    tok = AutoTokenizer.from_pretrained("google/mt5-xl", legacy=False)
    model = (
        models.MT5ForRegression.from_pretrained(MODEL, torch_dtype=torch.bfloat16)
        .to(DEVICE)
        .eval()
    )

    scores = []
    with torch.inference_mode():
        for source, hypothesis in zip(src, hyp):
            text = f"source: {source} candidate: {hypothesis}"
            enc = tok(text, return_tensors="pt", truncation=True, max_length=1536).to(DEVICE)
            enc["labels"] = torch.tensor([[0]], device=DEVICE)
            out = model(**enc)
            scores.append(float(out.predictions[0]))

    out_dir = ROOT / "results"
    out_dir.mkdir(exist_ok=True)
    (out_dir / "metricx.score").write_text(
        "\n".join(f"{s:.6f}" for s in scores) + "\n",
        encoding="utf-8",
    )
    print(f"wrote {len(scores)} scores -> results/metricx.score (lower=better, 0-25)")


if __name__ == "__main__":
    main()
