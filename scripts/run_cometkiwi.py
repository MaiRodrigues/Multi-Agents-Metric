"""Run CometKiwi (wmt22-cometkiwi-da) QE on src.en / hyp.pt -> results/cometkiwi.score."""
from pathlib import Path

from comet import download_model, load_from_checkpoint

ROOT = Path(__file__).resolve().parent.parent


def main():
    """Load CometKiwi, score each (src, hyp) pair, write results/cometkiwi.score."""
    src = (ROOT / "src.en").read_text(encoding="utf-8").splitlines()
    hyp = (ROOT / "hyp.pt").read_text(encoding="utf-8").splitlines()
    assert len(src) == len(hyp), f"line mismatch: {len(src)} vs {len(hyp)}"

    ckpt = download_model("Unbabel/wmt22-cometkiwi-da")
    model = load_from_checkpoint(ckpt)

    data = [{"src": s, "mt": h} for s, h in zip(src, hyp)]
    out = model.predict(data, batch_size=16, gpus=1, devices=[0])

    out_dir = ROOT / "results"
    out_dir.mkdir(exist_ok=True)
    (out_dir / "cometkiwi.score").write_text(
        "\n".join(f"{s:.6f}" for s in out.scores) + "\n",
        encoding="utf-8",
    )
    print(f"wrote {len(out.scores)} scores -> results/cometkiwi.score")


if __name__ == "__main__":
    main()
