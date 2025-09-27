import typer
from rich import print

from ..eval.runner import run_once, sample_inputs
from ..utils.config import load_config

app = typer.Typer(add_completion=False)


@app.command()
def main(config: str = "configs/default.yaml"):
    cfg = load_config(config)
    d, c, B = sample_inputs(
        seed=getattr(cfg, "seed", 123), B=float(getattr(cfg, "bandwidth_mhz", 20.0))
    )
    b = run_once(getattr(cfg, "scheduler", "cstarpp"), d, c, float(B), ctx={})
    print(
        {"scheduler": getattr(cfg, "scheduler", "cstarpp"), "sum_bw": float(b.sum()), "n": len(b)}
    )


if __name__ == "__main__":
    app()
