from __future__ import annotations

import typer
from rich import print

from ..eval.runner import evaluate_schedulers
from ..plotting.plots import cdf_rates, knee_plot, radar_mean

app = typer.Typer(add_completion=False)


@app.command()
def main(
    n_snaps: int = typer.Option(300, help="Number of snapshots"),
    n_users: int = typer.Option(12, help="Users per snapshot"),
    seed: int = typer.Option(123, help="RNG seed"),
    out: str = typer.Option("results/bench.csv", help="CSV output"),
    make_figs: bool = typer.Option(True, help="Save default figures"),
):
    df = evaluate_schedulers(n_snaps=n_snaps, n_users=n_users, seed=seed)
    df.to_csv(out, index=False)
    print(f"[green]Saved[/green] {out}  ({len(df)} rows)")
    if make_figs:
        fig1 = knee_plot(df)
        fig1.savefig("figures/fairness_vs_throughput.png", dpi=150)
        fig2 = radar_mean(df)
        fig2.savefig("figures/radar_mean_metrics.png", dpi=150)
        fig3 = cdf_rates(df)
        fig3.savefig("figures/cdf_throughput.png", dpi=150)
        print(
            "Saved figures: fairness_vs_throughput.png, radar_mean_metrics.png, cdf_throughput.png"
        )
