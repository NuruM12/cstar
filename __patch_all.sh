set -Eeuo pipefail

# 0) dirs
mkdir -p figures results src/cstar/plotting

# 1) metrics: complete set used by reports
cat > src/cstar/objectives/metrics.py <<'PY_EOF'
from __future__ import annotations
import numpy as np

def jain_fairness(r: np.ndarray, eps: float = 1e-12) -> float:
    r = np.asarray(r, float)
    s1 = float(r.sum())
    s2 = float((r ** 2).sum())
    n = r.size
    return (s1 * s1) / (n * s2 + eps) if s2 > 0 else 0.0

def throughput_sum(r: np.ndarray) -> float:
    return float(np.asarray(r, float).sum())

def unmet_demand_U(r: np.ndarray, d: np.ndarray, eps: float = 1e-12) -> float:
    """1 - normalized shortfall (higher is better)."""
    r = np.asarray(r, float)
    d = np.asarray(d, float)
    short = np.clip(d - r, 0.0, None).sum()
    denom = d.sum() + eps
    return 1.0 - (short / denom if denom > 0 else 0.0)

def cvar_shortfall(r: np.ndarray, d: np.ndarray, alpha: float = 0.8, eps: float = 1e-12) -> float:
    """Return 1 - CVaR_alpha of shortfall normalized by mean demand (higher is better)."""
    r = np.asarray(r, float); d = np.asarray(d, float)
    s = np.clip(d - r, 0.0, None)
    if s.size == 0:
        return 1.0
    q = np.quantile(s, alpha)
    tail = s[s >= q]
    cvar = float(tail.mean()) if tail.size else 0.0
    denom = float(d.mean()) + eps
    return 1.0 - cvar / denom

def entropy_alloc(bw: np.ndarray, eps: float = 1e-12) -> float:
    """Normalized allocation entropy."""
    b = np.asarray(bw, float)
    Z = b.sum() + eps
    p = b / Z
    p = p[p > 0]
    if p.size == 0:
        return 0.0
    H = float(-(p * np.log(p)).sum())
    import math
    return H / (math.log(bw.size) + eps)

def stability_L2(b_now: np.ndarray, b_prev: np.ndarray | None, B: float, eps: float = 1e-12) -> float:
    if b_prev is None:
        return 1.0
    diff = np.linalg.norm(np.asarray(b_now, float) - np.asarray(b_prev, float))
    return 1.0 - (diff * diff) / (B * B + eps)

def spectral_efficiency_sum(r: np.ndarray, B: float, eps: float = 1e-12) -> float:
    return float(np.asarray(r, float).sum() / (B + eps))
PY_EOF

# 2) eval runner: generate snapshots, compute metrics, bootstrap CIs, CSV
cat > src/cstar/eval/runner.py <<'PY_EOF'
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd

from ..schedulers.uniform import Uniform
from ..schedulers.proportional import Proportional
from ..schedulers.greedy_sinr import GreedySINR
from ..schedulers.wrr import WRR
from ..schedulers.optimal_fair import OptimalFair
from ..schedulers.cstarpp import CStarPP
from ..objectives.metrics import (
    jain_fairness, throughput_sum, unmet_demand_U,
    cvar_shortfall, entropy_alloc, stability_L2, spectral_efficiency_sum
)

REGISTRY = {
    "uniform": Uniform(),
    "proportional": Proportional(),
    "greedy_sinr": GreedySINR(),
    "wrr": WRR(),
    "optimal_fair": OptimalFair(),
    "cstarpp": CStarPP(),
}

@dataclass
class Snapshot:
    demand: np.ndarray
    capacity: np.ndarray
    B: float

def sample_inputs(n: int = 12, B: float = 20.0, seed: int = 123) -> Tuple[np.ndarray, np.ndarray, float]:
    rng = np.random.default_rng(seed)
    demand = np.clip(rng.lognormal(mean=2.0, sigma=0.6, size=n), 0.5, 40.0)
    sinr_db = np.clip(rng.normal(10.0, 5.0, size=n), -5.0, 30.0)
    capacity = np.log2(1.0 + 10 ** (sinr_db / 10.0))
    return demand, capacity, B

def gen_snapshots(n_snaps: int = 200, n_users: int = 12, B: float = 20.0, seed: int = 123) -> List[Snapshot]:
    rng = np.random.default_rng(seed)
    snaps: List[Snapshot] = []
    for k in range(n_snaps):
        d, c, Bb = sample_inputs(n=n_users, B=B, seed=int(rng.integers(0, 1_000_000)))
        snaps.append(Snapshot(d, c, Bb))
    return snaps

def evaluate_schedulers(n_snaps: int = 200, n_users: int = 12, B: float = 20.0, seed: int = 123,
                        schedulers: List[str] | None = None) -> pd.DataFrame:
    snaps = gen_snapshots(n_snaps, n_users, B, seed)
    rows = []
    for sname in (schedulers or list(REGISTRY.keys())):
        sch = REGISTRY[sname]
        b_prev = None
        for snap in snaps:
            b = sch.allocate(snap.demand, snap.capacity, snap.B, ctx={})
            r = b * snap.capacity
            row = dict(
                scheduler=sname,
                users=len(b),
                B=snap.B,
                throughput=throughput_sum(r),
                fairness=jain_fairness(r),
                unmet=unmet_demand_U(r, snap.demand),
                cvar=cvar_shortfall(r, snap.demand, alpha=0.8),
                entropy=entropy_alloc(b),
                stability=stability_L2(b, b_prev, snap.B),
                se=spectral_efficiency_sum(r, snap.B),
            )
            rows.append(row)
            b_prev = b
    return pd.DataFrame(rows)

def bootstrap_ci(x: np.ndarray, iters: int = 1000, alpha: float = 0.05, rng: np.random.Generator | None = None):
    rng = rng or np.random.default_rng(0)
    means = []
    n = len(x)
    for _ in range(iters):
        idx = rng.integers(0, n, size=n)
        means.append(np.mean(x[idx]))
    lo = float(np.quantile(means, alpha/2))
    hi = float(np.quantile(means, 1-alpha/2))
    return lo, hi
PY_EOF

# 3) plotting: knee + radar + CDF + spectrogram/PSD demo
cat > src/cstar/plotting/plots.py <<'PY_EOF'
from __future__ import annotations
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from scipy import signal

def knee_plot(df: pd.DataFrame):
    fig, ax = plt.subplots()
    ax.set_title("Fairness vs Throughput")
    for name, sub in df.groupby("scheduler"):
        ax.scatter(sub["throughput"], sub["fairness"], s=8, alpha=0.35, label=name)
    ax.set_xlabel("Throughput (Mb/s)")
    ax.set_ylabel("Jain Fairness")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, ncols=2, frameon=False)
    return fig

def radar_mean(df: pd.DataFrame):
    metrics = ["fairness","throughput","unmet","cvar","entropy","stability","se"]
    means = df.groupby("scheduler")[metrics].mean()
    th = np.linspace(0, 2*np.pi, len(metrics)+1)
    fig = plt.figure()
    ax = fig.add_subplot(111, polar=True)
    for name, row in means.iterrows():
        vals = np.r_[row.values, row.values[0]]
        ax.plot(th, vals, label=name)
        ax.fill(th, vals, alpha=0.08)
    ax.set_xticks(th[:-1]); ax.set_xticklabels(metrics, fontsize=8)
    ax.set_title("Mean metrics (radar)", pad=18)
    ax.legend(fontsize=8, bbox_to_anchor=(1.2,1.05))
    return fig

def cdf_rates(df: pd.DataFrame):
    fig, ax = plt.subplots()
    ax.set_title("Distribution of snapshot throughput")
    for name, sub in df.groupby("scheduler"):
        x = np.sort(sub["throughput"].values)
        y = np.linspace(0, 1, len(x), endpoint=True)
        ax.plot(x, y, label=name)
    ax.set_xlabel("Throughput (Mb/s)")
    ax.set_ylabel("CDF")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, ncols=2, frameon=False)
    return fig

def spectrogram_demo(save_prefix: str = "figures/spec_demo"):
    """Synthesize IQ with a tone + noise + bursty interferer and produce PSD & spectrogram."""
    fs = 30_000  # Hz
    T = 0.5      # seconds
    t = np.arange(int(fs*T)) / fs
    # clean signal: QPSK-like tone + wide noise
    sig = 0.3*np.sin(2*np.pi*2000*t) + 0.2*np.sin(2*np.pi*4500*t)
    noise = 0.5*np.random.default_rng(0).normal(size=t.size)
    x = sig + noise
    # add narrowband burst
    burst = (t>0.18)&(t<0.28)
    x[burst] += 1.0*np.sin(2*np.pi*7000*t[burst])
    # Welch PSD
    f, Pxx = signal.welch(x, fs=fs, nperseg=1024)
    fig1, ax1 = plt.subplots()
    ax1.semilogy(f, Pxx)
    ax1.set_xlabel("Hz"); ax1.set_ylabel("PSD")
    ax1.set_title("Welch PSD (demo)")
    fig1.tight_layout()
    fig1.savefig(f"{save_prefix}_psd.png", dpi=150)
    # Spectrogram
    f2, t2, Sxx = signal.spectrogram(x, fs=fs, nperseg=256, noverlap=192)
    fig2, ax2 = plt.subplots()
    im = ax2.pcolormesh(t2, f2, 10*np.log10(Sxx+1e-12), shading='auto')
    ax2.set_xlabel("Time (s)"); ax2.set_ylabel("Hz")
    ax2.set_title("Spectrogram (demo)")
    fig2.colorbar(im, ax=ax2, label="dB")
    fig2.tight_layout()
    fig2.savefig(f"{save_prefix}_spectrogram.png", dpi=150)
    return (f"{save_prefix}_psd.png", f"{save_prefix}_spectrogram.png")
PY_EOF

# 4) CLI: compare → run sims, write CSV, make figures
cat > src/cstar/cli/compare.py <<'PY_EOF'
from __future__ import annotations
import typer, pandas as pd
from rich import print
from ..eval.runner import evaluate_schedulers
from ..plotting.plots import knee_plot, radar_mean, cdf_rates

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
        fig1 = knee_plot(df);         fig1.savefig("figures/fairness_vs_throughput.png", dpi=150)
        fig2 = radar_mean(df);        fig2.savefig("figures/radar_mean_metrics.png", dpi=150)
        fig3 = cdf_rates(df);         fig3.savefig("figures/cdf_throughput.png", dpi=150)
        print("Saved figures: fairness_vs_throughput.png, radar_mean_metrics.png, cdf_throughput.png")
PY_EOF

# 5) CLI: make_plots → spectrogram/PSD demo
cat > src/cstar/cli/make_plots.py <<'PY_EOF'
from __future__ import annotations
import typer
from rich import print
from ..plotting.plots import spectrogram_demo

app = typer.Typer(add_completion=False)

@app.command()
def main():
    psd, spec = spectrogram_demo("figures/spec_demo")
    print(f"[green]Saved[/green] {psd} and {spec}")
PY_EOF

# 6) default config (if missing)
test -f configs/default.yaml || cat > configs/default.yaml <<'YAML_EOF'
seed: 123
scheduler: cstarpp
bandwidth_mhz: 20.0
YAML_EOF

# 7) smoke: reinstall & quick run
python -m pip install -e ".[plots]" >/dev/null
cstar-compare --n-snaps 150 --seed 42 --out results/bench.csv
cstar-make-plots
echo "✔ Patch applied."
