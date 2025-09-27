C-STAR++ — Context-Aware Channel Allocation (Signal Scheduling)






C-STAR++ is a lightweight, context-aware, multi-objective channel allocator for a single-cell downlink.
It optimizes a composite of throughput, fairness, unmet-demand shortfall, tail-risk (CVaR), allocation entropy,
and temporal stability — with ML-gated weights that adapt to context (e.g., demand/SE dispersion, mobility, anomalies/interference).
The repo includes simple baselines (Uniform, Proportional/WRR-style) and a projected optimization core (capped simplex).

✨ Highlights

Composite objective: throughput (T), fairness (F), unmet-demand (U), tail-risk CVaRα (R), entropy (H), stability (S)

Context gate: κ → weights w on the simplex (guardrails + temperature calibration)

Projected optimizer: Adam-like ascent + one-sided finite differences + capped-simplex projection

Baselines: Uniform, Proportional (demand-weighted / WRR mix)

Reproducibility: fixed seeds, deterministic sweeps, CSV artifacts

🗂️ Repo layout
<details> <summary><strong>Show structure</strong></summary>
cstar/
├─ src/cstar/
│  ├─ schedulers/
│  │  ├─ uniform.py
│  │  ├─ proportional.py
│  │  └─ cstarpp.py
│  ├─ optimization/
│  │  └─ projection.py
│  ├─ eval/
│  │  └─ runner.py
│  └─ ...
├─ .github/workflows/ci.yml
├─ pyproject.toml (or setup.cfg)
└─ README.md

</details>

If some files are not yet committed, the runner still works with the included baselines and projection utilities.

🚀 Quick start
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -e ".[dev]"                   # install package + dev extras
python -m pip install -U pre-commit && pre-commit install

Run the demo experiment
python -m cstar.eval.runner


The runner:

generates (or loads) a small synthetic dataset,

executes selected schedulers (Uniform / Proportional / C-STAR++ if present),

writes metrics to results/ (metrics/, CSVs, summaries).

🧠 Using the schedulers directly
import numpy as np
from cstar.schedulers.cstarpp import CStarPP
from cstar.optimization.projection import project_capped_simplex

B = 100.0
demand = np.array([30., 20., 10., 40.])
capacity = np.array([3., 2., 1., 4.])
ctx = {"lambda": 0.6}

sched = CStarPP()
b = sched.allocate(demand, capacity, B, ctx)
print(b, b.sum())


Baselines:

from cstar.schedulers.uniform import Uniform
from cstar.schedulers.proportional import Proportional

Uniform().allocate(demand, capacity, B, {})
Proportional().allocate(demand, capacity, B, {"lambda": 0.6})

📦 Results & artifacts

results/metrics/ – per-run CSVs (means, CIs where applicable)

results/ – summaries (summary.csv), any dashboards or plots you generate

(If you integrate the ML gate + dashboards: place outputs under cstar_results_plus/)

⚙️ Common knobs

Budget B (total bandwidth)

Caps per user d_i / c_i (enforced by projection)

Gate inputs κ (dispersion of demand/SE, mobility m, anomaly a)

Risk level α (CVaR tail; can be dynamic α(κ))

Stability weight (κ-modulated)

✅ Testing
pytest -q

🔁 CI

A minimal GitHub Actions workflow lives in .github/workflows/ci.yml and runs lint/tests on push & PR.
If Actions don’t trigger, ensure Actions are enabled in Settings → Actions and your credential had workflow permission when pushing workflow files.

📄 Citing

If you use C-STAR++ in academic work, please cite the associated manuscript once available.
(Add BibTeX here when the DOI is minted.)

🤝 Contributing

Fork → feature branch → commit with tests

Run pre-commit run --all-files locally

Open a PR against main

🛡️ License

MIT © 2025 NuruM12
EOF

git add README.md
git commit -m "docs: add polished README (collapsible layout, quickstart, usage, CI notes)"
git push
