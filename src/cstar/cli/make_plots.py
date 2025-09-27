from __future__ import annotations

import typer
from rich import print

from ..plotting.plots import spectrogram_demo

app = typer.Typer(add_completion=False)


@app.command()
def main():
    psd, spec = spectrogram_demo("figures/spec_demo")
    print(f"[green]Saved[/green] {psd} and {spec}")
