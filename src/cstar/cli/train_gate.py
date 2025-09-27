import typer
from rich import print

app = typer.Typer(add_completion=False)


@app.command()
def main():
    print("[green]Gate training stub[/green] — replace with Pareto-sweep pipeline later.")


if __name__ == "__main__":
    app()
