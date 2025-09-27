from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict


def _mini_yaml(text: str) -> Dict[str, Any]:
    """Ultra-light YAML: key: value per line, no nesting."""
    out: Dict[str, Any] = {}
    for ln in text.splitlines():
        s = ln.strip()
        if not s or s.startswith("#") or ":" not in s:
            continue
        k, v = s.split(":", 1)
        k, v = k.strip(), v.strip()
        if v.lower() in {"true", "false"}:
            out[k] = v.lower() == "true"
        else:
            try:
                out[k] = float(v) if "." in v else int(v)
            except ValueError:
                out[k] = v
    return out


def load_config(path: str):
    p = Path(path)
    if not p.exists():
        # sensible defaults
        data = {"scheduler": "cstarpp", "bandwidth_mhz": 20.0, "seed": 123}
        return SimpleNamespace(**data)
    txt = p.read_text()
    try:
        import yaml  # type: ignore

        data = yaml.safe_load(txt) or {}
    except Exception:
        data = _mini_yaml(txt)
    return SimpleNamespace(**data)
