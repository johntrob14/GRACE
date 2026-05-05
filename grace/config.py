"""Configuration: env loading and the model registry used by the paper."""
from __future__ import annotations

import os
from pathlib import Path


def load_env_file(env_path: str = ".env") -> None:
    """Populate os.environ from a dotenv file. Silent if the file is absent."""
    env_file = Path(env_path)
    if not env_file.exists():
        return
    with env_file.open() as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


load_env_file()


# The three steering models the paper evaluates on, plus the default judge.
MODEL_CONFIGS: dict[str, dict] = {
    "google/gemma-2-2b-it":            {"hidden_size": 2304, "num_layers": 26, "max_seq_length": 8192},
    "google/gemma-3-27b-it":           {"hidden_size": 4608, "num_layers": 62, "max_seq_length": 8192},
    "meta-llama/Llama-3.3-70B-Instruct": {"hidden_size": 8192, "num_layers": 80, "max_seq_length": 8192},
    # Default LLM judge.
    "google/gemma-3-12b-it":           {"hidden_size": 3840, "num_layers": 48, "max_seq_length": 8192},
}


def require_env(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(
            f"Environment variable {name!r} is required but not set. "
            f"Add it to .env (see .env.example)."
        )
    return value
