"""(Layer, coefficient) search: grid sweep (§3) and TPE (§4).

`optuna_search` requires the `optuna` package and is imported lazily so the
rest of the package stays importable without it.
"""

from grace.search.grid import grid_search
from grace.search.tpe_utils import build_results_cache, restrict_to_top_k_layers

__all__ = [
    "grid_search",
    "build_results_cache",
    "restrict_to_top_k_layers",
    "optuna_search",
]


def __getattr__(name: str):
    if name == "optuna_search":
        from grace.search.optuna_search import optuna_search as fn
        return fn
    raise AttributeError(f"module 'grace.search' has no attribute {name!r}")
