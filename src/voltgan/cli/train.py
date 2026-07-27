from __future__ import annotations

import sys

from voltgan.training import (
    train_estimator,
    train_generator,
    train_generator_mse,
)

_VALID = {"estimator", "generator", "generator-mse"}


def main() -> None:
    if len(sys.argv) < 3:
        raise SystemExit("usage: voltgan train <estimator|generator|generator-mse>")
    mode = sys.argv[2]
    if mode not in _VALID:
        raise SystemExit(f"unknown mode '{mode}'. valid: {sorted(_VALID)}")

    if mode == "estimator":
        train_estimator()
    elif mode == "generator":
        train_generator()
    else:
        train_generator_mse()

