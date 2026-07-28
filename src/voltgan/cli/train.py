from __future__ import annotations

import sys

from voltgan.training import (
    train_estimator,
    train_generator,
    train_generator_mse,
)

_VALID = {"estimator", "generator", "generator-mse"}


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit("usage: voltgan train <estimator|generator|generator-mse>")

    mode = sys.argv[1]

    if mode not in _VALID:
        raise SystemExit(f"unknown mode '{mode}'. valid: {sorted(_VALID)}")

    match mode:
        case "estimator":
            train_estimator()
        case "generator":
            train_generator()
        case "generator-mse":
            train_generator_mse()
