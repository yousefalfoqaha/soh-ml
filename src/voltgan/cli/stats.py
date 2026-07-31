from __future__ import annotations

from voltgan.config import (
    STATS_PATH,
    TRAINING_MCUS,
    WUPPERTAL_PROVIDER,
)
from voltgan.dataset import InstanceRepository, StatisticsCalculator


def main() -> None:
    print("Calculating training statistics...")

    wuppertal_repo = InstanceRepository(provider=WUPPERTAL_PROVIDER)

    training_instances = wuppertal_repo.load(TRAINING_MCUS)
    train_stats = StatisticsCalculator(save_path=STATS_PATH)
    train_stats.compute(training_instances)
    train_stats.save()

    print("\nCalculation complete.")


if __name__ == "__main__":
    main()
