from __future__ import annotations

from voltgan.config import (
    MAX_SEQUENCE_LENGTH,
    STATS_PATH,
    TRAINING_MCUS,
    TRAINING_PROVIDER,
)
from voltgan.dataset import InstanceRepository, StatisticsCalculator


def main() -> None:
    print("Calculating training statistics...")

    wuppertal_repo = InstanceRepository(provider=TRAINING_PROVIDER)

    training_instances = wuppertal_repo.load(
        TRAINING_MCUS, max_length=MAX_SEQUENCE_LENGTH
    )
    stats = StatisticsCalculator(save_path=STATS_PATH)
    stats.compute(training_instances)
    stats.save()

    print("\nCalculation complete.")


if __name__ == "__main__":
    main()
