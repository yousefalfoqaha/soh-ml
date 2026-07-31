from __future__ import annotations

from voltgan.config import (
    DATASET_PATH,
    MIN_SEQUENCE_LENGTH,
    OXFORD_MAT_PATH,
    OXFORD_PROVIDER,
    RASTER_FREQUENCY,
    REFERENCE_DISCHARGE_RATE,
    REFERENCE_TEMPERATURE,
    TESTING_MCUS,
    TRAINING_MCUS,
    VALIDATION_MCUS,
    WUPPERTAL_PROVIDER,
)
from voltgan.dataset import InstanceRepository, SohCurveFitter
from voltgan.ingestor import OxfordIngestor, WuppertalIngestor


def main() -> None:
    print("Starting data ingestion pipeline...")

    wuppertal_repo = InstanceRepository(provider=WUPPERTAL_PROVIDER)
    oxford_repo = InstanceRepository(provider=OXFORD_PROVIDER)

    fitter = SohCurveFitter(
        reference_temperature=REFERENCE_TEMPERATURE,
        reference_discharge_rate=REFERENCE_DISCHARGE_RATE,
    )

    print("\n--- Ingesting Wuppertal Data ---")
    WuppertalIngestor(
        mf4_dir=DATASET_PATH / "mf4",
        raster=RASTER_FREQUENCY,
        min_seq_len=MIN_SEQUENCE_LENGTH,
        repo=wuppertal_repo,
    ).ingest()

    print("\n--- Applying SOH Degradation Curves ---")
    fitter.apply(
        repo=wuppertal_repo, mcus=TRAINING_MCUS + VALIDATION_MCUS + TESTING_MCUS
    )

    print("\n--- Ingesting Oxford Data ---")
    OxfordIngestor(
        mat_path=OXFORD_MAT_PATH,
        min_seq_len=MIN_SEQUENCE_LENGTH,
        repo=oxford_repo,
    ).ingest()

    print("\nIngestion complete.")


if __name__ == "__main__":
    main()
