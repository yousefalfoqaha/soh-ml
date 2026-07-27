from __future__ import annotations

from voltgan.config import (
    ALL_MCUS,
    DATASET_PATH,
    HDF_ROOT,
    MIN_SEQUENCE_LENGTH,
    NOMINAL_CAPACITY,
    OXFORD_MAT_PATH,
    RASTER_FREQUENCY,
    REFERENCE_CURRENT_RANGE,
    REFERENCE_TEMPERATURE_RANGE,
    STATS_PATH,
    TRAINING_MCUS,
)
from voltgan.dataset import InstanceRepository, SohCurveFitter, StatisticsCalculator
from voltgan.ingestor import OxfordIngestor, WuppertalIngestor


def main() -> None:
    print("Starting data ingestion pipeline...")

    repo = InstanceRepository(root=HDF_ROOT)
    fitter = SohCurveFitter(
        ref_temp_range=REFERENCE_TEMPERATURE_RANGE,
        ref_current_range=REFERENCE_CURRENT_RANGE,
    )

    print("\n--- Ingesting Wuppertal Data ---")
    WuppertalIngestor(
        raw_dir=DATASET_PATH / "mf4",
        mcus=ALL_MCUS,
        nominal_capacity=NOMINAL_CAPACITY,
        raster=RASTER_FREQUENCY,
        min_seq_len=MIN_SEQUENCE_LENGTH,
        repo=repo,
        fitter=fitter,
    ).ingest()

    print("\n--- Ingesting Oxford Data ---")
    OxfordIngestor(
        mat_path=OXFORD_MAT_PATH,
        min_seq_len=MIN_SEQUENCE_LENGTH,
        repo=repo,
    ).ingest()

    print("\n--- Calculating Statistics ---")
    training_instances = repo.load(TRAINING_MCUS)
    stats = StatisticsCalculator(save_path=STATS_PATH)
    stats.compute(training_instances)
    stats.save()

    print("\nIngestion complete.")

