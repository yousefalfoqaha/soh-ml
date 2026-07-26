from voltgan.config import (
    DATASET_PATH,
    REFERENCE_CURRENT_RANGE,
    REFERENCE_TEMPERATURE_RANGE,
    TESTING_MCUS,
    TRAINING_MCUS,
    VALIDATION_MCUS,
)
from voltgan.pipeline.soh_curve import fit_soh_curves


def main() -> None:
    all_mcus = TRAINING_MCUS + VALIDATION_MCUS + TESTING_MCUS
    print(f"Refitting SoH curves for {len(all_mcus)} MCUs...")
    fit_soh_curves(
        hdf_root=DATASET_PATH / "hdf",
        mcus=all_mcus,
        ref_temp_range=REFERENCE_TEMPERATURE_RANGE,
        ref_current_range=REFERENCE_CURRENT_RANGE,
    )
    print("Done.")


if __name__ == "__main__":
    main()
