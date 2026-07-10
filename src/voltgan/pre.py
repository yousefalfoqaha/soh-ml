from voltgan.config import (
    CHANNELS,
    DATA_PATH,
    NOMINAL_CAPACITY,
    RASTER_FREQUENCY,
    REFERENCE_CURRENT_RANGE,
    REFERENCE_TEMPERATURE_RANGE,
    TESTING_MCUS,
    TRAINING_MCUS,
    VALIDATION_MCUS,
)
from voltgan.data import Standardizer
from voltgan.pipeline import (
    ChannelValidationHandler,
    ExtractDischargePeriodsHandler,
    HdfConvertHandler,
    Pipeline,
    SohHandler,
    StatsEnrichHandler,
)
from voltgan.pipeline.ambient_temperature import AmbientTemperatureHandler
from voltgan.pipeline.soh_curve import fit_soh_curves

_PIPELINE_HANDLERS = [
    ChannelValidationHandler(CHANNELS, "ClimaTemp"),
    ExtractDischargePeriodsHandler(),
    SohHandler(nominal_capacity=NOMINAL_CAPACITY),
    AmbientTemperatureHandler(),
    HdfConvertHandler(DATA_PATH, RASTER_FREQUENCY),
    StatsEnrichHandler(),
]


def main():
    print("Starting data preprocessing pipeline...")

    pipeline = Pipeline(DATA_PATH, _PIPELINE_HANDLERS)
    pipeline.run(TRAINING_MCUS + VALIDATION_MCUS + TESTING_MCUS)

    print("Fitting SoH curves...")
    fit_soh_curves(
        hdf_root=DATA_PATH / "hdf",
        mcus=TRAINING_MCUS + VALIDATION_MCUS + TESTING_MCUS,
        ref_temp_range=REFERENCE_TEMPERATURE_RANGE,
        ref_current_range=REFERENCE_CURRENT_RANGE,
    )

    standardizer = Standardizer(DATA_PATH)
    standardizer.compute(TRAINING_MCUS)
    standardizer.save()

    print("Preprocessing complete.")


if __name__ == "__main__":
    main()