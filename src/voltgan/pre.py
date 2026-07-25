from voltgan.config import (
    CHANNELS,
    DATA_PATH,
    MIN_SEQUENCE_LENGTH,
    NOMINAL_CAPACITY,
    RASTER_FREQUENCY,
    REFERENCE_CURRENT_RANGE,
    REFERENCE_TEMPERATURE_RANGE,
    STATS_PATH,
    TESTING_MCUS,
    TRAINING_MCUS,
    VALIDATION_MCUS,
)
from voltgan.data import StatisticsCalculator
from voltgan.pipeline import (
    ChannelValidationHandler,
    ExtractDischargePeriodsHandler,
    HdfConvertHandler,
    Pipeline,
    ShortSequenceFilterHandler,
    SohHandler,
    StatsEnrichHandler,
)
from voltgan.pipeline.ambient_temperature import AmbientTemperatureHandler
from voltgan.pipeline.soh_curve import fit_soh_curves
from voltgan.utils.discover import load_instances

_PIPELINE_HANDLERS = [
    ChannelValidationHandler(CHANNELS, "ClimaTemp"),
    ExtractDischargePeriodsHandler(),
    SohHandler(nominal_capacity=NOMINAL_CAPACITY),
    AmbientTemperatureHandler(),
    HdfConvertHandler(DATA_PATH, RASTER_FREQUENCY),
    ShortSequenceFilterHandler(MIN_SEQUENCE_LENGTH),
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

    instances = load_instances(DATA_PATH / "hdf", TRAINING_MCUS)
    statistics = StatisticsCalculator(STATS_PATH)
    statistics.compute(instances)
    statistics.save()

    print("Preprocessing complete.")


if __name__ == "__main__":
    main()
