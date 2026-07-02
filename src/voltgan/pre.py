from voltgan.config import (
    CHANNELS,
    DATA_PATH,
    NOMINAL_CAPACITY,
    RASTER_FREQUENCY,
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
    pipeline.run(TRAINING_MCUS + VALIDATION_MCUS)

    standardizer = Standardizer(DATA_PATH)
    standardizer.compute(TRAINING_MCUS)
    standardizer.save()

    print("Preprocessing complete.")


if __name__ == "__main__":
    main()
