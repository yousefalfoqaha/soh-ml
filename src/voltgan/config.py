from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_PATH = Path("/mnt/ssd/datasets/wuppertal")
HDF_ROOT = DATA_PATH / "hdf"
STATS_PATH = DATA_PATH / "stats.json"
CHECKPOINT_PATH = PROJECT_ROOT / "model.pt"
PLOTS_PATH = PROJECT_ROOT / "plots"

RANDOM_SEED = 42

MAX_SEQUENCE_LENGTH = 15000
CHUNK_SIZE = 2000

NOMINAL_CAPACITY = 18000.0
CHANNELS = ["U", "I", "Temp[1]"]
RASTER_FREQUENCY = 1

TRAINING_MCUS = ["mcu1", "mcu2", "mcu3", "mcu4", "mcu5", "mcu6"]
VALIDATION_MCUS = ["mcu7"]
TESTING_MCUS = ["mcu8"]

# sequence generator
INPUT_FEATURES = 1
N_CONDITIONS = 2
HIDDEN_SIZE = 64
N_LAYERS = 2

# hyperparams
N_EPOCHS = 100
BATCH_SIZE = 128
LEARNING_RATE = 5e-4
DROPOUT = 0.1
