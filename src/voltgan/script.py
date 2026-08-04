import sys

import matplotlib.pyplot as plt

from voltgan.config import (
    TESTING_MCUS,
    TRAINING_MCUS,
    TRAINING_PROVIDER,
    VALIDATION_MCUS,
)
from voltgan.dataset.repository import InstanceRepository

NAME = "Cyc054_Aging_Constant_1.0C_Temp25_20250221.hdf"

repo = InstanceRepository(provider=TRAINING_PROVIDER)
instances = repo.load(TRAINING_MCUS + VALIDATION_MCUS + TESTING_MCUS)

instance = next((i for i in instances if i.filepath.name == NAME), None)
if instance is None:
    sys.exit("Nah")

fig, ax = plt.subplots()
ax.plot(instance.voltage[:3350], linewidth=3)
ax.set_xlabel("Time (s)")
ax.set_ylabel("Voltage (V)")
plt.savefig("yes.svg")
