from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from asammdf import MDF

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "mf4"
PLOTS_PATH = Path(__file__).resolve().parent.parent / "plots"
CHANNELS = ["U", "I", "Temp[1]"]
PLOT_CONFIG = {
    "U": {"label": "Voltage [V]", "color": "#2ecc71"},
    "I": {"label": "Current [A]", "color": "#3498db"},
    "Temp[1]": {"label": "Temperature [°C]", "color": "#e74c3c"},
}

FOOTPRINT_FILES = [
    "mcu1/after/sample01/Report_Samsung_INR21700-50E_BT1_MCU1_2025-04-02_13.51.41.mf4",
]


def plot_file_footprint(mf4_path: Path, output_png: Path):
    print(f"Streaming data from {mf4_path.name}...")

    mdf = MDF(mf4_path, channels=CHANNELS)

    signals = {}
    for ch in CHANNELS:
        sig = mdf.get(ch)
        signals[ch] = (sig.timestamps, sig.samples)

    mdf.close()

    fig, axes = plt.subplots(len(CHANNELS), 1, figsize=(14, 10), dpi=150, sharex=True)
    fig.patch.set_facecolor("#111111")

    for ax, ch in zip(axes, CHANNELS):
        t, y = signals[ch]
        cfg = PLOT_CONFIG[ch]
        ax.set_facecolor("#1a1a1a")
        ax.plot(t, y, color=cfg["color"], linewidth=0.8, label=cfg["label"])
        ax.set_ylabel(cfg["label"], color=cfg["color"], fontsize=10)
        ax.tick_params(colors="white", labelsize=9)
        ax.grid(True, color="#333333", linestyle="-", linewidth=0.5)
        ax.legend(
            loc="upper right",
            facecolor="#222222",
            edgecolor="#444444",
            labelcolor="white",
            fontsize=9,
        )

    axes[-1].set_xlabel("Elapsed Test Time [Seconds]", color="#aaaaaa", fontsize=10)
    fig.suptitle(
        f"Telemetry Footprint: {mf4_path.name}", color="white", fontsize=12, y=0.98
    )

    fig.tight_layout()
    fig.savefig(output_png, facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close()

    print(f"Footprint saved -> {output_png}")


if __name__ == "__main__":
    PLOTS_PATH.mkdir(parents=True, exist_ok=True)

    for rel_path in FOOTPRINT_FILES:
        mf4_path = DATA_PATH / rel_path
        if not mf4_path.exists():
            print(f"File not found, skipping: {mf4_path}")
            continue

        tag = mf4_path.stem.split("_")[-1]
        output_png = PLOTS_PATH / f"footprint_{tag}.png"
        plot_file_footprint(mf4_path, output_png)

