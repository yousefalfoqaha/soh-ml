from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from asammdf import MDF

DATA_PATH = Path(__file__).resolve().parent.parent / "data"

CHANNELS = ["U", "Temp[1]", "I"]
CHANNEL_LABELS = ["Voltage (V)", "Temperature (°C)", "Current (A)"]

METADATA_KEYS = [
    ("sgl_SOHC", "%"),
    ("sgl_q_mess_ch", "Ah"),
    ("sgl_q_mess_dch", "Ah"),
    ("sgl_e_mess_ch", "Wh"),
    ("sgl_e_mess_dch", "Wh"),
    ("sgl_energy_efficiency", "%"),
    ("sgl_dod", ""),
    ("sgl_cycletype", ""),
]

INITIAL_FILES = [
    (
        "Plain",
        "mcu1/initial/sample01/2024-09-08_06.05.50 Pulse_Test_SamsungINR2170050E_Cell 1 Zelltester_1.mf4",
    ),
    (
        "Ch5",
        "mcu1/initial/sample01/Ch5 2024-09-08_00.35.09 Pulse_Test_SamsungINR2170050E_Cell 1 Zelltester_1.mf4",
    ),
    (
        "Ch6",
        "mcu1/initial/sample01/Ch6 2024-09-08_13.07.10 Pulse_Test_SamsungINR2170050E_Cell 1 Zelltester_1.mf4",
    ),
    (
        "ChDch",
        "mcu1/initial/sample01/ChDch1 2024-09-05_13.18.30 Pulse_Test_SamsungINR2170050E_Cell 1 Zelltester_1.mf4",
    ),
    (
        "WLTC",
        "mcu1/initial/sample01/WLTC 2024-09-07 21.45.06 Pulse_Test_SamsungINR2170050E_Cell 1 Zelltester_1.mf4",
    ),
    (
        "HPPC",
        "mcu1/initial/sample02/HPPC 2024-09-15_16.21.32 Pulse_Test_SamsungINR2170050E_Cell 1 Zelltester_1.mf4",
    ),
]

AGING_FILES = [
    (
        "Cycle 1",
        "mcu1/aging/sample01/2025-02-12_13.11.28 Aging_SamsungINR2170050E_Cell 1 Zelltester_1.mf4",
    ),
    (
        "Cycle 4",
        "mcu1/aging/sample01/2025-02-13_02.10.11 Aging_SamsungINR2170050E_Cell 1 Zelltester_1.mf4",
    ),
    (
        "Cycle 7",
        "mcu1/aging/sample01/2025-02-13_15.09.31 Aging_SamsungINR2170050E_Cell 1 Zelltester_1.mf4",
    ),
    (
        "Cycle 11",
        "mcu1/aging/sample01/2025-02-14_08.29.44 Aging_SamsungINR2170050E_Cell 1 Zelltester_1.mf4",
    ),
]


def get_metadata(mdf):
    available = set(mdf.channels_db.keys())
    meta = {}
    for key, unit in METADATA_KEYS:
        if key in available:
            groups = mdf.channels_db[key]
            ch = mdf.get(key, group=groups[0][0], index=groups[0][1])
            vals = np.asarray(ch.samples)
            if len(vals) <= 4:
                meta[key] = f"{', '.join(f'{v:.1f}' for v in vals)} {unit}".strip()
            else:
                meta[key] = f"[{len(vals)} vals] {float(np.nanmean(vals)):.1f} {unit}".strip()
        else:
            meta[key] = "—"
    return meta


def plot_bracket(file_list, title, out_path):
    n = len(file_list)
    fig, axes = plt.subplots(4, n, figsize=(4 * n, 12),
                              gridspec_kw={"height_ratios": [3, 3, 3, 2]})

    for col, (label, rel_path) in enumerate(file_list):
        fpath = DATA_PATH / rel_path
        print(f"  [{col + 1}/{n}] {label}: {fpath.name}")

        mdf = MDF(fpath)
        df = mdf.to_dataframe(channels=CHANNELS, raster="U")
        meta = get_metadata(mdf)
        mdf.close()

        t = df.index - df.index[0]

        for row, (ch, ch_label) in enumerate(zip(CHANNELS, CHANNEL_LABELS)):
            ax = axes[row, col]
            ax.plot(t, df[ch], linewidth=0.4)
            ax.set_title(label if row == 0 else "")
            ax.set_ylabel(ch_label if col == 0 else "")
            ax.set_xlabel("Time (s)")
            ax.grid(True, alpha=0.3)
            ax.tick_params(labelbottom=True)

        ax_meta = axes[3, col]
        ax_meta.axis("off")
        lines = []
        for key, val in meta.items():
            if val != "—":
                lines.append(f"{key}: {val}")
        ax_meta.text(0.05, 0.95, "\n".join(lines), transform=ax_meta.transAxes,
                     fontsize=7, verticalalignment="top", fontfamily="monospace",
                     bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.8))

    fig.suptitle(title, fontsize=14, y=1.01)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved to {out_path}")
    plt.close(fig)


plot_bracket(
    INITIAL_FILES,
    "Initial — mcu1",
    Path(__file__).resolve().parent.parent / "plots" / "initial_sample01.png",
)

plot_bracket(
    AGING_FILES,
    "Aging — mcu1/aging",
    Path(__file__).resolve().parent.parent / "plots" / "aging_sample01.png",
)

