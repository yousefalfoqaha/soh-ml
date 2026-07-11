from pathlib import Path
from typing import Generator

from voltgan.data.instance import DischargeInstance


def discover(
    root: Path, mcus: list[str], extensions: tuple[str, ...]
) -> Generator[Path, None, None]:
    for mcu in mcus:
        mcu_path = root / mcu
        if not mcu_path.exists():
            print(f"Source directory missing, skipping MCU: {mcu_path}")
            continue

        for path in sorted(mcu_path.rglob("*")):
            if path.is_file() and path.suffix.lower() in extensions:
                yield path


def load_instances(
    root: Path,
    mcus: list[str],
    extensions: tuple[str, ...] = (".hdf",),
) -> list[DischargeInstance]:
    return [DischargeInstance(p) for p in discover(root, mcus, extensions)]


def filter_by_temperature(
    instances: list[DischargeInstance],
    temp_range: tuple[float, float],
    exclude: bool = False,
) -> list[DischargeInstance]:
    lo, hi = temp_range
    if exclude:
        return [i for i in instances if not (lo <= i.ambient_temperature <= hi)]
    return [i for i in instances if lo <= i.ambient_temperature <= hi]

