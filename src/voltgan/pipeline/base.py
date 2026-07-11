import os
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from asammdf import MDF

from voltgan.utils.discover import discover

_DATETIME_PATTERN = re.compile(r"(\d{4}-\d{2}-\d{2})[_ ](\d{2}\.\d{2}\.\d{2})")


@dataclass
class SampleContext:
    source_path: Path
    output_path: Path | None = None
    interrupted: str | None = None
    skipped: bool = False
    metadata: dict = field(default_factory=dict)
    mdf: "MDF" = MDF()


class PipelineHandler(ABC):
    @property
    @abstractmethod
    def order(self) -> int: ...
    @abstractmethod
    def handle(self, context: SampleContext) -> SampleContext: ...


def _parse_datetime(path: Path) -> datetime:
    match = _DATETIME_PATTERN.search(path.name)
    if not match:
        raise ValueError(f"Cannot parse date-time from filename: {path.name}")
    date_str, time_str = match.group(1), match.group(2)
    return datetime.strptime(
        f"{date_str} {time_str.replace('.', ':')}", "%Y-%m-%d %H:%M:%S"
    )


def _discover_sorted(root: Path, mcu: str, extensions: tuple[str, ...]) -> list[Path]:
    mcu_path = root / mcu
    if not mcu_path.exists():
        print(f"Source directory missing, skipping MCU: {mcu_path}")
        return []

    paths = [
        path
        for path in mcu_path.rglob("*")
        if path.is_file() and path.suffix.lower() in extensions
    ]
    return sorted(paths, key=_parse_datetime)


class Pipeline:
    def __init__(self, data_path: Path, handlers: list[PipelineHandler]):
        self.data_path = data_path
        self.handlers = sorted(handlers, key=lambda h: h.order)
        self.mf4_root = data_path / "mf4"
        self.hdf_root = data_path / "hdf"

    def run(self, mcus: list[str]):
        for mcu in mcus:
            discharge_cycle_index: int = 0

            for mf4_path in _discover_sorted(self.mf4_root, mcu, (".mf4", ".dat")):
                relative_path = mf4_path.relative_to(self.mf4_root)
                base_path = (self.hdf_root / relative_path).with_suffix("")
                stem = base_path.name
                parent = base_path.parent

                existing = list(parent.glob(f"{stem}.hdf")) + list(
                    parent.glob(f"{stem}_*.hdf")
                )
                if existing:
                    continue

                context = SampleContext(source_path=mf4_path)
                context.metadata["discharge_cycle_index"] = discharge_cycle_index

                print(f"Processing {mf4_path.name}...")

                try:
                    for handler in self.handlers:
                        context = handler.handle(context)

                        if context.interrupted:
                            for target in context.metadata.get("target_files", []):
                                if target.is_file():
                                    os.remove(target)

                            raise ValueError(
                                f"[{handler.__class__.__name__}] {context.interrupted}"
                            )
                finally:
                    context.mdf.close()

                discharge_cycle_index = context.metadata.get(
                    "discharge_cycle_index", discharge_cycle_index
                )
