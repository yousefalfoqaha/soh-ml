import os
import shutil
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Generator

from asammdf import MDF


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


class Pipeline:
    def __init__(self, data_path: Path, handlers: list[PipelineHandler]):
        self.data_path = data_path
        self.handlers = sorted(handlers, key=lambda h: h.order)
        self.mf4_root = data_path / "mf4"
        self.hdf_root = data_path / "hdf"

    def run(self, mcus: list[str]):
        for mf4_path in discover(self.mf4_root, mcus, (".mf4", ".dat")):
            relative_path = mf4_path.relative_to(self.mf4_root)
            hdf_path = (self.hdf_root / relative_path).with_suffix(".hdf")
            dir_path = (self.hdf_root / relative_path).with_suffix("")

            if dir_path.exists() or hdf_path.exists():
                continue

            context = SampleContext(source_path=mf4_path)

            print(f"Processing {mf4_path.name}...")

            try:
                for handler in self.handlers:
                    context = handler.handle(context)

                    if context.interrupted:
                        if context.output_path and context.output_path.is_file():
                            os.remove(context.output_path)
                        elif context.output_path and context.output_path.is_dir():
                            shutil.rmtree(context.output_path)

                        raise ValueError(
                            f"[{handler.__class__.__name__}] {context.interrupted}"
                        )
            finally:
                context.mdf.close()
