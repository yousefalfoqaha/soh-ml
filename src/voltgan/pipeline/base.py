from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Generator


@dataclass
class SampleContext:
    source_path: Path
    output_path: Path | None = None
    stage: str = "unknown"
    interrupted: str | None = None
    metadata: dict = field(default_factory=dict)


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

            if hdf_path.exists():
                continue

            context = SampleContext(source_path=mf4_path)

            for handler in self.handlers:
                context = handler.handle(context)

                if context.interrupted:
                    raise ValueError(
                        f"[{handler.__class__.__name__}] {context.interrupted}"
                    )
