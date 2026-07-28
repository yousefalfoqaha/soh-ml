from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class Window:
    start: float
    end: float
    protocol: str
    soh: float = 0.0
    amb: float = float("nan")
    discharge_rate: float | None = None


class DatasetIngestor(ABC):
    @abstractmethod
    def ingest(self) -> None:
        pass
