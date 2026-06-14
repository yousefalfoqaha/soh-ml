from abc import ABC, abstractmethod
from dataclasses import dataclass

from asammdf import MDF


@dataclass
class SohResult:
    soh_file: float
    method: str = ""


class SohStrategy(ABC):
    @abstractmethod
    def can_handle(self, mdf: MDF) -> bool: ...

    @abstractmethod
    def calculate(self, mdf: MDF, qnom: float, raster: float) -> SohResult: ...

