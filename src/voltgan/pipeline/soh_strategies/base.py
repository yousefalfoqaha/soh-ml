from abc import ABC, abstractmethod

from asammdf import MDF

from voltgan.pipeline.base import SampleContext


class SohStrategy(ABC):
    @abstractmethod
    def can_handle(self, mdf: MDF) -> bool: ...

    @abstractmethod
    def calculate(
        self, mdf: MDF, nominal_capacity: float, raster: float, context: SampleContext
    ) -> SampleContext: ...
