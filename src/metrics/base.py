from abc import ABC, abstractmethod
from typing import List


class BaseMetric(ABC):
    """All metrics share the same interface so Evaluator can call them uniformly."""

    name: str = "base"

    @abstractmethod
    def update(self, preds: List[str], targets: List[str]) -> None:
        ...

    @abstractmethod
    def compute(self) -> float:
        ...

    def reset(self) -> None:
        self.__init__()
