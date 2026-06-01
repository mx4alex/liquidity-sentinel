from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import pandas as pd


class LSIStatus(str, Enum):
    GREEN = 'green'
    YELLOW = 'yellow'
    RED = 'red'

    @classmethod
    def from_value(cls, lsi: float, green_max: float=40, yellow_max: float=70) -> LSIStatus:
        if lsi < green_max:
            return cls.GREEN
        if lsi < yellow_max:
            return cls.YELLOW
        return cls.RED

    @property
    def label_ru(self) -> str:
        return {LSIStatus.GREEN: 'Норма', LSIStatus.YELLOW: 'Напряжение', LSIStatus.RED: 'Стресс'}[self]

@dataclass
class ModuleSignals:
    module_id: str
    date: pd.Timestamp
    mad_scores: dict[str, float] = field(default_factory=dict)
    flags: dict[str, bool | int | float] = field(default_factory=dict)
    raw_metrics: dict[str, float] = field(default_factory=dict)

@dataclass
class LSIResult:
    date: pd.Timestamp
    lsi: float
    status: LSIStatus
    module_contributions: dict[str, float]
    active_flags: list[str]
    seasonal_factor: float = 1.0
    overlap_adjusted: bool = False
