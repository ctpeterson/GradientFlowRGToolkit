from __future__ import annotations

from dataclasses import dataclass, field

from .setup import ProcessingConfiguration
from .stages.process import ProcessingResult, process_dataset


@dataclass
class RunningCoupling:
    """Facade for analysis of running coupling"""
    nf: int
    nc: int = 3
    gauge_action: str = "s"
    _processing_result: ProcessingResult | None = field(
        default = None,
        init = False,
        repr = False,
    )

    def __post_init__(self) -> None:
        if isinstance(self.nf, bool) or not isinstance(self.nf, int) or self.nf < 0:
            raise ValueError("nf must be a non-negative integer")
        if isinstance(self.nc, bool) or not isinstance(self.nc, int) or self.nc < 2:
            raise ValueError("nc must be an integer of at least two")
        if self.gauge_action not in {"s", "w"}:
            raise ValueError("gauge_action must be 's' (Symanzik) or 'w' (Wilson)")

    @property
    def processing_result(self) -> ProcessingResult | None:
        return self._processing_result

    def process(self, configuration: ProcessingConfiguration) -> ProcessingResult:
        result = process_dataset(
            configuration,
            nc=self.nc,
            gauge_action=self.gauge_action,
        )
        self._processing_result = result
        return result
