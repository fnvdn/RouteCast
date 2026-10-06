from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass
class CalibrationResult:
    threshold: float
    empirical_risk: float
    target_risk: float
    requests: int


class RequestRiskCalibrator:
    """Grouped calibration for request-level expert-set miss risk.

    Each calibration item is one request-level risk curve. The selected
    threshold is the smallest candidate that satisfies the requested empirical
    risk; request-level grouping prevents token rows from acting as independent
    samples.
    """

    def __init__(self, target_risk: float = 0.25, delta: float = 0.1):
        if not 0 < target_risk < 1 or not 0 < delta < 1:
            raise ValueError("target_risk and delta must be in (0, 1)")
        self.target_risk = target_risk
        self.delta = delta
        self.result: CalibrationResult | None = None

    def fit(self, request_risks: list[tuple[float, float]]) -> CalibrationResult:
        if not request_risks:
            raise ValueError("at least one request is required")
        # Conservative finite-sample correction for a request-level split.
        allowance = self.target_risk - math.sqrt(
            math.log(2.0 / self.delta) / (2.0 * len(request_risks)))
        allowance = max(0.0, allowance)
        feasible = [(threshold, risk) for threshold, risk in request_risks
                    if risk <= allowance]
        if feasible:
            threshold, risk = min(feasible, key=lambda x: x[0])
        else:
            threshold, risk = min(request_risks, key=lambda x: x[1])
        self.result = CalibrationResult(threshold, risk, self.target_risk,
                                        len(request_risks))
        return self.result

    def transform(self, scores, threshold: float | None = None):
        if threshold is None:
            if self.result is None:
                raise RuntimeError("fit the calibrator first")
            threshold = self.result.threshold
        return scores >= threshold
