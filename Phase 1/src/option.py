from dataclasses import dataclass


@dataclass
class Option:
    S: float      # spot price
    K: float      # strike
    T: float      # time to maturity (years)
    r: float      # risk-free rate (continuous, annualised)
    sigma: float  # volatility (annualised)
    kind: str     # 'call' or 'put'

    def __post_init__(self):
        if self.S <= 0:
            raise ValueError(f"S must be > 0, got {self.S}")
        if self.K <= 0:
            raise ValueError(f"K must be > 0, got {self.K}")
        if self.T <= 0:
            raise ValueError(f"T must be > 0, got {self.T}")
        if self.sigma <= 0:
            raise ValueError(f"sigma must be > 0, got {self.sigma}")
        if self.kind not in ('call', 'put'):
            raise ValueError(f"kind must be 'call' or 'put', got '{self.kind}'")
