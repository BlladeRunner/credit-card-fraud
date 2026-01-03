from dataclasses import dataclass

@dataclass(frozen=True)
class Config:
    seed: int = 42
    target_col: str = "Class"
    test_size: float = 0.2

    # Threshold tuning defaults
    min_precision: float = 0.80
    min_recall: float = 0.80

    # Business costs (example; adjust later)
    cost_false_negative: float = 100.0  # missed fraud
    cost_false_positive: float = 1.0    # manual review / customer friction

CFG = Config()
