from dataclasses import dataclass

@dataclass(frozen=True)
class ShapConfig:
    """
    Configuration for SHAP computation and visualisation.

    Defines the background sampling, reproducibility, and display settings used for SHAP feature attribution and visualisation.

    Args:
        n_samples: Maximum number of samples drawn for the background dataset. Defaults to 100.
        seed: Random seed used for background data sampling. Defaults to 42.
        max_display: Maximum number of top features displayed in the visualisation. Defaults to 15.
        tick_intervals: Number of intervals used for the visualisation's axis ticks. Defaults to 10.
    """
    n_samples: int = 100
    seed: int = 42
    max_display: int = 15
    tick_intervals: int = 10