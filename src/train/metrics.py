from abc import ABC, abstractmethod
from typing import Any

import numpy as np

from preprocessing.label_transformer import BaseTargetTransform


class BaseMetricsTracker(ABC):
    """
    Abstract evaluation engine for multi-dimensional, multi-participant metrics.

    Provides a unified interface for computing metrics at global, dimensional,
    and participant levels, as well as aggregating metric distributions across
    cross-validation runs.
    """

    @property
    @abstractmethod
    def metric_keys(self) -> list[str]:
        """Return the names of the metrics computed by this tracker."""
        pass

    @property
    @abstractmethod
    def dim_names(self) -> list[str]:
        """Return the names of the target dimensions."""
        pass
    
    @classmethod
    @abstractmethod
    def _calculate_base_metrics(
        cls,
        y_t: np.ndarray,
        y_p: np.ndarray,
        label_transformer: BaseTargetTransform,
    ) -> dict[str, float]:
        """
        Calculate the domain-specific metrics for a set of predictions.

        Args:
            y_t: Ground-truth target values.
            y_p: Predicted target values.
            label_transformer: Transformer used to decode targets before metric computation.

        Returns:
            Dictionary mapping metric names to their calculated values. The keys must match metric_keys.
        """
        pass

    @classmethod
    def evaluate_predictions(
        cls,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        p_ids: np.ndarray,
        label_transformer: BaseTargetTransform,
    ) -> dict[str, Any]:
        """
        Compute global, dimensional, and participant-level metrics.

        Args:
            y_true: Ground-truth values with shape (N, Label_Dim).
            y_pred: Predicted values with shape (N, Label_Dim).
            p_ids: Participant identifier for each sample, with shape (N,).
            label_transformer: Transformer used to decode targets before metric computation.

        Returns:
            Nested dictionary containing
                - global: Metrics computed across all samples and dimensions.
                - by_dimension: Metrics computed independently for each target dimension.
                - by_participant: Metrics for each participant, indexed by stringified participant IDs.
                        Each participant entry contains global and by_dimension dictionaries.
        """
        results = {
            "global": cls._calculate_base_metrics(y_true, y_pred, label_transformer),
            "by_dimension": {},
            "by_participant": {}
        }

        # Per-dimension metrics
        for d, name in enumerate(cls.dim_names):
            results["by_dimension"][name] = cls._calculate_base_metrics(y_true[:, d], y_pred[:, d], label_transformer)

        # Per-participant metrics
        unique_ids = np.unique(p_ids)
        for pid in unique_ids:
            mask = (p_ids == pid)
            yt_p = y_true[mask]
            yp_p = y_pred[mask]

            participant_entry = {
                "global": cls._calculate_base_metrics(yt_p, yp_p, label_transformer),
                "by_dimension": {}
            }

            for d, name in enumerate(cls.dim_names):
                participant_entry["by_dimension"][name] = cls._calculate_base_metrics(yt_p[:, d], yp_p[:, d], label_transformer)
            results["by_participant"][str(pid)] = participant_entry

        return results

    @staticmethod
    def _aggregate_distribution(
        vals: list[float] | np.ndarray,
        confidence: float = 0.95,
        n_bootstraps: int = 1000,
        random_state: int = 42,
    ) -> dict[str, float]:
        """
        Calculate descriptive statistics and a bootstrap confidence interval.

        The confidence interval is estimated by non-parametric bootstrapping of the supplied values. 
        Each bootstrap replicate contains the same number of observations as the original input.

        Args:
            vals: Metric values across folds or participants.
            confidence: Confidence level for the bootstrap interval. Must be strictly between 0 and 1. Defaults to 0.95.
            n_bootstraps: Number of bootstrap replicates. Defaults to 1000.
            random_state: Seed for reproducible bootstrap sampling. Defaults to 42.

        Returns:
            Dictionary containing
                - mean: Arithmetic mean.
                - std: Sample standard deviation (ddof=1).
                - ci_lower: Lower bootstrap confidence bound.
                - ci_upper: Upper bootstrap confidence bound.
                - median: Median.
                - iqr: Interquartile range.
                - q25: 25th percentile.
                - q75: 75th percentile.
        """
        arr = np.asarray(vals, dtype=np.float64)
        n = len(arr)

        # Non-parametric Point Estimates
        mean_val = float(np.mean(arr))
        std_val = float(np.std(arr, ddof=1))
        median_val = float(np.median(arr))
        q25, q75 = np.percentile(arr, [25, 75])
        iqr_val = float(q75 - q25)

        # Handle trivial single-value or zero-variance cases
        if n <= 1 or std_val == 0.0:
            return {
                "mean": mean_val, "std": 0.0,
                "ci_lower": mean_val, "ci_upper": mean_val,
                "median": median_val, "iqr": 0.0, "q25": float(q25), "q75": float(q75)
            }

        # Vectorized Bootstrap for 95% Confidence Interval
        rng = np.random.default_rng(random_state)
        # Generate all random resample indices in one 2D matrix (shape: n_bootstraps x n)
        boot_indices = rng.integers(0, n, size=(n_bootstraps, n))
        boot_samples = arr[boot_indices]
        boot_means = np.mean(boot_samples, axis=1)
        # Calculate interval
        alpha = 1.0 - confidence
        ci_lower = float(np.percentile(boot_means, (alpha / 2.0) * 100))
        ci_upper = float(np.percentile(boot_means, (1.0 - alpha / 2.0) * 100))

        return {
            "mean": mean_val,
            "std": std_val,
            "ci_lower": ci_lower,
            "ci_upper": ci_upper,
            "median": median_val,
            "iqr": iqr_val,
            "q25": float(q25),
            "q75": float(q75)
        }


    @staticmethod
    def _populate_summary_dict(
        target_dict: dict[str, float],
        metric_name: str,
        agg_res: dict[str, float],
    ) -> None:
        """
        Insert aggregated statistics into a summary dictionary.

        Args:
            target_dict: Dictionary receiving the formatted statistics.
            metric_name: Base name of the metric being summarized.
            agg_res: Aggregated statistics returned by_aggregate_distribution.
        """
        for statistic, value in agg_res.items():
            target_dict[f"{metric_name}_{statistic}"] = value


    @classmethod
    def _aggregate_history_levels(
        cls,
        records: list[dict[str, Any]],
        confidence: float = 0.95,
        n_bootstraps: int = 1000,
    ) -> dict[str, Any]:
        """
        Aggregate global and dimensional metrics across metric records.

        Args:
            records: Collection of metric dictionaries, each potentially containing global and by_dimension entries.
            confidence: Confidence level for bootstrap intervals. Defaults to 0.95.
            n_bootstraps: Number of bootstrap replicates. Defaults to 1000.

        Returns:
            Nested dictionary containing
                - global: Aggregated statistics for each global metric.
                - by_dimension: Aggregated statistics for each metric within each target dimension.

            Metrics absent from all records are omitted.
        """
        summary = {"global": {}, "by_dimension": {}}

        # Global Metric Scales
        for metric in cls.metric_keys:
            vals = [r["global"][metric] for r in records if "global" in r and metric in r["global"]]
            if vals:
                agg = cls._aggregate_distribution(vals, confidence=confidence, n_bootstraps=n_bootstraps)
                cls._populate_summary_dict(summary["global"], metric, agg)

        # Dimensional Metrics
        for dim in cls.dim_names:
            summary["by_dimension"][dim] = {}
            for metric in cls.metric_keys:
                vals = [
                    r["by_dimension"][dim][metric]
                    for r in records
                    if "by_dimension" in r and dim in r["by_dimension"] and metric in r["by_dimension"][dim]
                ]
                if vals:
                    agg = cls._aggregate_distribution(vals, confidence=confidence, n_bootstraps=n_bootstraps)
                    cls._populate_summary_dict(summary["by_dimension"][dim], metric, agg)

        return summary


    @classmethod
    def aggregate_cross_validation_runs(
        cls,
        train_histories: list[dict[str, Any]],
        val_histories: list[dict[str, Any]],
        test_histories: list[dict[str, Any]],
        confidence: float = 0.95,
        n_bootstraps: int = 1000,
    ) -> tuple[dict[str, Any] | None, dict[str, Any] | None, dict[str, Any] | None]:
        """
        Aggregate metric histories across cross-validation runs.

        Metrics are summarized independently for the training, validation, and test splits. 
        Participant-level summaries are computed across the folds in which each participant occurs.

        Args:
            train_histories: Metric dictionaries from the training split, one per fold.
            val_histories: Metric dictionaries from the validation split, one per fold.
            test_histories: Metric dictionaries from the test split, one per fold.
            confidence: Confidence level for bootstrap intervals. Defaults to 0.95.
            n_bootstraps: Number of bootstrap replicates. Defaults to 1000.

        Returns:
            Tuple containing
                - train_summary: Aggregated training metrics, or None if no training histories are available.
                - val_summary: Aggregated validation metrics, or None if no validation histories are available.
                - test_summary: Aggregated test metrics, or None if no test histories are available.

            Each non-empty summary contains global, by_dimension, and by_participant entries. 
            Participant entries contain aggregated global and by_dimension metrics.
        """
        def aggregate_split(history: list[dict[str, Any]]) -> dict[str, Any] | None:
            """Aggregate metrics for one dataset split.

            Args:
                history: Metric dictionaries, one per fold.

            Returns:
                Aggregated split summary, or None if the history is empty.
            """
            if not history or len(history) == 0:
                return None

            # Aggregate base global and dimensional metrics
            history_summary = cls._aggregate_history_levels(history, confidence=confidence, n_bootstraps=n_bootstraps)
            history_summary["by_participant"] = {}

            # Participant-Level Metrics across folds
            all_pids = set()
            for h in history:
                if "by_participant" in h:
                    all_pids.update(h["by_participant"].keys())

            for pid in sorted(all_pids):
                pid_histories = [
                    h["by_participant"][pid]
                    for h in history
                    if "by_participant" in h and pid in h["by_participant"]
                ]
                if pid_histories:
                    history_summary["by_participant"][pid] = cls._aggregate_history_levels(pid_histories, confidence=confidence, n_bootstraps=n_bootstraps)

            return history_summary

        return aggregate_split(train_histories), aggregate_split(val_histories), aggregate_split(test_histories)


    @classmethod
    def aggregate_participant_metrics(
        cls,
        cv_participant_metrics: list[dict[str, Any]],
        confidence: float = 0.95,
        n_bootstraps: int = 1000,
    ) -> dict[str, Any]:
        """
        Aggregates metrics across all individual participants collected during participant-level CV.

        Args:
            cv_participant_metrics: Fold dictionaries, each containing a by_participant mapping from stringified participant IDs to participant metric dictionaries.
            confidence: Confidence level for bootstrap intervals. Defaults to 0.95.
            n_bootstraps: Number of bootstrap replicates. Defaults to 1000.

        Returns:
            Dictionary containing aggregated global and by_dimension metrics across participants, together with participant_count.
            Returns an empty dictionary if no participant records are found.
        """
        all_participants = {}
        for fold in cv_participant_metrics:
            if "by_participant" in fold:
                all_participants.update(fold["by_participant"])

        if not all_participants:
            return {}

        summary = cls._aggregate_history_levels(list(all_participants.values()), confidence=confidence, n_bootstraps=n_bootstraps)
        summary["participant_count"] = len(all_participants)

        return summary



class RegressionMetricsTracker(BaseMetricsTracker):
    """
    Evaluation engine for multi-dimensional regression tasks.

    Computes regression metrics in the original target scale and supports
    statistical aggregation across cross-validation folds and participants.
    """
    metric_keys = ["mae", "mse", "rmse", "r2"]
    dim_names = ["EA", "V", "C"]

    @classmethod
    def _calculate_base_metrics(
        cls,
        y_t: np.ndarray,
        y_p: np.ndarray,
        label_transformer: BaseTargetTransform,
    ) -> dict[str, float]:
        """
        Calculate regression metrics after decoding target values.

        Args:
            y_t: Ground-truth target values.
            y_p: Predicted target values.
            label_transformer: Transformer used to decode ground-truth and predicted values into the original target scale.

        Returns:
            Dictionary containing
                - mae: Mean absolute error.
                - mse: Mean squared error.
                - rmse: Root mean squared error.
                - r2: Coefficient of determination.

            Returns zero for every metric if the input contains no samples.
            The R² value is set to zero when the ground-truth values have zero variance.
        """
        if len(y_t) == 0:
            return {k: 0.0 for k in cls.metric_keys}

        # Transform back to original 0-100 scale
        y_t = label_transformer.decode_truth(y_t)
        y_p = label_transformer.decode_predicted(y_p)
            
        mae = np.mean(np.abs(y_p - y_t))
        mse = np.mean((y_p - y_t) ** 2)
        rmse = np.sqrt(mse)

        ss_res = np.sum((y_t - y_p) ** 2)
        ss_tot = np.sum((y_t - np.mean(y_t)) ** 2)
        r2 = 1.0 - (ss_res / ss_tot) if ss_tot != 0 else 0.0

        return {
            "mae": float(mae),
            "mse": float(mse),
            "rmse": float(rmse),
            "r2": float(r2),
        }
