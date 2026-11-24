import logging
from abc import ABC, abstractmethod

import numpy as np
from sklearn.model_selection import GroupKFold, GroupShuffleSplit


logger = logging.getLogger(__name__)


class BaseSplitter(ABC):
    """
    Abstract base class defining the interface for dataset partitioning strategies.

    Subclasses implement a partitioning strategy that generates training, validation, and test indices.
    """
    @abstractmethod
    def split(
        self,
        data: np.ndarray,
        labels: np.ndarray,
        participant_ids: np.ndarray,
    ) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """
        Generate training, validation, and test indices.

        Args:
            data: Feature array with shape (N, ...), where N is the number of samples.
            labels: Target array with shape (N, ...).
            participant_ids: Participant identifier for each sample, with shape (N,).

        Returns:
            A list of tuples, with each tuple containing
            - train_idx: Indices of training samples.
            - val_idx: Indices of validation samples.
            - test_idx: Indices of test samples.

            All indices refer to positions in the original input arrays.
        """
        pass


    @property
    @abstractmethod
    def is_between_splitter(self) -> bool:
        """
        Indicate whether participants are isolated across data partitions.

        Returns:
            True for between-subject splitting and False for within-subject splitting.
        """
        pass



class BetweenSubjectSplitter(BaseSplitter):
    """
    Perform three-way between-subject cross-validation using nested GroupKFold.

    The outer GroupKFold generates the test partition, while an inner GroupKFold divides 
    the remaining participants into training and validation partitions.

    Participants are strictly isolated across the three partitions.

    Args:
        n_splits: Number of folds used for the cross-validation. Must be at least 3. Defaults to 10.
        verbose: Whether to log partition statistics for each fold. Defaults to False.
    """

    def __init__(
        self,
        n_splits: int = 10,
        verbose: bool = False,
    ) -> None:
        if n_splits < 3:
            raise ValueError("n_splits must be at least 3 to support a Train/Val/Test split.")
        
        self.n_splits = n_splits
        self.verbose = verbose

        logger.info(f"Initialized {self} \n")


    def __str__(self):
        """Returns a string representation of the cross-validation configuration."""
        return f"BetweenSubjectSplitter(n_splits={self.n_splits}, verbose={self.verbose})"

    
    @property
    def is_between_splitter(self) -> bool:
        """
        Indicate whether the splitting strategy separates participants.

        Returns:
            True, indicating between-subject partitioning.
        """
        return True


    def split(
        self,
        data: np.ndarray,
        labels: np.ndarray,
        participant_ids: np.ndarray,
    ) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """
        Generate three-way between-subject partitions using nested GroupKFold.

        The outer GroupKFold generates the test partition. A second GroupKFold is applied to the remaining participants 
        to generate the training and validation partitions.

        Args:
            data: Feature array with shape (N, ...).
            labels: Target array with shape (N, ...).
            participant_ids: Participant identifiers with shape (N,).

        Returns:
            A list of tuples, with each tuple containing
            - train_idx: Indices of training samples.
            - val_idx: Indices of validation samples.
            - test_idx: Indices of test samples.

            All indices refer to the original input arrays. Participants are mutually exclusive across the three partitions.

        Raises:
            ValueError: If the number of unique participants is smaller than n_splits.
        """
        # Ensure enough participants
        unique_participants = np.unique(participant_ids)
        if len(unique_participants) < self.n_splits:
            raise ValueError(
                f"Number of folds ({self.n_splits}) cannot exceed unique subjects ({len(unique_participants)})."
            )
        
        splits = [] # List of folds

        # Outer Fold creates the final holdout Test set (1/K of total data)
        outer_gkf = GroupKFold(n_splits=self.n_splits)

        for fold_idx, (outer_train_val_idx, test_idx) in enumerate(outer_gkf.split(data, labels, groups=participant_ids)):
            # Isolate data for train-validation
            train_val_data = data[outer_train_val_idx]
            train_val_labels = labels[outer_train_val_idx]
            train_val_groups = participant_ids[outer_train_val_idx]

            # Inner Fold creates the Val set 
            # To match the Test set size, it takes 1 out of the remaining (K - 1) folds
            inner_gkf = GroupKFold(n_splits=self.n_splits - 1)
            inner_generator = inner_gkf.split(train_val_data, train_val_labels, groups=train_val_groups)
            
            # Use modulo to stay within the inner split bounds (0 to n_splits-2)
            target_inner_fold = fold_idx % (self.n_splits - 1)
            
            # Step through the generator to pick different val subsets (else almost always the same subset is selected)
            for _ in range(target_inner_fold + 1):
                inner_train_idx_local, val_idx_local = next(inner_generator)
            
            # Map local sub-arrays back to absolute dataset index space
            train_idx = outer_train_val_idx[inner_train_idx_local]
            val_idx = outer_train_val_idx[val_idx_local]

            splits.append((train_idx, val_idx, test_idx))   # Add fold to list

            # Optional inline logging summary per fold
            if self.verbose:
                self._log_splits(fold_idx, train_idx, val_idx, test_idx, participant_ids)

        return splits

    
    def _log_splits(
        self,
        fold_idx: int,
        train_idx: np.ndarray,
        val_idx: np.ndarray,
        test_idx: np.ndarray,
        participant_ids: np.ndarray,
    ) -> None:
        """
        Log sample counts and participant assignments for a fold.

        Args:
            fold_idx: Zero-based index of the current fold.
            train_idx: Training sample indices.
            val_idx: Validation sample indices.
            test_idx: Test sample indices.
            participant_ids: Participant identifiers for the full dataset.
        """
        train_p = sorted(list(set(participant_ids[train_idx])))
        val_p = sorted(list(set(participant_ids[val_idx])))
        test_p = sorted(list(set(participant_ids[test_idx])))
        total_samples = len(train_idx) + len(val_idx) + len(test_idx)

        logger.info(f"\n🔹 FOLD {fold_idx + 1}:")
        logger.info(f"  | Set   | Rows | Sample % | Unique Participants")
        logger.info(f"  |-------|------|----------|-------------------")
        logger.info(f"  | Train | {len(train_idx):<4} | {len(train_idx)/total_samples*100:>.1f}%    | {train_p}")
        logger.info(f"  | Val   | {len(val_idx):<4} | {len(val_idx)/total_samples*100:>.1f}%    | {val_p}")
        logger.info(f"  | Test  | {len(test_idx):<4} | {len(test_idx)/total_samples*100:>.1f}%    | {test_p} \n")



class BetweenSubjectMonteCarloSplitter(BaseSplitter):
    """
    Perform three-way between-subject Monte Carlo cross-validation.

    Uses two nested GroupShuffleSplit operations to generate randomized training, validation, and test partitions.

    Participants are strictly isolated across all three partitions.
    The requested proportions determine the approximate relative sizes of the partitions in terms of participant groups.

    Args:
        n_splits: Number of randomized partitions to generate. Defaults to 10.
        train_ratio: Target proportion of participants assigned to training. Must be positive. Defaults to 0.70.
        val_ratio: Target proportion of participants assigned to validation. Must be positive. Defaults to 0.15.
        test_ratio: Target proportion of participants assigned to testing. Must be positive. Defaults to 0.15.
        random_state: Random seed controlling the outer split sequence. If None, the splits are not reproducible. Defaults to 42.
        verbose: Whether to log partition statistics for each split. Defaults to False.
    """
    def __init__(
        self,
        n_splits: int = 10,
        train_ratio: float = 0.70,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        random_state: int | None = 42,
        verbose: bool = False,
    ) -> None:
        if not np.isclose(train_ratio + val_ratio + test_ratio, 1.0):
            raise ValueError("The sum of train_ratio, val_ratio, and test_ratio must equal 1.0")
            
        self.n_splits = n_splits
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.random_state = random_state
        self.verbose = verbose

        logger.info(f"Initialized {self} \n")


    def __str__(self):
        """Returns a string representation of the ShuffleSplit dataset configuration."""
        return (f"BetweenSubjectMonteCarloSplitter(n_splits={self.n_splits}, "
                f"train_ratio={self.train_ratio}, val_ratio={self.val_ratio}, "
                f"test_ratio={self.test_ratio}, random_state={self.random_state}, "
                f"verbose={self.verbose})")

    
    @property
    def is_between_splitter(self) -> bool:
        """
        Indicate whether the splitting strategy separates participants.

        Returns:
            True, indicating between-subject partitioning.
        """
        return True


    def split(
        self,
        data: np.ndarray,
        labels: np.ndarray,
        participant_ids: np.ndarray,
    ) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """
        Generate randomized three-way between-subject partitions.

        The outer GroupShuffleSplit generates the test partition.
        A second GroupShuffleSplit divides the remaining participants into training and validation partitions.

        A different random seed is used for each inner split when random_state is specified.

        Args:
            data: Feature array with shape (N, ...).
            labels: Target array with shape (N, ...).
            participant_ids: Participant identifiers with shape (N,).

        Returns:
            A list of tuples, with each tuple containing
            - train_idx: Indices of training samples.
            - val_idx: Indices of validation samples.
            - test_idx: Indices of test samples.

            All indices refer to the original input arrays. Participants are mutually exclusive across the three partitions.
        """
        splits = [] # List of folds
        
        # Outer Fold creates the final holdout Test set of relative size test_ratio
        outer_gss = GroupShuffleSplit(
            n_splits=self.n_splits,  
            test_size=self.test_ratio, 
            random_state=self.random_state
        )

        for fold_idx, (outer_train_val_idx, test_idx) in enumerate(outer_gss.split(data, labels, groups=participant_ids)):
            # Isolate data for train-validation
            train_val_data = data[outer_train_val_idx]
            train_val_labels = labels[outer_train_val_idx]
            train_val_groups = participant_ids[outer_train_val_idx]

            # Inner Fold creates the Val set 
            # Because the outer split took a chunk away, we must scale the validation ratio relative to the REMAINING pool size
            adjusted_val_ratio = self.val_ratio / (self.train_ratio + self.val_ratio)

            # Use an updated random_state per fold so the inner splits vary across folds
            fold_seed = None if self.random_state is None else self.random_state + fold_idx
            
            inner_gss = GroupShuffleSplit(
                n_splits=1,                     # one split
                test_size=adjusted_val_ratio,   # with adjusted_val_ratio percent for validation subset
                random_state=fold_seed
            )
            
            # Extract the single inner split partition
            inner_train_idx_local, val_idx_local = next(inner_gss.split(train_val_data, train_val_labels, groups=train_val_groups))
            
            # Map local sub-arrays back to absolute dataset index space
            train_idx = outer_train_val_idx[inner_train_idx_local]
            val_idx = outer_train_val_idx[val_idx_local]

            splits.append((train_idx, val_idx, test_idx))   # Add fold to list

            # Optional inline logging summary per fold
            if self.verbose:
                self._log_splits(fold_idx, train_idx, val_idx, test_idx, participant_ids)

        return splits

    
    def _log_splits(
        self,
        fold_idx: int,
        train_idx: np.ndarray,
        val_idx: np.ndarray,
        test_idx: np.ndarray,
        participant_ids: np.ndarray,
    ) -> None:
        """
        Log sample counts and participant assignments for a split.

        Args:
            fold_idx: Zero-based index of the current split.
            train_idx: Training sample indices.
            val_idx: Validation sample indices.
            test_idx: Test sample indices.
            participant_ids: Participant identifiers for the full dataset.
        """
        train_p = sorted(list(set(participant_ids[train_idx])))
        val_p = sorted(list(set(participant_ids[val_idx])))
        test_p = sorted(list(set(participant_ids[test_idx])))
        total_samples = len(train_idx) + len(val_idx) + len(test_idx)

        logger.info(f"\n🔹 FOLD {fold_idx + 1}:")
        logger.info(f"  | Set   | Rows | Sample % | Unique Participants")
        logger.info(f"  |-------|------|----------|-------------------")
        logger.info(f"  | Train | {len(train_idx):<4} | {len(train_idx)/total_samples*100:>.1f}%    | {train_p}")
        logger.info(f"  | Val   | {len(val_idx):<4} | {len(val_idx)/total_samples*100:>.1f}%    | {val_p}")
        logger.info(f"  | Test  | {len(test_idx):<4} | {len(test_idx)/total_samples*100:>.1f}%    | {test_p} \n")
    


class WithinSubjectSplitter(BaseSplitter):
    """
    Perform individualized temporal expanding-window cross-validation.

    Generates sequential training, validation, and test partitions relative to each participant's individual timeline.

    Each configuration specifies the cumulative training, validation, and test proportions for a particular temporal horizon. 
    Training windows expand across successive configurations, while validation and test windows are positioned sequentially after the training  period.

    Input samples must already be ordered chronologically within each participant.

    Args:
        windows: Sequence of partition configurations, each represented as (train_fraction, val_fraction, test_fraction).
            Defaults to five configurations with progressively expanding training periods.
        verbose: Whether to log partition statistics and timeline information for each horizon. Defaults to False.
    """
    def __init__(
        self,
        windows: list[tuple[float, float, float]] | None = None,
        verbose: bool = False,
    ) -> None:
        # Configuration: (train_fraction, val_fraction, test_fraction)
        self.windows = windows or [
            (0.40, 0.10, 0.10),
            (0.50, 0.10, 0.10),
            (0.60, 0.10, 0.10),
            (0.70, 0.10, 0.10),
            (0.80, 0.10, 0.10)
        ]
        self.verbose = verbose

        logger.info(f"Initialized {self} \n")


    def __str__(self):
        """Returns a string representation of the expanding windows timeline parameters."""
        return f"WithinSubjectSplitter(windows={self.windows}, verbose={self.verbose})"

    
    @property
    def is_between_splitter(self) -> bool:
        """Indicate whether the splitting strategy separates participants.

        Returns:
            False, indicating within-subject partitioning.
        """
        return False


    def split(
        self,
        data: np.ndarray,
        labels: np.ndarray,
        participant_ids: np.ndarray,
    ) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """
        Generate sequential expanding-window partitions for each participant.

        For each temporal configuration, the samples belonging to every participant are divided into 
        non-overlapping training, validation, and test segments.

        Partition boundaries are calculated independently for each participant using the specified fractions. 
        The resulting indices are mapped back to the original dataset.

        Args:
            data: Feature array with shape (N, ...).
            labels: Target array with shape (N, ...).
            participant_ids: Participant identifiers with shape (N,). Samples must be chronologically ordered within each participant.

        Returns:
            A list of tuples, with each tuple containing
            - train_idx: Indices of training samples.
            - val_idx: Indices of validation samples.
            - test_idx: Indices of test samples.

            All indices refer to the original input arrays. 
            The partitions are temporally ordered and non-overlapping within each participant.

        Notes:
            The method does not sort the input samples. Chronological
            ordering must be ensured before calling this method.
        """
        unique_ids = np.unique(participant_ids)
        splits = [] # List of folds

        for fold_idx, (t_train, t_val, t_test) in enumerate(self.windows):
            train_indices, val_indices, test_indices = [], [], []

            for p_id in unique_ids:
                # Find global indices where this participant appears
                p_idx = np.where(participant_ids == p_id)[0]
                n_samples = len(p_idx)

                # Track timeline cuts proportionally against the absolute group size footprint
                idx_train_end = int(np.floor(n_samples * t_train))
                idx_val_end = int(np.floor(n_samples * (t_train + t_val)))
                idx_test_end = int(np.floor(n_samples * (t_train + t_val + t_test)))

                # Map back to the absolute global data positions
                train_indices.extend(p_idx[:idx_train_end])
                val_indices.extend(p_idx[idx_train_end:idx_val_end])
                test_indices.extend(p_idx[idx_val_end:idx_test_end])

            train_idx = np.array(train_indices)
            val_idx = np.array(val_indices)
            test_idx = np.array(test_indices)

            splits.append((train_idx, val_idx, test_idx))

            if self.verbose:
                self._log_splits(fold_idx, t_train, t_val, t_test, train_idx, val_idx, test_idx, unique_ids, participant_ids)

        return splits

    
    def _log_splits(
        self,
        fold_idx: int,
        t_train: float,
        t_val: float,
        t_test: float,
        train_idx: np.ndarray,
        val_idx: np.ndarray,
        test_idx: np.ndarray,
        unique_ids: np.ndarray,
        participant_ids: np.ndarray,
    ) -> None:
        """
        Log partition sizes and temporal index ranges for selected participants.

        Displays the relative training, validation, and test index ranges for the first two participants, 
        together with the overall sample counts and percentages.

        Args:
            fold_idx: Zero-based index of the current temporal configuration.
            t_train: Training fraction for the current configuration.
            t_val: Validation fraction for the current configuration.
            t_test: Test fraction for the current configuration.
            train_idx: Training sample indices.
            val_idx: Validation sample indices.
            test_idx: Test sample indices.
            unique_ids: Unique participant identifiers.
            participant_ids: Participant identifiers for the full dataset.
        """
        total_samples = len(participant_ids)
        logger.info(f"\n🔹 HORIZON WINDOW {fold_idx + 1} (Target: {t_train*100:.0f}% Train / {t_val*100:.0f}% Val / {t_test*100:.0f}% Test):")
        logger.info(f"  | Set   | Total Rows | Sample % | Relative Indices Showcase (P1 vs P2)")
        logger.info(f"  |-------|------------|----------|-----------------------------------------------------")
        
        # Handle index extraction checks safely for multi-subject arrays
        p1_id, p2_id = unique_ids[0], unique_ids[1]
        p1_global_idx = np.where(participant_ids == p1_id)[0]
        p2_global_idx = np.where(participant_ids == p2_id)[0]
        
        p1_tr = np.where(np.isin(p1_global_idx, train_idx))[0]
        p1_va = np.where(np.isin(p1_global_idx, val_idx))[0]
        p1_te = np.where(np.isin(p1_global_idx, test_idx))[0]

        p2_tr = np.where(np.isin(p2_global_idx, train_idx))[0]
        p2_va = np.where(np.isin(p2_global_idx, val_idx))[0]
        p2_te = np.where(np.isin(p2_global_idx, test_idx))[0]

        logger.info(f"  | Train | {len(train_idx):<10} | {len(train_idx)/total_samples*100:>.1f}%    | P{p1_id}: Indices {p1_tr[0]} to {p1_tr[-1]}  vs  P{p2_id}: Indices {p2_tr[0]} to {p2_tr[-1]}")
        logger.info(f"  | Val   | {len(val_idx):<10} | {len(val_idx)/total_samples*100:>.1f}%    | P{p1_id}: Indices {p1_va[0]} to {p1_va[-1]}  vs  P{p2_id}: Indices {p2_va[0]} to {p2_va[-1]}")
        logger.info(f"  | Test  | {len(test_idx):<10} | {len(test_idx)/total_samples*100:>.1f}%    | P{p1_id}: Indices {p1_te[0]} to {p1_te[-1]}  vs  P{p2_id}: Indices {p2_te[0]} to {p2_te[-1]} \n")
