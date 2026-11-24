import os
import copy
import logging
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from data_extractors.study_5 import DataProcessor5
from preprocessing.label_transformer import BaseTargetTransform


logger = logging.getLogger(__name__)


class Extractor:
    """
    Coordinate multi-study data extraction and cross-study aggregation.

    Initializes study-specific data processors, harmonizes sampling frequencies, validates metadata consistency, 
    and aggregates extracted data into unified data and label arrays.

    Args:
        study_configs: Mapping of study IDs to their corresponding study-specific processor configurations.
        label_transformer: Target transformation instance used to process labels across all configured studies.
        movement_type: Type of movement data to extract. Supported values are raw, move, tsfel, tsfel-move,and fitbit. 
            Defaults to raw.
        window_min: Duration of the lookback context window in minutes. Defaults to 5.
        include_magnitude: Whether to calculate and include raw movement magnitude as an additional feature. Defaults to True.
        include_jerk: Whether to calculate and include raw movement jerk as an additional feature. Defaults to True.
        top_k: Maximum number of non-redundant features selected per target label. If None, no top-k feature selection is applied.
            Defaults to None.
        corr_threshold: Pearson correlation coefficient threshold used for feature selection. If None, no correlation-based filtering
            is applied. Defaults to None.
        one_hot_encoding: Whether categorical features are one-hot encoded. If False, categorical features retain their integer encoding.
            Defaults to True.
        sampling_rate_raw: Target sampling frequency in Hz used to harmonize raw movement data across studies. Defaults to 64.
        minimum_samples: Minimum number of valid windows required per participant. Participants below this threshold are excluded from 
            the aggregated dataset. Defaults to 10.
    """
    def __init__(
            self,
            study_configs: Mapping[int, dict[str, Any]],
            label_transformer: BaseTargetTransform,
            movement_type: str = "raw",
            window_min: int = 5,
            include_magnitude: bool = True,
            include_jerk: bool = True,
            top_k: int | None = None,
            corr_threshold: float | None = None,
            one_hot_encoding: bool = True,
            sampling_rate_raw: int = 64,
            minimum_samples: int = 10,
    ) -> None:
        self.label_transformer = label_transformer
        self.movement_type = movement_type
        self.window_min = window_min
        self.include_magnitude = include_magnitude
        self.include_jerk = include_jerk
        self.top_k = top_k
        self.corr_threshold = corr_threshold
        self.one_hot_encoding = one_hot_encoding
        self.target_fs = sampling_rate_raw
        self.minimum_samples = minimum_samples
        self.processors = {}

        # Resolve study-specific root directories
        base_path = Path("/data")   # TODO: change path to actual data directory
        study_root_registry = {
            5: base_path / "Studie_5",
            # Register additional study root directories here
        }
        resolved_configs = self._resolve_root_paths( study_configs, study_root_registry)

        # Initialize the registered study-specific data processors
        processor_registry = {
            5: DataProcessor5,
            # Register additional study-specific processors here
        }
        self.processors = self._instantiate_processors(resolved_configs, processor_registry)

        logger.info(f"Initialized {self} \n")


    def __str__(self):
        """Returns a string representation of the extractor configuration."""
        active_studies = sorted(list(self.processors.keys()))
        return (
            f"Extractor(studies={active_studies}, "
            f"label_transformer={self.label_transformer}, "
            f"movement_type='{self.movement_type}', "
            f"window_min={self.window_min}min, "
            f"include_magnitude={self.include_magnitude}, "
            f"include_jerk={self.include_jerk}, "
            f"top_k={self.top_k}, "
            f"corr_threshold={self.corr_threshold}, "            
            f"one_hot_encoding={self.one_hot_encoding}, "
            f"target_fs={self.target_fs}Hz)"
        )
    

    def _resolve_root_paths(
        self,
        study_configs: Mapping[int, dict[str, Any]],
        study_root_registry: Mapping[int, Path],
    ) -> dict[int, dict[str, Any]]:
        """
        Resolve and assign root directories for configured studies.

        Creates a deep copy of the supplied study configurations and adds the corresponding root directory to 
        each configuration using the study root registry.

        Args:
            study_configs: Mapping of study IDs to their study-specific processor configurations.
            study_root_registry: Mapping of study IDs to their respective root directories.

        Returns:
            Deep-copied configuration dictionary containing the resolved root_path entry for each study.

        Raises:
            KeyError: If a configured study ID does not have a corresponding root directory registered.
        """
        # Deep copy parameters to avoid modifying the original input
        updated_configs = copy.deepcopy(study_configs)
        
        # Assign the corresponding root directory to each study
        for study_id, params in updated_configs.items():
            if study_id in study_root_registry:
                params['root_path'] = study_root_registry[study_id]
            else:
                raise KeyError(f"No root path mapping found for Study {study_id}.")
                    
        return updated_configs
    

    def _instantiate_processors(
        self,
        resolved_configs: Mapping[int, dict[str, Any]],
        processor_registry: Mapping[int, type],
    ) -> dict[int, Any]:
        """
        Instantiate the registered data processors for the configured studies.

        Combines shared extraction parameters with study-specific configuration
        parameters and initializes the corresponding DataProcessor instances.

        Args:
            resolved_configs: Mapping of study IDs to their resolved processor configurations.
            processor_registry: Mapping of study IDs to their corresponding DataProcessor classes.

        Returns:
            Dictionary mapping study IDs to their initialized DataProcessor instances.

        Raises:
            KeyError: If a configured study ID is not registered in the processor registry.
        """
        instantiated_processors = {}

        for study_id, params in resolved_configs.items():
            if study_id not in processor_registry:
                raise KeyError(f"Study ID {study_id} is not registered in the processor library.")
                
            ProcessorClass = processor_registry[study_id]
            
            # Combine core structural parameters with custom study-specific overrides
            full_params = {
                "label_transformer": self.label_transformer,
                "movement_type": self.movement_type,
                "window_min": self.window_min,
                "include_magnitude": self.include_magnitude,
                "include_jerk": self.include_jerk,
                "top_k": self.top_k,
                "corr_threshold": self.corr_threshold,
                "one_hot_encoding": self.one_hot_encoding,
                **params  # Captures unique resolved root_path, individual sampling_rate_raw, etc.
            }
            
            instantiated_processors[study_id] = ProcessorClass(**full_params)
            
        return instantiated_processors
    

    def _filter_by_minimum_samples(
        self,
        data: np.ndarray,
        labels: np.ndarray,
        participant_ids: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Remove samples belonging to participants below the minimum threshold.

        Counts the number of available samples for each participant and retains only those participants meeting the 
        configured minimum sample requirement. Filtering is applied consistently across data, labels, and participant IDs.

        Args:
            data: Input data array of shape (N, Window_Size, Feature_Dim).
            labels: Corresponding target array of shape (N, Label_Dim).
            participant_ids: Participant identifier array of shape (N,).

        Returns:
            Tuple containing
                - filtered_data: Filtered input data array.
                - filtered_labels: Corresponding filtered target array.
                - filtered_ids: Filtered participant identifier array.

            All returned arrays retain their original dimensional structure.

        Notes:
            If minimum_samples is zero, no filtering is performed.
        """
        # Skip filtering if no minimum sample requirement is configured
        if self.minimum_samples == 0:
            return data, labels, participant_ids
        
        # Count samples belonging to each unique participant
        unique_ids, sample_counts = np.unique(participant_ids, return_counts=True)
        id_to_count = dict(zip(unique_ids, sample_counts))

        # Build a boolean filter mask across the absolute sample size N
        # True means keep the row, False means drop it
        keep_mask = np.array([id_to_count[p_id] >= self.minimum_samples for p_id in participant_ids])

        # Log details about excluded participants
        for p_id, count in id_to_count.items():
            if count < self.minimum_samples:
                logger.warning(f"Participant {p_id} excluded from dataset. Sample count ({count}) is below required minimum ({self.minimum_samples}).")

        # Slice data
        filtered_data = data[keep_mask]
        filtered_labels = labels[keep_mask]
        filtered_ids = participant_ids[keep_mask]

        # Log summary data metrics
        total_dropped = len(data) - len(filtered_data)
        if total_dropped > 0:
            logger.info(f"Dropped Windows: {total_dropped}")

        return filtered_data, filtered_labels, filtered_ids


    def load_data_label_pairs(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str], list[str], list[int]]:
        """
        Extract, harmonize, and aggregate data across configured studies.

        Executes the extraction pipeline for each configured study, aligns sampling frequencies where required, 
        validates cross-study metadata consistency, and consolidates the resulting data streams into unified arrays.

        For raw movement data, studies operating above the target sampling  frequency are downsampled using block-wise averaging.

        Participant identifiers are prefixed with their corresponding study IDs to prevent identifier collisions across studies. 
        Participants not meeting the configured minimum sample requirement are excluded.

        Returns:
            Tuple containing
                - unified_data: Aggregated data array of shape (N, Window_Size, Feature_Dim).
                - unified_labels: Aggregated target array of shape (N, Label_Dim).
                - unified_ids: Participant identifiers of shape (N,), including their corresponding study prefixes.
                - unified_mask: Binary feature mask of shape (Feature_Dim,),
                    where 1 denotes continuous features and 0 denotes ategorical or discrete features.
                - unified_feature_names: Ordered list of feature names.
                - unified_label_names: Ordered list of target label names.
                - unified_cardinalities: List of cardinalities for categorical features.

        Raises:
            ValueError: If sampling frequencies cannot be aligned, cross-study
                metadata is inconsistent, or no valid data records are available.
        """
        all_study_data, all_study_labels, all_study_ids = [], [], []
        unified_mask, unified_feature_names, unified_label_names = None, None, None
        unified_cardinalities = []

        # Iterate over studies
        for study_id, processor in self.processors.items():
            logger.info(f"Beginning pipeline extraction for Study {study_id}...")
            
            # Extract data from the current study processor
            study_data, study_labels, p_ids, binary_mask, feature_names, label_names, cardinalities = processor.process_all_participants()
            
            if len(study_data) == 0:
                logger.warning(f"Study {study_id} returned no valid participant data pairs. Skipping aggregation.")
                continue

            # Frequency Resampling Realignment
            if self.movement_type == "raw":
                current_fs = processor.sampling_rate_raw
                
                if current_fs > self.target_fs:
                    if current_fs % self.target_fs != 0:
                        raise ValueError(f"Incompatible sample rate manipulation: Study {study_id} operates at {current_fs}Hz, which cannot be cleanly averaged to target {self.target_fs}Hz.")
                    
                    decimation_factor = current_fs // self.target_fs    # Compute downsampling interval factor (e.g., 64 // 32 = 2)
                    n_windows, original_samples, n_features = study_data.shape  # Capture current array dimensions
                    target_samples = original_samples // decimation_factor  # Calculate the new target length for the time axis
                    
                    # Reshape to isolate the blocks, take the mean, and collapse back
                    # Shape evolution: (N_Windows, Samples, Dim) -> (N_Windows, Target_Samples, Factor, Dim) -> Mean along Axis 2 -> (N_Windows, Target_Samples, Dim)
                    study_data = study_data.reshape(n_windows, target_samples, decimation_factor, n_features).mean(axis=2)

                    logger.info(f"Downsampled Study {study_id} raw data from {current_fs}Hz to {self.target_fs}Hz (Factor: {decimation_factor}).")

            # Data Strem Alignment Validation
            if unified_mask is None:
                # Establish baselines from the first valid data stream
                unified_mask = binary_mask
                unified_feature_names = feature_names
                unified_label_names = label_names
                unified_cardinalities = cardinalities
                unified_window_samples = study_data.shape[1]  # Capture baseline window size
            else:
                # Feature Names Alignment
                if feature_names != unified_feature_names:
                    raise ValueError(f"Feature names mismatch in Study {study_id}. Expected {unified_feature_names}, got {list(feature_names)}.")
                
                # Label Names Alignment
                if label_names != unified_label_names:
                    raise ValueError(f"Label names mismatch in Study {study_id}. Expected {unified_label_names}, got {list(label_names)}.")

                # Cardinalities Alignment
                if cardinalities != unified_cardinalities:
                    raise ValueError(f"Categorical cardinalities mismatch in Study {study_id}. Expected {unified_cardinalities}, got {cardinalities}.")
                
                # Binary Mask Alignment
                if not np.array_equal(binary_mask, unified_mask):
                    raise ValueError(f"Binary data mask mismatch in Study {study_id}. Target masking arrays must be identical.")
                
                # Time Window Size Validation
                if study_data.shape[1] != unified_window_samples:
                    raise ValueError(f"Time window size mismatch in Study {study_id}. Expected {unified_window_samples} samples per window (after resampling), got {study_data.shape[1]}.")
                
                # Feature Dimension Verification
                if study_data.shape[2] != len(unified_feature_names):
                    raise ValueError(f"Data matrix feature dimension ({study_data.shape[2]}) does not match feature name list length ({len(unified_feature_names)}) in Study {study_id}.")
                
                # Label Dimension Verification
                if study_labels.shape[1] != len(unified_label_names):
                    raise ValueError(f"Label matrix dimension ({study_labels.shape[1]}) does not match label name list length ({len(unified_label_names)}) in Study {study_id}.")
            
            # Prefix participant identifiers to securely prevent identity duplication
            prefixed_ids = np.array([f"{study_id}_{pid}" for pid in p_ids])

            # Filer for minimum samples per participants
            filtered_data, filtered_labels, filtered_ids = self._filter_by_minimum_samples(study_data, study_labels, prefixed_ids)

            # Accumulate processed arrays
            all_study_data.append(filtered_data)
            all_study_labels.append(filtered_labels)
            all_study_ids.append(filtered_ids)

        if not all_study_data:
            raise ValueError("Multi-study extraction loop completed with zero matching valid data records.")

        # Final Matrix Consolidation
        unified_data = np.concatenate(all_study_data, axis=0)
        unified_labels = np.concatenate(all_study_labels, axis=0)
        unified_ids = np.concatenate(all_study_ids, axis=0)

        # Logging
        logger.info("Data Extraction complete.")
        logger.info(f"Samples: {unified_data.shape}")
        logger.info(f"Labels: {unified_labels.shape}")
        logger.info(f"Participants: {len(np.unique(unified_ids))}")
        logger.info(f"Data Mask (1=Continuous): {unified_mask}")
        logger.info(f"Feature Names: {unified_feature_names}")
        logger.info(f"Label Names: {unified_label_names}")
        logger.info(f"Cardinalities: {unified_cardinalities}")

        return unified_data, unified_labels, unified_ids, unified_mask, unified_feature_names, unified_label_names, unified_cardinalities
