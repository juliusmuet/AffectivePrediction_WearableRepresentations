import logging

import numpy as np
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.base import clone


logger = logging.getLogger(__name__)


class FeatureScaler:
    """
    Scales continuous features using global or participant-specific baselines.

    Supports full-dataset normalization, within-subject temporal splits, between-subject splits with participant-specific calibration, 
    and participant fine-tuning. Discrete features are preserved unchanged.

    Scaling parameters are fitted independently within each method call. No fitted scaler state is retained between calls.

    Args:
        method (str): Scaling strategy. Supported values are "standard" (StandardScaler) and "robust" (RobustScaler).
            Defaults to "standard".
        calibration_ratio (float): Fraction of initial records used to calibrate unseen participants (validation and test) 
            in between-subject splits. Defaults to 0.50 (50%).
        min_calibration_samples (int): Minimum absolute samples required for calibration in between-subject splits. 
            If total samples of participant is smaller, all available samples of participant are used. Defaults to 5.
    """
    def __init__(
        self,
        method: str = "standard",
        calibration_ratio: float = 0.50,
        min_calibration_samples: int = 5,
    ):
        if method.lower() == 'standard':
            self.base_scaler = StandardScaler()
        elif method.lower() == 'robust':
            self.base_scaler = RobustScaler()
        else:
            raise ValueError("Method must be 'standard' or 'robust'")
            
        if not (0.0 < calibration_ratio <= 1.0):
            raise ValueError("calibration_ratio must be strictly in (0.0, 1.0]")
        if min_calibration_samples < 1:
            raise ValueError("min_calibration_samples must be at least 1")

        self.method = method.lower()
        self.calibration_ratio = calibration_ratio
        self.min_calibration_samples = min_calibration_samples
        logger.info(f"Initialized {self} \n")


    def __str__(self):
        """Returns a string representation of the feature scaler configuration."""
        return f"FeatureScaler(method='{self.method}', calibration_ratio={self.calibration_ratio}, min_calibration_samples={self.min_calibration_samples})"


    def _calculate_calibration_cutoff(self, total_samples: int) -> int:
        """
        Determines the number of records used for participant calibration.

        Uses the configured calibration fraction while enforcing the minimum calibration size whenever enough records are available.

        Args:
            total_samples (int): Amount of samples of participant.

        Returns:
            int: Amount of samples used for fitting the participant's scaler.
        """
        if total_samples <= self.min_calibration_samples:
            return total_samples  # Use all available samples
        
        # Otherwise, calculate calibration_ratio% samples but guarantee it doesn't drop below min_calibration_samples
        percent_cutoff = int(np.floor(total_samples * self.calibration_ratio))
        return max(self.min_calibration_samples, percent_cutoff)


    def _transform_subset(
        self,
        data: np.ndarray,
        p_ids: np.ndarray,
        continuous_indices: np.ndarray,
        scalers_dict: dict,
        global_scaler,
        W: int,
    ) -> np.ndarray:
        """
        Applies fitted participant-specific or global fallback scalers.

        Continuous features are transformed independently for each participant. If a participant-specific scaler is unavailable,
        the global training-data scaler is used instead. Discrete features remain unchanged.

        Args:
            data (np.ndarray): The raw data segment to scale of shape (N_subset, Window_Size, Feature_Dim).
            p_ids (np.ndarray): Participant IDs corresponding to the subset of shape (N_subset,).
            continuous_indices (np.ndarray): Extracted index positions where binary_mask == 1.
            scalers_dict (dict): Dictionary mapping participant IDs to their fitted scalers.
            global_scaler (sklearn.base.BaseEstimator): Fitted global scaler used as a fallback.
            W (int): The temporal window size dimension length.

        Returns:
            np.ndarray: The data subset with scaled continuous features and unaltered discrete features of shape (N_subset, Window_Size, Feature_Dim).
        """
        normed_data = data.copy()
        unique_p_ids = np.unique(p_ids)
        
        for p_id in unique_p_ids:
            idx = np.where(p_ids == p_id)[0]
            p_continuous = normed_data[idx][:, :, continuous_indices].reshape(-1, len(continuous_indices))
            
            # Retrieve specific dictionary scaler; fallback to global baseline if missing
            scaler = scalers_dict.get(p_id, global_scaler)
            scaled_features = scaler.transform(p_continuous)
            
            # Reshape data back into 3D shape and insert into data array
            normed_data[idx[:, None, None], np.arange(W)[:, None], continuous_indices] = (
                scaled_features.reshape(len(idx), W, len(continuous_indices))
            )
        return normed_data


    def process_full_dataset(
        self,
        data: np.ndarray,
        p_ids: np.ndarray,
        binary_mask: np.ndarray,
    ) -> np.ndarray:
        """
        Fits participant-specific scalers on the complete dataset.

        A global scaler is also fitted on all available samples and serves as a fallback for participants without an individual scaler.
        Each participant's scaler is fitted using all of their available records, including all temporal windows.

        Args:
            data (np.ndarray): The complete dataset features of shape (N, Window_Size, Feature_Dim).
            p_ids (np.ndarray): Participant IDs list of shape (N,).
            binary_mask (np.ndarray): Mask of shape (Feature_Dim,) where 1=continuous, 0=discrete.

        Returns:
            np.ndarray: Fully normalized array of shape (N, Window_Size, Feature_Dim).
        """
        if len(binary_mask) != data.shape[2]:
            raise ValueError("Binary mask length must match feature channel counts.")
            
        N, W, F = data.shape    # Get window size
        continuous_indices = np.where(binary_mask == 1)[0]  # Indices of continuous features
        
        # Edge case: No continuous metrics found to normalize
        if len(continuous_indices) == 0:
            return data.copy()

        # Instantiate scalers
        participant_scalers = {}
        global_scaler = clone(self.base_scaler)

        logger.info("Executing Scaling of whole Dataset...")

        # Establish global baseline fallback using all data
        # Get continuous features and flatten over time for fitting
        flat_all_continuous = data[:, :, continuous_indices].reshape(-1, len(continuous_indices))
        global_scaler.fit(flat_all_continuous)

        # Fit individual scalers for each participant using all of their data
        unique_p_ids = np.unique(p_ids)
        for p_id in unique_p_ids:
            # Get continuous features of participant and flatten over time for fitting
            idx = np.where(p_ids == p_id)[0]
            p_continuous = data[idx][:, :, continuous_indices].reshape(-1, len(continuous_indices))
            
            scaler = clone(self.base_scaler)
            scaler.fit(p_continuous)
            participant_scalers[p_id] = scaler

        # Transform and return data
        return self._transform_subset(data, p_ids, continuous_indices, participant_scalers, global_scaler, W)
    

    def scale_split(
        self,
        data: np.ndarray,
        p_ids: np.ndarray,
        train_idx: np.ndarray,
        val_idx: np.ndarray,
        test_idx: np.ndarray,
        binary_mask: np.ndarray,
        mode_between: bool,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Fits and applies scalers independently within a CV split.

        In between-subject mode, participant-specific scalers are fitted using all training records for training participants. 
        Previously unseen validation and test participants are calibrated using only their initial records, as determined by the 
        configured calibration ratio and minimum sample count.

        In within-subject mode, participant-specific scalers are fitted using only each participant's training records.
        The global fallback is fitted on all training records.

        Calibration for unseen participants uses the order of records in the corresponding validation or test subset. 
        Therefore, the supplied indices must preserve chronological order when temporal calibration is intended.

        All fitted scalers are local to this method call and are discarded when it returns.

        Args:
            data (np.ndarray): The raw sensor/feature data of shape (N, Window_Size, Feature_Dim).
            p_ids (np.ndarray): Participant IDs corresponding to each window of shape (N,).
            train_idx, val_idx, test_idx (np.ndarray): Indices of data for training, validation and testing.
            binary_mask (np.ndarray): Array mapping 1 for continuous features, 0 for discrete of shape (Feature_Dim,).
            mode_between (bool): True if Between-Subject (GroupKFold), False if Within-Subject (Temporal).

        Returns:
            tuple: (normed_train, normed_val, normed_test) arrays matching input subset shapes.
        """
        if len(binary_mask) != data.shape[2]:
            raise ValueError("Binary mask length must match feature channel counts.")
            
        N, W, F = data.shape    # Get window size
        continuous_indices = np.where(binary_mask == 1)[0]  # Indices of continuous features
        
        # Slice target subsets immediately
        train_data, train_p_ids = data[train_idx], p_ids[train_idx]
        val_data, val_p_ids = data[val_idx], p_ids[val_idx]
        test_data, test_p_ids = data[test_idx], p_ids[test_idx]

        # Edge case: No continuous metrics found to normalize
        if len(continuous_indices) == 0:
            return train_data.copy(), val_data.copy(), test_data.copy()

        # Instantiate scalers
        participant_scalers = {}
        global_scaler = clone(self.base_scaler)

        # Mode A: Between-Subject Scaling (Unseen Participants in Val / Test)
        if mode_between:
            logger.info("Executing Uniform Between-Subject Dynamic Scaling Strategy...")

            # Fit Global Baseline on all training data across all training participants
            flat_train_continuous = train_data[:, :, continuous_indices].reshape(-1, len(continuous_indices))
            global_scaler.fit(flat_train_continuous)
            
            # Fit individual scalers for Training participants using all their training data
            for p_id in np.unique(train_p_ids):
                p_idx = np.where(train_p_ids == p_id)[0]
                p_train_data = train_data[p_idx][:, :, continuous_indices].reshape(-1, len(continuous_indices))
                
                scaler = clone(self.base_scaler)
                scaler.fit(p_train_data)
                participant_scalers[p_id] = scaler

            # Fit individual scalers for Val & Test participants using only their Calibration Window
            for subset_data, subset_p_ids in [(val_data, val_p_ids), (test_data, test_p_ids)]:
                for p_id in np.unique(subset_p_ids):
                    if p_id in participant_scalers:
                        continue

                    p_idx = np.where(subset_p_ids == p_id)[0]
                    cutoff = self._calculate_calibration_cutoff(len(p_idx))
                    
                    # Take calibration window for fitting
                    calib_slice = subset_data[p_idx[:cutoff]]
                    flat_calib_data = calib_slice[:, :, continuous_indices].reshape(-1, len(continuous_indices))

                    scaler = clone(self.base_scaler)
                    scaler.fit(flat_calib_data)
                    participant_scalers[p_id] = scaler

        # Mode B: Within-Subject Scaling (Temporal Splits per Participant)
        else:
            logger.info("Executing Within-Subject Scaling Strategy...")

            # Global fallback fit on all training data
            flat_train_continuous = train_data[:, :, continuous_indices].reshape(-1, len(continuous_indices))
            global_scaler.fit(flat_train_continuous)
            
            # Fit individual scalers per participant using their training data
            for p_id in np.unique(train_p_ids):
                p_idx = np.where(train_p_ids == p_id)[0]
                p_train_data = train_data[p_idx][:, :, continuous_indices].reshape(-1, len(continuous_indices))
                
                scaler = clone(self.base_scaler)
                scaler.fit(p_train_data)
                participant_scalers[p_id] = scaler

        # Perform data transformation / normalization
        normed_train = self._transform_subset(train_data, train_p_ids, continuous_indices, participant_scalers, global_scaler, W)
        normed_val = self._transform_subset(val_data, val_p_ids, continuous_indices, participant_scalers, global_scaler, W)
        normed_test = self._transform_subset(test_data, test_p_ids, continuous_indices, participant_scalers, global_scaler, W)
        
        return normed_train, normed_val, normed_test


    def scale_finetune(
        self,
        data: np.ndarray,
        p_ids: np.ndarray,
        train_idx: np.ndarray,
        val_idx: np.ndarray,
        test_idx: np.ndarray,
        binary_mask: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Fits scalers on fine-tuning training data and transforms all splits.

        Participant-specific scalers are fitted exclusively on the fine-tuning training subset. 
        The global fallback is fitted on the same training subset. 
        Both are then applied to training, validation, and testing data without using evaluation records during fitting.

        Args:
            data (np.ndarray): The raw sensor/feature data of shape (N, Window_Size, Feature_Dim).
            p_ids (np.ndarray): Participant IDs corresponding to each window of shape (N,).
            train_idx (np.ndarray): Indices corresponding to the participant's fine-tuning calibration subset (`p_train_idx`).
            val_idx (np.ndarray): Indices for validation subset (can be identical to test_idx if unused).
            test_idx (np.ndarray): Indices corresponding to the participant's evaluation subset (`p_test_idx`).
            binary_mask (np.ndarray): Mask mapping 1 for continuous features, 0 for discrete of shape (Feature_Dim,).

        Returns:
            tuple: (normed_train, normed_val, normed_test) normalized feature arrays matching input subset shapes.
        """
        if len(binary_mask) != data.shape[2]:
            raise ValueError("Binary mask length must match feature channel counts.")
            
        N, W, F = data.shape
        continuous_indices = np.where(binary_mask == 1)[0]

        # Slice target subsets
        train_data, train_p_ids = data[train_idx], p_ids[train_idx]
        val_data, val_p_ids = data[val_idx], p_ids[val_idx]
        test_data, test_p_ids = data[test_idx], p_ids[test_idx]

        # Edge case: No continuous features found
        if len(continuous_indices) == 0:
            return train_data.copy(), val_data.copy(), test_data.copy()

        logger.info("Executing Fine-Tuning Scaling Strategy...")

        # Instantiate scalers
        participant_scalers = {}
        global_scaler = clone(self.base_scaler)

        # Fit global fallback on fine-tuning train data
        flat_train_continuous = train_data[:, :, continuous_indices].reshape(-1, len(continuous_indices))
        global_scaler.fit(flat_train_continuous)

        # Fit specific participant scaler on all available fine-tuning training samples
        for p_id in np.unique(train_p_ids):
            p_idx = np.where(train_p_ids == p_id)[0]
            p_train_data = train_data[p_idx][:, :, continuous_indices].reshape(-1, len(continuous_indices))

            scaler = clone(self.base_scaler)
            scaler.fit(p_train_data)
            participant_scalers[p_id] = scaler

        # Apply parameters fitted on train_idx to all three splits
        normed_train = self._transform_subset(train_data, train_p_ids, continuous_indices, participant_scalers, global_scaler, W)
        normed_val = self._transform_subset(val_data, val_p_ids, continuous_indices, participant_scalers, global_scaler, W)
        normed_test = self._transform_subset(test_data, test_p_ids, continuous_indices, participant_scalers, global_scaler, W)

        return normed_train, normed_val, normed_test
