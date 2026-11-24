import time
import logging
import warnings
from pathlib import Path
from datetime import timedelta
from typing import Any

import numpy as np
import pandas as pd
import unisens
from tqdm import tqdm
import tsfel
import mrmr
from sklearn.preprocessing import OneHotEncoder
from scipy.signal import lfilter

from preprocessing.label_transformer import BaseTargetTransform


logger = logging.getLogger(__name__)


# Suppress the openpyxl default style warning
warnings.filterwarnings(
    "ignore", 
    category=UserWarning, 
    module="openpyxl.styles.stylesheet"
)


class DataProcessor5:
    """
    Extract and align EMA labels and movement data for Study 5.

    Processes participant-specific EMA responses and movement recordings from Excel, Unisens, Move, and Fitbit data sources. 
    Aligns sensor data with chronological EMA timestamps to generate data-label pairs for downstream machine learning applications.

    Supports five movement data representations: raw accelerometer signals,
    Move features, TSFEL features, combined TSFEL-Move features, and Fitbit features.

    Args:
        root_path: Root directory containing the participant-specific subdirectories of Study 5.
        label_transformer: Target transformation instance used to encode the extracted mood labels.
        movement_type: Movement data representation to extract. Supported values are raw, fitbit, move,
            tsfel, and tsfel-move. Defaults to raw.
        window_min: Duration of the movement data window preceding each EMA timestamp, in minutes. Defaults to 5.
        include_magnitude: Whether to calculate and include dynamic acceleration and total acceleration magnitudes. Defaults to True.
        include_jerk: Whether to calculate and include jerk features. Defaults to True.
        top_k: Maximum number of non-redundant features selected per target label during feature selection. If None, no top-k
            selection is performed. Defaults to None.
        corr_threshold: Absolute Pearson correlation threshold used for collinearity filtering. If None, correlation-based
            feature selection is not performed. Defaults to None.
        one_hot_encoding: Whether categorical Move features are one-hot encoded. If False, categorical features retain integer
            category representations. Defaults to True.
    """
    def __init__(
        self,
        root_path: str | Path,
        label_transformer: BaseTargetTransform,
        movement_type: str = "raw",
        window_min: int = 5,
        include_magnitude: bool = True,
        include_jerk: bool = True,
        top_k: int | None = None,
        corr_threshold: float | None = None,
        one_hot_encoding: bool = True,
    ) -> None:
        # Validation check of data type
        valid_movement_types = {'raw', 'fitbit', 'move', 'tsfel', 'tsfel-move'}
        if movement_type.lower() not in valid_movement_types:
            raise ValueError(f"Invalid movement_type: '{movement_type}'. Supported types are: {list(valid_movement_types)}")

        self.root_path = Path(root_path)
        self.label_transformer = label_transformer
        self.movement_type = movement_type.lower()
        self.window_min = window_min
        self.include_magnitude = include_magnitude
        self.include_jerk = include_jerk
        self.top_k = top_k
        self.corr_threshold = corr_threshold
        self.one_hot_encoding = one_hot_encoding
        self.sampling_rate_raw = 64 # Is constant
        
        # Expected mood columns
        self.mood_cols = ["Mood_EA1", "Mood_V1", "Mood_C1", "Mood_V2", "Mood_EA2", "Mood_C2"]

        logger.info(f"Initialized {self} \n")


    def __str__(self):
        """Returns a string representation of the dataset configuration."""
        return (f"DataProcessor5(root_path={self.root_path}, "
                f"label_transformer={self.label_transformer}, "
                f"movement_type={self.movement_type}, "
                f"window_min={self.window_min}, "
                f"include_magnitude={self.include_magnitude}, "
                f"include_jerk={self.include_jerk}, "
                f"top_k={self.top_k}, "
                f"corr_threshold={self.corr_threshold}, "
                f"one_hot_encoding={self.one_hot_encoding}, "
                f"sampling_rate_raw={self.sampling_rate_raw})")


    def process_all_participants(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str], list[str], list[int]]:
        """
        Execute the complete extraction pipeline across all participants.

        Iterates over the numeric participant directories, extracts EMA labels and the configured movement data, 
        and aligns both data streams temporally to generate valid data-label pairs.

        Depending on movement_type, applies raw signal preprocessing, TSFEL feature extraction, Move feature extraction, 
        Fitbit alignment, and optional global feature selection.

        Participants that cannot be processed successfully are skipped.
        Extracted participant-level arrays are concatenated into unified study-level arrays.

        Returns:
            Tuple containing
                - data: Aggregated movement data of shape (N, Window_Size, Feature_Dim).
                - labels: Corresponding target labels of shape (N, Label_Dim).
                - participant_ids: Participant identifiers of shape (N,).
                - feature_mask: Binary feature-type mask of shape (Feature_Dim,), where 1 denotes continuous and 0 denotes categorical features.
                - feature_names: Ordered names of the extracted features.
                - label_names: Ordered names of the target labels.
                - cardinalities: Cardinalities of categorical features in their corresponding feature order. Empty if no integer-
                    encoded categorical features remain.

        Notes:
            If no valid participant data can be extracted, empty arrays and empty metadata lists are returned.
            Global feature selection is applied only when both top_k and corr_threshold are configured.
        """
        all_data, all_labels, all_ids = [], [], []
        final_mask, final_feature_names, final_label_names = None, None, None
        final_cardinalities = []
        
        # Identify participant folders (numeric folder names)
        p_folders = self._get_participant_folders()

        # Iterate over participant folders
        for p_folder in tqdm(p_folders, desc="Overall Progress DataProcessor5"):
            p_id = int(p_folder.name)
            logger.info(f"--- Processing Participant {p_id} ---")
            
            try:
                # Extract Labels
                labels_df, label_names = self.extract_labels(p_folder)
                if labels_df.empty:
                    logger.warning(f"No valid EMA labels found for Participant {p_id}. Skipping.")
                    continue
                
                # Extract Movement Data and align Data and Labels
                p_x, p_y, ts_list = np.array([]), np.array([]), np.array([])
                p_mask = None
                p_feature_names = []
                p_cardinalities = []

                # Branch 1: Processing pipelines requiring raw accelerometer extraction
                if self.movement_type in {'raw', 'tsfel', 'tsfel-move'}:
                    movement_segments, raw_mask, raw_feature_names = self.extract_data_raw(p_folder)
                    if not movement_segments:
                        logger.warning(f"No raw movement data found for Participant {p_id}. Skipping.")
                        continue
                    
                    raw_X, p_y, ts_list = self.create_pairs_raw(labels_df, movement_segments)
                    if len(raw_X) == 0:
                        logger.warning(f"No Raw pairs matched for Participant {p_id}. Skipping.")
                        continue

                    if self.movement_type == 'raw':
                        p_x, p_mask, p_feature_names = raw_X, raw_mask, raw_feature_names

                    elif self.movement_type == 'tsfel':
                        p_x, p_y, p_mask, p_feature_names = self.create_pairs_tsfel(raw_X, p_y)

                    elif self.movement_type == 'tsfel-move':
                        p_x, p_y, p_mask, p_feature_names, p_cardinalities = self.create_pairs_tsfel_move(p_folder, raw_X, p_y, ts_list)

                # Branch 2: Move data
                elif self.movement_type == 'move':
                    window_end_timestamps = labels_df['timestamp']
                    p_y_labels = np.vstack(labels_df['labels'])
                    p_x, p_y, p_mask, p_feature_names, p_cardinalities = self.create_pairs_move(p_folder, window_end_timestamps, p_y_labels)
                    if len(p_x) == 0:
                        logger.warning(f"No Move features extracted for Participant {p_id}. Skipping.")
                        continue

                # Branch 3: Fitbit data
                elif self.movement_type == 'fitbit':
                    p_x, p_y, p_mask, p_feature_names = self.create_pairs_fitbit(p_folder, labels_df)
                    if len(p_x) == 0:
                        logger.warning(f"No FitBit features extracted for Participant {p_id}. Skipping.")
                        continue
                
                # Add participant's data to full data list
                if len(p_x) > 0:
                    all_data.append(p_x)
                    all_labels.append(p_y)
                    all_ids.append(np.full(len(p_y), p_id))
                    if final_mask is None:
                        final_mask = p_mask
                        final_feature_names = p_feature_names
                        final_label_names = label_names
                        final_cardinalities = p_cardinalities
                    logger.info(f"Extracted Pairs: {len(p_y)} for Participant {p_id}.")
                else:
                    logger.warning(f"No temporal matches found for Participant {p_id}.")           

            except Exception:
                logger.exception(f"Failed to process participant {p_id}.")

        if not all_data:
            logger.warning(f"No data extracted from study 5.")
            return np.array([]), np.array([]), np.array([]),  np.array([]), [], [], []

        # Concatenate all participant data
        final_X = np.concatenate(all_data)
        final_y = np.concatenate(all_labels)
        final_ids = np.concatenate(all_ids)

        # Global Feature Selection Step
        if self.top_k is not None and self.corr_threshold is not None:
            if self.movement_type == 'tsfel':
                logger.info("Running global feature selection on concatenated TSFEL features...")
                final_X, final_feature_names, selected_indices = self.select_features_collinear_mrmr(
                    final_X, final_y, final_feature_names, self.top_k, self.corr_threshold
                )
                final_mask = final_mask[selected_indices]
                logger.info(f"TSFEL Global Feature Selection: Reduced to {len(final_feature_names)} features.")

            elif self.movement_type == 'tsfel-move':
                logger.info("Running global feature selection on TSFEL sub-array in TSFEL-Move dataset...")
                # Infer number of TSFEL columns
                if self.one_hot_encoding:
                    n_tsfel_cols = len(final_feature_names) - 28
                else:
                    n_tsfel_cols = len(final_feature_names) - 6
                
                tsfel_X = final_X[:, :, :n_tsfel_cols]
                move_X = final_X[:, :, n_tsfel_cols:]
                
                tsfel_names = final_feature_names[:n_tsfel_cols]
                move_names = final_feature_names[n_tsfel_cols:]
                
                tsfel_mask = final_mask[:n_tsfel_cols]
                move_mask = final_mask[n_tsfel_cols:]

                # Select features only from the TSFEL block
                selected_tsfel_X, selected_tsfel_names, selected_indices = self.select_features_collinear_mrmr(
                    tsfel_X, final_y, tsfel_names, self.top_k, self.corr_threshold
                )
                selected_tsfel_mask = tsfel_mask[selected_indices]

                # Recombine selected TSFEL features with Move features
                final_X = np.concatenate([selected_tsfel_X, move_X], axis=2)
                final_feature_names = selected_tsfel_names + move_names
                final_mask = np.concatenate([selected_tsfel_mask, move_mask])
                logger.info(f"TSFEL-Move Global Selection: TSFEL features reduced from {n_tsfel_cols} to {len(selected_tsfel_names)} features.")

        # Logging
        logger.info("Successfully extracted data from study 5:")
        logger.info(f"  - X shape (Data):   {final_X.shape}") 
        logger.info(f"  - y shape (Labels): {final_y.shape}")
        logger.info(f"  - IDs shape:        {final_ids.shape}")
        logger.info(f"  - Participants:     {len(np.unique(final_ids))}")
        logger.info(f"  - Data Mask (1=Continuous): {final_mask}")
        logger.info(f"  - Feature Names: {final_feature_names}")
        logger.info(f"  - Label Names: {final_label_names}")
        logger.info(f"  - Cardinalities: {final_cardinalities}")
        
        return final_X, final_y, final_ids, final_mask, final_feature_names, final_label_names, final_cardinalities

    
    def _get_participant_folders(
        self,
        retries: int = 7,
        delay: int = 180,
    ) -> list[Path]:
        """
        Retrieve participant directories with retry handling.

        Identifies numeric subdirectories in the study root directory and sorts them by their integer participant identifiers.

        Retries directory access when a PermissionError occurs, which can happen with intermittently unavailable network file systems.

        Args:
            retries: Maximum number of directory access attempts. Defaults to 7.
            delay: Waiting time between failed attempts, in seconds. Defaults to 180.

        Returns:
            Numerically sorted list of participant directory paths.

        Raises:
            PermissionError: If directory access remains unavailable after all configured attempts.
        """
        for attempt in range(retries):
            try:
                return sorted(
                    [f for f in self.root_path.iterdir() if f.is_dir() and f.name.isdigit()],
                    key=lambda x: int(x.name)
                )
            except PermissionError as e:
                if attempt == retries - 1:
                    logger.error(f"Failed to read directory {self.root_path} after {retries} attempts.")
                    raise e
                logger.warning(f"Storage access error on attempt {attempt + 1}/{retries}. Retrying in {delay} seconds...")
                time.sleep(delay)


    def extract_labels(self, p_path: Path) -> tuple[pd.DataFrame, list[str]]:
        """
        Extract and preprocess EMA mood labels for a participant.

        Reads Excel files from the participant's EMA directory, selects valid EMA entries, validates the six raw mood subscales, 
        and transforms them into three composite mood dimensions.

        Extracted entries are chronologically sorted and duplicate timestamps are removed.

        Args:
            p_path: Path to the participant's root directory.

        Returns:
            Tuple containing
                - labels_df: DataFrame with the columns timestamp and labels. The timestamp column contains EMA timestamps, 
                    and each entry in labels is an array of shape (Label_Dim,).
                - label_names: Ordered names of the three target dimensions: Energetic Arousal, Valence, and Calmness.
            
            If the participant's EMA directory does not exist, returns an empty DataFrame and an empty list.

        Notes:
            EMA entries containing missing mood values or values outside the permitted range of 0 to 100 are excluded.
        """
        # Construct paths to EMA folders
        xs_path = p_path / "AA-Phase" / "XS"
        if not xs_path.exists():
            return pd.DataFrame(), []
        xlsx_files = sorted(xs_path.glob("*.xlsx"))

        extracted_rows = []
        skipped_count = 0

        # Iterate over EMA xlsx files
        for file in xlsx_files:
            try:
                df = pd.read_excel(file)
                # Filter for EMA forms only
                ema_df = df[df["Form"].str.contains("EMA", na=False)].copy()
                
                # Iterate over EMA rows
                for _, row in ema_df.iterrows():
                    # Extract mood values
                    mood_values = row[self.mood_cols]
                    # Validate mood ranges and check for NaNs
                    if mood_values.isnull().any() or not mood_values.between(0, 100).all():
                        skipped_count += 1
                        continue

                    # Process 6 raw label values into 3 dimensions [EA, V, C]
                    processed_mood = self.calculate_mood_dimensions(mood_values)
                    if len(processed_mood) == 0:
                        return pd.DataFrame(), []
                    
                    # Construct timestamp assuming dd.mm.yyyy hh:mm:ss format
                    ts = pd.to_datetime(row['Form_start_date'])
                    
                    # Save EMA values with timestamp
                    extracted_rows.append({
                        "timestamp": ts,
                        "labels": processed_mood
                    })

            except Exception as e:
                logger.error(f"Error reading label file {file.name}: {e}")

        logger.info(f"Labels: {len(extracted_rows)} extracted, {skipped_count} skipped for Participant {int(p_path.name)}.")

        return pd.DataFrame(extracted_rows).sort_values("timestamp").drop_duplicates('timestamp'), ["Energetic Arousal", "Valence", "Calmness"]
    

    def calculate_mood_dimensions(self, raw_mood: pd.Series) -> np.ndarray:
        """
        Transforms 6 raw mood sub-scales into 3 composite dimensions using inversion formulas.
        
        The formulas account for the reverse-coded nature of the second item in each pair
        to create a balanced score for Energetic Arousal (EA), Valence (V), and Calmness (C).

        The 6 raw mood sub-scales are:
        EA1 – tired to awake; V1 – content to discontent; C1 – agitated to calm;
        EA2 – full energy to without energy; V2 – unwell to well; C2 – relaxed to tense

        Applies the configured target transformation to the resulting composite scores.

        Args:
            raw_mood: Series containing the six expected mood values: Mood_EA1, Mood_EA2, Mood_V1, Mood_V2, Mood_C1, Mood_C2.

        Returns:
            Encoded mood target array of shape (3,), ordered as Energetic Arousal, Valence, and Calmness.
            Returns an empty float32 array if an expected mood key is missing.

        Notes:
            The composite scores are calculated as:
            - Energetic Arousal: (EA1 + 100 - EA2) / 2
            - Valence: (V2 + 100 - V1) / 2
            - Calmness: (C1 + 100 - C2) / 2
        """
        try:
            ea = (raw_mood["Mood_EA1"] + 100 - raw_mood["Mood_EA2"]) / 2
            v  = (raw_mood["Mood_V2"] + 100 - raw_mood["Mood_V1"]) / 2
            c  = (raw_mood["Mood_C1"] + 100 - raw_mood["Mood_C2"]) / 2

            processed_labels = np.array([ea, v, c], dtype=np.float32)

            # Apply label transformation
            processed_labels = self.label_transformer.encode(processed_labels)
            
            return processed_labels
            
        except KeyError as e:
            logger.error(f"Missing expected mood column for calculation: {e}")
            return np.array([], dtype=np.float32)


    def preprocess_acc_segment(
        self,
        data: np.ndarray,
        cutoff_gravity: float = 0.5,
        cutoff_noise: float = 20.0,
    ) -> tuple[np.ndarray, list[str]]:
        """
        Preprocess a three-axis accelerometer recording.

        Applies low-pass EMA IIR filtering to separate gravity and dynamic acceleration components, attenuate high-frequency noise, 
        and optionally calculate jerk and acceleration magnitude features.

        Args:
            data: Raw accelerometer array of shape (Window Size, 3), containing the x, y, and z acceleration components.
            cutoff_gravity: Cutoff frequency in Hz for gravity isolation. Defaults to 0.5.
            cutoff_noise: Cutoff frequency in Hz for high-frequency noise attenuation. Defaults to 20.0.

        Returns:
            Tuple containing
                - feature_matrix: Preprocessed acceleration array of shape (Window_Size, Feature_Dim).
                - feature_names: Ordered names corresponding to the generated feature channels.

        Notes:
            The base channels contain dynamic and gravity acceleration.
            Jerk and magnitude channels are added according to the corresponding instance configuration flags.
        """
        fs = float(self.sampling_rate_raw)

        # Compute Alpha Values Dynamically from Cutoff Frequencies
        alpha_grav = 1.0 - np.exp(-2.0 * np.pi * cutoff_gravity / fs)
        alpha_noise = 1.0 - np.exp(-2.0 * np.pi * cutoff_noise / fs)

        # Low-pass Noise Filtering
        b_noise, a_noise = [alpha_noise], [1.0, -(1.0 - alpha_noise)]
        clean_data = lfilter(b_noise, a_noise, data, axis=0)

        # Gravity Isolation
        b_grav, a_grav = [alpha_grav], [1.0, -(1.0 - alpha_grav)]
        gravity_data = lfilter(b_grav, a_grav, clean_data, axis=0)

        # Dynamic Acceleration
        dynamic_data = clean_data - gravity_data

        # Base features (always included)
        channels = [dynamic_data, gravity_data]
        feature_names = ["dyn_x", "dyn_y", "dyn_z", "grav_x", "grav_y", "grav_z"]

        # Conditional: Jerk signal calculation
        if self.include_jerk:
            dt = 1.0 / self.sampling_rate_raw
            jerk_data = np.gradient(dynamic_data, dt, axis=0)
            channels.append(jerk_data)
            feature_names.extend(["jerk_x", "jerk_y", "jerk_z"])

        # Conditional: Magnitude calculations
        if self.include_magnitude:
            # Notice [:, None] to ensure 2D column arrays (samples, 1)
            dynamic_mag = np.sqrt(dynamic_data[:, 0]**2 + dynamic_data[:, 1]**2 + dynamic_data[:, 2]**2)[:, None]
            total_mag = np.sqrt(data[:, 0]**2 + data[:, 1]**2 + data[:, 2]**2)[:, None]
            channels.extend([dynamic_mag, total_mag])
            feature_names.extend(["dyn_mag", "total_mag"])

        # Concatenate feature channels along column axis
        feature_matrix = np.concatenate(channels, axis=1)

        return feature_matrix, feature_names


    def extract_data_raw(self, p_path: Path) -> tuple[list[dict[str, Any]], np.ndarray, list[str]]:
        """
        Extract and preprocess raw accelerometer recordings.

        Locates Unisens recordings within the participant's raw movement directory, reads available acc.bin files, 
        and applies the configured accelerometer preprocessing pipeline.

        Duplicate recordings are identified by their start timestamps. Successfully extracted segments are sorted chronologically.

        Args:
            p_path: Path to the participant's root directory.

        Returns:
            Tuple containing
                - segments: Chronologically sorted movement recordings. Each segment contains start, end, and data.
                - feature_mask: Binary mask of shape (F,), containing ones because all extracted accelerometer features are continuous.
                - feature_names: Ordered names of the extracted features.

            If the raw movement directory does not exist, empty collections and an empty mask are returned.
        """
        # Construct path to raw data folder 
        raw_dir = p_path / "AA-Phase" / "move4" / "Raw-Daten"
        if not raw_dir.exists():
            return [], np.array([]), []

        segments = []
        seen_starts = set() # Track unique start times to skip duplicates
        skipped_count = 0
        feature_names = []
        
        # Find folders containing "result" in their names
        result_roots = [f for f in raw_dir.iterdir() if f.is_dir() and "result" in f.name.lower()]
        # Recursively find any subfolder containing unisens.xml
        result_folders = []
        for root in result_roots:
            for xml_path in root.rglob("unisens.xml"):
                result_folders.append(xml_path.parent)

        # Iterate over all unisens result folders
        for folder in result_folders:
            try:
                u = unisens.Unisens(str(folder))
                acc_entry = u["acc.bin"]
                if acc_entry:
                    start_ts = pd.to_datetime(u.timestampStart) # Get start timestamp
                    
                    # Check for duplicate start times
                    if start_ts in seen_starts:
                        logger.warning(f"Skipping duplicate segment found in: {folder.name}")
                        skipped_count += 1
                        continue

                    # Load 3-axis accelerometer data
                    raw_data = np.asarray(acc_entry.get_data()).transpose() #  Shape (samples, 3)

                    # Apply preprocessing and compute features
                    data, feature_names_seg = self.preprocess_acc_segment(raw_data)
                    if not feature_names:
                        feature_names = feature_names_seg
                    
                    # Add data segment extracted from this unisens results folder
                    segments.append({
                        "start": start_ts,
                        "data": data,
                        "end": start_ts + timedelta(seconds=(len(data)-1)/self.sampling_rate_raw)
                    })
                    seen_starts.add(start_ts)
                
                else:
                    logger.warning(f"No acc.bin file found in {folder.name}")

            except Exception as e:
                logger.error(f"Error reading Unisens in {folder.name}: {e}")

        # Ensure chronological order across different bin files
        sorted_segments = sorted(segments, key=lambda x: x['start'])

        # Create binary mask for continuous data
        num_features = len(feature_names)
        binary_mask = np.ones(num_features, dtype=int)

        # Logging
        logger.info(f"Data Segments: {len(sorted_segments)} extracted, {skipped_count} skipped.")
        for i, seg in enumerate(sorted_segments):
            logger.info(f"  Segment {i+1}: {seg['start']} to {seg['end']}")

        return sorted_segments, binary_mask, feature_names


    def create_pairs_raw(
        self,
        labels_df: pd.DataFrame,
        segments: list[dict[str, Any]],
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Align raw accelerometer recordings with EMA labels.

        Extracts fixed-duration movement windows preceding each EMA timestamp. Multiple overlapping recording segments 
        are combined to reconstruct the complete window.

        Windows containing missing sensor samples or entirely zero-valued movement data are excluded.

        Args:
            labels_df: EMA DataFrame containing timestamp and labels columns.
            segments: Chronologically sorted accelerometer recording segments.

        Returns:
            Tuple containing
                - data: Aligned raw movement windows of shape (N, Window_Size, Feature_Dim).
                - labels: Corresponding target labels of shape (N, Label_Dim).
                - timestamps: EMA timestamps associated with the valid data-label pairs, of shape (N,).
        """
        X, y, ts_list = [], [], []
        samples_per_window = int(self.window_min * 60 * self.sampling_rate_raw)
        num_features = segments[0]['data'].shape[1] if segments else 3
        zero_drop_count = 0
        
        # Iterate over labels
        for _, row in labels_df.iterrows():
            # Get label time window
            label_ts = row['timestamp']
            window_start = label_ts - timedelta(minutes=self.window_min)

            # Initialize a canvas for this window using NaNs to detect gaps
            window_canvas = np.full((samples_per_window, num_features), np.nan)
            data_found = False
            
            # Find data segment(s) that contain this time range
            for seg in segments:
                # Check if segment overlaps with time window [window_start, label_ts]
                if seg['start'] < label_ts and seg['end'] > window_start:
                    # Calculate indices within this specific segment
                    offset_sec = (window_start - seg['start']).total_seconds()
                    start_idx = int(offset_sec * self.sampling_rate_raw)
                    end_idx = int(start_idx + samples_per_window)
                    
                    # Clipping to available data and extract data
                    actual_start = max(0, start_idx)
                    actual_end = min(len(seg['data']), end_idx)
                    chunk = seg['data'][actual_start:actual_end]

                    # Calculate position within the canvas
                    canvas_start = max(0, -start_idx)
                    canvas_end = canvas_start + len(chunk)
                    
                    # Fill the canvas at the precise temporal position
                    window_canvas[canvas_start:canvas_end] = chunk
                    data_found = True

            if not data_found:
                logger.warning(f"No accelerometer data found for label {row['labels']} in time window [{window_start},{label_ts}].")
                continue
                
            # Check for gaps (NaNs) in the canvas before adding data to output
            if not np.isnan(window_canvas).any():
                # Check if movement data is all 0
                if np.all(window_canvas == 0):
                    zero_drop_count += 1
                    continue
                X.append(window_canvas)
                y.append(row['labels'])
                ts_list.append(label_ts)
            else:
                num_missing = np.isnan(window_canvas).any(axis=1).sum()
                logger.warning(f"Not enough accelerometer data found for label {row['labels']} in time window [{window_start},{label_ts}]. Needed {samples_per_window} samples, got {samples_per_window-num_missing} samples.")
            
        if zero_drop_count > 0:
            logger.info(f"Dropped {zero_drop_count} pairs because the movement data was entirely zeros.")

        logger.info(f"Raw Pairs: {len(y)} matched, {len(labels_df)-len(y)} skipped.")
        
        return np.array(X), np.array(y), np.array(ts_list)
    

    def get_feature_window(
        self,
        end_ts: pd.Timestamp,
        move_df: pd.DataFrame,
        cols: list[str],
    ) -> np.ndarray:
        """
        Extract a feature window of length window_min ending at a given timestamp.

        Selects feature observations within the configured lookback interval and orders them chronologically. 
        Incomplete windows are represented by NaN-filled arrays to preserve a consistent output shape.

        Args:
            end_ts: Timestamp marking the end of the extraction window.
            move_df: DataFrame containing a timestamp column and the requested feature columns.
            cols: Ordered names of the feature columns to extract.

        Returns:
            Feature window of shape (window_min, len(cols)).
            Returns a NaN-filled array if the number of observations does not match the configured window duration.
        """
        start_ts = end_ts - timedelta(minutes=self.window_min)
        mask = (move_df['timestamp'] > start_ts) & (move_df['timestamp'] <= end_ts)
        window = move_df.loc[mask].sort_values('timestamp')
        
        # If the window is incomplete, fill with NaNs to keep (window_min, features) shape
        if len(window) == self.window_min:
            return window[cols].values
        else:
            return np.full((self.window_min, len(cols)), np.nan)


    def extract_data_move(self, p_path: Path) -> tuple[pd.DataFrame, np.ndarray, list[str], list[int]]:
        """
        Extract and encode Move features from Unisens result directories.

        Locates Results.xlsx files, reads their movement measurements, removes duplicate recordings based on their start timestamps,
        and aligns the resulting feature streams chronologically.

        Continuous features are retained in their original representation.
        Categorical features are either one-hot encoded or remapped to sequential integer categories.

        Args:
            p_path: Path to the participant's root directory.

        Returns:
            Tuple containing
                - move_data: Chronologically sorted feature DataFrame, including a timestamp column.
                - feature_mask: Binary feature-type mask of shape (Feature_Dim,), where 1 denotes continuous and 0 denotes categorical features.
                - feature_names: Ordered names of the encoded features.
                - cardinalities: Cardinalities of integer-encoded categorical features. Empty when one-hot encoding is enabled.

            If no valid Move recordings are available, empty data and metadata structures are returned.
        """
        # Construct path to raw data folder 
        raw_dir = p_path / "AA-Phase" / "move4" / "Raw-Daten"
        if not raw_dir.exists():
            return pd.DataFrame(), np.array([]), [], []

        all_move_data = []
        seen_starts = set()
        move_feature_names = []
        binary_mask = None
        cardinalities = []
        
        # Locate Unisens folders containing Results.xlsx
        result_roots = [f for f in raw_dir.iterdir() if f.is_dir() and "result" in f.name.lower()]
        result_folders = []
        for root in result_roots:
            for xml_path in root.rglob("unisens.xml"):
                if (xml_path.parent / "Results.xlsx").exists():
                    result_folders.append(xml_path.parent)

        # Columns configuration
        continuous_cols = ["MovementAcceleration [g]", "StepCount [steps]", "tempMean []"]
        categorical_cols = {
            "ActivityClass []": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 99],
            "BodyPosition []": [0, 1, 2, 3, 4, 5, 6, 7, 99],
            "Sedentariness []": [0, 1, 2]
        }
        
        # Precompute mappings from original values to strictly sequential 0-indexed integers, e.g., {0: 0, 1: 1, ..., 11: 11, 99: 12}
        category_mappings = {
            col: {val: i for i, val in enumerate(cats)} for col, cats in categorical_cols.items()
        }

        # Determine cardinalities order
        if not self.one_hot_encoding:
            cardinalities = [len(cats) for cats in categorical_cols.values()]
        else:
            cardinalities = []  # No categorical integer features remain when One-Hot encoded

        # Iterate over all unisens folders containing a Results.xlsx file
        for folder in result_folders:
            try:
                # Duplicate check using unisens start timestamp
                u = unisens.Unisens(str(folder))
                start_ts = pd.to_datetime(u.timestampStart)
                if start_ts in seen_starts:
                    logger.warning(f"Skipping duplicate Move results in: {folder.name}")
                    continue
                
                # Read Results.xlsx and process timestamps
                df = pd.read_excel(folder / "Results.xlsx")
                df['timestamp'] = pd.to_datetime(df["Date abs [yyyy-mm-dd]"])

                # Remap original categorical values to sequential 0..N-1 indices
                for col, mapping in category_mappings.items():
                    if col in df.columns:
                        df[col] = df[col].map(mapping)
                
                # Setup list of dataframes for concatenation
                encoded_parts = [df[['timestamp']]]
                mask_parts = []
                current_file_feature_names = []

                # Append Continuous Columns
                encoded_parts.append(df[continuous_cols])
                mask_parts.append(np.ones(len(continuous_cols))) # 1 for continuous
                current_file_feature_names.extend(continuous_cols)

                # Append Categorical Columns
                if self.one_hot_encoding:   # One-Hot Encoding for most NNs (exceptions e.g. FT-Transformer)
                    for col, cats in categorical_cols.items():
                        seq_cats = list(range(len(cats)))  # Use sequential categories [0..len(cats)-1]
                        encoder = OneHotEncoder(categories=[seq_cats], sparse_output=False, handle_unknown='ignore')
                        out = encoder.fit_transform(df[[col]])
                        col_names = [f"{col}_{c}" for c in cats]
                        encoded_parts.append(pd.DataFrame(out, columns=col_names, index=df.index))
                        mask_parts.append(np.zeros(len(cats))) # 0s for categorical
                        current_file_feature_names.extend(col_names)
                
                else:   # Sequential integer values for other models which can handle categorical indices
                    for col in categorical_cols.keys():
                        cat_df = df[[col]].copy()
                        cat_df[col] = cat_df[col].astype("category")
                        encoded_parts.append(cat_df)
                        mask_parts.append(np.zeros(1))  # 0 for a single discrete column
                        current_file_feature_names.append(col)

                # Combine into a single dataframe
                segment_df = pd.concat(encoded_parts, axis=1)
                all_move_data.append(segment_df)
                
                # Mark this start timestamp as processed
                seen_starts.add(start_ts)

                if not move_feature_names:
                    move_feature_names = current_file_feature_names
                    binary_mask = np.concatenate(mask_parts)

            except Exception as e:
                logger.error(f"Error processing Results.xlsx in {folder.name}: {e}")

        if not all_move_data:
            return pd.DataFrame(), np.array([]), [], []

        # Combine all dataframes and sort by time
        full_move_df = pd.concat(all_move_data).sort_values('timestamp').drop_duplicates('timestamp')

        logger.info(f"Move Features: Extracted {len(full_move_df)} rows from {len(seen_starts)} unique files.")

        return full_move_df, binary_mask, move_feature_names, cardinalities


    def _extract_matched_move_data(
        self,
        p_folder: Path,
        window_end_timestamps: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, list[str], list[int]]:
        """
        Extract Move features and align them with EMA window timestamps.

        Calls the Move extraction pipeline and generates one fixed-duration feature window for each supplied EMA timestamp.

        Args:
            p_folder: Path to the participant's root directory.
            window_end_timestamps: EMA timestamps marking the ends of the movement windows, of shape (N,).

        Returns:
            Tuple containing
                - matched_move_data: Aligned Move feature windows of shape (N, Window_Size, Feature_Dim).
                - feature_mask: Binary feature-type mask of shape (Feature_Dim,).
                - feature_names: Ordered names of the Move features.
                - cardinalities: Cardinalities of integer-encoded categorical features.

            If no valid Move recordings are available, empty arrays and metadata lists are returned.
        """
        # Extract Move Data from Results.xlsx
        move_df, move_mask, move_feature_names, cardinalities = self.extract_data_move(p_folder)
        
        if move_df.empty:
            logger.warning(f"No Results.xlsx features found or missing columns for participant at {p_folder.name}.")
            return np.array([]), np.array([]), [], []

        # Match Move Data to the existing Windows
        matched_move_features = []
        feature_cols = [c for c in move_df.columns if c != 'timestamp']
        for ts in window_end_timestamps:
            window_move = self.get_feature_window(ts, move_df, feature_cols)
            matched_move_features.append(window_move)
        
        matched_move_np = np.array(matched_move_features)  # (N_windows, time_window, Move_features)
        return matched_move_np, move_mask, move_feature_names, cardinalities


    def create_pairs_move(
        self,
        p_folder: Path,
        window_end_timestamps: np.ndarray,
        p_y: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str], list[int]]:
        """
        Generate aligned Move data-label pairs.

        Extracts Move feature windows for the supplied EMA timestamps and removes windows containing missing values.

        Args:
            p_folder: Path to the participant's root directory.
            window_end_timestamps: EMA timestamps marking the ends of the movement windows, of shape (N,).
            p_y: Corresponding target labels of shape (N, Label_Dim).

        Returns:
            Tuple containing
                - data: Valid Move feature windows of shape (N_valid, Window_Size, Feature_Dim).
                - labels: Corresponding target labels of shape (N_valid, Label_Dim).
                - feature_mask: Binary feature-type mask of shape (Feature_Dim,).
                - feature_names: Ordered names of the Move features.
                - cardinalities: Cardinalities of integer-encoded categorical features.

            If no Move features are available, empty arrays and lists are returned.
        """
        matched_move_np, move_mask, move_feature_names, cardinalities = self._extract_matched_move_data(p_folder, window_end_timestamps)
        
        if matched_move_np.size == 0:
            return np.array([]), np.array([]), np.array([]), [], []

        # Filter out NaN rows across time and feature dimensions
        valid_mask = ~np.isnan(matched_move_np).any(axis=(1, 2))
        cleaned_X = matched_move_np[valid_mask]
        cleaned_y = p_y[valid_mask]

        logger.info(f"Move Pairs: {len(cleaned_X)} matched, {len(matched_move_np) - len(cleaned_X)} skipped due to NaNs.")
        return cleaned_X, cleaned_y, move_mask, move_feature_names, cardinalities


    def compute_tsfel(self, X_raw: np.ndarray) -> tuple[np.ndarray, list[str]]:
        """
        Extract TSFEL features from raw accelerometer windows.

        Computes the configured statistical, spectral, and temporal time-series features independently for each minute of every raw window.

        Args:
            X_raw: Raw movement windows of shape (N, Window_Size, Feature_Dim_raw).

        Returns:
            Tuple containing
                - features: Extracted TSFEL feature array of shape (N, Window_Size, Feature_Dim_tsfel).
                - feature_names: Ordered names of the extracted TSFEL features, including their corresponding sensor channels.
        """
        # Load tsfel features
        all_feature_set = tsfel.get_features_by_domain()
        desired_features = {
            "statistical": ["Mean", "Standard deviation", "Max", "Min", "Entropy", "Skewness", "Kurtosis", "Absolute energy"],
            "spectral": ["Max power spectrum", "Fundamental frequency", "Spectral entropy"],
            "temporal": ["Neighbourhood peaks", "Zero crossing rate", "Autocorrelation"]
        }
        feature_set = {}
        for domain, features in desired_features.items():
            feature_set[domain] = {feature: all_feature_set[domain][feature] for feature in features}

        samples_per_min = 60 * self.sampling_rate_raw
        all_windows_features = []
        feature_names = []

        # Baseline dimension mapping sequence
        raw_dims = ["dyn_x", "dyn_y", "dyn_z", "grav_x", "grav_y", "grav_z"]
        if self.include_jerk:
            raw_dims.extend(["jerk_x", "jerk_y", "jerk_z"])
        if self.include_magnitude:
            raw_dims.extend(["dyn_mag", "total_mag"])

        # Iterate over raw data windows
        for window in X_raw:
            minute_features = []
            # Compute features per minute
            for i in range(self.window_min):
                chunk = window[i*samples_per_min : (i+1)*samples_per_min]
                f = tsfel.time_series_features_extractor(feature_set, chunk, fs=self.sampling_rate_raw, verbose=0)
                minute_features.append(f.values.flatten())

                # Store feature names
                if not feature_names:
                    for col in f.columns:
                        # TSFEL prefixes columns via integers matching channel axes, e.g. "0_Mean" -> "x_Mean"
                        channel_idx = int(col.split('_')[0])
                        feat_name = col.split('_', 1)[1]
                        feature_names.append(f"{raw_dims[channel_idx]}_{feat_name}")

            all_windows_features.append(minute_features)
            
        return np.array(all_windows_features), feature_names


    def create_pairs_tsfel(
        self,
        raw_X: np.ndarray,
        p_y: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
        """
        Generate aligned TSFEL feature and target pairs.

        Computes TSFEL features from raw movement windows and removes windows containing NaN values.

        Args:
            raw_X: Raw movement windows of shape (N, Window_Size, Feature_Dim_raw).
            p_y: Corresponding target labels of shape (N, Label_Dim).

        Returns:
            Tuple containing
                - data: Valid TSFEL feature windows of shape (N_valid, Window_Size, Feature_Dim_tsfel).
                - labels: Corresponding target labels of shape (N_valid, Label_Dim).
                - feature_mask: Continuous feature mask containing ones, of shape (Feature_Dim_tsfel,).
                - feature_names: Ordered names of the TSFEL features.
        """
        tsfel_X, tsfel_feature_names = self.compute_tsfel(raw_X)  # Shape: (N_windows, time_window, TSFEL_features)
        
        # All TSFEL calculated features are continuous
        tsfel_mask = np.ones(tsfel_X.shape[2])

        # Filter out NaN rows across time and feature dimensions
        valid_mask = ~np.isnan(tsfel_X).any(axis=(1, 2))
        cleaned_X = tsfel_X[valid_mask]
        cleaned_y = p_y[valid_mask]

        logger.info(f"TSFEL Pairs: {len(cleaned_X)} matched, {len(tsfel_X) - len(cleaned_X)} skipped due to NaNs.")
        return cleaned_X, cleaned_y, tsfel_mask, tsfel_feature_names


    def create_pairs_tsfel_move(
        self,
        p_folder: Path,
        raw_X: np.ndarray,
        p_y: np.ndarray,
        window_end_timestamps: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str], list[int]]:
        """
        Combine TSFEL and Move features into aligned data-label pairs.

        Extracts TSFEL features from raw accelerometer windows, retrieves the corresponding Move feature windows, 
        and concatenates both feature representations along the feature dimension.

        Windows containing missing values in either feature representation are excluded.

        Args:
            p_folder: Path to the participant's root directory.
            raw_X: Raw movement windows of shape (N, Window_Size, Feature_Dim_raw).
            p_y: Corresponding target labels of shape (N, Label_Dim).
            window_end_timestamps: EMA timestamps marking the ends of the movement windows, of shape (N,).

        Returns:
            Tuple containing
                - data: Combined valid feature windows of shape (N_valid, Window_Size, Feature_Dim_tsfel + Feature_Dim_move).
                - labels: Corresponding target labels of shape (N_valid, Label_Dim).
                - feature_mask: Combined feature-type mask.
                - feature_names: Ordered names of the TSFEL and Move features.
                - cardinalities: Cardinalities of integer-encoded categorical Move features.
        """
        # Calculate TSFEL Features
        tsfel_X, tsfel_feature_names = self.compute_tsfel(raw_X)  # Shape: (N_windows, time_window, TSFEL_features)
        
        # Extract and align Move Data
        matched_move_np, move_mask, move_feature_names, cardinalities = self._extract_matched_move_data(p_folder, window_end_timestamps)
        
        if matched_move_np.size == 0:
            logger.warning("No Move features found for TSFEL-Move combination.")
            return np.array([]), np.array([]), np.array([]), [], []

        # Concatenate TSFEL and Move features along the feature dimension
        combined_X = np.concatenate([tsfel_X, matched_move_np], axis=2)
        
        # NaN Check with boolean mask: True for windows that have NO NaNs anywhere
        valid_mask = ~np.isnan(combined_X).any(axis=(1, 2))
        
        # Filter X and y
        cleaned_X = combined_X[valid_mask]
        cleaned_y = p_y[valid_mask]
        
        # Combine masks and feature names
        tsfel_mask = np.ones(tsfel_X.shape[2])
        full_mask = np.concatenate([tsfel_mask, move_mask])
        combined_feature_names = tsfel_feature_names + move_feature_names

        logger.info(f"TSFEL-Move Pairs: {len(cleaned_X)} matched, {len(combined_X) - len(cleaned_X)} skipped due to NaNs.")
        return cleaned_X, cleaned_y, full_mask, combined_feature_names, cardinalities
    

    def create_pairs_fitbit(
        self,
        p_path: Path,
        labels_df: pd.DataFrame,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
        """
        Extract and align Fitbit sensor data with EMA labels.

        Loads the available Fitbit minute-level sensor streams, including  calories, heart rate, steps, intensity, and metabolic equivalents.
        Combines the sensor streams into synchronized feature windows preceding each EMA timestamp.

        Only windows containing complete observations across all required sensor streams are retained.

        Args:
            p_path: Path to the participant's root directory.
            labels_df: EMA DataFrame containing timestamp and labels columns.

        Returns:
            Tuple containing
                - data: Valid Fitbit feature windows of shape (N_valid, Window_Size, 5).
                - labels: Corresponding target labels of shape (N_valid, Label_Dim).
                - feature_mask: Continuous feature mask of shape (5,), containing ones.
                - feature_names: Ordered Fitbit feature names: Calories, Heart Rate, Steps, Intensity, and METs.

            If required Fitbit data are unavailable, empty data and label arrays are returned alongside the feature metadata.
        """
        participant_id = p_path.name
        feature_names = ['Calories', 'Heart_Rate', 'Steps', 'Intensity', 'METs']
        binary_mask = np.ones(len(feature_names))   # All features are continuous, so entirely 1s
        
        # Construct path to FitBit data
        fitbit_path = p_path / "AA-Phase" / "Fitbit"
        if not fitbit_path.exists():
            logger.warning(f"Fitbit directory missing for participant {participant_id}. Skipping.")
            return np.array([]), np.array([]), binary_mask, feature_names

        # Schema: { sensor_name: (sub_dir, file_glob_pattern, target_column_name) }
        sensor_mapping = {
            'Calories': ('Calories', f"{participant_id}_minuteCaloriesNarrow_*.csv", 'Calories'),
            'Heart_Rate': ('Heart_Rate', f"{participant_id}_heartrate_1min_*.csv", 'Value'),
            'Steps': ('Steps', f"{participant_id}_minuteStepsNarrow_*.csv", 'Steps'),
            'Intensity': ('Intensity', f"{participant_id}_minuteIntensitiesNarrow_*.csv", 'Intensity'),
            'METs': ('Intensity', f"{participant_id}_minuteMETsNarrow_*.csv", 'METs')
        }

        sensor_dfs = {} # Store dataframes per sensors

        # Load FitBit data of each sensor
        for sensor, (sub_dir, glob_pattern, val_col) in sensor_mapping.items():
            # Check if sub-directory exists
            target_dir = fitbit_path / sub_dir
            if not target_dir.exists():
                logger.warning(f"Required Fitbit directory {sub_dir} missing for participant {participant_id}. Skipping.")
                return np.array([]), np.array([]), binary_mask, feature_names

            # Check if files exist
            file_list = sorted(target_dir.glob(glob_pattern))
            if not file_list:
                logger.warning(f"No CSV matches found for {sensor} ({glob_pattern}). Skipping.")
                return np.array([]), np.array([]), binary_mask, feature_names

            try:
                # Accumulate files if split across multiple parts
                chunks = []
                for file in file_list:
                    raw_df = pd.read_csv(file)
                    # Build standardized sensor DataFrame with "timestamp" column assuming mm/dd/yyyy hh:mm:ss AM/PM format and the value column
                    df_chunk = pd.DataFrame({
                        'timestamp': pd.to_datetime(raw_df.iloc[:, 0], format="%m/%d/%Y %I:%M:%S %p"),
                        val_col: raw_df[val_col]
                    })
                    chunks.append(df_chunk)
                
                # Merge chunks, sort chronologically, and drop duplicate timestamps
                full_sensor_df = pd.concat(chunks).sort_index().drop_duplicates(subset='timestamp',keep='first')
                sensor_dfs[sensor] = full_sensor_df
                logger.info(f"{sensor} records: {len(full_sensor_df)} extracted in time window [{full_sensor_df['timestamp'].iloc[0]},{full_sensor_df['timestamp'].iloc[-1]}].")
                
            except Exception as e:
                logger.error(f"Failed loading Fitbit stream {sensor} for participant {participant_id}: {e}")
                return np.array([]), np.array([]), binary_mask, feature_names

        # Slice Windows & Sync with EMA Labels
        windows_accumulator, aligned_label_rows = [], []
        skipped_count = 0

        # Iterate over labels
        for _, label_row in labels_df.iterrows():
            # Get time window of label
            window_end = label_row['timestamp']
            window_start = window_end - pd.Timedelta(minutes=self.window_min)

            window_is_valid = True
            sliced_sensor_data = []

            # Iterate over each sensor
            for sensor in feature_names:
                sensor_df = sensor_dfs[sensor]  # Get dataframe of sensor
                
                sensor_window = self.get_feature_window(window_end, sensor_df, [sensor_mapping[sensor][2]]) # Get sensor data in label time window

                # Check if enough data samples found
                if np.isnan(sensor_window).any():
                    window_is_valid = False
                    skipped_count += 1
                    logger.warning(f"Incomplete {sensor} data found for label {label_row['labels']} in time window [{window_start},{window_end}].")
                    break   # Skip label if not enough data for one sensor found
                
                sliced_sensor_data.append(sensor_window)

            if window_is_valid:
                # Combine the 5 data arrays (one for each sensor) and add to output
                window_matrix = np.hstack(sliced_sensor_data)
                windows_accumulator.append(window_matrix)
                aligned_label_rows.append(label_row['labels'])

        if not windows_accumulator:
            logger.warning(f"No valid time windows with FitBit data could be aligned for participant {participant_id}.")
            return np.array([]), np.array([]), binary_mask, feature_names
        
        # Convert lists to arrays
        fitbit_features = np.stack(windows_accumulator, axis=0)
        valid_labels = np.array(aligned_label_rows)

        logger.info(f"FitBit Pairs: {len(valid_labels)} matched, {skipped_count} skipped.")
        
        return fitbit_features, valid_labels, binary_mask, feature_names


    def select_features_collinear_mrmr(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: list[str],
        top_k: int = 20,
        corr_threshold: float = 0.85,
    ) -> tuple[np.ndarray, list[str], list[int]]:
        """
        Select non-redundant features using collinearity filtering and mRMR.

        Applies a two-stage feature selection procedure to three-dimensional feature arrays.

        First, aggregates feature values across the temporal dimension and removes highly correlated features. 
        Subsequently, performs multi-target mRMR regression and combines the selected features across all target dimensions.

        Args:
            X: Input feature tensor of shape (N, Window_Size, Feature_Dim).
            y: Target array of shape (N, Label_Dim).
            feature_names: Ordered names corresponding to the input feature dimension.
            top_k: Maximum number of features selected per target label. Defaults to 20.
            corr_threshold: Absolute Pearson correlation threshold above which features are considered redundant. Defaults to 0.85.

        Returns:
            Tuple containing
                - selected_data: Feature tensor containing only selected features, of shape (N, Window_Size, Feature_Dim_selected).
                - selected_feature_names: Ordered names of the selected features.
                - selected_indices: Original feature indices corresponding to the selected features.
        """
        # Aggregate across time dimension (mean feature state over the window)
        X_agg = np.mean(X, axis=1)  # Shape: (Samples, TSFEL_features)
        df_agg = pd.DataFrame(X_agg, columns=feature_names)

        # Stage 1: Collinearity Filter (Remove redundant duplicate channels)
        corr_matrix = df_agg.corr().abs()
        upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        
        to_drop = [col for col in upper_tri.columns if any(upper_tri[col] > corr_threshold)]
        df_filtered = df_agg.drop(columns=to_drop)

        # Stage 2: Multi-Label mRMR
        selected_indices_set = set()
        num_labels = y.shape[1]

        for l_idx in range(num_labels):
            target_series = pd.Series(y[:, l_idx], name=f"label_{l_idx}")
            
            # Select top-k features for this specific target label
            mrmr_selected = mrmr.mrmr_regression(
                X=df_filtered, 
                y=target_series, 
                K=min(top_k, df_filtered.shape[1]),
                show_progress=False
            )
            
            for feat in mrmr_selected:
                selected_indices_set.add(feature_names.index(feat))

        # Convert unique selected feature indices back to array mapping
        selected_indices = sorted(list(selected_indices_set))
        selected_feature_names = [feature_names[i] for i in selected_indices]

        # Slice raw 3D input tensor along feature axis
        X_selected = X[:, :, selected_indices]

        return X_selected, selected_feature_names, selected_indices
