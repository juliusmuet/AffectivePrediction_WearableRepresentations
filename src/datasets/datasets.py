import logging
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset, default_collate

from preprocessing.timeseries_scaler import FeatureScaler
from datasets.data_splitter import BaseSplitter
from datasets.data_scheduler import BaseLabelScheduler


logger = logging.getLogger(__name__)


class MovementMoodDataset(Dataset):
    """
    Dataset for time-series movement data and multidimensional mood labels.

    Stores movement features, corresponding mood targets, and participant identifiers. 
    Individual samples are converted to PyTorch tensors when retrieved. 
    The dataset also provides functionality for generating data partitions and constructing DataLoaders with optional feature scaling and label-aware sampling.

    Args:
        data: Movement feature array of shape (N, Window_Size, Feature_Dim).
        labels: Target array of shape (N, Label_Dim).
        participant_ids: Participant identifiers corresponding to each sample, with shape (N,).
        binary_mask: Feature-type mask of shape (Feature_Dim,). A value of 1 indicates a continuous feature, 
            while 0 indicates a discrete or categorical feature.
        feature_names: Ordered names of the Feature_Dim input features.
        label_names: Ordered names of the Label_Dim target dimensions.
        cardinality: Cardinalities of categorical features.
        dataloader_args: Keyword arguments forwarded to PyTorch's DataLoader. The custom collate_fn is assigned internally. 
            During standard evaluation, the validation and test batch sizes are doubled, and shuffling and drop_last are disabled.
        scaler: Optional feature scaler for on-the-fly data scaling. Defaults to None.
        splitter: Optional dataset splitter implementing the split() method and is_between_splitter property. Defaults to None.
        scheduler: Optional label scheduler implementing the get_sampler() method. Defaults to None.
        verbose: Whether to log dataset initialization information. Defaults to True.
    """
    def __init__(
        self,
        data: np.ndarray,
        labels: np.ndarray,
        participant_ids: np.ndarray,
        binary_mask: np.ndarray,
        feature_names: list[str],
        label_names: list[str],
        cardinality: list[int],
        dataloader_args: dict[str, Any],
        scaler: FeatureScaler | None = None,
        splitter: BaseSplitter | None = None,
        scheduler: BaseLabelScheduler | None = None,
        verbose: bool = True,
    ) -> None:
        self.data = data
        self.labels = labels
        self.participant_ids = participant_ids
        self.binary_mask = binary_mask
        self.feature_names = feature_names
        self.label_names = label_names
        self.cardinality = cardinality
        self.scaler = scaler
        self.scheduler = scheduler
        self.splitter = splitter

        # Copy to avoid modifying the original dict passed from outside
        self.dataloader_args = dataloader_args.copy()
        self.dataloader_args['collate_fn'] = self.numpy_id_collate

        if verbose:
            logger.info(f"Initialized {self}  \n")
    

    def __str__(self):
        """Returns a string representation of the dataset configuration."""
        return (f"MovementMoodDataset(samples={len(self)}, "
                f"features={self.data.shape[2]}, "
                f"window_size={self.data.shape[1]}, "
                f"participants={len(np.unique(self.participant_ids))})")


    def __len__(self) -> int:
        """
        Return the total number of samples in the dataset.

        Returns:
            Number of samples in the dataset.
        """
        return len(self.labels)


    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, Any]:
        """
        Retrieve a movement sample, its target labels, and participant ID.

        Converts the selected movement data and labels into floating-point PyTorch tensors. 
        The participant identifier is returned without tensor conversion.

        Args:
            idx: Index of the sample to retrieve.

        Returns:
            Tuple containing
            - x: Movement feature tensor of shape (Window_Size, Feature_Dim).
            - y: Target tensor of shape (L,).
            - p_id: Participant identifier associated with the sample.
        """
        # Convert NumPy slice to PyTorch Tensor on the fly
        x = torch.from_numpy(self.data[idx]).float()
        y = torch.from_numpy(self.labels[idx]).float()
        
        # Participant ID can remain a numpy array because not used in loss
        p_id = self.participant_ids[idx] 
        
        return x, y, p_id
    

    def get_io_dimensions(self) -> tuple[int, int, int]:
        """
        Retrieve the input and output dimensions of the dataset.

        Returns:
            Tuple containing
            - input_dim: Number of input features (Feature_Dim).
            - output_dim: Number of target dimensions (Label_Dim).
            - seq_len: Number of time steps per input window (Window_Size).
        """
        input_dim = self.data.shape[2]
        output_dim = self.labels.shape[1]
        seq_len = self.data.shape[1]
        return input_dim, output_dim, seq_len
    

    def get_splits(self) -> list[tuple[np.ndarray | None,  np.ndarray | None, np.ndarray | None]]:
        """
        Generate training, validation, and test indices.

        Uses the configured splitter to generate dataset partitions.
        If no splitter is configured, returns a single entry representing the complete dataset without partitioning.

        Returns:
            A list of tuples, with each tuple containing
            - train_idx: Training sample indices, or None if the complete dataset is used.
            - val_idx: Validation sample indices, or None if the complete dataset is used.
            - test_idx: Test sample indices, or None if the complete dataset is used.

            If no splitter is configured, returns [(None, None, None)].
        """
        # If no split configured return list of one tuple containing None to create only one dataloader for full dataset in get_dataloaders()
        if self.splitter == None:
            logger.info(f"No dataset split configured.")
            return [(None, None, None)]

        return self.splitter.split(self.data, self.labels, self.participant_ids)
    

    def get_dataloaders(
        self,
        train_idx: np.ndarray | None = None,
        val_idx: np.ndarray | None = None,
        test_idx: np.ndarray | None = None,
        is_finetuning: bool = False,
    ) -> tuple[DataLoader, DataLoader | None, DataLoader | None]:
        """
        Create PyTorch DataLoaders for training, validation, and testing.

        Supports optional feature scaling and label-aware sampling.
        When no training indices are provided, a single DataLoader is created for the complete dataset.

        For split datasets, scaling depends on the configured splitting strategy. Between-subject splitting uses the corresponding 
        between-subject scaling procedure, while within-subject splitting uses the corresponding within-subject procedure. 
        During fine-tuning, the dedicated fine-tuning scaling procedure is applied instead.

        The training DataLoader can optionally use a label scheduler. Validation and test DataLoaders always use sequential sampling.

        During standard evaluation, the validation and test batch sizes are doubled relative to the configured batch size. 
        During fine-tuning, the configured batch size is retained and drop_last is disabled for all DataLoaders.

        Args:
            train_idx: Training sample indices. If None, the complete dataset is used without partitioning. Defaults to None.
            val_idx: Validation sample indices. Required when train_idx is provided. Defaults to None.
            test_idx: Test sample indices. Required when train_idx is provided. Defaults to None.
            is_finetuning: Whether the DataLoaders are being created for participant-specific fine-tuning. Ensures that training
                samples are not discarded through drop_last and that the configured evaluation batch size is retained. Defaults to False.

        Returns:
            Tuple containing
            - train_loader: DataLoader for the training data, or for the complete dataset if no split indices are provided.
            - val_loader: DataLoader for the validation data, or None when no partitioning is performed.
            - test_loader: DataLoader for the test data, or None when no partitioning is performed.

        Raises:
            ValueError: If training indices are provided but validation or test indices are missing.

        Notes:
            If a label scheduler is configured, its sampler is applied only to the training DataLoader. 
            For an unpartitioned dataset, the sampler is applied to the complete dataset.

            If scaling is enabled for a partitioned dataset, a splitter must be configured to determine the appropriate scaling mode.
        """
        # Case 1: Simple loader without splitting (e.g. for initial data inspection)
        if train_idx is None:
            full_args = self.dataloader_args.copy()
            
            # Apply label scheduler to the full dataset if provided
            if self.scheduler is not None:
                full_args['sampler'] = self.scheduler.get_sampler(self.labels)
                full_args['shuffle'] = False

            # Case 1a: No scaling
            if self.scaler is None:
                logger.info(f"DataLoader for full raw dataset created with arguments {full_args}")
                return DataLoader(self, **full_args), None, None

            # Case 1b: Scaling applied
            else:
                full_scaled = self.scaler.process_full_dataset(self.data, self.participant_ids, self.binary_mask)
                full_dataset = MovementMoodDataset(full_scaled, self.labels, self.participant_ids, self.binary_mask, self.feature_names, 
                                                   self.label_names, self.cardinality, self.dataloader_args, verbose=False)
                logger.info(f"DataLoader for full scaled dataset created with arguments {full_args}")
                return DataLoader(full_dataset, **full_args), None, None
        
        if val_idx is None or test_idx is None:
            raise ValueError("val_idx and test_idx must not be None if train_idx is given.")
        
        # Case 2a: Splitting without a scaler
        if self.scaler is None:
            logger.info("No scaler provided. Using raw data for loaders.")
            train_subset = Subset(self, train_idx)
            val_subset = Subset(self, val_idx)
            test_subset = Subset(self, test_idx)

        else:
        # Case 2b: Splitting wit a scaler
            mode = "between mode" if self.splitter.is_between_splitter else "within mode"
            logger.info(f"Applying {self.scaler.base_scaler} scaling for {mode}")

            if is_finetuning:
                # Scale using all fine-tuning training samples as calibration baseline
                train_scaled, val_scaled, test_scaled = self.scaler.scale_finetune(self.data, self.participant_ids, train_idx, val_idx, test_idx, self.binary_mask)
            else:
                # Scale using standard outer CV fold rules (between/within)
                train_scaled, val_scaled, test_scaled = self.scaler.scale_split(self.data, self.participant_ids, train_idx, val_idx, test_idx, 
                                                                                self.binary_mask, self.splitter.is_between_splitter)
            
            # Create temporary subsets with scaled data
            train_subset = MovementMoodDataset(train_scaled, self.labels[train_idx], self.participant_ids[train_idx], self.binary_mask, 
                                               self.feature_names, self.label_names, self.cardinality, self.dataloader_args, verbose=False)
            val_subset = MovementMoodDataset(val_scaled, self.labels[val_idx], self.participant_ids[val_idx], self.binary_mask, 
                                             self.feature_names, self.label_names, self.cardinality, self.dataloader_args, verbose=False)
            test_subset = MovementMoodDataset(test_scaled, self.labels[test_idx], self.participant_ids[test_idx], self.binary_mask, 
                                              self.feature_names, self.label_names, self.cardinality, self.dataloader_args, verbose=False)

        # Configure training dataloader arguments
        train_dataloader_args = self.dataloader_args.copy()
        if self.scheduler is not None:
            train_dataloader_args['sampler'] = self.scheduler.get_sampler(self.labels[train_idx])
            train_dataloader_args['shuffle'] = False

        # Configure validation and test dataloader arguments
        val_test_dataloader_args = self.dataloader_args.copy()
        val_test_dataloader_args['shuffle'] = False
        val_test_dataloader_args['drop_last'] = False
        val_test_dataloader_args.pop('sampler', None)

        # Safety checks applied during fine-tuning mode
        if is_finetuning:
            # Guarantee training data is never discarded when participant samples < batch_size
            train_dataloader_args['drop_last'] = False
            
            # Avoid inflating batch size for small evaluation subsets
            val_test_dataloader_args['batch_size'] = self.dataloader_args.get('batch_size', 16)
        else:
            # Standard population training behavior
            val_test_dataloader_args['batch_size'] = self.dataloader_args.get('batch_size', 16) * 2

        # Create the dataloaders
        train_loader = DataLoader(train_subset, **train_dataloader_args)
        val_loader = DataLoader(val_subset, **val_test_dataloader_args)
        test_loader = DataLoader(test_subset, **val_test_dataloader_args)
        
        logger.info(f"Dataloader created: Train ({len(train_idx)} samples) with arguments {train_dataloader_args}")
        logger.info(f"Dataloaders created: Val ({len(val_idx)} samples), Test ({len(test_idx)} samples) with arguments {val_test_dataloader_args}")

        return train_loader, val_loader, test_loader

    
    @staticmethod
    def numpy_id_collate(batch: list[tuple[torch.Tensor, torch.Tensor, Any]]) -> tuple[torch.Tensor, torch.Tensor, np.ndarray]:
        """
        Collate samples while preserving participant IDs as a NumPy array.

        Uses PyTorch's default collation procedure for movement features and target labels, while collecting participant identifiers into a NumPy array.

        Args:
            batch: List of individual samples, each containing movement features, target labels, and a participant identifier.

        Returns:
            Tuple containing
            - x_batch: Batched movement feature tensor of shape (B, Window_Size, Feature_Dim).
            - y_batch: Batched target tensor of shape (B, Label_Dim).
            - p_id_batch: NumPy array containing the participant identifiers of shape (B,).
        """
        # batch is a list of tuples: [(x1, y1, p_id1), (x2, y2, p_id2), ...]
        transposed = list(zip(*batch))
        
        # Let PyTorch handle x and y using its standard optimized routine
        x_batch = default_collate(transposed[0])
        y_batch = default_collate(transposed[1])
        
        # Force participant IDs to stay a NumPy array
        p_id_batch = np.array(transposed[2])
        
        return x_batch, y_batch, p_id_batch
