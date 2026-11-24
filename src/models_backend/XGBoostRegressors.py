import logging
import copy
from abc import ABC, abstractmethod

import numpy as np
import xgboost as xgb
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from models_backend.base_backend_model import BaseNonNNBackendModel
from train.metrics import RegressionMetricsTracker
from explainability.shap_strategies import TreeSHAPStrategy
from preprocessing.label_transformer import BaseTargetTransform


logger = logging.getLogger(__name__)


class BaseXGBoostBackend(BaseNonNNBackendModel, ABC):
    """
    Abstract Intermediate Base Class for XGBoost-driven architectures.
    Handles data processing, inference structures, metrics, and importance extraction.

    Args:
        input_dim (int): Number of features of the input (not flattened over time). Placeholder for interface compatability.
        output_dim (int): Number of features of the output. Placeholder for interface compatability.
        seq_len (int): Lookback window sequence length. Placeholder for interface compatability.
        label_transformer (BaseTargetTransform): Instance of a label transformer.
    """
    def __init__(self, input_dim: int, output_dim: int, seq_len: int, label_transformer: BaseTargetTransform):
        super().__init__(input_dim, output_dim, seq_len, label_transformer)
        self.model = None
        self.params = {}    # Needs to be initialised by child classes
        self.feature_mask = []


    @property
    def requires_one_hot_encoding(self) -> bool:
        """
        Flag identifying whether categorical input features should be one-hot-encoded.

        Returns:
            bool: Always False since XGBoost processes categories internally.
        """
        return False


    @abstractmethod
    def train(self, train_loader: DataLoader, val_loader: DataLoader, binary_mask: list, **kwargs):
        """
        Abstract method. Must be implemented by child classes to handle 
        specific boosting or parallel tree ensemble training execution.

        Args:
            train_loader (DataLoader): PyTorch DataLoader streaming training batches.
            val_loader (DataLoader): PyTorch DataLoader streaming validation batches.
            binary_mask (list): List indicating feature types (e.g., 1 for continuous, 0 for categorical).
            **kwargs: Arbitrary keyword arguments passed to tree-specific routines.
        """
        pass

    
    @abstractmethod
    def fine_tune(self, train_loader: DataLoader, lr: float = 0.01, device: str = "cpu", **kwargs):
        """
        Fine-tunes the tree model on a small dataset.
        """
        pass


    def _flatten_input(self, X_tensor: torch.Tensor) -> torch.Tensor:
        """
        Flattens 3D window data (N, Window, Features) to 2D (N, Window * Features).

        Args:
            X_tensor (torch.Tensor): Input tensor of shape (N, Window, Features).

        Returns:
            torch.Tensor: A flattened tensor of shape (N, Window * Features).
        """
        if len(X_tensor.shape) == 3:
            N, W, F = X_tensor.shape
            return X_tensor.reshape(N, W * F).contiguous()  # .contiguous() prevents CUDA memory allocation issues in DMatrix
        return X_tensor.contiguous()


    def _infer_and_set_feature_mask(self, train_loader: DataLoader, binary_mask: list) -> None:
        """
        Infers the operational window size and scales the feature mask.

        Args:
            train_loader (DataLoader): PyTorch DataLoader used to peek at the input shape.
            binary_mask (list): Numeric binary mask representing base frame feature types (1 for continuous, 0 for categorical).

        Raises:
            ValueError: If the sampled batch tensor does not have exactly 3 dimensions.
        """
        loader_iter = iter(train_loader)
        first_batch_x, _, _ = next(loader_iter)

        if len(first_batch_x.shape) != 3:
            raise ValueError(f"Expected 3D tensor (N, Window, Features), got shape {first_batch_x.shape}")
        window_size = first_batch_x.shape[1] 

        base_string_mask = ['q' if x == 1 else 'c' for x in binary_mask]    # q -> continuous, c -> categorial
        self.feature_mask = base_string_mask * window_size

        # Cleanup GPU/Host memory references
        del first_batch_x
        del loader_iter


    class _WindowDataIter(xgb.core.DataIter):
        """
        Internal iterator definition for data streaming across batches.
        Enables efficient memory management for PyTorch DataLoaders within XGBoost.

        Args:
            dataloader (DataLoader): Source PyTorch DataLoader streaming batches.
            flattener (callable): Method/function used to flatten 3D batches to 2D.
            feature_types (list): Feature mask matching the shape of flattened rows.
            device (str): Device to move the data to. Defaults to "cpu".
        """
        def __init__(self, dataloader, flattener, feature_types, device="cpu"):
            self.dataloader = dataloader
            self.flattener = flattener
            self.feature_types = feature_types
            self.device = device
            self.iterator = iter(self.dataloader)
            super().__init__()

        def next(self, input_data):
            try:
                X_batch, y_batch, _ = next(self.iterator)
                
                X_batch = X_batch.to(self.device)
                X_flat = self.flattener(X_batch).float()

                y_batch = y_batch.to(self.device).float()
                
                # Handle PyTorch CPU vs GPU ingestion for XGBoost proxy iterator
                if X_flat.is_cuda:
                    # Native CUDA ingestion via PyTorch CUDA interface
                    input_data(data=X_flat, label=y_batch, feature_types=self.feature_types)
                else:
                    # CPU PyTorch Tensors must be converted to NumPy arrays
                    input_data(data=X_flat.numpy(), label=y_batch.numpy(), feature_types=self.feature_types)
                return 1
            except StopIteration:
                return 0

        def reset(self):
            self.iterator = iter(self.dataloader)


    def forward(self, X_tensor: torch.Tensor, device: str = None) -> torch.Tensor:
        """
        Runs model inference on an input tensor, adjusting iteration ranges based on architecture.

        Args:
            X_tensor (torch.Tensor): Input sample batch tensor of shape (N, window_size, features).
            device (str): Device to move the data to. If None, uses the device of X_tensor. Defaults to None.

        Returns:
            torch.Tensor: Evaluated predictions of shape (N, output_dim).

        Raises:
            ValueError: If the underlying booster has not been trained yet.
        """
        if self.model is None:
            raise ValueError(f"Cannot run inference. {self.__class__.__name__} is not trained. Call train() or run_full_training() first.")

        # Move data to target device
        target_device = device if device is not None else X_tensor.device
        X_tensor = X_tensor.to(target_device)
        X_flat = self._flatten_input(X_tensor).float().contiguous()

        # Determine iteration range
        best_iter = getattr(self.model, 'best_iteration', None)
        total_rounds = self.model.num_boosted_rounds()

        if best_iter is not None:
            iter_range = (0, best_iter + 1)
        elif total_rounds == 1:
            iter_range = (0, 1) # RandomForest path
        else:
            iter_range = (0, total_rounds)

        # Handle inplace_predict data types (CPU vs CUDA)
        if X_flat.is_cuda:
            # In-place prediction directly on CUDA memory without creating a CPU DMatrix
            preds = self.model.inplace_predict(X_flat, iteration_range=iter_range)
        else:
            # CPU PyTorch Tensors must be converted to NumPy arrays for inplace_predict
            preds = self.model.inplace_predict(X_flat.numpy(), iteration_range=iter_range)

        if not isinstance(preds, torch.Tensor):
            preds = torch.as_tensor(preds, dtype=torch.float32, device=target_device)
        
        return preds.float()


    def evaluate(self, dataloader: DataLoader, device: str = "cpu", description: str = "Train Evaluation"):
        """
        Iterates over a dataloader to collect all predictions and ground truth labels.

        Args:
            dataloader (DataLoader): The collection dataloader to compute metrics against.
            device (str): Device to move the data to. Defaults to "cpu".
            description (str): Description of the progress bar.

        Returns:
            dict: Nested dictionary containing evaluation metrics with keys:
                - "global" (dict): Overall metrics across all samples and dimensions.
                - "by_dimension" (dict): Metrics broken down per dimension name (EA, V, C).
                - "by_participant" (dict): Nested metrics mapped to stringified participant IDs,
                  each containing its own "global" and "by_dimension" metric dicts.
        """
        all_preds, all_labels, all_pids = [], [], []

        with torch.no_grad():
            for X_batch, y_batch, p_ids in tqdm(dataloader, desc=description):
                X_batch = X_batch.to(device)
                preds = self.forward(X_batch, device)

                all_preds.append(preds.cpu().numpy())
                all_labels.append(y_batch.cpu().numpy())
                all_pids.append(p_ids)

        all_preds = np.concatenate(all_preds, axis=0)
        all_labels = np.concatenate(all_labels, axis=0)
        all_pids = np.concatenate(all_pids)

        return RegressionMetricsTracker.evaluate_predictions(all_labels, all_preds, all_pids, self.label_transformer)


    def run_full_training(
        self, 
        train_loader: DataLoader, 
        val_loader: DataLoader, 
        test_loader: DataLoader, 
        binary_mask: list, 
        eval_metrics: list = ["mae", "rmse"], 
        verbose: bool = True, 
        device: str = "cpu", 
        **kwargs
    ):
        """
        Orchestrates full cycle training followed by evaluations across 
        train, validation, and test datasets.

        Args:
            train_loader (DataLoader): PyTorch DataLoader streaming training batches.
            val_loader (DataLoader): PyTorch DataLoader streaming validation batches.
            test_loader (DataLoader): PyTorch DataLoader streaming test batches.
            binary_mask (list): Numeric binary mask representing base frame feature types (1 for continuous, 0 for categorical).
            eval_metrics (list): Metric evaluation labels parsed by the booster. Defaults to ["mae", "rmse"].
            verbose (bool): Whether evaluation progress blocks display during iterations. Defaults to True.
            device (str): Computations backend processor device targeted ("cpu"/"cuda"). Defaults to "cpu".
            **kwargs: Arbitrary parameters passed down directly to the trainer.
        
        Returns:
            Tuple[dict, dict, dict]: Metrics of best model for train, validation and test subset (see BaseMetricsTracker).
        """
        logger.info(f"Starting training workflow via {self.__class__.__name__} with device {device}...")
        
        # This will call the child's specific train implementation
        self.train(
            train_loader=train_loader, 
            val_loader=val_loader, 
            binary_mask=binary_mask, 
            eval_metrics=eval_metrics, 
            verbose=verbose, 
            device=device,
            **kwargs  # For passing early_stopping_rounds / num_boost_round cleanly
        )

        # Calculate metrics for all given dataloaders
        train_metrics = self.evaluate(train_loader, device, "Train Evaluation")
        val_metrics = self.evaluate(val_loader, device, "Validation") if val_loader is not None else None
        test_metrics = self.evaluate(test_loader, device, "Test") if test_loader is not None else None
        
        # Summary logging blocks
        summary_msg = (
            f"\n==================== FOLD TRAINING COMPLETE ====================\n"
            f"Best Train Metrics: " + " | ".join(f"{k.upper()}: {v:.4f}" for k, v in train_metrics.get('global', {}).items())
        )
        if val_metrics:
            summary_msg += f"\nBest Val   Metrics: " + " | ".join(f"{k.upper()}: {v:.4f}" for k, v in val_metrics.get('global', {}).items())
        if test_metrics:
            summary_msg += f"\nBest Test  Metrics: " + " | ".join(f"{k.upper()}: {v:.4f}" for k, v in test_metrics.get('global', {}).items())
        summary_msg += f"\n================================================================"
        logger.info(summary_msg)
        
        return train_metrics, val_metrics, test_metrics
    

    def get_shap_strategy(self) -> TreeSHAPStrategy:
        """
        Returns its own compatible SHAP explainability strategy.

        Returns: 
            TreeSHAPStrategy: Instance of the SHAP calculator.
        """
        return TreeSHAPStrategy()



class XGBoostRegressor(BaseXGBoostBackend):
    """
    Implementation of a gradient boosted sequential tree regressor via XGBoost.

    Args:
        input_dim (int): Number of features of the input (not flattened over time). Placeholder for interface compatability.
        output_dim (int): Number of features of the output. Placeholder for interface compatability.
        seq_len (int): Lookback window sequence length. Placeholder for interface compatability.
        label_transformer (BaseTargetTransform): Instance of a label transformer.
        objective (str): The loss function for training (e.g. "reg:squarederror" or "reg:logistic").
        max_depth (int): Maximum depth of a tree structure during splitting. Defaults to 6.
        random_state (int): Seed used by the random number generator tracking splits. Defaults to 42.
    """
    def __init__(
        self, 
        input_dim: int, 
        output_dim: int, 
        seq_len: int, 
        label_transformer: BaseTargetTransform, 
        objective: str, 
        max_depth: int = 6, 
        random_state: int = 42,
    ):
        super().__init__(input_dim, output_dim, seq_len, label_transformer)
        
        self.params = {
            "objective": objective,
            "max_depth": max_depth,
            "tree_method": "hist",
            "seed": random_state
        }

        logger.info(f"Initialized {self}")


    def __str__(self):
        """Returns a string representation of the XGBoostRegressorBackend configuration."""
        params_str = ", ".join(f"{k}={v}" for k, v in self.params.items())
        return (f"XGBoostRegressor(\n"
                f"  input_dim={self.input_dim}, output_dim={self.output_dim}, seq_len={self.seq_len},\n"
                f"  params={params_str}\n"
                f")")


    def train(
        self, 
        train_loader: DataLoader, 
        val_loader: DataLoader, 
        binary_mask: list, 
        early_stopping_rounds: int = 5, 
        num_boost_round: int = 100, 
        eval_metrics: list = ["mae", "rmse"], 
        verbose: bool = True, 
        device: str = "cpu",
    ):
        """
        Executes sequential gradient boosting iterations over streamed loaders.

        Args:
            train_loader (DataLoader): PyTorch DataLoader streaming training batches.
            val_loader (DataLoader): PyTorch DataLoader streaming validation batches.
            binary_mask (list): Numeric binary mask representing base frame feature types (1 for continuous, 0 for categorical).
            early_stopping_rounds (int): Minimum iteration threshold without improvement to halt. Defaults to 5.
            num_boost_round (int): Maximum number of boosting tree iterations. Defaults to 100.
            eval_metrics (list): Metric evaluation labels parsed by the booster. Defaults to ["mae", "rmse"].
            verbose (bool): Whether evaluation progress blocks display during iterations. Defaults to True.
            device (str): Computations backend processor device targeted ("cpu"/"cuda"). Defaults to "cpu".
        """
        # Update class attributes
        self.params.update({"eval_metric": eval_metrics, "device": device})
        self._infer_and_set_feature_mask(train_loader, binary_mask)

        # Initialize native Quantile stream iterators for training data
        train_iter = self._WindowDataIter(train_loader, self._flatten_input, self.feature_mask, device)
        dtrain = xgb.QuantileDMatrix(train_iter)

        # Set up validation monitoring targets
        eval_set = [(dtrain, 'train')]
        stop_rounds = None
        if val_loader is not None:
            val_iter = self._WindowDataIter(val_loader, self._flatten_input, self.feature_mask, device)
            dval = xgb.QuantileDMatrix(val_iter, ref=dtrain)
            eval_set.append((dval, 'val'))  # Last entry of eval_set used for early stopping
            stop_rounds = early_stopping_rounds

        # Core training call to native api execution structure
        self.model = xgb.train(
            params=self.params,
            dtrain=dtrain,
            num_boost_round=num_boost_round,
            evals=eval_set,
            early_stopping_rounds=stop_rounds,
            verbose_eval=verbose
        )


    def fine_tune(
        self, 
        train_loader: DataLoader, 
        lr: float = 0.01, 
        device: str = "cpu", 
        epochs: int = 5,
    ):
        """
        Continues training the existing booster on new data by appending trees.

        Args:
            train_loader (DataLoader): Calibration/fine-tuning PyTorch DataLoader.
            lr (float): Reduced learning rate for fine-tuning iterations. Defaults to 0.01.
            device (str): Device to run computations on. Defaults to "cpu".
            epochs (int): Number of additional boosting trees to append. Defaults to 5.
        """
        if self.model is None:
            raise ValueError("Cannot fine-tune an uninitialized model. Call train() first.")

        # Build stream iterator and QuantileDMatrix for calibration data
        ft_iter = self._WindowDataIter(train_loader, self._flatten_input, self.feature_mask, device)
        dft = xgb.QuantileDMatrix(ft_iter)

        # Create parameter copy with reduced learning rate
        ft_params = copy.deepcopy(self.params)
        ft_params.update({
            "learning_rate": lr,
            "device": device
        })

        logger.info(f"Fine-tuning {self.__class__.__name__} for {epochs} rounds with lr={lr}...")

        # Continue training using xgb_model parameter
        self.model = xgb.train(
            params=ft_params,
            dtrain=dft,
            num_boost_round=epochs,
            xgb_model=self.model,  # Keeps existing trees and appends new ones
            verbose_eval=False
        )



class XGBoostRandomForestRegressor(BaseXGBoostBackend):
    """
    Implementation of a parallel random forest tree regressor via XGBoost structures.

    Args:
        input_dim (int): Number of features of the input (not flattened over time). Placeholder for interface compatability.
        output_dim (int): Number of features of the output. Placeholder for interface compatability.
        seq_len (int): Lookback window sequence length. Placeholder for interface compatability.
        label_transformer (BaseTargetTransform): Instance of a label transformer.
        objective (str): The loss function for training (e.g. "reg:squarederror" or "reg:logistic").
        num_parallel_tree (int): Number of parallel trees built within a forest configuration. Defaults to 100.
        max_depth (int): Maximum depth of tree split structures. Defaults to 8.
        subsample (float): Subsample ratio of the training instances. Defaults to 0.8.
        colsample_bynode (float): Subsample ratio of columns per tree splitting action. Defaults to 0.8.
        random_state (int): Seed used by the random number generator tracking splits. Defaults to 42.
    """
    def __init__(
        self, 
        input_dim: int, 
        output_dim: int, 
        seq_len: int, 
        label_transformer: BaseTargetTransform, 
        objective: str, 
        num_parallel_tree: int = 100, 
        max_depth: int = 8, 
        subsample: float = 0.8, 
        colsample_bynode: float = 0.8, 
        random_state: int = 42,
    ):
        super().__init__(input_dim, output_dim, seq_len, label_transformer)

        self.params = {
            "objective": objective,
            "tree_method": "hist",
            "max_depth": max_depth,
            "learning_rate": 1.0,   # Constant for Random Forest         
            "subsample": subsample,       
            "colsample_bynode": colsample_bynode, 
            "num_parallel_tree": num_parallel_tree,    
            "seed": random_state
        }

        logger.info(f"Initialized {self}")


    def __str__(self):
        """Returns a string representation of the XGBoostRandomForestRegressor configuration."""
        params_str = ", ".join(f"{k}={v}" for k, v in self.params.items())
        return (f"XGBoostRandomForestRegressor(\n"
                f"  input_dim={self.input_dim}, output_dim={self.output_dim}, seq_len={self.seq_len},\n"
                f"  params={params_str}\n"
                f")")


    def train(
        self, 
        train_loader: DataLoader, 
        val_loader: DataLoader, 
        binary_mask: list, 
        eval_metrics: list = ["mae", "rmse"], 
        verbose: bool = True, 
        device: str = "cpu",
    ):
        """
        Executes parallel ensemble building in a single training iteration step.

        Args:
            train_loader (DataLoader): PyTorch DataLoader streaming training batches.
            val_loader (DataLoader): PyTorch DataLoader streaming validation batches.
            binary_mask (list): Numeric binary mask representing base frame feature types (1 for continuous, 0 for categorical).
            eval_metrics (list): Metric evaluation labels parsed by the booster. Defaults to ["mae", "rmse"].
            verbose (bool): Whether evaluation progress blocks display during iterations. Defaults to True.
            device (str): Computations backend processor device targeted ("cpu"/"cuda"). Defaults to "cpu".
        """
        # Update class attributes
        self.params.update({"eval_metric": eval_metrics, "device": device})
        self._infer_and_set_feature_mask(train_loader, binary_mask)

        # Initialize native Quantile stream iterators for training data
        train_iter = self._WindowDataIter(train_loader, self._flatten_input, self.feature_mask, device)
        dtrain = xgb.QuantileDMatrix(train_iter)

        # Set up validation monitoring targets
        eval_set = [(dtrain, 'train')]
        if val_loader is not None:
            val_iter = self._WindowDataIter(val_loader, self._flatten_input, self.feature_mask, device)
            dval = xgb.QuantileDMatrix(val_iter, ref=dtrain)
            eval_set.append((dval, 'val'))

        # Run training for exactly 1 round containing all parallel standalone estimators
        self.model = xgb.train(
            params=self.params,
            dtrain=dtrain,
            num_boost_round=1,  # RF constructs the full parallel ensemble in 1 iteration
            evals=eval_set,
            verbose_eval=verbose
        )


    def fine_tune(
        self, 
        train_loader: DataLoader, 
        lr: float = 0.05, 
        device: str = "cpu", 
        num_ft_trees: int = 10,
    ):
        """
        Continues training the existing booster on new data by appending trees.

        Args:
            train_loader (DataLoader): Calibration/fine-tuning PyTorch DataLoader.
            lr (float): Reduced learning rate for fine-tuning iterations. Defaults to 0.05.
            device (str): Device to run computations on. Defaults to "cpu".
            num_ft_trees (int): Number of parallel trees to fit during fine-tuning.
        """
        if self.model is None:
            raise ValueError("Cannot fine-tune an uninitialized model. Call train() first.")

        # Build stream iterator and QuantileDMatrix for calibration data
        ft_iter = self._WindowDataIter(train_loader, self._flatten_input, self.feature_mask, device)
        dft = xgb.QuantileDMatrix(ft_iter)

        # Create parameter copy with reduced learning rate
        ft_params = copy.deepcopy(self.params)
        ft_params.update({
            "learning_rate": lr,
            "num_parallel_tree": num_ft_trees,
            "device": device
        })

        logger.info(f"Fine-tuning {self.__class__.__name__} by appending {num_ft_trees} parallel trees with lr={lr}...")

        # Continue training
        self.model = xgb.train(
            params=ft_params,
            dtrain=dft,
            num_boost_round=1,
            xgb_model=self.model,
            verbose_eval=False
        )
