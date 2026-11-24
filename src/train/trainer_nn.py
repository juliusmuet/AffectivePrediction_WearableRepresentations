import copy
import logging
from dataclasses import dataclass
from typing import Any, Callable, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from train.metrics import RegressionMetricsTracker
from utils.utils import get_elapsed_remaining_time_from_progress_bar


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class NeuralConfig:
    """
    Configuration profile for neural network training.

    Defines optimizer and scheduler factories, loss construction, hardware settings, and training-loop parameters.

    Args:
        optimizer_factory: Factory that creates a new optimizer for a model and its trainable parameters.
        lr_scheduler_factory: Optional factory that creates a learning rate scheduler for an optimizer. Receives the optimizer, the number
            of optimizer updates per epoch, and the total number of epochs.
        loss_factory: Factory that creates the loss function.
        use_mixed_precision: Whether to enable automatic mixed-precision training.
        num_epochs: Maximum number of training epochs.
        early_stopping_patience: Number of consecutive epochs without validation improvement before stopping.
        accumulation_interval: Number of batches over which gradients are accumulated before an optimizer update.
        log_interval: Number of batches between training log messages. Set to 0 to disable batch-level logging.
        device: Execution device, such as "cuda", "cpu", or "mps".
    """
    optimizer_factory: Callable[[nn.Module, list[nn.Parameter]], torch.optim.Optimizer]
    lr_scheduler_factory: Optional[Callable[[torch.optim.Optimizer, int, int], Any]]
    loss_factory: Callable[[], nn.Module]
    use_mixed_precision: bool
    num_epochs: int
    early_stopping_patience: int
    accumulation_interval: int
    log_interval: int
    device: str


class NeuralExecutionEngine:
    """
    Manage the training and evaluation lifecycle of a neural network.

    Executes training for a single data split, supporting gradient accumulation, optional automatic mixed precision, learning rate
    scheduling, validation-based early stopping, and best-checkpoint restoration.

    Args:
        model: Neural network model to train. It must expose a backend attribute, a get_trainable_parameters_frontend() method,
            and a state_dict() compatible with PyTorch modules.
        train_dataloader: DataLoader providing training batches.
        val_dataloader: Optional DataLoader providing validation batches.
        test_dataloader: Optional DataLoader providing test batches.
        neural_config: Configuration defining the training procedure.
    """

    def __init__(
        self,
        model: nn.Module,
        train_dataloader: DataLoader,
        val_dataloader: DataLoader | None,
        test_dataloader: DataLoader | None,
        neural_config: NeuralConfig,
    ):
        self.model = model
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.test_dataloader = test_dataloader
        
        self.num_epochs = neural_config.num_epochs
        self.early_stopping_patience = neural_config.early_stopping_patience
        self.log_interval = neural_config.log_interval
        self.accumulation_interval = neural_config.accumulation_interval
        self.steps_per_epoch = int(np.ceil(len(train_dataloader) / self.accumulation_interval))
        
        self.optimizer = neural_config.optimizer_factory(model.backend, model.get_trainable_parameters_frontend())
        self.lr_scheduler = neural_config.lr_scheduler_factory(self.optimizer, self.steps_per_epoch, self.num_epochs) if neural_config.lr_scheduler_factory is not None else None
        self.scaler = torch.amp.GradScaler(enabled=neural_config.use_mixed_precision)
        self.device = neural_config.device
        
        self.criterion = neural_config.loss_factory()


    def train_epoch(self, epoch_num: int) -> dict[str, Any]:
        """
        Run one training epoch and calculate training metrics.

        Performs forward and backward passes over the training dataloader, accumulates gradients, 
        updates model parameters, and optionally advances the learning rate scheduler.

        Args:
            epoch_num: Current epoch index, starting at 1.

        Returns:
            Nested dictionary containing training metrics
                - global: Metrics across all training samples and target dimensions.
                - by_dimension: Metrics computed independently for each target dimension.
                - by_participant: Metrics grouped by participant ID. Each participant entry contains global and by_dimension dictionaries.

        Raises:
            ValueError: If a batch does not contain exactly three elements.
        """
        self.model.to(self.device)  # Move model to device
        self.model.train()  # Set model to train mode
        
        # Define variables
        num_batches = len(self.train_dataloader)
        epoch_preds, epoch_trues, epoch_pids = [], [], []

        # Set current steps in schedulers if defined (needed if epoch_num is not 1)
        current_step = (epoch_num - 1) * self.steps_per_epoch
        if self.lr_scheduler:
            self.lr_scheduler.step(current_step)

        # Wrap the data loader with tqdm for a progress bar
        progress_bar = tqdm(self.train_dataloader, desc=f"Epoch {epoch_num} Train", leave=False)

        # Batch training loop
        for batch_idx, batch_data in enumerate(progress_bar):
            # Ensure batch_data has the correct number of elements
            if len(batch_data) != 3:
                raise ValueError(f"DataLoader expected to yield 3 items (inputs, mood labels, p_ids), but got {len(batch_data)} items.")
            
            # Get data and put on device
            x, y, p_ids = batch_data
            x, y = x.to(self.device), y.to(self.device)

            with torch.amp.autocast(self.device):
                predictions = self.model(x) # Get predictions
                batch_loss = self.criterion(predictions, y) # Calculate batch loss
                batch_loss = batch_loss / self.accumulation_interval # Normalize loss for gradient accumulation

            self.scaler.scale(batch_loss).backward()    # Accumulate gradients
            
            # Store predictions, labels and p_ids for evaluation (detached to prevent memory leaks)
            epoch_preds.append(predictions.detach().cpu().numpy())
            epoch_trues.append(y.detach().cpu().numpy())
            epoch_pids.append(p_ids)

            # Update weights and schedulers after accumulation_steps or after last batch
            if (batch_idx + 1) % self.accumulation_interval == 0 or (batch_idx + 1) == num_batches:
                # Unscale before clipping gradients
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

                self.scaler.step(self.optimizer)    # Update weights
                self.scaler.update()    # Update scaler
                self.optimizer.zero_grad()  # Delete all gradients
                
                # Step the schedulers
                current_step += 1
                if self.lr_scheduler:
                    self.lr_scheduler.step(current_step)
            
            # Log after self.log_interval many batches and after first batch if self.log_interval is not 0
            if self.log_interval != 0 and ((batch_idx + 1) % self.log_interval == 0 or batch_idx == 0):
                log_msg = (
                    f"Epoch: {epoch_num} [{batch_idx + 1}/{num_batches} ({100. * (batch_idx + 1) / num_batches:.0f}%)] | "
                    f"Batch Loss: {batch_loss.item()*self.accumulation_interval:.4f}"
                )
                log_msg += f" | LR:" + ", ".join(f"{pg['lr']:.6e}" for pg in self.optimizer.param_groups)
                elapsed_str, remaining_str = get_elapsed_remaining_time_from_progress_bar(progress_bar)
                log_msg += f" | Elapsed: {elapsed_str} | ETA: {remaining_str}"
                logger.info(log_msg)

            # Update progress bar after every batch
            progress_bar.set_postfix({
                'Loss': f"{batch_loss.item()*self.accumulation_interval:.4f}",
                'LR': ", ".join(f"{pg['lr']:.6e}" for pg in self.optimizer.param_groups)
            })

        # Concatenate individual batch arrays into complete epoch matrices
        y_pred_all = np.concatenate(epoch_preds, axis=0)
        y_true_all = np.concatenate(epoch_trues, axis=0)
        p_ids_all = np.concatenate(epoch_pids, axis=0)
        
        # Compute & return train metrics
        return RegressionMetricsTracker.evaluate_predictions(y_true_all, y_pred_all, p_ids_all, self.model.backend.label_transformer)


    def evaluate(self, dataloader: DataLoader, description: str = "Validation") -> dict[str, Any]:
        """
        Evaluate the current model on a validation or test dataset.

        Runs inference without gradient tracking and computes global, dimensional, and participant-level metrics.

        Args:
            dataloader: DataLoader providing evaluation batches.
            description: Description used in the inference progress bar. Defaults to "Validation".

        Returns:
            Nested dictionary containing evaluation metrics
                - global: Metrics across all evaluation samples and target dimensions.
                - by_dimension: Metrics computed independently for each target dimension.
                - by_participant: Metrics grouped by participant ID.

        Raises:
            ValueError: If  a batch does not contain exactly three elements.
        """
        self.model.to(self.device)  # Move model to device
        self.model.eval()   # Set model to evaluation mode
        epoch_preds, epoch_trues, epoch_pids = [], [], []   # Declare variables
        
        # Iterate over evaluation samples
        with torch.no_grad():   # No gradients tracked during evaluation
            progress_bar_inference = tqdm(dataloader, desc=f"Batch Inference {description}")
            for batch_data in progress_bar_inference:
                if len(batch_data) != 3:
                    raise ValueError(f"DataLoader expected to yield 3 items (inputs, mood labels, p_ids), but got {len(batch_data)} items.")
                
                # Get data and put on device
                x, y, p_ids = batch_data
                x, y = x.to(self.device), y.to(self.device)
                
                # Get & Store predictions, labels and p_ids for evaluation (detached to prevent memory leaks)
                predictions = self.model(x)
                epoch_preds.append(predictions.cpu().numpy())
                epoch_trues.append(y.cpu().numpy())
                epoch_pids.append(p_ids)

        # Concatenate individual batch arrays into matrices for complete evaluation dataset
        y_pred_all = np.concatenate(epoch_preds, axis=0)
        y_true_all = np.concatenate(epoch_trues, axis=0)
        p_ids_all = np.concatenate(epoch_pids, axis=0)
        
        # Compute & return evaluation metrics
        return RegressionMetricsTracker.evaluate_predictions(y_true_all, y_pred_all, p_ids_all, self.model.backend.label_transformer)


    def run_full_training(self) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
        """
        Execute the complete training, validation, and testing workflow.

        Trains the model for the configured number of epochs, optionally applies validation-based early stopping, 
        restores the best model checkpoint, and evaluates the restored model on the test dataset.

        When no validation dataloader is provided, the best checkpoint is selected using training MSE instead.

        Returns:
            Tuple containing
                - best_train_metrics: Training metrics from the epoch associated with the best checkpoint.
                - best_val_metrics: Validation metrics from the best checkpoint, or an empty dictionary if no validation dataloader is provided.
                - test_metrics: Metrics on the test dataset using the restored best checkpoint, or an empty dictionary if no test dataloader is provided.
        """
        # Initialise variables
        best_train_metrics, best_val_metrics, test_metrics = {}, {}, {}
        best_epoch = 1
        lowest_val_loss = float('inf')
        early_stopping_counter = 0
        best_model_weights = copy.deepcopy(self.model.state_dict())

        # Iterate over epochs
        for epoch in range(1, self.num_epochs + 1):
            # Train model for one epoch
            current_train_metrics = self.train_epoch(epoch)
            
            # Log train metrics of epoch
            logger.info(f"--- Epoch {epoch} Summary ---")
            train_str = " | ".join(f"{key.upper()}: {value:.4f}" for key, value in current_train_metrics['global'].items())
            logger.info(f"Train: {train_str}")
            
            # Early stopping if validation dataloader is given
            if self.val_dataloader:
                current_val_metrics = self.evaluate(self.val_dataloader, description="Validation")
                val_str = " | ".join(f"{key.upper()}: {value:.4f}" for key, value in current_val_metrics['global'].items())
                logger.info(f"Val:   {val_str}")
                
                # Monitor Early Stopping via Validation MSE criteria
                current_val_loss = current_val_metrics['global']['mse']
                # Update values of new best epoch
                if current_val_loss < lowest_val_loss:
                    lowest_val_loss = current_val_loss
                    best_epoch = epoch
                    early_stopping_counter = 0
                    best_model_weights = copy.deepcopy(self.model.state_dict())
                    best_train_metrics = copy.deepcopy(current_train_metrics)
                    best_val_metrics = copy.deepcopy(current_val_metrics)
                # Not new best epoch, so step the patience
                else:
                    if self.early_stopping_patience is not None:
                        early_stopping_counter += 1
                        logger.info(f" | Patience: [{early_stopping_counter}/{self.early_stopping_patience}]")
            
            # No validation dataloader, so track only best train metrics
            else:
                current_train_loss = current_train_metrics['global']['mse']
                if current_train_loss < lowest_val_loss:
                    lowest_val_loss = current_train_loss
                    best_epoch = epoch
                    best_model_weights = copy.deepcopy(self.model.state_dict())
                    best_train_metrics = copy.deepcopy(current_train_metrics)

            logger.info("\n")

            # Trigger early stopping if threshold boundaries are breached
            if self.early_stopping_patience and early_stopping_counter >= self.early_stopping_patience:
                logger.info(f"Early stopping triggered at epoch {epoch}. Restoring weights from epoch {best_epoch}.")
                break

        # Restore optimal parameters back to the model instance before final testing
        self.model.load_state_dict(best_model_weights)

        # Process final validation holdout tests
        if self.test_dataloader:
            logger.info(f"Evaluating optimal model checkpoint (Epoch {best_epoch}) against holdout test loader...")
            test_metrics = self.evaluate(self.test_dataloader, description="Test")

        # Summary logging blocks
        summary_msg = (
            f"\n==================== FOLD TRAINING COMPLETE ====================\n"
            f"Best Epoch: {best_epoch}\n"
            f"Best Train Metrics: " + " | ".join(f"{k.upper()}: {v:.4f}" for k, v in best_train_metrics.get('global', {}).items())
        )
        if best_val_metrics:
            summary_msg += f"\nBest Val   Metrics: " + " | ".join(f"{k.upper()}: {v:.4f}" for k, v in best_val_metrics.get('global', {}).items())
        if test_metrics:
            summary_msg += f"\nBest Test  Metrics: " + " | ".join(f"{k.upper()}: {v:.4f}" for k, v in test_metrics.get('global', {}).items())
        summary_msg += f"\n================================================================\n"
        logger.info(summary_msg)

        return best_train_metrics, best_val_metrics, test_metrics
