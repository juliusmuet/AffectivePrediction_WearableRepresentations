import copy
import logging
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from train.metrics import RegressionMetricsTracker
from train.trainer_nn import NeuralConfig, NeuralExecutionEngine
from explainability.shap_config import ShapConfig
from models_unified.mood_regressor import MoodRegressor
from datasets.datasets import MovementMoodDataset


logger = logging.getLogger(__name__)


@dataclass
class TrainingResults:
    """Standardized output container for trainer fit methods."""
    metrics_files: dict[str, Any] = field(default_factory=dict)
    summary_msg: str = ""
    global_shap_matrix: Optional[np.ndarray] = None
    global_x_matrix: Optional[np.ndarray] = None



class PopulationTrainer:
    """
    Orchestrate population-level model training across data splits.

    Routes training to the appropriate neural or hybrid-tabular execution pipeline, aggregates cross-validation metrics, 
    and optionally computes out-of-fold SHAP attributions.

    Args:
        neural_config: Configuration for neural network training. Defaults to None.
        custom_train_config: Keyword arguments passed to custom training routines for non-neural models. Defaults to None.
        shap_config: Configuration for SHAP computation. If None, SHAP values are not computed. Defaults to None.

    Raises:
        ValueError: If neither a neural nor a custom training configuration is provided.
    """
    def __init__(
        self,
        neural_config: NeuralConfig | None = None,
        custom_train_config: dict[str, Any] | None = None,
        shap_config: ShapConfig | None = None,
    ) -> None:
        if neural_config is None and custom_train_config is None:
            raise ValueError("Both 'neural_config' and 'custom_train_config' cannot be None. At least one must be provided.")
        
        self.neural_config = neural_config
        self.custom_train_config = custom_train_config
        self.shap_config = shap_config
        
        logger.info(f"Initialized {self}  \n")

    
    def __str__(self):
        """Returns a string representation of the trainer configuration."""
        return (f"PopulationTrainer(neural_config={self.neural_config}, custom_train_config={self.custom_train_config}, shap_config={self.shap_config})")


    def fit(self, model: MoodRegressor, dataset: MovementMoodDataset) -> TrainingResults:
        """
        Train a population model across the dataset's cross-validation splits.

        Creates an independent model copy for each split, routes training according to the model backend, collects fold-level metrics, 
        and aggregates results across folds. Optionally computes and concatenates out-of-fold SHAP attributions.

        Args:
            model: Model instance to be trained. Its backend must expose an is_neural_network attribute. 
                The model must also implement calculate_shap() if SHAP computation is enabled.
            dataset: Dataset providing get_splits(), get_dataloaders(), and binary_mask attributes.

        Returns:
            TrainingResults containing
                - metrics_files: Fold-level and aggregated training, validation, and test metrics.
                - summary_msg: Formatted cross-validation summary.
                - global_shap_matrix: Concatenated out-of-fold SHAP attributions with shape (N, Window_Size, Feature_Dim, Label_Dim), or None.
                - global_x_matrix: Corresponding input samples with shape (N, Window_Size, Feature_Dim), or None.
        """
        splits = dataset.get_splits()   # List of (train_idx, val_idx, test_idx)
        cv_train_metrics, cv_val_metrics, cv_test_metrics = [], [], []  # Store metrics of folds

        # SHAP value accumulators to stack fold-wise evaluation splits cleanly 
        shap_fold_accumulator = []
        x_fold_accumulator = []

        # Iterate through data partitions (Single Train, Single Train/Val/Test, or Multi-fold CV)
        progress_bar_folds = tqdm(enumerate(splits), desc=f"Training Folds")
        for fold_idx, (train_idx, val_idx, test_idx) in progress_bar_folds:
            logging.info(f"\n ==================== Starting training of fold {fold_idx+1} ====================")
            
            # Create a pristine clone of the model for this specific fold
            fold_model = copy.deepcopy(model)
            
            # Get dataloaders for given indices
            train_loader, val_loader, test_loader = dataset.get_dataloaders(train_idx, val_idx, test_idx)

            # Route data flow based on backend type
            if fold_model.backend.is_neural_network:
                train_metrics, val_metrics, test_metrics = self._fit_neural_network(fold_model, train_loader, val_loader, test_loader)
            else:
                train_metrics, val_metrics, test_metrics = self._fit_hybrid_tabular(fold_model, train_loader, val_loader, test_loader, dataset.binary_mask)

            # Collect metrics
            cv_train_metrics.append(train_metrics)
            if val_loader is not None:
                cv_val_metrics.append(val_metrics)
            if test_loader is not None:
                cv_test_metrics.append(test_metrics)

            # Out-of-Fold (OOF) SHAP tracking
            if self.shap_config is not None:
                # Determine which evaluation loader is active (prefers test, falls back to validation)
                eval_loader = test_loader if test_loader is not None else val_loader
                
                if eval_loader is not None:
                    # Extract 3D tensors of input data X from DataLoaders as numpy arrays
                    X_train_raw = torch.cat([batch[0] for batch in train_loader], dim=0).detach().cpu().numpy()
                    X_eval_raw = torch.cat([batch[0] for batch in eval_loader], dim=0).detach().cpu().numpy()
                    
                    # Execute strategy-specific end-to-end SHAP calculations on the trained fold model
                    fold_shap_values = fold_model.calculate_shap(X_train_raw, X_eval_raw, self.shap_config.n_samples, self.shap_config.seed)
                    
                    # Collect SHAP and eval data
                    shap_fold_accumulator.append(fold_shap_values)
                    x_fold_accumulator.append(X_eval_raw)

        # Compute aggregate metrics across folds
        summarised_cv_train_metrics, summarised_cv_val_metrics, summarised_cv_test_metrics = RegressionMetricsTracker.aggregate_cross_validation_runs(cv_train_metrics, cv_val_metrics, cv_test_metrics)
        
        # OOF SHAP: Merge individual segments
        if shap_fold_accumulator:
            global_shap_matrix = np.concatenate(shap_fold_accumulator, axis=0)
            global_x_matrix = np.concatenate(x_fold_accumulator, axis=0)
        else:
            global_shap_matrix, global_x_matrix = None, None

        # Construct output file map
        metrics_files = {
            "train_metrics.json": cv_train_metrics,
            "val_metrics.json": cv_val_metrics,
            "test_metrics.json": cv_test_metrics,
            "summarised_train_metrics.json": summarised_cv_train_metrics,
            "summarised_val_metrics.json": summarised_cv_val_metrics,
            "summarised_test_metrics.json": summarised_cv_test_metrics,
        }

        # Format summary log
        summary_msg = (
            f"\n==================== POPULATION TRAINING COMPLETE ====================\n"
            f"Train Metrics: " + " | ".join(f"{k.upper()}: {v:.4f}" for k, v in summarised_cv_train_metrics.get('global', {}).items())
        )
        if summarised_cv_val_metrics:
            summary_msg += f"\nVal   Metrics: " + " | ".join(f"{k.upper()}: {v:.4f}" for k, v in summarised_cv_val_metrics.get('global', {}).items())
        if summarised_cv_test_metrics:
            summary_msg += f"\nTest  Metrics: " + " | ".join(f"{k.upper()}: {v:.4f}" for k, v in summarised_cv_test_metrics.get('global', {}).items())
        summary_msg += f"\n================================================================"

        return TrainingResults(
            metrics_files=metrics_files,
            summary_msg=summary_msg,
            global_shap_matrix=global_shap_matrix,
            global_x_matrix=global_x_matrix
        )


    def _fit_neural_network(
        self,
        model: MoodRegressor,
        train_loader: DataLoader,
        val_loader: DataLoader | None,
        test_loader: DataLoader | None,
    ) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
        """
        Train a neural network model for a cross-validation split.

        Creates a fresh execution engine to initialize independent optimizer, scheduler, and gradient-scaling states for the current fold.

        Args:
            model: Neural network model instance for the current fold.
            train_loader: DataLoader providing training samples.
            val_loader: Optional DataLoader providing validation samples.
            test_loader: Optional DataLoader providing test samples.

        Returns:
            Tuple containing
                - train_metrics: Metrics associated with the best checkpoint.
                - val_metrics: Validation metrics associated with the best checkpoint, or an empty dictionary if no validation loader is provided.
                - test_metrics: Test metrics evaluated using the best checkpoint, or an empty dictionary if no test loader is provided.

        Raises:
            ValueError: If the neural network training configuration is missing.
        """
        if self.neural_config is None:
            raise ValueError("Requested neural network training but 'neural_config' is not defined.")
        
        # Factory Reset: Instantiates clean NN trainer with optimizers, scalers, learning rate scheduler to prevent parameter leaks between folds
        engine = NeuralExecutionEngine(model, train_loader, val_loader, test_loader, self.neural_config)
        return engine.run_full_training()


    def _fit_hybrid_tabular(
        self,
        model: MoodRegressor,
        train_loader: DataLoader,
        val_loader: DataLoader | None,
        test_loader: DataLoader | None,
        binary_mask: np.ndarray,
    ) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
        """
        Train a hybrid-tabular model for a cross-validation split.

        Delegates training to the model backend using the configured custom training parameters.

        Args:
            model: Hybrid-tabular model instance for the current fold.
            train_loader: DataLoader providing training samples.
            val_loader: Optional DataLoader providing validation samples.
            test_loader: Optional DataLoader providing test samples.
            binary_mask: Feature mask with shape (Feature_Dim,). A value of 1 indicates a continuous feature and 0 indicates a discrete feature.

        Returns:
            Tuple containing
                - train_metrics: Metrics associated with the best checkpoint.
                - val_metrics: Validation metrics associated with the best checkpoint, or an empty dictionary if no validation loader is provided.
                - test_metrics: Test metrics evaluated using the best checkpoint, or an empty dictionary if no test loader is provided.
        
        Raises:
            ValueError: If the custom training configuration is missing.
        """
        if self.custom_train_config is None:
            raise ValueError("Requested hybrid-tabular training , but 'custom_train_config' is not defined.")
        
        return model.backend.run_full_training(train_loader, val_loader, test_loader, binary_mask, **self.custom_train_config)



class ParticipantTrainer:
    """
    Orchestrate participant-level personalization and evaluation.

    For each outer cross-validation fold, trains a population model and evaluates it on held-out participants.
    Each participant's data is then divided chronologically into personalization and evaluation subsets.
    The model is evaluated before and after participant-specific fine-tuning.

    Optionally computes SHAP attributions using the fine-tuned models.

    Args:
        neural_config: Configuration for general neural network training. Defaults to None.
        custom_train_config: Keyword arguments for general non-neural model training. Defaults to None.
        neural_ft_config: Configuration for neural network fine-tuning. Defaults to None.
        custom_ft_config: Keyword arguments for non-neural model fine-tuning. Defaults to None.
        shap_config: Configuration for SHAP computation. If None, SHAP values are not computed. Defaults to None.
        finetune_ratio: Fraction of each held-out participant's samples allocated to personalization. 
            Must be strictly between 0 and 1. Defaults to 0.5.

    Raises:
        ValueError: If no training configuration is provided, a required fine-tuning configuration is missing, 
            or finetune_ratio is outside its valid range.
    """
    def __init__(
        self,
        neural_config: NeuralConfig | None = None,
        custom_train_config: dict[str, Any] | None = None,
        neural_ft_config: NeuralConfig | None = None,
        custom_ft_config: dict[str, Any] | None = None,
        shap_config: ShapConfig | None = None,
        finetune_ratio: float = 0.5,
    ):
        if neural_config is None and custom_train_config is None:
            raise ValueError("Either 'neural_config' or 'custom_train_config' must be provided.")
        if neural_config is not None and neural_ft_config is None:
            raise ValueError("'neural_ft_config' must be provided")
        if custom_train_config is not None and custom_ft_config is None:
            raise ValueError("'custom_ft_config' must be provided")
        
        self.neural_config = neural_config
        self.custom_train_config = custom_train_config
        self.neural_ft_config = neural_ft_config
        self.custom_ft_config = custom_ft_config
        self.shap_config = shap_config
        self.finetune_ratio = finetune_ratio

        logger.info(f"Initialized {self}\n")
        
            
    def __str__(self):
        """Returns a string representation of the trainer configuration."""
        return (f"ParticipantTrainer(neural_config={self.neural_config}, custom_train_config={self.custom_train_config}, "
                f"neural_ft_config={self.neural_ft_config}, custom_ft_config={self.custom_ft_config}, "
                f"shap_config={self.shap_config}, finetune_ratio={self.finetune_ratio})")


    def fit(self, model: MoodRegressor, dataset: MovementMoodDataset) -> TrainingResults:
        """
        Train and evaluate population models with participant personalization.

        For each outer cross-validation fold, trains a population model, evaluates zero-shot performance on held-out participants, 
        fine-tunes an independent copy of the population model for each participant, and evaluates the adapted model on the participant's evaluation data.

        Optionally computes SHAP attributions using each fine-tuned model.

        Args:
            model: Model instance used to initialize the population models. Its backend must expose an is_neural_network attribute.
            dataset: Dataset providing get_splits(), get_dataloaders(), and participant_ids attributes.
                Participant samples are assumed to be chronologically ordered within each outer test split.

        Returns:
            TrainingResults containing
                - metrics_files: Fold-level and participant-level zero-shot and fine-tuned metrics and their aggregated summaries.
                - summary_msg: Formatted comparison of zero-shot and fine-tuned performance.
                - global_shap_matrix: Concatenated fine-tuned SHAP attributions with shape (N, Window_Size, Feature_Dim, Label_Dim), or None.
                - global_x_matrix: Corresponding participant evaluation inputs with shape (N, Window_Size, Feature_Dim), or None.

        Raises:
            ValueError: If no cross-validation splits are available or a required model-specific training configuration is missing.
        """
        splits = dataset.get_splits()
        if not splits:
            raise ValueError("The dataset must provide at least one split.")
        
        cv_zero_shot_metrics = []
        cv_fine_tuned_metrics = []

        # SHAP value accumulators across folds and participants
        shap_fold_accumulator = []
        x_fold_accumulator = []

        for fold_idx, (train_idx, val_idx, test_idx) in tqdm(enumerate(splits), desc="Training Folds"):
            logging.info(f"\n ==================== Starting training of fold {fold_idx+1} ====================")
            
            # Train General Population Model
            pop_model = copy.deepcopy(model)
            train_loader, val_loader, test_loader = dataset.get_dataloaders(train_idx, val_idx, test_idx)
            
            if pop_model.backend.is_neural_network:
                if self.neural_config is None:
                    raise ValueError("Requested neural network training but 'neural_config' is not defined.")
                engine = NeuralExecutionEngine(pop_model, train_loader, val_loader, test_loader, self.neural_config)
                engine.run_full_training()
            else:
                if self.custom_train_config is None:
                    raise ValueError("Requested hybrid-tabular training , but 'custom_train_config' is not defined.")
                pop_model.backend.run_full_training(train_loader, val_loader, test_loader, dataset.binary_mask, **self.custom_train_config)

            # Iterate over held-out test participants
            test_participant_ids = np.unique(dataset.participant_ids[test_idx])
            
            fold_zero_shot = {"by_participant": {}}
            fold_fine_tuned = {"by_participant": {}}

            for pid in tqdm(test_participant_ids, desc=f"Fine-Tuning Fold {fold_idx+1} Participants", leave=False):
                logger.info(f"Fine-tuning of participant {pid}")
                
                # Isolate indices for the specific participant within test_idx
                pid_mask = (dataset.participant_ids[test_idx] == pid)
                pid_indices = test_idx[pid_mask]

                if len(pid_indices) < 2:
                    logger.warning(f"Participant {pid} has insufficient samples ({len(pid_indices)}) for fine-tuning split. Skipping.")
                    continue

                # Sequential Temporal Split (First X% for fine-tuning, remaining for evaluation)
                n_ft_samples = max(1, int(len(pid_indices) * self.finetune_ratio))
                p_train_idx = pid_indices[:n_ft_samples]
                p_test_idx = pid_indices[n_ft_samples:]

                if len(p_test_idx) == 0:
                    continue

                # Create personal DataLoaders
                p_train_loader, _, p_test_loader = dataset.get_dataloaders(p_train_idx, p_test_idx, p_test_idx, is_finetuning=True)

                # Zero-Shot Evaluation
                zero_shot_eval = self._evaluate_model(pop_model, p_train_loader, p_test_loader, description="Zero-Shot Evaluation")
                zero_shot_eval.pop("by_participant", None)  # As there is only one participant, the "by_participant" data is already covered by "global" and "by_dimension" keys
                fold_zero_shot["by_participant"][str(pid)] = zero_shot_eval

                # Fine-Tune General Model
                adapted_model = copy.deepcopy(pop_model)
                self._fine_tune_model(adapted_model, p_train_loader)

                # Post Fine-Tuning Evaluation
                fine_tuned_eval = self._evaluate_model(adapted_model, p_train_loader, p_test_loader, description="Fine-Tune Evaluation")
                fine_tuned_eval.pop("by_participant", None) # As there is only one participant, the "by_participant" data is already covered by "global" and "by_dimension" keys
                fold_fine_tuned["by_participant"][str(pid)] = fine_tuned_eval

                # Fine-Tuned SHAP Calculation
                if self.shap_config is not None:
                    # Extract 3D tensors of input data X from DataLoaders as numpy arrays from participant train and test DataLoaders
                    X_p_train_raw = torch.cat([batch[0] for batch in p_train_loader], dim=0).detach().cpu().numpy()
                    X_p_test_raw = torch.cat([batch[0] for batch in p_test_loader], dim=0).detach().cpu().numpy()

                    # Calculate SHAP values using the fine-tuned participant model
                    p_shap_values = adapted_model.calculate_shap(X_p_train_raw, X_p_test_raw, self.shap_config.n_samples, self.shap_config.seed)

                    shap_fold_accumulator.append(p_shap_values)
                    x_fold_accumulator.append(X_p_test_raw)

            cv_zero_shot_metrics.append(fold_zero_shot)
            cv_fine_tuned_metrics.append(fold_fine_tuned)

        # Aggregate across participants
        sum_zero_shot = RegressionMetricsTracker.aggregate_participant_metrics(cv_zero_shot_metrics)
        sum_fine_tuned = RegressionMetricsTracker.aggregate_participant_metrics(cv_fine_tuned_metrics)

        # Merge participant SHAP segments across folds
        if shap_fold_accumulator:
            global_shap_matrix = np.concatenate(shap_fold_accumulator, axis=0)
            global_x_matrix = np.concatenate(x_fold_accumulator, axis=0)
        else:
            global_shap_matrix, global_x_matrix = None, None

        metrics_files = {
            "zero_shot_metrics.json": cv_zero_shot_metrics,
            "fine_tuned_metrics.json": cv_fine_tuned_metrics,
            "summarised_zero_shot_metrics.json": sum_zero_shot,
            "summarised_fine_tuned_metrics.json": sum_fine_tuned,
        }

        # Pre-format summary for logging
        zs_mae = sum_zero_shot.get("global", {}).get("mae_mean", 0.0)
        ft_mae = sum_fine_tuned.get("global", {}).get("mae_mean", 0.0)
        zs_rmse = sum_zero_shot.get("global", {}).get("rmse_mean", 0.0)
        ft_rmse = sum_fine_tuned.get("global", {}).get("rmse_mean", 0.0)
        zs_r2 = sum_zero_shot.get("global", {}).get("r2_mean", 0.0)
        ft_r2 = sum_fine_tuned.get("global", {}).get("r2_mean", 0.0)
        summary_msg = (
            f"\n==================== PARTICIPANT PERSONALIZATION COMPLETE ====================\n"
            f"Evaluated Participants : {sum_fine_tuned.get('participant_count', 0)}\n"
            f"Zero-Shot  Metrics     : MAE: {zs_mae:.4f} | RMSE: {zs_rmse:.4f} | R^2: {zs_r2:.4f}\n"
            f"Fine-Tuned Metrics     : MAE: {ft_mae:.4f} | RMSE: {ft_rmse:.4f} | R^2: {ft_r2:.4f}\n"
            f"=============================================================================="
        )

        return TrainingResults(
            metrics_files=metrics_files,
            summary_msg=summary_msg,
            global_shap_matrix=global_shap_matrix,
            global_x_matrix=global_x_matrix
        )


    def _fine_tune_model(self, model: MoodRegressor, train_loader: DataLoader) -> None:
        """
        Fine-tune a model using participant-specific calibration data.

        Routes fine-tuning to the neural or hybrid-tabular training pipeline according to the model backend.

        Args:
            model: Population model copy to be adapted for the participant.
            train_loader: DataLoader providing participant-specific personalization samples.

        Raises:
            ValueError: If the required fine-tuning configuration is missing.
        """
        if model.backend.is_neural_network:
            # Fine-tune network using engine
            if self.neural_ft_config is None:
                raise ValueError("Requested neural network fine-tuning, but 'neural_ft_config' is not defined.")
            engine = NeuralExecutionEngine(model, train_loader, None, None, self.neural_ft_config)
            engine.run_full_training()
        else:
            if self.custom_ft_config is None:
                raise ValueError("Requested hybrid-tabular fine-tuning, but 'custom_ft_config' is not defined.")
            model.backend.fine_tune(train_loader, **self.custom_ft_config)


    def _evaluate_model(
        self,
        model: MoodRegressor,
        train_loader: DataLoader,
        eval_loader: DataLoader,
        description: str = "Validation",
    ) -> dict[str, Any]:
        """
        Evaluate a model on a participant-specific evaluation dataset.

        Neural models are evaluated through NeuralExecutionEngine.
        Hybrid-tabular models delegate evaluation to their backend.

        Args:
            model: Model instance to evaluate.
            train_loader: DataLoader used to initialize the neural execution engine. Its samples are not used for metric calculation.
            eval_loader: DataLoader providing evaluation samples.
            description: Description used for evaluation progress reporting. Defaults to "Validation".
            neural_config: Neural execution configuration used for device and inference settings. Defaults to None.

        Returns:
            Nested dictionary containing global, dimensional, and participant-level evaluation metrics.
        """
        if model.backend.is_neural_network:
            engine = NeuralExecutionEngine(model, train_loader, eval_loader, eval_loader, self.neural_config)
            return engine.evaluate(eval_loader, description=description)
        else:
            return model.backend.evaluate(eval_loader, device=self.custom_train_config.get("device", "cpu"), description=description)
