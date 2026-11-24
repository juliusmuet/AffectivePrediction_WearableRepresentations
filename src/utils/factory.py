import logging
import inspect
from pathlib import Path
from typing import Any
from collections.abc import Callable

from torch import nn
from torch.optim import Optimizer

from utils.utils import set_seed
import utils.config_loader as config_loader
from data_extractors.extractor import Extractor
from datasets.datasets import MovementMoodDataset
from preprocessing.timeseries_scaler import FeatureScaler
from preprocessing.label_transformer import BaseTargetTransform
from train.lr_scheduler import SchedulerWithFixedLR
from train.trainer_nn import NeuralConfig
from train.trainer import PopulationTrainer, ParticipantTrainer
from models_unified.mood_regressor import MoodRegressor
from explainability.shap_config import ShapConfig
import utils.model_getter as model_getter
from utils.utils import get_device


logger = logging.getLogger(__name__)


class Factory:
    """
    Factory class responsible for creating and initialising all components of the pipeline.
    It loads configuration from a YAML file and uses it to instantiate data, models, loss functions,
    optimisers and schedulers.

    Args:
        config_path (str): Path to the configuration YAML file.
        checkpoint_path (str): Path to a model checkpoint file (default: None). Used for testing and inference.
    """
    def __init__(self, config_path: str, checkpoint_path: str = None):
        # Get root config
        self.config_directory = Path(config_path).parent
        self.config = config_loader.load_config_file(Path(config_path))
        
        # Setup checkpoint path if given
        self.model_checkpoint_path = Path(checkpoint_path) if checkpoint_path else None
        
        # Get device for Neural Network Training & Inference
        self.device = get_device()

        # Declare attributes (initialised in create methods)
        self.label_transformer = None
        self.movement_mood_dataset = None
        self.optimizer_factory = None
        self.lr_scheduler_factory = None
        self.loss_factory = None
        self.neural_config = None
        self.neural_config_finetune = None
        self.shap_config = None
        self.model = None
        self.trainer = None

        # Get configs for components
        self.seed = self.config.get('seed', 42)
        self.config_studies = config_loader.get_study_configs(self.config)
        self.config_extractor = config_loader.get_extractor_config(self.config)
        self.config_label_transformer = config_loader.get_label_transformer_config(self.config)
        self.config_dataset_split = config_loader.get_dataset_split_config(self.config)
        self.config_dataloader = config_loader.get_dataloader_config(self.config)
        self.config_scaler = config_loader.get_data_scaler_config(self.config)
        self.config_data_scheduler = config_loader.get_data_scheduler_config(self.config)
        self.config_train = config_loader.get_train_config(self.config)
        self.config_loss = config_loader.get_loss_config(self.config)
        self.config_frontend = config_loader.get_frontend_config(self.config)
        self.config_backend = config_loader.get_backend_config(self.config)
        self.config_optimiser = config_loader.get_optimiser_config(self.config)
        self.config_lr_scheduler = config_loader.get_lr_scheduler_config(self.config)
        self.config_shap = config_loader.get_shap_config(self.config)
        self.config_finetune = config_loader.get_finetune_config(self.config)

        # set seed
        set_seed(self.seed)


    def create_label_transformer(self, fresh: bool = False) -> BaseTargetTransform:
        """
        Creates the label transformer instance.

        Args:
            fresh (bool): If True, bypasses the cached label transformer and forces recreation. Defaults to False.

        Returns:
            BaseTargetTransform: The configured label transformer instance.
        """
        if self.label_transformer is not None:
            return self.label_transformer

        label_transformer_class = model_getter.get_label_transformer(self.config_label_transformer.get("transformer_name"))
        self.label_transformer = label_transformer_class(**self.config_label_transformer.get("transformer_args", {}))

        return self.label_transformer


    def create_movement_mood_dataset(self, fresh: bool = False) -> MovementMoodDataset:
        """
        Loads data and builds the MovementMoodDataset instance.

        Args:
            fresh (bool): If True, bypasses the cached dataset and forces recreation. Defaults to False.

        Returns:
            MovementMoodDataset: The configured movement mood dataset instance.
        """
        if not fresh and self.movement_mood_dataset:
            return self.movement_mood_dataset
        
        # Determine one-hot-encoding of categorical input features
        backend_class = model_getter.get_backend_model(self.config_backend.get("model_name"))
        attr = getattr(backend_class, "requires_one_hot_encoding", False)
        if isinstance(attr, property):
            one_hot_encoding = attr.fget(None)  # Call the underlying function of the property, passing None as 'self'
        elif callable(attr):
            one_hot_encoding = attr(None)
        else:
            one_hot_encoding = bool(attr)

        # Create label transformer
        label_transformer = self.create_label_transformer()
        
        # Load data
        extractor = Extractor(self.config_studies, one_hot_encoding=one_hot_encoding, label_transformer=label_transformer, **self.config_extractor)
        data, labels, participant_ids, binary_mask, feature_names, label_names, cardinality = extractor.load_data_label_pairs()

        # Setup data scaler
        feature_scaler = None
        if self.config_scaler is None:
            logger.info("No feature scaler configured.")
        else:
            feature_scaler = FeatureScaler(**self.config_scaler)

        # Setup data scheduler
        data_scheduler = None
        if self.config_data_scheduler is None:
            logger.info("No data scheduler configured.")
        else:
            scheduler_class = model_getter.get_data_scheduler(self.config_data_scheduler.get("scheduler_name"))
            scheduler_args = self.config_data_scheduler.get("scheduler_args", {}) or {}
            data_scheduler = scheduler_class(**scheduler_args)

        # Setup data splitter
        dataset_splitter = None
        if self.config_dataset_split is None:
            logger.info("No dataset split configured.")
        else:
            dataset_splitter = model_getter.get_dataset_splitter(self.config_dataset_split.get('splitter_name'))(**self.config_dataset_split.get('splitter_args'))

        # Setup Dataset
        self.movement_mood_dataset = MovementMoodDataset(data, labels, participant_ids, binary_mask, feature_names, label_names, cardinality, 
                                                         self.config_dataloader, scaler=feature_scaler, splitter=dataset_splitter, scheduler=data_scheduler)
        return self.movement_mood_dataset
    

    def create_optimizer_factory(self, config_optimiser: dict = None, fresh: bool = False) -> Callable[[nn.Module, nn.Module | None], Optimizer]:
        """
        Builds a blueprint factory function for an optimizer.

        Args:
            config_optimiser (dict, optional): Optimizer configuration dict. 
                                               If None, defaults to self.config_optimiser.
            fresh (bool): If True, forces creation of a new factory function.

        Returns:
            callable: A function expecting (backend_model, frontend_params) that returns an initialized Optimizer.
        """
        # Use supplied config or fall back to class instance attribute
        target_config = config_optimiser if config_optimiser is not None else self.config_optimiser

        if target_config is None:
            raise ValueError("No optimizer configuration provided or configured.")

        # Extract args from the chosen config dict
        optimiser_args = target_config.get("optimiser_args", {})
        backend_cfg = optimiser_args.get("backend", {})
        frontend_cfg = optimiser_args.get("frontend", {})
        other_args = optimiser_args.get("other_args", {})

        # Get Optimizer class
        OptimizerClass = model_getter.get_optimiser(target_config.get("optimiser_name"))

        def build_optimizer(backend_model, frontend_params=None):
            """
            Creates optimizer with backend + frontend parameter groups.
            Supports models with custom parameter group rules (e.g. FT-Transformer).
            """
            param_groups = []

            # Handle Backend Model Custom Parameter Groups
            if hasattr(backend_model, "make_parameter_groups") and callable(backend_model.make_parameter_groups):
                backend_groups = backend_model.make_parameter_groups()
                for idx, group in enumerate(backend_groups):
                    param_groups.append({
                        **backend_cfg,                      # Global backend settings (lr, etc.)
                        **group,                            # Group overrides (e.g. weight_decay=0.0)
                        "name": f"backend_group_{idx}",
                    })
            else:
                # Fallback for standard models
                param_groups.append({
                    **backend_cfg,
                    "params": backend_model.parameters(),
                    "name": "backend",
                })

            # Handle Frontend Model Parameter Groups (Optional)
            if frontend_params:
                param_groups.append({
                    **frontend_cfg,
                    "params": frontend_params,
                    "name": "frontend",
                })

            # Logging & Instantiation
            num_backend_groups = len(param_groups) - (1 if frontend_params else 0)
            log_msg = (f"Initialized optimizer: {OptimizerClass.__name__}({num_backend_groups} backend group(s) with cfg {backend_cfg}")
            if frontend_params:
                log_msg += f", 1 frontend group with cfg {frontend_cfg})"
            else:
                log_msg += ", no frontend group)"
            logger.info(log_msg)

            return OptimizerClass(param_groups, **other_args)

        # Cache only if creating the default base factory
        if config_optimiser is None and not fresh:
            self.optimizer_factory = build_optimizer

        return build_optimizer


    def create_lr_scheduler_factory(self, fresh: bool = False) -> Callable[[Optimizer, int, int], SchedulerWithFixedLR] | None:
        """
        Returns a factory function that builds a fresh LR scheduler.

        Args:
            fresh (bool): If True, bypasses the cached factory and forces recreation. Defaults to False.

        Returns:
            callable or None: A function expecting (optimizer, steps_per_epoch, total_epochs) that returns an initialized learning rate scheduler or None if unconfigured.
        """
        if not fresh and self.lr_scheduler_factory:
            return self.lr_scheduler_factory

        if self.config_lr_scheduler is None:
            logger.info("No learning rate scheduler configured.")
            return None

        # Get frontend LR from config
        frontend_lr = self.config_optimiser.get("optimiser_args", {}).get("frontend", {}).get("lr", None)

        # Get LR Scheduler class
        SchedulerClass = model_getter.get_lr_scheduler(self.config_lr_scheduler.get("scheduler_name"))

        def build_scheduler(optimizer, steps_per_epoch: int, total_epochs: int):
            """
            Creates a fresh LR scheduler + wraps it for frontend parameter handling.
            """
            scheduler_args = self.config_lr_scheduler.get("scheduler_args", {})
            
            backend_scheduler = SchedulerClass(
                optimizer,
                steps_per_epoch=steps_per_epoch,
                total_epochs=total_epochs,
                **scheduler_args
            )

            return SchedulerWithFixedLR(
                backend_scheduler,
                optimizer,
                fixed_group_name="frontend",
                fixed_lr=frontend_lr,
            )

        self.lr_scheduler_factory = build_scheduler
        return self.lr_scheduler_factory


    def create_loss_factory(self, fresh: bool = False) -> Callable[..., nn.Module] | None:
        """
        Returns a factory function that builds a fresh loss function instance.

        Args:
            fresh (bool): If True, bypasses the cached factory and forces recreation. Defaults to False.

        Returns:
            callable or None: A function expecting no mandatory args (or optional **override_kwargs) 
                            that returns an initialized loss object, or None if unconfigured.
        """
        if not fresh and self.loss_factory:
            return self.loss_factory

        # Retrieve Loss class and configuration
        loss_name = self.config_loss.get("loss_name", "SmoothL1Loss")
        LossClass = model_getter.get_loss(loss_name)
        loss_args = self.config_loss.get("loss_args", {"beta": 0.1})

        def build_loss(**override_kwargs):
            """
            Creates a fresh Loss instance using the configured arguments.
            Allows runtime keyword overrides if needed.
            """
            logger.info(f"Initialized {loss_name}({loss_args})")

            # Combine default args with any runtime overrides
            final_args = {**loss_args, **override_kwargs}
            return LossClass(**final_args)

        self.loss_factory = build_loss
        return self.loss_factory


    def create_neural_config(self, fresh: bool = False) -> NeuralConfig:
        """
        Generates the configuration object required for neural network pipelines.

        Args:
            fresh (bool): Plaaceholder for interface compatability.

        Returns:
            NeuralConfig: The initialized neural network configuration object.
        """
        if self.config_train is None:
            raise ValueError("No training arguments configured.")
        
        if self.optimizer_factory is None:
            self.create_optimizer_factory()

        if self.lr_scheduler_factory is None:
            self.create_lr_scheduler_factory()

        if self.loss_factory is None:
            self.create_loss_factory()

        self.neural_config = NeuralConfig(optimizer_factory=self.optimizer_factory, lr_scheduler_factory=self.lr_scheduler_factory, 
                                          loss_factory=self.loss_factory, device=self.device, **self.config_train)
        return self.neural_config


    def create_neural_config_finetune(self, fresh: bool = False) -> NeuralConfig:
        """
        Generates the configuration object required for neural network pipelines for participant-level finetuning.

        Args:
            fresh (bool): Plaaceholder for interface compatability.

        Returns:
            NeuralConfig: The initialized neural network configuration object for participant-level finetuning.
        """
        if self.config_finetune.get("train_args") is None:
            raise ValueError("No training arguments for finetuning configured.")

        if self.loss_factory is None:
            self.create_loss_factory()

        self.neural_config_finetune = NeuralConfig(optimizer_factory=self.create_optimizer_factory(config_optimiser=self.config_finetune.get("optimiser")), 
                                                   lr_scheduler_factory=None, 
                                                   loss_factory=self.loss_factory, 
                                                   device=self.device, 
                                                   **self.config_finetune.get("train_args"))
        return self.neural_config_finetune
    

    def create_custom_train_config(self, fresh: bool = False) -> dict[str, Any]:
        """
        Returns the raw configuration dictionary for custom/non-neural training methods.

        Args:
            fresh (bool): Plaaceholder for interface compatability.

        Returns:
            dict: The custom training configurations.
        """
        self.config_train["device"] = self.device   # Add device, which gets determined at runtime
        return self.config_train


    def create_custom_ft_config(self, fresh: bool = False) -> dict[str, Any]:
        """
        Returns the raw configuration dictionary for custom/non-neural finetuning.

        Args:
            fresh (bool): Plaaceholder for interface compatability.

        Returns:
            dict: The custom finetuning configurations.
        """
        self.config_finetune.get("train_args")["device"] = self.device   # Add device, which gets determined at runtime
        return self.config_finetune.get("train_args")


    def create_shap_config(self, fresh: bool = False) -> ShapConfig:
        """
        Return the configuration for SHAP computation and plotting.

        Args:
            fresh (bool): Plaaceholder for interface compatability.
        
        Returns:
            ShapConfig: The SHAP configuration instance.
        """
        if self.shap_config:
            return self.shap_config

        if self.config_shap is None:
            self.shap_config = None
        else:
            self.shap_config = ShapConfig(**self.config_shap)

        return self.shap_config
    

    def create_model(self, fresh: bool = False) -> MoodRegressor:
        """
        Assembles the MoodRegressor model architecture.

        Args:
            fresh (bool): If True, bypasses the cached model instance and forces recreation. Defaults to False.

        Returns:
            MoodRegressor: The configured mood regressor model.
        """
        if not fresh and self.model:
            return self.model
        
        if not self.movement_mood_dataset:
            self.create_movement_mood_dataset()

        # Get runtime parameters from dataset
        input_dim, output_dim, seq_len = self.movement_mood_dataset.get_io_dimensions()
        cardinality = self.movement_mood_dataset.cardinality

        # Fetch model class
        backend_class = model_getter.get_backend_model(self.config_backend.get("model_name"))
        
        # Bundle user config args with runtime extracted args
        config_args = self.config_backend.get("model_args", {})
        runtime_args = {
            "input_dim": input_dim,
            "output_dim": output_dim,
            "seq_len": seq_len,
            "label_transformer": self.create_label_transformer(),
            "cardinality": cardinality,
        }
        combined_args = {**config_args, **runtime_args}

        # Inspect the model's __init__ parameters
        sig = inspect.signature(backend_class.__init__)

        # Filter down to only arguments explicitly defined in __init__
        valid_params = sig.parameters.keys()
        filtered_args = {k: v for k, v in combined_args.items() if k in valid_params}

        # Instantiate backend model safely
        backend_model = backend_class(**filtered_args)

        # Create frontend model if configured
        #TODO: Frontend Implementation
        frontend_model = None
        if self.config_frontend:
            pass    

        # Create unified model with backend and optional frontend
        self.model = MoodRegressor(backend_model, frontend_model)
        return self.model


    def create_trainer(self, fresh: bool = False) -> PopulationTrainer | ParticipantTrainer:
        """
        Assembles the training coordinator.

        Args:
            fresh (bool): If True, bypasses the cached trainer and builds a new pipeline workflow. Defaults to False.

        Returns:
            PopulationTrainer or ParticipantTrainer: The initialised trainer runner.
        """
        if not fresh and self.trainer:
            return self.trainer
        
        if self.model is None:
            self.create_model()

        # Get SHAP config
        shap_config = self.create_shap_config()

        # Create configuration based on the backend models type (neural network or something else)
        neural_config = None
        custom_train_config = None
        if self.model.backend.is_neural_network:
            neural_config = self.create_neural_config()
        else:
            custom_train_config = self.create_custom_train_config()

        # Check if fine-tuning is configured and create participant-level trainer
        neural_ft_config = None
        custom_ft_config = None
        if self.config_finetune is not None:
            if self.model.backend.is_neural_network:
                neural_ft_config = self.create_neural_config_finetune()
            else:
                custom_ft_config = self.create_custom_ft_config()
            self.trainer = ParticipantTrainer(neural_config, custom_train_config, neural_ft_config, custom_ft_config, shap_config, self.config_finetune.get("finetune_ratio"))
            return self.trainer

        # Create population-level trainer
        self.trainer = PopulationTrainer(neural_config, custom_train_config, shap_config)
        return self.trainer