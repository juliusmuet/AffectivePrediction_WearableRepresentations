from abc import ABC, abstractmethod

import numpy as np
import shap
import torch
import torch.nn as nn
from captum.attr import GradientShap

from utils.utils import get_device


class BaseSHAPStrategy(ABC):
    """
    Abstract interface for model-specific SHAP attribution strategies.

    Defines a unified computation interface for feature attribution across models with potentially multiple output dimensions.
    """
    @abstractmethod
    def compute_shap(
        self,
        mood_model: nn.Module,
        X_train_raw: np.ndarray,
        X_test_raw: np.ndarray,
        n_samples: int = 100,
        seed: int = 42,
    ) -> np.ndarray:
        """
        Compute signed SHAP attributions in the original input space.

        Args:
            mood_model: Model for which feature attributions are computed.
            X_train_raw: Background reference data with shape (N_train, Window_Size, Feature_Dim)
            X_test_raw: Evaluation data with shape (N_test, Window_Size, Feature_Dim).
            n_samples: Number of background samples or attribution samples, depending on the strategy. Defaults to 100.
            seed: Random seed used for reproducible background sampling. Defaults to 42.

        Returns:
            Array of signed SHAP attributions with shape (N_test, Window_Size, Feature_Dim, Label_Dim).
        """
        pass



class TreeSHAPStrategy(BaseSHAPStrategy):
    """
    SHAP strategy for tree-based model architectures.

    Uses SHAP's TreeExplainer with tree-path-dependent perturbation.
    Falls back to a model-agnostic SHAP explainer when tree-path perturbation is not supported.
    """
    def compute_shap(
        self,
        mood_model: nn.Module,
        X_train_raw: np.ndarray,
        X_test_raw: np.ndarray,
        n_samples: int = 100,
        seed: int = 42,
    ) -> np.ndarray:
        """
        Compute SHAP attributions for a tree-based model.

        Flattens the temporal and feature dimensions before attribution computation and 
        reshapes the resulting attributions back to the original input dimensions.

        Args:
            mood_model: Model container whose tree-based estimator is available through mood_model.backend.model.
            X_train_raw: Background reference data with shape (N_train, Window_Size, Feature_Dim). Used only by the model-agnostic fallback.
            X_test_raw: Evaluation data with shape (N_test, Window_Size, Feature_Dim).
            n_samples: Maximum number of background samples used by the fallback explainer. Defaults to 100.
            seed: Random seed used to select fallback background samples. Defaults to 42.

        Returns:
            Array of signed SHAP attributions with shape (N_test, Window_Size, Feature_Dim, Label_Dim).

        Notes:
            The primary TreeExplainer path does not use X_train_raw, n_samples, or seed. 
            These arguments are retained for compatibility with the common strategy interface.
        """
        # Flatten 3D data to 2D
        N_train, W, F = X_train_raw.shape
        N_test = X_test_raw.shape[0]
        X_test_flat = X_test_raw.reshape(N_test, W * F)

        try:
            # Method 1: Default Tree-Path perturbation (Supported for XGBoost Random Forests)
            explainer = shap.TreeExplainer(mood_model.backend.model, feature_perturbation="tree_path_dependent", model_output="raw")
            raw_explanation = explainer(X_test_flat)    # Multi-Output: Shape (#num_samples, #num_flattened_features, #num_outputs)
                                                        # Single-Output: Shape (#num_samples, #num_flattened_features)
        except NotImplementedError:
            # Method 2: Generic Explainer fallback if tree-path fails
            # Sample background data
            X_train_flat = X_train_raw.reshape(N_train, W * F)
            rng = np.random.default_rng(seed)
            bg_idx = rng.choice(N_train, size=min(n_samples, N_train), replace=False)
            background_data = X_train_flat[bg_idx]

            # Custom prediction wrapper around forward() for general model compatibility
            def model_predict(x_flat):
                tensor_input = torch.as_tensor(x_flat, dtype=torch.float32)
                # Reshape back to 3D tensor expected by model's forward
                tensor_3d = tensor_input.reshape(-1, W, F)
                with torch.no_grad():
                    preds = mood_model.backend(tensor_3d)
                return preds.cpu().numpy()

            explainer = shap.Explainer(model_predict, background_data)
            raw_explanation = explainer(X_test_flat)

        # Standardize output into numpy array
        if hasattr(raw_explanation, "values"):
            shap_values = raw_explanation.values
        elif isinstance(raw_explanation, list):
            # Fallback for legacy list outputs in multi-target trees
            shap_values = np.stack(raw_explanation, axis=-1)    # (N, W*F, Outputs)
        else:
            shap_values = np.array(raw_explanation)

        # Standardize output tensor dimensions: (N_test, W * F, Outputs)
        if shap_values.ndim == 2:
            # Single-output model: (N_test, W * F) -> (N_test, W * F, 1)
            shap_values = np.expand_dims(shap_values, axis=-1)
        elif shap_values.ndim == 3 and shap_values.shape[1] != (W * F):
            # If shape returned as (N_test, Outputs, W * F), transpose to (N_test, W * F, Outputs)
            shap_values = np.transpose(shap_values, (0, 2, 1))

        # Reshape back to raw input dimensions: (N_test, Window, Features, Outputs)
        return shap_values.reshape(N_test, W, F, -1)



class ShortSeqNeuralSHAPStrategy(BaseSHAPStrategy):
    """
    GradientSHAP strategy for short-sequence neural architectures.

    Uses Captum's GradientSHAP to compute feature attributions for architectures such as FT-Transformer and TSMixer.

    Args:
        batch_size: Number of evaluation samples processed per batch. Defaults to 64.
    """
    def __init__(self, batch_size: int = 64) -> None:
        super().__init__()
        self.batch_size = batch_size

    def compute_shap(
        self,
        mood_model: nn.Module,
        X_train_raw: np.ndarray,
        X_test_raw: np.ndarray,
        n_samples: int = 100,
        seed: int = 42,
    ) -> np.ndarray:
        """
        Compute GradientSHAP attributions for all model outputs.

        Samples background baselines from the training data and computes attributions for the evaluation data in mini-batches 
        to limit GPU memory consumption.

        Args:
            mood_model: Neural network model to explain.
            X_train_raw: Background reference data with shape (N_train, Window_Size, Feature_Dim).
            X_test_raw: Evaluation data with shape (N_test, Window_Size, Feature_Dim).
            n_samples: Number of background samples and Monte Carlo samples used by GradientSHAP. Defaults to 100.
            seed: Random seed used to select background samples. Defaults to 42.

        Returns:
            Array of signed GradientSHAP attributions with shape (N_test, Window_Size, Feature_Dim, Label_Dim).
        """
        # Set model evaluation mode and device
        mood_model.eval()
        device = get_device()
        mood_model.to(device)

        # Sample background dataset baselines
        N_train = len(X_train_raw)
        rng = np.random.default_rng(seed)
        bg_idx = rng.choice(N_train, size=min(n_samples, N_train), replace=False)
        
        # Baselines are small (n_samples), so keep on GPU
        baselines_tensor = torch.from_numpy(X_train_raw[bg_idx]).float().to(device)

        # Keep X_test on CPU to preserve VRAM
        X_test_tensor = torch.from_numpy(X_test_raw).float()
        N_test, W, F = X_test_raw.shape

        # Dynamic output dimension detection
        with torch.no_grad():
            dummy_out = mood_model(X_test_tensor[:1].to(device))
            if isinstance(dummy_out, tuple):
                dummy_out = dummy_out[0]
            num_outputs = dummy_out.shape[-1] if dummy_out.ndim > 1 else 1

        # Instantiate Captum's GradientShap
        explainer = GradientShap(mood_model)

        # Pre-allocate NumPy output array (N_test, W, F, Outputs)
        all_attributions = np.zeros((N_test, W, F, num_outputs), dtype=np.float32)

        # Compute attributions with mini-batching
        for start_idx in range(0, N_test, self.batch_size):
            end_idx = min(start_idx + self.batch_size, N_test)
            
            # Transfer only the current batch to GPU VRAM
            batch_inputs = X_test_tensor[start_idx:end_idx].to(device)

            for target_idx in range(num_outputs):
                target = target_idx if num_outputs > 1 else None
                
                attr = explainer.attribute(
                    inputs=batch_inputs,
                    baselines=baselines_tensor,
                    target=target,
                    n_samples=n_samples
                )
                
                # Copy GPU result directly into pre-allocated NumPy block
                all_attributions[start_idx:end_idx, ..., target_idx] = attr.cpu().detach().numpy()

        return all_attributions



class LongSeqNeuralSHAPStrategy(BaseSHAPStrategy):
    """
    Patch-level GradientSHAP strategy for long time-series inputs.

    Aggregates the input sequence into non-overlapping temporal patches and computes GradientSHAP attributions in patch space. 
    Patch-level  attributions are subsequently distributed uniformly across the corresponding input timesteps.

    Args:
        batch_size: Number of evaluation samples processed per batch. Defaults to 2.
        patch_len: Number of consecutive timesteps represented by each patch. Defaults to 64.
    """
    def __init__(
        self,
        batch_size: int = 2,
        patch_len: int = 64,
    ) -> None:
        super().__init__()
        self.batch_size = batch_size
        self.patch_len = patch_len


    def compute_shap(
        self,
        mood_model: nn.Module,
        X_train_raw: np.ndarray,
        X_test_raw: np.ndarray,
        n_samples: int = 100,
        seed: int = 42,
    ) -> np.ndarray:
        """
        Compute patch-level GradientSHAP attributions for all outputs.

        Converts the training and evaluation sequences into patch-averaged representations. GradientSHAP is then applied to the patch-level
        inputs, and each patch attribution is distributed uniformly across the timesteps belonging to that patch.

        Args:
            mood_model: Neural network model to explain.
            X_train_raw: Background reference data with shape (N_train, Window_Size, Feature_Dim).
            X_test_raw: Evaluation data with shape (N_test, Window_Size, Feature_Dim).
            n_samples: Number of background samples and Monte Carlo  samples used by GradientSHAP. Defaults to 100.
            seed: Random seed used to select background samples. Defaults to 42.

        Returns:
            Array of signed attributions with shape (N_test, Window_Size, Feature_Dim, Label_Dim).

        Raises:
            ValueError: If the window size is not divisible by patch_len.
        """
        # Set model evaluation mode and device
        mood_model.eval()
        device = get_device()
        mood_model.to(device)

        N_train, W, F = X_train_raw.shape
        N_test = X_test_raw.shape[0]

        assert W % self.patch_len == 0, f"Window size {W} must be divisible by patch_len {self.patch_len}"
        num_patches = W // self.patch_len

        # Wrapper module that operates in Patch-Space (N, Num_Patches, F)
        class PatchWrapper(nn.Module):
            def __init__(self, model, W, F, patch_len):
                super().__init__()
                self.model = model
                self.W = W
                self.F = F
                self.patch_len = patch_len

            def forward(self, patch_tokens):
                # patch_tokens shape: (B, Num_Patches, F)
                # Expand patch tokens across length dimension to reconstruct window sequence
                x = patch_tokens.repeat_interleave(self.patch_len, dim=1) # (B, W, F)
                out = self.model(x)
                return out[0] if isinstance(out, tuple) else out

        # Instantiate Captum's GradientShap with model wrapper
        patch_model = PatchWrapper(mood_model, W, F, self.patch_len)
        explainer = GradientShap(patch_model)

        # Reshape data to patch space and downsample input sequence to patch-averages (N, Num_Patches, F)
        X_train_patched = X_train_raw.reshape(N_train, num_patches, self.patch_len, F).mean(axis=2)
        X_test_patched = X_test_raw.reshape(N_test, num_patches, self.patch_len, F).mean(axis=2)

        # Sample baselines
        rng = np.random.default_rng(seed)
        bg_idx = rng.choice(N_train, size=min(n_samples, N_train), replace=False)
        baselines_tensor = torch.from_numpy(X_train_patched[bg_idx]).float().to(device)

        # Create tensor for test data
        X_test_tensor = torch.from_numpy(X_test_patched).float()

        # Dynamic output dimension detection
        with torch.no_grad():
            dummy_out = mood_model(X_test_tensor[:1].to(device))
            if isinstance(dummy_out, tuple):
                dummy_out = dummy_out[0]
            num_outputs = dummy_out.shape[-1] if dummy_out.ndim > 1 else 1

        # Pre-allocate NumPy output array (N_test, W, F, Outputs)
        all_attributions = np.zeros((N_test, W, F, num_outputs), dtype=np.float32)

        # Compute attributions with mini-batching
        for start_idx in range(0, N_test, self.batch_size):
            end_idx = min(start_idx + self.batch_size, N_test)
            batch_inputs = X_test_tensor[start_idx:end_idx].to(device)

            for target_idx in range(num_outputs):
                target = target_idx if num_outputs > 1 else None

                # Compute patch attributions: (Batch, Num_Patches, F)
                patch_attr = explainer.attribute(
                    inputs=batch_inputs,
                    baselines=baselines_tensor,
                    target=target,
                    n_samples=n_samples
                )

                # Broadcast patch attributions back across time-steps inside the patch
                # Shape: (Batch, Num_Patches, F) -> (Batch, W, F)
                attr_expanded = patch_attr.repeat_interleave(self.patch_len, dim=1).cpu().detach().numpy()
                all_attributions[start_idx:end_idx, ..., target_idx] = attr_expanded / self.patch_len

        return all_attributions
