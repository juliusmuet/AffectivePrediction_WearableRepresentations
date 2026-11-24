import argparse
from datetime import datetime
import json
from pathlib import Path
import logging

import numpy as np

from utils.factory import Factory
from utils.utils import setup_logging
from explainability.shap_plots import *
from explainability.shap_config import ShapConfig


class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder to handle NumPy scalar types and arrays."""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def main():
    # === Get input parameters ===
    parser = argparse.ArgumentParser(description="Train a speaker embedding model.")
    parser.add_argument("base_path", type=str, help="Base path to the config directory (must contain config.yml)")
    args = parser.parse_args()

    # === Get paths ===
    base_path = Path(args.base_path)
    config_path = base_path / "config.yml"
    date_str = str(datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))

    # === Setup logger ===
    setup_logging(base_path / f"{date_str}_train.txt")
    logger = logging.getLogger(__name__)
    logger.info(f"Training with path {config_path}")

    # === Build all components for training from config ===
    factory = Factory(config_path)
    dataset = factory.create_movement_mood_dataset()
    model = factory.create_model()
    trainer = factory.create_trainer()
    
    # === Execute Training Workflow (Polymorphic) ===
    results = trainer.fit(model, dataset)

    # === Log Summary ===
    if results.summary_msg:
        logger.info(results.summary_msg)

    # === Save Metric Files ===
    base_path.mkdir(parents=True, exist_ok=True)
    for filename, data in results.metrics_files.items():
        if data:  # Guard against saving empty evaluation dictionaries
            output_file = base_path / f"{date_str}_{filename}"
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=4, cls=NumpyEncoder)
    logger.info(f"Saved metric evaluation files to {base_path}")

    # === Render SHAP Visualisations (If active) ===
    if results.global_shap_matrix is not None:
        shap_visualisation(
            global_shap_matrix=results.global_shap_matrix,
            global_x_matrix=results.global_x_matrix,
            base_feature_names=dataset.feature_names,
            label_names=dataset.label_names,
            base_save_path=args.base_path,
            date_str=date_str,
            shap_config=getattr(trainer, "shap_config", None)
        )


def shap_visualisation(
    global_shap_matrix: np.ndarray | None,
    global_x_matrix: np.ndarray,
    base_feature_names: list[str],
    label_names: list[str],
    base_save_path: str | Path,
    date_str: str,
    shap_config: ShapConfig,
) -> None:
    """
    Generate and save SHAP visualisations for each model output.

    Creates a separate set of SHAP plots and feature-importance summaries for each output label. 
    Results are saved in a date-specific subdirectory of base_save_path.

    Args:
        global_shap_matrix: SHAP values with shape (Samples, Window_Size, Feature_Dim, Label_Dim). If None, no visualisations are generated.
        global_x_matrix: Evaluation input features with shape (Samples, Window_Size, Feature_Dim). 
            Must correspond to the samples and features represented in global_shap_matrix.
        base_feature_names: Original feature names before temporal unrolling.
            Feature names containing a time-related suffix in square brackets are reduced to their base names for plotting.
        label_names: Names of the model outputs corresponding to the final dimension of global_shap_matrix.
        base_save_path: Root directory in which the visualisation output directory is created.
        date_str: Date or timestamp used to identify the output directory and generated files.
        shap_config: Configuration providing max_display and tick_intervals settings for the visualisations.

    Returns:
        None. Saves the generated plots as PDF files and feature-importance summaries as JSON files. 
        If global_shap_matrix is None, returns without generating any files.
    """
    if global_shap_matrix is None:
        return

    # Clean feature names (remove everything after a "[" for move features)
    cleaned_feature_names = [name.split(" [")[0] for name in base_feature_names]

    # Create the output directory inside base_save_path
    output_dir = Path(base_save_path) / f"{date_str}_shap"
    output_dir.mkdir(parents=True, exist_ok=True)

    sample_idx = 0  #TODO: find representative samples

    for idx, label_name in enumerate(label_names):
        save_path = output_dir / f"{date_str}_shap_{label_name}"

        plot_shap_beeswarm(
            global_shap_matrix=global_shap_matrix[:,:,:,idx], 
            global_x_matrix=global_x_matrix, 
            base_feature_names=cleaned_feature_names, 
            save_path=Path(f"{save_path}_beeswarm.pdf"), 
            label_name=label_name, 
            max_display=shap_config.max_display)

        plot_shap_beeswarm_time_accumulated(
            global_shap_matrix=global_shap_matrix[:,:,:,idx], 
            global_x_matrix=global_x_matrix, 
            base_feature_names=cleaned_feature_names, 
            save_path=Path(f"{save_path}_beeswarm_accumulated.pdf"), 
            label_name=label_name, 
            max_display=shap_config.max_display)
        
        plot_temporal_absolute_heatmap(
            global_shap_matrix=global_shap_matrix[:,:,:,idx], 
            base_feature_names=cleaned_feature_names, 
            save_path=Path(f"{save_path}_abs_heatmap.pdf"), 
            label_name=label_name, 
            tick_intervals=shap_config.tick_intervals, 
            max_display=shap_config.max_display)
        
        plot_temporal_signed_heatmap(
            global_shap_matrix=global_shap_matrix[:,:,:,idx], 
            base_feature_names=cleaned_feature_names, 
            save_path=Path(f"{save_path}_sign_heatmap.pdf"), 
            label_name=label_name, 
            tick_intervals=shap_config.tick_intervals, 
            max_display=shap_config.max_display)
        
        plot_shap_force_overlay(
            global_shap_matrix=global_shap_matrix[:,:,:,idx], 
            global_x_matrix=global_x_matrix, 
            base_feature_names=cleaned_feature_names, 
            sample_idx=sample_idx, 
            save_path=Path(f"{save_path}_force_overlay.pdf"), 
            label_name=label_name, 
            tick_intervals=shap_config.tick_intervals)
        
        records_flattened = get_top_flattened_features(global_shap_matrix[:,:,:,idx], base_feature_names)
        records_flattened_file = Path(f"{save_path}_top_flat.json")
        with open(records_flattened_file, "w", encoding="utf-8") as f:
            json.dump(records_flattened, f, indent=4, cls=NumpyEncoder)

        records_accumulated = get_top_base_features_over_time(global_shap_matrix[:,:,:,idx], base_feature_names)
        records_accumulated_file = Path(f"{save_path}_top_acc.json")
        with open(records_accumulated_file, "w", encoding="utf-8") as f:
            json.dump(records_accumulated, f, indent=4, cls=NumpyEncoder)
    

if __name__ == "__main__":
    main()
