import logging
from pathlib import Path

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import shap
from matplotlib.collections import LineCollection

from utils.plotting_style import apply_style, apply_style_to_shap_plot, get_mood_dimensions_abbreviations


logger = logging.getLogger(__name__)


# Set global rcParams for npj Digital Medicine submission compliance
apply_style("npj")


def plot_shap_beeswarm(
    global_shap_matrix: np.ndarray,
    global_x_matrix: np.ndarray,
    base_feature_names: list[str],
    save_path: str | Path | None = None,
    label_name: str = "",
    max_display: int = 15,
) -> None:
    """
    Render a SHAP beeswarm plot with temporally flattened features.

    Flattens the time and feature dimensions of the input matrices so that each feature-timestep combination is displayed as a separate feature.
    The resulting beeswarm plot shows the direction and magnitude of SHAP  contributions in relation to the corresponding feature values.

    Args:
        global_shap_matrix: SHAP values with shape (N, Window_Size, Feature_Dim).
        global_x_matrix: Input feature values with shape (N, Window_Size, Feature_Dim).
        base_feature_names: Names of the base features, in the same order as the final dimension of the input matrices.
        save_path: Destination path for the generated figure. If None, the figure is displayed instead of being saved. Defaults to None.
        label_name: Name of the output dimension associated with the SHAP values. Used to annotate the plot title. Defaults to "".
        max_display: Maximum number of top features displayed. Defaults to 15.

    Returns:
        None. Displays the plot or saves it to save_path.
    """
    num_samples, window_size, num_features = global_shap_matrix.shape
    
    # Internal flattening for SHAP beeswarm requirements
    global_shap_flat = global_shap_matrix.reshape(num_samples, window_size * num_features)
    global_x_flat = global_x_matrix.reshape(num_samples, window_size * num_features)
    
    # Compute feature names for flattened features over time matching -N notation
    flattened_feature_names = [f"{feat}_t-{window_size - t}" for t in range(window_size) for feat in base_feature_names]
    
    # Setup title dynamically based on label_name presence
    base_title = "Top Time-Flattened Features"
    label_abbreviations = get_mood_dimensions_abbreviations()
    short_name = label_abbreviations.get(label_name, label_name)
    title_string = f"{base_title}\nDimension: {short_name}" if label_name else base_title
    
    # Setup beeswarm plot and enforce canvas size
    fig = plt.figure(figsize=(5.5, 2.8))
    
    explanation = shap.Explanation(
        values=global_shap_flat,
        data=global_x_flat,
        feature_names=flattened_feature_names
    )

    # Fetch standard matplotlib 'bwr' colormap object to override SHAP default pink/blue
    bwr_cmap = plt.get_cmap('bwr')

    # Render beeswarm with explicit color property injection
    shap.plots.beeswarm(
        explanation, 
        max_display=max_display, 
        color=bwr_cmap, 
        show=False,
        plot_size=None
    )

    # Override SHAP's hardcoded large fonts
    apply_style_to_shap_plot(fig, title_string)
    plt.tight_layout()
    
    # Saving / Showing
    if save_path is None:
        plt.show()
    else:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logging.info(f"Generated Beeswarm plot at {save_path}")

    plt.close(fig)


def plot_shap_beeswarm_time_accumulated(
    global_shap_matrix: np.ndarray,
    global_x_matrix: np.ndarray,
    base_feature_names: list[str],
    save_path: str | Path | None = None,
    label_name: str = "",
    max_display: int = 15,
) -> None:
    """
    Render a SHAP beeswarm plot with contributions accumulated over time.

    Sums SHAP values across the temporal dimension to obtain one contribution per base feature and sample. 
    The corresponding feature values are averaged across time and used for the beeswarm colour scale.

    Args:
        global_shap_matrix: SHAP values with shape (N, Window_Size, Feature_Dim).
        global_x_matrix: Input feature values with shape (N, Window_Size, Feature_Dim).
        base_feature_names: Names of the base features, in the same order as the final dimension of the input matrices.
        save_path: Destination path for the generated figure. If None, the figure is displayed instead of being saved. Defaults to None.
        label_name: Name of the output dimension associated with the SHAP values. Used to annotate the plot title. Defaults to "".
        max_display: Maximum number of top features displayed. Defaults to 15.

    Returns:
        None. Displays the plot or saves it to save_path.
    """
    # Sum SHAP values across the time dimension
    global_shap_summed = global_shap_matrix.sum(axis=1) # (Samples, Window, Features) -> (Samples, Features)
    
    # Compute mean feature value across time for dot color scale representation
    global_x_mean = global_x_matrix.mean(axis=1)
    
    # Setup title dynamically based on label_name presence
    base_title = "Top Features Over Time"
    label_abbreviations = get_mood_dimensions_abbreviations()
    short_name = label_abbreviations.get(label_name, label_name)
    title_string = f"{base_title}\nDimension: {short_name}" if label_name else base_title
    
    # Setup beeswarm plot and enforce canvas size
    fig = plt.figure(figsize=(5.5, 2.8))
    
    explanation = shap.Explanation(
        values=global_shap_summed,
        data=global_x_mean,
        feature_names=base_feature_names
    )

    # Fetch standard matplotlib 'bwr' colormap object
    bwr_cmap = plt.get_cmap('bwr')

    # Render beeswarm with explicit color property injection
    shap.plots.beeswarm(
        explanation, 
        max_display=max_display, 
        color=bwr_cmap, 
        show=False,
        plot_size=None
    )

    # Override SHAP's hardcoded large fonts
    apply_style_to_shap_plot(fig, title_string)
    plt.tight_layout()
    
    # Saving / Showing
    if save_path is None:
        plt.show()
    else:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logging.info(f"Generated Accumulated Beeswarm plot at {save_path}")

    plt.close(fig)


def plot_temporal_absolute_heatmap(
    global_shap_matrix: np.ndarray,
    base_feature_names: list[str],
    save_path: str | Path | None = None,
    label_name: str = "",
    tick_intervals: int = 10,
    max_display: int = 15,
) -> None:
    """
    Render a heatmap of mean absolute SHAP values over time.

    Ranks base features by their mean absolute SHAP value across all samples and timesteps, 
    then displays the mean absolute SHAP value of the highest-ranked features at each timestep. 

    Args:
        global_shap_matrix: SHAP values with shape (N, Window_Size, Feature_Dim).
        base_feature_names: Names of the base features, in the same order as the final dimension of global_shap_matrix.
        save_path: Destination path for the generated figure. If None, the figure is displayed instead of being saved. Defaults to None.
        label_name: Name of the output dimension associated with the SHAP values. Used to annotate the plot title. Defaults to "".
        tick_intervals: Interval between labelled lookback timesteps when the window is sufficiently large. Defaults to 10.
        max_display: Maximum number of top-ranked base features displayed. Defaults to 15.

    Returns:
        None. Displays the plot or saves it to save_path.
    """
    _, window_size, _ = global_shap_matrix.shape
    
    # Calculate overall importance per feature to rank and truncate top features
    mean_abs_per_feature = np.mean(np.abs(global_shap_matrix), axis=(0, 1))
    top_indices = np.argsort(mean_abs_per_feature)[::-1][:max_display]
    
    # Filter matrix and feature names
    filtered_shap = global_shap_matrix[:, :, top_indices]
    filtered_feature_names = [base_feature_names[i] for i in top_indices]
    
    # Take absolute values and average across dataset dimension (axis=0)
    abs_global_shap = np.abs(filtered_shap)
    mean_abs_shap_spatial = np.mean(abs_global_shap, axis=0)  # Shape: (Window, Filtered_Features)
    temporal_matrix = mean_abs_shap_spatial.T                 # Shape: (Filtered_Features, Window)

    base_title = "Absolute (Magnitude) Temporal Feature Importance"
    label_abbreviations = get_mood_dimensions_abbreviations()
    short_name = label_abbreviations.get(label_name, label_name)
    title_string = f"{base_title}\nDimension: {short_name}" if label_name else base_title
    
    # Scale figure height based on rendered features
    fig, ax = plt.subplots(figsize=(5.5, max(2.8, 0.25 * len(filtered_feature_names))))

    sns.heatmap(
        temporal_matrix, 
        cmap="YlOrRd", 
        xticklabels=False, 
        yticklabels=filtered_feature_names,
        annot=False, 
        cbar_kws={
            "label": "Mean Absolute SHAP Value",
            "pad": 0.02,  # Reduced spacing between heatmap and right-side colorbar legend
        },
        ax=ax
    )
    
    # Calculate matching lookback sequence and map to Seaborn centers (i + 0.5)
    all_tick_positions = np.arange(window_size) + 0.5
    lookback_values = [window_size - i for i in range(window_size)]
    
    # Dynamic tick filtering rule: track every step if short window, else multiples of tick_intervals
    if window_size < tick_intervals:
        major_indices = list(range(window_size))
    else:
        major_indices = [i for i, val in enumerate(lookback_values) if val % tick_intervals == 0]
        
    major_positions = all_tick_positions[major_indices]
    tick_labels = [f"-{lookback_values[idx]}" for idx in major_indices]
    
    # Apply structural major and minor coordinate states
    ax.set_xticks(major_positions)
    ax.set_xticklabels(tick_labels, rotation=90)
    ax.set_xticks(all_tick_positions, minor=True)

    # Force feature names (y-axis) to be rendered horizontally
    ax.tick_params(axis="y", rotation=0)
    
    # Styling
    ax.figure.axes[-1].yaxis.label.set_fontweight("normal")
    plt.title(title_string, fontweight="bold", pad=8)
    plt.xlabel("Lookback Window Timesteps")
    plt.ylabel("Features")
    plt.tight_layout()
    
    # Saving / Showing
    if save_path is None:
        plt.show()
    else:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logging.info(f"Generated Absolute Heatmap plot at {save_path}")

    plt.close(fig)


def plot_temporal_signed_heatmap(
    global_shap_matrix: np.ndarray,
    base_feature_names: list[str],
    save_path: str | Path | None = None,
    label_name: str = "",
    tick_intervals: int = 10,
    max_display: int = 15,
) -> None:
    """
    Render a heatmap of mean signed SHAP values over time.

    Ranks base features by their mean absolute SHAP value across all samples and timesteps, 
    then displays the mean signed SHAP value of the highest-ranked features at each timestep. 

    Args:
        global_shap_matrix: SHAP values with shape (N, Window_Size, Feature_Dim).
        base_feature_names: Names of the base features, in the same order as the final dimension of global_shap_matrix.
        save_path: Destination path for the generated figure. If None, the figure is displayed instead of being saved. Defaults to None.
        label_name: Name of the output dimension associated with the SHAP values. Used to annotate the plot title. Defaults to "".
        tick_intervals: Interval between labelled lookback timesteps when the window is sufficiently large. Defaults to 10.
        max_display: Maximum number of top-ranked base features displayed. Defaults to 15.

    Returns:
        None. Displays the plot or saves it to save_path.
    """
    _, window_size, _ = global_shap_matrix.shape
    
    # Rank features based on total absolute magnitude over time and samples
    mean_abs_per_feature = np.mean(np.abs(global_shap_matrix), axis=(0, 1))
    top_indices = np.argsort(mean_abs_per_feature)[::-1][:max_display]
    
    filtered_shap = global_shap_matrix[:, :, top_indices]
    filtered_feature_names = [base_feature_names[i] for i in top_indices]
    
    # Mean signed SHAP over samples
    mean_signed_shap_spatial = np.mean(filtered_shap, axis=0)  # Shape: (Window, Filtered_Features)
    temporal_matrix_signed = mean_signed_shap_spatial.T        # Shape: (Filtered_Features, Window)
    
    max_val = np.max(np.abs(temporal_matrix_signed))

    # Setup title dynamically based on label_name presence
    base_title = "Signed (Directional) Temporal Feature Importance"
    label_abbreviations = get_mood_dimensions_abbreviations()
    short_name = label_abbreviations.get(label_name, label_name)
    title_string = f"{base_title}\nDimension: {short_name}" if label_name else base_title
    
    fig, ax = plt.subplots(figsize=(5.5, max(2.8, 0.25 * len(filtered_feature_names))))
    
    sns.heatmap(
        temporal_matrix_signed, 
        cmap="bwr", 
        vmin=-max_val, 
        vmax=max_val,
        xticklabels=False, 
        yticklabels=filtered_feature_names,
        annot=False, 
        cbar_kws={
            "label": "Mean Signed SHAP Value",
            "pad": 0.02,  # Reduced spacing between heatmap and right-side colorbar legend
        },
        ax=ax
    )
    
    # Calculate matching lookback sequence and map to Seaborn centers (i + 0.5)
    all_tick_positions = np.arange(window_size) + 0.5
    lookback_values = [window_size - i for i in range(window_size)]
    
    # Dynamic tick filtering rule: track every step if short window, else multiples of tick_intervals
    if window_size < tick_intervals:
        major_indices = list(range(window_size))
    else:
        major_indices = [i for i, val in enumerate(lookback_values) if val % tick_intervals == 0]
    
    major_positions = all_tick_positions[major_indices]
    tick_labels = [f"-{lookback_values[idx]}" for idx in major_indices]
    
    # Apply structural major and minor coordinate states
    ax.set_xticks(major_positions)
    ax.set_xticklabels(tick_labels, rotation=90)
    ax.set_xticks(all_tick_positions, minor=True)

    # Force feature names (y-axis) to be rendered horizontally
    ax.tick_params(axis="y", rotation=0)

    # Styling
    ax.figure.axes[-1].yaxis.label.set_fontweight("normal")
    plt.title(title_string, fontweight="bold", pad=8)
    plt.xlabel("Lookback Window Timesteps")
    plt.ylabel("Features")
    plt.tight_layout()
    
    # Saving / Showing
    if save_path is None:
        plt.show()
    else:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logging.info(f"Generated Signed Heatmap plot at {save_path}")

    plt.close(fig)


def plot_shap_force_overlay(
    global_shap_matrix: np.ndarray,
    global_x_matrix: np.ndarray,
    base_feature_names: list[str],
    sample_idx: int = 0,
    save_path: str | Path | None = None,
    label_name: str = "",
    tick_intervals: int = 10,
    max_display: int = 5,
) -> None:
    """
    Render a SHAP impact overlay for a single time-series sample.

    Selects the most important features for the specified sample and displays each feature's time series in a separate subplot.
    The line segments are coloured according to their corresponding SHAP values.

    Args:
        global_shap_matrix: SHAP values with shape (N, Window_Size, Feature_Dim).
        global_x_matrix: Input feature values with shape (N, Window_Size, Feature_Dim).
        base_feature_names: Names of the base features, in the same order as the final dimension of the input matrices.
        sample_idx: Index of the sample to visualise along the sample dimension. Defaults to 0.
        save_path: Destination path for the generated figure. If None, the figure is displayed instead of being saved. Defaults to None.
        label_name: Name of the output dimension associated with the SHAP values. Used to annotate the plot title. Defaults to "".
        tick_intervals: Interval between labelled lookback timesteps when the window is sufficiently large. Defaults to 10.
        max_display: Maximum number of top-ranked base features displayed. Defaults to 5.

    Returns:
        None. Displays the plot or saves it to save_path.
    """
    _, window_size, _ = global_shap_matrix.shape
    
    # Extract the single sample matrices
    sample_shap = global_shap_matrix[sample_idx]  # Shape: (Window, Features)
    sample_x = global_x_matrix[sample_idx]        # Shape: (Window, Features)
    
    # Rank features based on the sample's mean absolute SHAP values across time
    sample_feat_importance = np.mean(np.abs(sample_shap), axis=0)
    top_indices = np.argsort(sample_feat_importance)[::-1][:max_display]
    
    # Filter features for this sample
    sample_shap = sample_shap[:, top_indices]
    sample_x = sample_x[:, top_indices]
    filtered_feature_names = [base_feature_names[i] for i in top_indices]
    num_features = len(filtered_feature_names)
    
    time_steps = np.array([- (window_size - t) for t in range(window_size)])
    
    # Setup title dynamically based on label_name presence
    base_title = f"Local Feature Impact Overlay"
    label_abbreviations = get_mood_dimensions_abbreviations()
    short_name = label_abbreviations.get(label_name, label_name)
    title_string = f"{base_title}\nDimension: {short_name}" if label_name else base_title
    
    fig, axes = plt.subplots(num_features, 1, figsize=(5.5, 1.2 * num_features), sharex=True)
    if num_features == 1:
        axes = [axes]
        
    # Find global SHAP min/max for this sample to standardize colorbars
    max_shap_val = max(abs(sample_shap.min()), abs(sample_shap.max()))

    cmap_blue_gray_red = mcolors.LinearSegmentedColormap.from_list(
        'blue_gray_red', ['#1f77b4', '#e7e7e7', '#d62728']
    )
    
    if window_size < tick_intervals:
        # Include every single timestep as a major tick
        major_ticks = list(time_steps)
        tick_labels = [f"-{abs(ts)}" for ts in major_ticks]
    else:
        # Define major tick locations strictly on multiples of tick_intervals for clean grids
        major_ticks = [ts for ts in time_steps if abs(ts) % tick_intervals == 0]
        tick_labels = [f"-{abs(ts)}" for ts in major_ticks]
    
    for i, feat_name in enumerate(filtered_feature_names):
        ax = axes[i]
        
        # Get signal and SHAP arrays for the current channel
        y_signal = sample_x[:, i]
        shap_vals = sample_shap[:, i]
        
        # Segment the time-series into continuous line chunks to map colors smoothly
        points = np.array([time_steps, y_signal]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        
        # Generate the LineCollection
        norm = plt.Normalize(-max_shap_val, max_shap_val)
        lc = LineCollection(segments, cmap=cmap_blue_gray_red, norm=norm, linewidths=1.2)
        lc.set_array(shap_vals)
        
        # Plot signals
        line = ax.add_collection(lc)
        
        # Explicit bounds adjustment since LineCollections do not autoscale
        ax.set_xlim(time_steps.min(), time_steps.max())
        ax.set_ylim(
            y_signal.min() - (abs(y_signal.min()) * 0.1 + 1e-5),
            y_signal.max() + (abs(y_signal.max()) * 0.1 + 1e-5),
        )

        ax.set_xticks(major_ticks)
        ax.set_xticks(time_steps, minor=True)

        # Styling
        ax.set_ylabel(f"{feat_name}", fontweight="normal")
        ax.grid(True, which='major', linestyle='--', alpha=0.6)
        ax.grid(False, which='minor')  # Keep the individual steps clean

    # Configure major labels only on the final bottom subplot's x-axis
    axes[-1].set_xticklabels(tick_labels, rotation=90)
    axes[-1].set_xlabel("Lookback Window Timesteps")

    # Positioning of the plots
    fig.subplots_adjust(top=0.92, bottom=0.15, left=0.12, right=0.85, hspace=0.25)

    # Insert unified colorbar aligned colorbar bounds with the expanded subplot region
    cbar_ax = fig.add_axes([0.87, 0.12, 0.02, 0.80])
    cbar = fig.colorbar(line, cax=cbar_ax)

    # Styling
    cbar.set_label("SHAP Impact Magnitude", rotation=90, labelpad=10, fontweight="normal")

    # Attach title directly to the top subplot frame with pad=6 to eliminate vertical whitespace
    axes[0].set_title(title_string, fontweight="bold", fontsize=8, pad=6)

    if save_path is None:
        plt.show()
    else:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logging.info(f"Generated Force Overlay plot at {save_path}")

    plt.close(fig)


def get_top_flattened_features(global_shap_matrix: np.ndarray, base_feature_names: list[str]) -> list[dict[str, int | str | float]]:
    """
    Rank temporally flattened features by mean absolute SHAP importance.

    Treats each feature-timestep combination as a separate feature and calculates its mean absolute SHAP value across all samples. 
    Returns records in descending order of importance.

    Args:
        global_shap_matrix: SHAP values with shape (N, Window_Size, Feature_Dim).
        base_feature_names: Names of the base features, in the same order as the final dimension of global_shap_matrix.

    Returns:
        List of feature records ordered by descending mean absolute SHAP value. Each record contains:
        - rank: One-based importance rank.
        - feature_name: Name of the feature-timestep combination.
        - mean_abs_shap: Mean absolute SHAP value across samples.
    """
    num_samples, window_size, num_features = global_shap_matrix.shape
    
    # Reconstruct dynamic string array for the flattened dimensions
    flattened_feature_names = [f"{feat}_t-{window_size - t}" for t in range(window_size) for feat in base_feature_names]
    
    # Flatten across the temporal dimensions to compute ranked importances
    global_shap_flat = global_shap_matrix.reshape(num_samples, window_size * num_features)
    mean_abs_shap = np.mean(np.abs(global_shap_flat), axis=0)
    
    # Sort indices in descending order
    sorted_indices = np.argsort(mean_abs_shap)[::-1]

    # Pack metrics into dictionaries
    records = []
    for rank, idx in enumerate(sorted_indices, start=1):
        records.append({
            "rank": rank,
            "feature_name": flattened_feature_names[idx],
            "mean_abs_shap": float(mean_abs_shap[idx])
        })
        
    return records


def get_top_base_features_over_time(global_shap_matrix: np.ndarray, base_feature_names: list[str]) -> list[dict[str, int | float | str]]:
    """
    Rank base features by their cumulative importance over time.

    Computes the mean absolute SHAP value for each base feature at every timestep, then sums these values across the temporal dimension. 
    Returns the resulting feature records in descending order of cumulative importance.

    Args:
        global_shap_matrix: SHAP values with shape (N, Window_Size, Feature_Dim).
        base_feature_names: Names of the F base features, in the same order as the final dimension of global_shap_matrix.

    Returns:
        List of base-feature records ordered by descending cumulative importance. Each record contains:
            - rank: One-based importance rank.
            - modality_name: Name of the base feature.
            - total_cumulative_shap: Sum of the mean absolute SHAP values across all timesteps.
            - avg_shap_per_step: Cumulative importance divided by the number of timesteps.
    """
    _, window_size, num_base_features = global_shap_matrix.shape
    
    # 1. Get mean absolute importance per timestep/feature node, shape: (Window, Base_Features)
    mean_abs_shap_spatial = np.mean(np.abs(global_shap_matrix), axis=0)
    
    # 2. Sum down the timesteps (axis=0) to collapse time and get total impact per base feature
    total_modality_importance = np.sum(mean_abs_shap_spatial, axis=0) # Shape: (num_base_features,)
    
    # 3. Sort the base features in descending order
    sorted_base_indices = np.argsort(total_modality_importance)[::-1]
    
    # Pack metrics into dictionaries
    records = []
    for rank, idx in enumerate(sorted_base_indices, start=1):
        total_score = float(total_modality_importance[idx])
        records.append({
            "rank": rank,
            "modality_name": base_feature_names[idx],
            "total_cumulative_shap": total_score,
            "avg_shap_per_step": total_score / window_size
        })
        
    return records
