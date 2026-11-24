import matplotlib.pyplot as plt

# -----------------------------------------------------------------------------
# Style Definitions
# -----------------------------------------------------------------------------
NPJ_STYLE_PARAMS = {
    # Typography & Elements
    'font.family': 'sans-serif',
    'font.sans-serif': ['Helvetica', 'Arial', 'DejaVu Sans'],
    'font.size': 8,
    'axes.labelsize': 8,
    'axes.titlesize': 8,
    'xtick.labelsize': 7,
    'ytick.labelsize': 7,
    'legend.fontsize': 6.5,
    'figure.titlesize': 8,
    # Line & Grid Styles
    'axes.linewidth': 0.6,
    'grid.linewidth': 0.4,
    'lines.linewidth': 1.0,
    'xtick.major.width': 0.6,
    'ytick.major.width': 0.6,
    # Vector Export
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
    # Custom Layout & Padding
    'custom.title_pad': 8,
}

# Master registry mapping style names to parameter dictionaries
STYLES = {
    'npj': NPJ_STYLE_PARAMS
}

# Active style variable (defaults to 'npj')
_CURRENT_STYLE = 'npj'

# -----------------------------------------------------------------------------
# Style Application Functions
# -----------------------------------------------------------------------------
def get_style_params(style_name: str = None):
    """Retrieves parameter dict for a given style name, falling back to current active style."""
    name = (style_name or _CURRENT_STYLE).lower()
    if name not in STYLES:
        raise ValueError(f"Unknown style '{name}'. Available styles: {list(STYLES.keys())}")
    return STYLES[name]


def apply_style(style_name: str = 'npj'):
    """Sets Matplotlib global rcParams for the chosen style."""
    global _CURRENT_STYLE
    _CURRENT_STYLE = style_name.lower()
    params = get_style_params(_CURRENT_STYLE)
    
    # Filter out custom non-matplotlib parameters
    rc_params = {k: v for k, v in params.items() if not k.startswith('custom.')}
    plt.rcParams.update(rc_params)


def apply_style_to_shap_plot(fig, title_string: str, style_name: str = None):
    """Post-processes SHAP plots dynamically using parameters from the specified style."""
    params = get_style_params(style_name)

    tick_size = params['xtick.labelsize']
    label_size = params['axes.labelsize']
    title_size = params['axes.titlesize']
    title_pad = params.get('custom.title_pad', 8)

    for ax in fig.axes:
        ax.tick_params(axis='both', labelsize=tick_size)
        ax.xaxis.label.set_size(label_size)
        ax.xaxis.label.set_fontweight('normal')
        ax.yaxis.label.set_size(label_size)
        ax.yaxis.label.set_fontweight('normal')

        if ax.get_ylabel():
            ax.set_ylabel(ax.get_ylabel(), fontsize=label_size, fontweight='normal')

    plt.title(title_string, fontsize=title_size, fontweight='bold', pad=title_pad)


def get_mood_dimensions_abbreviations() -> dict[str, str]:
    """Returns dict with keys "Energetic Arousal", "Valence" and "Calmness" and their short version names as values."""
    return {"Energetic Arousal": "EA", "Valence": "V", "Calmness": "C"}


def get_mood_dimensions_colours() -> dict[str, str]:
    """Returns dict with keys "EA", "V" and "C" and their colour values."""
    return {"EA": "#2B5C8F", "V": "#D95F02", "C": "#1B9E77"}
