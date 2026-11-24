
# Generalization and Personalization of Machine Learning Across Wearable Sensor Representations for Affect Digital Phenotyping

This repository contains the code and experimental configurations for investigating how wearable sensor representations influence the generalization and personalization of machine learning models for affective state prediction.

The work compares population-level prediction across individuals with participant-specific model adaptation using different wearable sensor representations, ranging from commercially available activity-tracker metrics to handcrafted features and high-frequency raw accelerometry.

> **Publication status:** The associated manuscript is currently in preparation. Please refer to this repository for updates and the final publication details.

---

## Overview

Wearable sensors enable continuous and unobtrusive monitoring of behavioral and physiological patterns associated with affective states. However, the choice of sensor representation may substantially influence the predictive performance of machine learning models, particularly when models are transferred between individuals or adapted to individual users.

This project investigates the relationship between sensor representation, model generalization, and personalization for predicting three dimensions of affect:

- **Energetic Arousal (EA)**
- **Valence (V)**
- **Calmness \(C\)**

Five-minute wearable sensor windows are paired with ecological momentary assessment (EMA) responses. Three principal sensor representation strategies are compared:

1. **Fitbit aggregates:** activity, cardiovascular, and metabolic estimates
2. **Handcrafted statistical features:** engineered temporal, spectral, and statistical features extracted from raw accelerometer data
3. **Raw accelerometer signals:** high-frequency tri-axial accelerometry and derived movement signals

The experimental framework evaluates more than 390 model configurations across different settings and architectures.

### Main findings

The experiments indicate that the most suitable sensor representation depends on the intended prediction setting.

| Finding | Result |
|---|---|
| Population-level generalization | Fitbit aggregates and handcrafted statistical features achieved competitive performance and outperformed raw-signal models in the zero-shot population-level settings |
| Participant-specific adaptation | Fine-tuning raw acceleration models substantially improved prediction performance |
| Energetic Arousal | Median MAE reduction of 20.45% following personalization |
| Valence | Median MAE reduction of 28.83% following personalization |
| Calmness | Median MAE reduction of 23.55% following personalization |

These findings highlight that representations optimized for generalization across individuals do not necessarily provide the greatest benefit when models are personalized.

---

## Research objectives

The study investigates three main research questions:

1. **Representation and generalization:** How do different wearable sensor representations affect the ability of machine learning models to generalize to previously unseen individuals?
2. **Temporal prediction:** How accurately can affective states be predicted using temporally ordered, within-subject wearable sensor data at the population-level given a specific data representation?
3. **Personalization:** To what extent can participant-specific fine-tuning improve prediction performance compared with an unadapted population model based on the used wearable sensor representation?

---

## Dataset and study design

### Study population and data collection

The study used ambulatory assessment data collected in a within-person encouragement design over at least 15 consecutive working days. Weekends were excluded from the monitoring schedule.

The initial study sample consisted of 222 participants:

- 55% female
- Mean age: 35.6 ± 10.6 years
- Mean BMI: 24.07 ± 3.3 kg/m²

Participants were university and office employees working in roles associated with elevated sedentary behavior.

Data collection was conducted using the movisensXS smartphone application (version 1.5.24).

### Ecological momentary assessment

EMA prompts were triggered after 30 consecutive minutes of sedentary behavior and were delivered between 8:40 a.m. and 9:40 p.m.

The protocol included:

- A maximum of six prompts per day
- A 30-minute cooldown between prompts
- The option to postpone prompts by up to 15 minutes in three five-minute increments

Participants were assigned to one of four conditions:

- **A:** participants were encouraged to continue sedentary behavior
- **B/C/D:** participants were encouraged to stand (B), walk slowly \(C\), or walk quickly (D)

The five-minute behavioral interval following a prompt was recorded by the wearable sensors. Participants then completed the corresponding EMA questionnaire. The sensor window was paired with the EMA response associated with that interval.

### Analytical sample

To support cross-validation, participants with fewer than ten paired sensor–EMA observations were excluded.
The resulting analytical dataset comprised:

| Characteristic | Value |
|---|---:|
| Initial participants | 222 |
| Participants included in the analysis | 215 |
| EMA questionnaires | 13,982 |
| Monitoring days | 3,225 |
| Average assessments per participant-day | 4.3 |

---

## Affective outcomes

Affective states were assessed using six items adapted from Wilhelm and Schoebi.

Each item was measured on a continuous scale ranging from 0 to 100.

The three target dimensions were calculated as follows:

| Target | Items and calculation |
|---|---|
| Energetic Arousal (EA) |$$ EA = \frac{EA_1 + 100 - EA_2}{2}$$ |
| Valence (V) | $$ V = \frac{V_2 + 100 - V_1}{2}$$ |
| Calmness \(C\) | $$ C = \frac{C_1 + 100 - C_2}{2}$$ |

The original item pairs were:

- **Energetic Arousal:** tired–awake ($EA_1$) and energy–without energy ($EA_2$)
- **Valence:** content–discontent ($V_1$) and unwell–well ($V_2$)
- **Calmness:** agitated–calm ($C_1$) and relaxed–tense ($C_2$)

The target distributions were clustered around moderate or neutral values. To address this imbalance, the framework includes target transformations and density-based sample scheduling.

### Target transformations

Two target representations are supported:

- **Min–Max scaling:** maps target values to the interval [0, 1]
- **Logit transformation:** first scales values from [0, 100] to $[\epsilon, 1-\epsilon]$, using $\epsilon = 10^{-4}$, and then applies ${logit}(y)=\ln\left(\frac{y}{1-y}\right)$
The logit transformation allows the framework to represent the target distribution on an unbounded scale to heavily penalize moderate predictions on extreme ground-truth values, while avoiding numerical problems at the boundaries.

### KDE-based label scheduling

The framework also includes a Gaussian kernel density estimation (KDE)-based label scheduler.
The scheduler estimates the distribution of target values and adjusts sample selection to give greater attention to sparsely represented target regions. This is intended to mitigate the concentration of observations around moderate affective states.

---

## Wearable sensor representations

Two wearable devices were used:

- **movisens move 4:** thigh-worn accelerometer with a measurement range of ±16 g and a sampling frequency of 64 Hz
- **Fitbit Inspire 2:** wrist-worn activity tracker providing aggregated cardiovascular, metabolic, and behavioral measurements

### 1. Fitbit aggregates

Fitbit data provide commercially available aggregated measurements describing cardiovascular activity, physical activity, and metabolic or behavioral characteristics.
These measurements represent a lower-dimensional and more readily accessible sensor representation than raw accelerometry.

### 2. Handcrafted statistical features

Handcrafted features were extracted from accelerometer data using TSFEL (Time Series Feature Extraction Library).
Feature extraction included:

- Statistical features
- Temporal features
- Spectral features

Feature extraction was performed on non-overlapping one-minute windows. Consequently, each five-minute observation was represented by five temporal feature steps.

Additional movement-related features were derived from the movisens data-processing outputs.

### 3. Raw acceleration signals

Raw accelerometry was recorded at 64 Hz across three axes.

For each five-minute window, the input therefore contains $L = 5 \times 60 \times 64 = 19{,}200$ time steps per acceleration channel.

#### Signal preprocessing

Raw acceleration was processed using single-pass exponential moving average infinite impulse response filters.
The filter coefficient is defined as $\alpha = 1-\exp\left(-2\pi\frac{f_c}{f_s}\right)$ where $f_c$ denotes the cutoff frequency and $f_s$ the sampling frequency.

The processing pipeline includes:

1. **Low-pass filtering:** a 20 Hz cutoff is used to attenuate high-frequency noise
2. **Gravity estimation:** a 0.5 Hz low-pass filter estimates the gravitational component
3. **Dynamic acceleration:** the estimated gravity component is subtracted from the cleaned acceleration signal
4. **Magnitude features:** total and dynamic acceleration magnitudes are derived
5. **Jerk:** first-order central differences are used to derive jerk signals

The resulting signals provide complementary information about body movement, orientation-related acceleration, and movement intensity.

---

## Feature selection and preprocessing

### Feature selection (for handcrafted statistical features)

Feature selection is conducted within the training data to avoid information leakage.
The selection procedure consists of two stages:

1. **Correlation-based pruning:** highly correlated features are identified using Pearson correlation. For pairs with an absolute correlation above 0.85, the feature with the higher mean inter-feature correlation is removed.
2. **Minimum Redundancy Maximum Relevance (mRMR):** the remaining features are ranked using mutual information, and the top 20 features are selected for the multidimensional prediction task.

### Participant-wise normalization

Continuous features are z-normalized on a participant-specific basis.
The normalization strategy depends on the evaluation setting:

- **Within-subject evaluation:** normalization parameters are estimated using the participant's training segment only and then applied to the training, validation, and test segments.
- **Between-subject evaluation and personalization:** normalization parameters for training participants are estimated from their training data. For unseen validation and test participants, the initial 50% of each participant's timeline is used to estimate normalization parameters. The resulting transformation is then applied to the participant's full evaluation sequence. This procedure represents an initial onboarding calibration step.

---

## Model architectures

The experimental framework combines conventional machine learning models, feature-based deep learning architectures, and neural architectures designed for high-frequency time-series data.

### Feature-based and aggregated representations

The following models are included:

| Model | Description |
|---|---|
| Random Forest (RF) | Ensemble of decision trees for nonlinear regression |
| XGBoost (XGB) | Gradient-boosted decision trees |
| TSMixer | Time-series architecture based on temporal and feature mixing |
| FT-Transformer | Feature-token-based transformer architecture using self-attention |

### Raw acceleration architectures

The following deep learning architectures are included for raw high-frequency movement data:

| Model | Description |
|---|---|
| BiMamba | Bidirectional state-space architecture |
| PatchTST | Patch-based transformer for time-series modeling |
| ModernTCN | Temporal convolutional network using large-kernel convolutions |
| TimesNet | Period-based time-series architecture using two-dimensional convolutional processing |

#### Shared architectural components

The raw-signal architectures use several tailored processing components.

**Instance normalization and signal context**

Instance normalization is used to normalize the input sequence. The original mean and standard deviation are retained and incorporated into the learned representation as contextual information.

**Multi-scale gated patching**

With the exception of ModernTCN, the architectures use multi-scale gated patching to reduce the temporal sequence length.
The patching module captures fine-, medium-, and coarse-scale temporal patterns using different receptive fields. A SwiGLU-based gated projection combines the resulting representations.
This transforms the original sequence length \(L\) into a smaller number of patches \(N\).

**Temporal attention pooling**

Multi-head attention pooling aggregates the patch-level representations into a global temporal representation.

**Contextual channel aggregation**

A nonlinear expansion and bottleneck module combines information across input channels. Softmax-normalized weights are used to aggregate the channel representations.

### Neural network training

The neural architectures use a training framework incorporating:

- AdamW optimization
- Automatic mixed precision (AMP)
- Cosine learning-rate scheduling with warmup
- SmoothL1 or binary cross-entropy loss, depending on the target transformation and model configuration
- Early stopping
- Pre-layer normalization
- Residual connections and LayerScale
- Gradient clipping
- Architecture-specific weight initialization

The experimental configurations also support ablation studies of architectural components.

---

## Evaluation protocols

Three complementary evaluation settings are used to distinguish population-level generalization, temporal within-person prediction, and participant-specific adaptation.

### 1. Between-subject generalization

The between-subject evaluation uses nested 10-fold group cross-validation.

- The outer cross-validation loop holds out approximately 10% of participants for testing.
- The remaining participants are used for model development.
- An inner group-based split allocates approximately 10% of participants for validation and 80% for training.

Participants are strictly separated between training, validation, and test partitions.
This setting evaluates how well a model trained on a group of participants generalizes to individuals not represented during model training.

### 2. Within-subject temporal evaluation

The within-subject evaluation uses expanding chronological training windows across participant timelines ($t_{training} < t_{validation} < t_{test}$).
Five temporal horizons are evaluated:

| Training proportion | Validation proportion | Test proportion |
|---:|---:|---:|
| 40% | Next 10% | Following 10% |
| 50% | Next 10% | Following 10% |
| 60% | Next 10% | Following 10% |
| 70% | Next 10% | Following 10% |
| 80% | Next 10% | Following 10% |

This setting evaluates how well a model trained on a group of participants generalizes to future data of these participants.

### 3. Participant-specific transfer learning

**Stage 1: Population model training**

A population model is trained and validated using the between-subject cross-validation procedure.

**Stage 2: Participant-specific adaptation**

For each held-out test participant, the population model is evaluated in two ways:

1. **Zero-shot prediction:** the original population model is applied without participant-specific fine-tuning
2. **Personalized prediction:** the population model is fine-tuned using an initial chronological segment of the participant's own data

The participant-specific training proportions range from 10% to 80%, increasing in increments of 10%.
Both the unadapted and personalized models are evaluated on the same participant-specific test segment. This allows the effect of fine-tuning to be assessed using identical test observations.

---

## Model training and configuration

Experiments are controlled through YAML configuration files.
The repository contains more than 390 experimental configurations.
Each configuration is intended to define a reproducible experimental setup. Users should inspect the corresponding YAML file before running an experiment to understand its model, representation, and training settings.

The configuration directory contains multiple configurations for different data representations and model architectures. For example:

```text
configs/
└── raw/
    └── BiMamba/
        └── 1/
            └── config.yml
```

---

## Explainability and feature attribution

Model interpretability is investigated using SHapley Additive exPlanations (SHAP).
The framework computes out-of-fold feature contributions to avoid evaluating explanations on data used to fit the corresponding model.

### Out-of-fold SHAP

The explanation procedure follows these principles:

- Background samples are drawn from the corresponding training partition
- Feature attributions are calculated on the held-out test partition
- The explanation procedure is performed separately for the cross-validation folds

### Tree-based models

TreeSHAP is used for tree-based architectures, including Random Forest and XGBoost.

### Raw acceleration models

For raw acceleration architectures, a hierarchical patch-level GradientSHAP procedure is used due to the long sequence length.

The procedure:

1. Divides the raw time series into temporal patches
2. Computes attribution values at the patch level
3. Distributes each patch attribution uniformly across its constituent fine-grained time steps

---

## Evaluation metrics and statistical analysis

Predictive performance is evaluated using:

- **Mean Absolute Error (MAE)**
- **Root Mean Squared Error (RMSE)**
- **Coefficient of determination (R²)**

Performance summaries include:

- Mean
- Standard deviation
- Median
- Interquartile range (IQR)
- 95% bootstrap confidence intervals using 1,000 resamples

Statistical comparisons use Wilcoxon signed-rank tests. Multiple comparisons are corrected using the Holm–Bonferroni procedure.

---

## Repository structure

### Main components

| Directory | Purpose |
|---|---|
| `configs/` | YAML configuration files for data representatio and model experiments |
| `datasets/` | Dataset handling, data scheduling, and cross-validation splitting |
| `data_extractors/` | Extraction and preparation of study data |
| `explainability/` | SHAP configuration, attribution strategies, and visualization |
| `models_backend/` | Model implementations and neural network layers |
| `models_unified/` | Unified model interface |
| `preprocessing/` | Target transformations and time-series normalization |
| `train/` | Training procedures, learning-rate scheduling, and evaluation metrics |
| `utils/` | Configuration loading, model construction, and shared utilities |

---

## Installation

### Prerequisites

The following are required for the general GPU installation:

- Python 3.11
- Linux operating system (if you want to use the Mamba model)
- GNU C/C++ compiler and build tools for packages that require compilation (if you want to use the Mamba model)
- NVIDIA GPU (if you want GPU acceleration)
- Compatible NVIDIA driver (if you want GPU acceleration)
- CUDA-compatible PyTorch installation (if you want GPU acceleration)

The exact CUDA and PyTorch combination must be compatible with the local GPU driver and the dependencies being compiled.

### (High-Performance Cluster) installation

The following installation procedure (`install_script.sh`) reflects the cluster environment used for the experiments. Adjustments may be needed for you specific environment.
It uses SLURM, the CUDA module, and a Python 3.11 virtual environment.

#### 1. Load the cluster environment

The installation script begins by clearing loaded modules and loading CUDA 12.9:

```bash
module purge
module load devel/cuda/12.9
```

It then configures the CUDA paths and GNU compilers.

#### 2. Create the Python environment

The provided installation script creates the environment at:

```text
~/envs/mamba_env
```

The environment is created using Python 3.11 if it does not already exist.

#### 3. Install the dependencies

The installation script installs the following explicitly pinned packages before installing the project requirements:

- `packaging==26.2`
- `ninja==1.13.0`
- `einops==0.8.2`
- `numpy==2.4.6`
- `huggingface_hub==1.25.1`
- `triton==3.7.1`
- `transformers==5.14.1`
- `torch==2.13.0+cu126`
- `causal-conv1d==1.6.2.post1`
- `mamba-ssm==2.3.2.post1`

The installation script uses the PyTorch CUDA 12.6 wheel and the locally loaded CUDA 12.9 toolkit for the cluster build environment. This combination reflects the supplied cluster setup and should not be assumed to work unchanged on other systems.
The script also sets CUDA-related environment variables and compilation targets specific to the tested cluster configuration.

#### 4. Run the installation script


```bash
sbatch install_script.sh
```

#### 5. Activate the environment

After successful installation:

```bash
source ~/envs/mamba_env/bin/activate
```

### Simplified installation without Mamba architecture

BiMamba is the only architecture in this framework that requires the additional `mamba-ssm` and `causal-conv1d` packages.
These packages involve CUDA-dependent compilation and can introduce compatibility challenges on different Linux systems.
**If you do not intend to use BiMamba, the framework can be simplified by removing its Mamba-specific components.**
This requires removing the `mamba_ssm` import from `models_backend/BiMamba.py`.
After these changes, the framework can be easily installed without `mamba-ssm` and `causal-conv1d` using only pip.

---

## Running experiments

Experiments are launched by passing a YAML configuration file to the training entry point.

For example:

```bash
python src/Train.py src/configs/raw/BiMamba/1/config.yml
```

### Running on cluster using SLURM

The provided cluster run script accepts a configuration file as its first argument and submits the experiment to the cluster.

For example:

```bash
sbatch run_script.sh src/configs/raw/BiMamba/1/config.yml
```

The run script activates the environment at `~/envs/mamba_env`, loads the CUDA module, and executes the training entry point.

---

## Data availability

The dataset used in this study is not included in the public repository.
It is available upon reasonable request, subject to the applicable data protection requirements and study conditions.
The repository provides the code and experimental configurations required to understand and reproduce the analysis when the corresponding data are available.

---

## Citation

If you use this repository in academic work, please cite the associated manuscript.

**Manuscript:**

*The full bibliographic reference and DOI will be added following publication..*

---

## Acknowledgements

We acknowledge the study participants and the researchers involved in the collection and preparation of the ambulatory assessment and wearable sensor data.
We also acknowledge the developers of the open-source libraries and model architectures used in this project.

---

## Contact

For questions about the framework, experimental configurations, or data access, please contact me.