import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split, TimeSeriesSplit, cross_val_score
from sklearn.model_selection import learning_curve
import logging
from typing import List, Dict, Any, Optional, Tuple
import os
from scipy.fft import fft
from scipy import signal
import pywt
from sklearn.exceptions import NotFittedError
import warnings

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('vortex_detection.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# =============================================================================
# Global Parameters and Data Files
# =============================================================================
FIXED_BEFORE = 150  # Number of rows before the matching index (historical data only)
SUB_WINDOW_SIZE = 20  # Number of rows per sliding (sub) window
STEP_SIZE = 1          # Slide one row at a time

DEFAULT_SIMPLE_PRESSURE_DROP_THRESHOLD = 0.08  # 8% drop: scheme 1 threshold (relative drop)
EXPERT_Z_THRESHOLD = 1.0                       # Scheme 2 threshold: tunable z-score threshold

VORTEX_FILE = "Jackson_vortex_detections_reformatted_augmented.csv"
ML_FILE = "ml_ready_vortex_data.csv"
OUTPUT_CSV = "address1.csv"

# =============================================================================
# Data Loading Functions
# =============================================================================
def read_vortex_sclk(file_path: str) -> List[float]:
    """
    Read the vortex CSV file and extract all SCLK values.
    
    Args:
        file_path: Path to the vortex CSV file
        
    Returns:
        List of SCLK values
        
    Raises:
        FileNotFoundError: If the file doesn't exist
        ValueError: If the file is empty or doesn't contain SCLK column
        pd.errors.EmptyDataError: If the CSV file is empty
    """
    try:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Vortex file not found: {file_path}")
            
        vortex_df = pd.read_csv(file_path)
        
        if vortex_df.empty:
            raise ValueError("Vortex file is empty")
            
        if "SCLK" not in vortex_df.columns:
            raise ValueError("SCLK column not found in vortex file")
            
        sclk_list = vortex_df["SCLK"].tolist()
        logger.info(f"Successfully read {len(sclk_list)} SCLK values from {file_path}")
        return sclk_list
        
    except pd.errors.EmptyDataError:
        logger.error(f"CSV file is empty: {file_path}")
        raise
    except Exception as e:
        logger.error(f"Error reading vortex file {file_path}: {str(e)}")
        raise

def read_ml_data(file_path: str) -> pd.DataFrame:
    """
    Read the ML CSV file, validate required columns, and convert types.
    
    Args:
        file_path: Path to the ML data CSV file
        
    Returns:
        DataFrame containing the ML data
        
    Raises:
        FileNotFoundError: If the file doesn't exist
        ValueError: If required columns are missing or data conversion fails
        pd.errors.EmptyDataError: If the CSV file is empty
    """
    try:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"ML data file not found: {file_path}")
            
        ml_df = pd.read_csv(file_path)
        
        if ml_df.empty:
            raise ValueError("ML data file is empty")
            
        required_cols = ["SCLK", "PRESSURE", "gt_detection_win", "gt_fwhm"]
        missing_cols = [col for col in required_cols if col not in ml_df.columns]
        if missing_cols:
            raise ValueError(f"Required columns missing in ML data: {missing_cols}")
        
        try:
            ml_df["gt_detection_win"] = ml_df["gt_detection_win"].astype(bool)
            ml_df["gt_fwhm"] = ml_df["gt_fwhm"].astype(bool)
        except Exception as e:
            raise ValueError(f"Error converting ground truth columns to boolean: {str(e)}")
            
        logger.info(f"Successfully loaded ML data from {file_path}")
        return ml_df
        
    except pd.errors.EmptyDataError:
        logger.error(f"CSV file is empty: {file_path}")
        raise
    except Exception as e:
        logger.error(f"Error reading ML data file {file_path}: {str(e)}")
        raise

def compute_frequency_features(pressure_series):
    """
    Compute frequency domain features using FFT and Wavelet transforms.
    
    Args:
        pressure_series: Array of pressure values
        
    Returns:
        Dictionary of frequency domain features
    """
    # FFT features
    fft_vals = fft(pressure_series)
    fft_magnitude = np.abs(fft_vals)
    
    # Get dominant frequencies
    freqs = np.fft.fftfreq(len(pressure_series))
    dominant_freq_idx = np.argsort(fft_magnitude)[-3:]  # Top 3 frequencies
    dominant_freqs = freqs[dominant_freq_idx]
    
    # Wavelet features
    coeffs = pywt.wavedec(pressure_series, 'db4', level=3)
    wavelet_energy = [np.sum(np.square(c)) for c in coeffs]
    
    return {
        'fft_energy': np.sum(np.square(fft_magnitude)),
        'dominant_freq_1': dominant_freqs[0],
        'dominant_freq_2': dominant_freqs[1],
        'dominant_freq_3': dominant_freqs[2],
        'wavelet_energy_1': wavelet_energy[0],
        'wavelet_energy_2': wavelet_energy[1],
        'wavelet_energy_3': wavelet_energy[2]
    }

def compute_temporal_features(pressure_series):
    """
    Compute temporal features including rate of change and acceleration.
    
    Args:
        pressure_series: Array of pressure values
        
    Returns:
        Dictionary of temporal features
    """
    # Rate of change (first derivative)
    rate_of_change = np.gradient(pressure_series)
    
    # Acceleration (second derivative)
    acceleration = np.gradient(rate_of_change)
    
    # Rate of change statistics
    roc_mean = np.mean(rate_of_change)
    roc_std = np.std(rate_of_change)
    roc_max = np.max(np.abs(rate_of_change))
    
    # Acceleration statistics
    acc_mean = np.mean(acceleration)
    acc_std = np.std(acceleration)
    acc_max = np.max(np.abs(acceleration))
    
    return {
        'rate_of_change_mean': roc_mean,
        'rate_of_change_std': roc_std,
        'rate_of_change_max': roc_max,
        'acceleration_mean': acc_mean,
        'acceleration_std': acc_std,
        'acceleration_max': acc_max
    }

def compute_enhanced_pressure_features(pressure_series):
    """
    Compute enhanced pressure-based features.
    
    Args:
        pressure_series: Array of pressure values
        
    Returns:
        Dictionary of enhanced pressure features
    """
    # Pressure gradients
    pressure_gradient = np.gradient(pressure_series)
    gradient_mean = np.mean(pressure_gradient)
    gradient_std = np.std(pressure_gradient)
    
    # Pressure peaks and valleys
    peaks, _ = signal.find_peaks(pressure_series)
    valleys, _ = signal.find_peaks(-pressure_series)
    
    # Peak statistics
    peak_heights = pressure_series[peaks]
    valley_heights = pressure_series[valleys]
    
    return {
        'pressure_gradient_mean': gradient_mean,
        'pressure_gradient_std': gradient_std,
        'peak_count': len(peaks),
        'valley_count': len(valleys),
        'peak_height_mean': np.mean(peak_heights) if len(peak_heights) > 0 else 0,
        'valley_height_mean': np.mean(valley_heights) if len(valley_heights) > 0 else 0
    }

def process_vortex_sclk(vortex_sclk: int, ml_df: pd.DataFrame) -> List[Dict[str, Any]]:
    """
    Process a single vortex SCLK value with enhanced feature engineering.
    """
    try:
        if not isinstance(vortex_sclk, (int, float)):
            raise ValueError(f"Invalid SCLK value type: {type(vortex_sclk)}")
            
        if ml_df.empty:
            raise ValueError("ML DataFrame is empty")
            
        labeled_windows = []
        matching_rows = ml_df[ml_df["SCLK"] == vortex_sclk]
        
        if matching_rows.empty:
            logger.warning(f"SCLK value {vortex_sclk} not found in ml data; skipping.")
            return labeled_windows

        matching_index = matching_rows.index[0]
        logger.info(f"Processing vortex SCLK {vortex_sclk} found at ml index: {matching_index}")

        fixed_start_index = max(matching_index - FIXED_BEFORE, 0)
        fixed_end_index = matching_index - 1
        
        if fixed_start_index >= fixed_end_index:
            logger.warning(f"Invalid window indices for SCLK {vortex_sclk}: start={fixed_start_index}, end={fixed_end_index}")
            return labeled_windows
            
        fixed_window_df = ml_df.iloc[fixed_start_index: fixed_end_index + 1]

        # Determine the detection ("red") region within the fixed window.
        red_region = fixed_window_df[fixed_window_df["gt_detection_win"]]
        if not red_region.empty:
            red_region_start = red_region.index.min()
            red_region_end = red_region.index.max()
        else:
            red_region_start, red_region_end = matching_index, matching_index

        # Slide a sub-window over the fixed window.
        for i in range(0, len(fixed_window_df) - SUB_WINDOW_SIZE + 1, STEP_SIZE):
            sub_window = fixed_window_df.iloc[i: i + SUB_WINDOW_SIZE]
            right_index = sub_window.index[-1]

            # Labeling logic
            if right_index < red_region_start:
                ml_label = False
            elif red_region_start <= right_index <= red_region_end:
                ml_label = True
            else:
                continue

            # Basic pressure features
            pressure_series = sub_window["PRESSURE"].values
            initial_pressure = pressure_series[0]
            final_pressure = pressure_series[-1]
            mean_pressure = np.mean(pressure_series)
            std_pressure = np.std(pressure_series)
            pressure_change = final_pressure - initial_pressure
            pressure_drop_ratio = (initial_pressure - final_pressure) / initial_pressure

            # Compute new features
            freq_features = compute_frequency_features(pressure_series)
            temporal_features = compute_temporal_features(pressure_series)
            enhanced_pressure_features = compute_enhanced_pressure_features(pressure_series)

            # Detection schemes
            scheme1_detection = pressure_drop_ratio >= DEFAULT_SIMPLE_PRESSURE_DROP_THRESHOLD
            z_score = (initial_pressure - final_pressure) / std_pressure if std_pressure > 0 else 0
            scheme2_detection = z_score >= EXPERT_Z_THRESHOLD

            # Long-term features
            long_term_window = fixed_window_df.iloc[:i + SUB_WINDOW_SIZE]
            long_term_mean = long_term_window["PRESSURE"].mean()
            long_term_std = long_term_window["PRESSURE"].std()
            span_val = max(len(long_term_window) // 2, 1)
            ema_pressure = long_term_window["PRESSURE"].ewm(span=span_val, adjust=False).mean().iloc[-1]
            trend = mean_pressure - long_term_mean

            # Combine all features
            row_data = sub_window.iloc[-1].to_dict()
            row_data.update({
                "sub_window_start_index": sub_window.index[0],
                "sub_window_end_index": right_index,
                "mean_pressure": mean_pressure,
                "std_pressure": std_pressure,
                "pressure_change": pressure_change,
                "pressure_drop_ratio": pressure_drop_ratio,
                "z_score": z_score,
                "scheme1_detection": scheme1_detection,
                "scheme2_detection": scheme2_detection,
                "ml_label": ml_label,
                "vortex_sclk": vortex_sclk,
                "long_term_mean": long_term_mean,
                "long_term_std": long_term_std,
                "ema_pressure": ema_pressure,
                "trend": trend,
                **freq_features,
                **temporal_features,
                **enhanced_pressure_features
            })
            labeled_windows.append(row_data)
        return labeled_windows
        
    except Exception as e:
        logger.error(f"Error processing vortex SCLK {vortex_sclk}: {str(e)}")
        raise

def process_all_vortices(vortex_sclk_list: list, ml_df: pd.DataFrame) -> pd.DataFrame:
    """Loop through all vortex SCLK values and compile a DataFrame of labeled sliding windows."""
    all_labeled_windows = []
    for vortex_sclk in vortex_sclk_list:
        labeled_windows = process_vortex_sclk(vortex_sclk, ml_df)
        all_labeled_windows.extend(labeled_windows)
    labeled_df = pd.DataFrame(all_labeled_windows)
    return labeled_df

# =============================================================================
# Modeling and Threshold Tuning Functions
# =============================================================================
def threshold_tuning(y_true: np.ndarray, y_scores: np.ndarray):
    """
    Tune the threshold by maximizing the F1 score.
    Returns:
      best_thresh: Best threshold found.
      best_f1: Best F1 score.
      best_precision: Precision at best threshold.
      best_recall: Recall at best threshold.
      thresholds: List of thresholds tried.
      f1_scores, precisions, recalls: Lists of metric values for each threshold.
    """
    thresholds = np.linspace(y_scores.min(), y_scores.max(), 101)
    f1_scores = []
    precisions = []
    recalls = []
    
    best_f1 = 0
    best_thresh = None

    for t in thresholds:
        y_pred = (y_scores >= t).astype(int)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        prec = precision_score(y_true, y_pred, zero_division=0)
        rec = recall_score(y_true, y_pred, zero_division=0)
        f1_scores.append(f1)
        precisions.append(prec)
        recalls.append(rec)
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = t

    # Metrics at the best threshold.
    y_pred_best = (y_scores >= best_thresh).astype(int)
    best_precision = precision_score(y_true, y_pred_best, zero_division=0)
    best_recall = recall_score(y_true, y_pred_best, zero_division=0)

    return best_thresh, best_f1, best_precision, best_recall, thresholds, f1_scores, precisions, recalls

# =============================================================================
# Plotting and Evaluation Functions
# =============================================================================
def plot_metrics(thresholds, f1_scores, precisions, recalls, best_thresh):
    """Plot threshold tuning metrics (F1, Precision, Recall) vs. thresholds (Image 1)."""
    plt.figure(figsize=(10, 6))
    plt.plot(thresholds, f1_scores, label="F1 Score", marker="o")
    plt.plot(thresholds, precisions, label="Precision", marker="x")
    plt.plot(thresholds, recalls, label="Recall", marker="s")
    plt.axvline(best_thresh, color='gray', linestyle='--', label=f"Best Threshold: {best_thresh:.4f}")
    plt.xlabel("Threshold")
    plt.ylabel("Metric Value")
    plt.title("Threshold Tuning on Classifier Output")
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_confusion_matrix(cm, classes, title='Confusion Matrix'):
    """Plot the confusion matrix as a heatmap (Image 2)."""
    plt.figure(figsize=(6, 5))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes)
    plt.yticks(tick_marks, classes)
    
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
    
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    plt.show()

def plot_cross_validation(cv_scores):
    """Plot cross validation scores as a bar chart (Image 3)."""
    plt.figure(figsize=(8, 5))
    folds = np.arange(1, len(cv_scores) + 1)
    plt.bar(folds, cv_scores, color="skyblue")
    plt.plot(folds, cv_scores, marker="o", color="black", linestyle="--")
    plt.xlabel("CV Fold")
    plt.ylabel("Accuracy")
    plt.title(f"Time Series CV Scores (Mean Accuracy: {cv_scores.mean():.3f})")
    plt.ylim(0, 1)
    plt.xticks(folds)
    plt.grid(True, axis='y')
    plt.show()

def plot_feature_importance(feature_names, importances):
    """Plot feature importance as a bar chart (Image 4)."""
    plt.figure(figsize=(10, 6))
    indices = np.argsort(importances)
    plt.barh(range(len(importances)), importances[indices], align="center", color="green")
    plt.yticks(range(len(importances)), np.array(feature_names)[indices])
    plt.xlabel("Relative Importance")
    plt.title("Feature Importances")
    plt.show()
    
def plot_learning_curve(estimator, X, y, cv, scoring="f1", train_sizes=np.linspace(0.1, 1.0, 8)):
    """
    Draws a learning curve showing model performance vs. training-set size.

    Parameters
    ----------
    estimator : fitted model (supports .fit and .score)
    X, y      : full feature matrix and label vector
    cv        : cross-validation splitter (TimeSeriesSplit recommended)
    scoring   : metric to evaluate, default "f1"
    train_sizes : iterable of fractions/integers for training set sizes
    """
    train_sizes_abs, train_scores, val_scores = learning_curve(
        estimator,
        X,
        y,
        cv=cv,
        train_sizes=train_sizes,
        scoring=scoring,
        n_jobs=-1,
        shuffle=False,     # keep chronological order for time-series data
        verbose=0,
    )

    train_mean = train_scores.mean(axis=1)
    train_std  = train_scores.std(axis=1)
    val_mean   = val_scores.mean(axis=1)
    val_std    = val_scores.std(axis=1)

    plt.figure(figsize=(8, 5))
    plt.fill_between(train_sizes_abs, train_mean - train_std, train_mean + train_std, alpha=0.15)
    plt.fill_between(train_sizes_abs, val_mean  - val_std,  val_mean  + val_std,  alpha=0.15)
    plt.plot(train_sizes_abs, train_mean, marker="o", label="Training F1")
    plt.plot(train_sizes_abs, val_mean,  marker="s", label="Validation F1")
    plt.xlabel("Training-set size (samples)")
    plt.ylabel("F1 Score")
    plt.title("Learning Curve – Random Forest")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def vortex_level_validation(labeled_df: pd.DataFrame, threshold: float, prob_column="classifier_prob"):
    """
    Evaluate detection at the vortex level.
    A vortex is considered detected if any of its sub-windows cross the probability threshold.
    """
    vortex_groups = labeled_df.groupby("vortex_sclk")
    vortex_labels = []
    vortex_preds = []

    for vortex_sclk, group in vortex_groups:
        true_label = group["ml_label"].any()
        predicted_label = (group[prob_column] >= threshold).any()
        vortex_labels.append(int(true_label))
        vortex_preds.append(int(predicted_label))

    v_f1 = f1_score(vortex_labels, vortex_preds, zero_division=0)
    v_precision = precision_score(vortex_labels, vortex_preds, zero_division=0)
    v_recall = recall_score(vortex_labels, vortex_preds, zero_division=0)

    print("\n🔍 Vortex-Level Validation Results:")
    print(f"Vortex F1 Score: {v_f1:.3f}")
    print(f"Vortex Precision: {v_precision:.3f}")
    print(f"Vortex Recall: {v_recall:.3f}")

def analyze_class_distribution(y, stage=""):
    """
    Analyze and print class distribution in the data.
    
    Args:
        y: Array of labels
        stage: String indicating which stage of the pipeline this analysis is for
    """
    unique, counts = np.unique(y, return_counts=True)
    total = len(y)
    print(f"\n📊 Class Distribution Analysis - {stage}")
    print("=" * 50)
    for class_label, count in zip(unique, counts):
        percentage = (count/total) * 100
        print(f"Class {class_label}: {count} samples ({percentage:.2f}%)")
    print("=" * 50)

# =============================================================================
# Data Validation Functions
# =============================================================================
def validate_data_quality(ml_df: pd.DataFrame) -> Tuple[bool, List[str]]:
    """
    Validate the quality of the input data.
    
    Args:
        ml_df: DataFrame containing the ML data
        
    Returns:
        Tuple of (is_valid, list of warnings)
    """
    warnings_list = []
    
    # Check for missing values
    missing_values = ml_df.isnull().sum()
    if missing_values.any():
        warnings_list.append(f"Missing values found: {missing_values[missing_values > 0].to_dict()}")
    
    # Check for infinite values
    inf_values = np.isinf(ml_df.select_dtypes(include=np.number)).sum()
    if inf_values.any():
        warnings_list.append(f"Infinite values found: {inf_values[inf_values > 0].to_dict()}")
    
    # Check for pressure range
    pressure_range = ml_df["PRESSURE"].max() - ml_df["PRESSURE"].min()
    if pressure_range == 0:
        warnings_list.append("Pressure values are constant")
    elif pressure_range < 0.1:
        warnings_list.append("Pressure range is very small")
    
    # Check for duplicate SCLK values
    duplicate_sclk = ml_df["SCLK"].duplicated().sum()
    if duplicate_sclk > 0:
        warnings_list.append(f"Found {duplicate_sclk} duplicate SCLK values")
    
    # Check for class imbalance
    if "gt_detection_win" in ml_df.columns:
        class_ratio = ml_df["gt_detection_win"].mean()
        if class_ratio < 0.1 or class_ratio > 0.9:
            warnings_list.append(f"Severe class imbalance detected: {class_ratio:.2%} positive samples")
    
    return len(warnings_list) == 0, warnings_list

def validate_feature_matrix(X: np.ndarray, feature_names: List[str]) -> Tuple[bool, List[str]]:
    """
    Validate the feature matrix before model training.
    
    Args:
        X: Feature matrix
        feature_names: List of feature names
        
    Returns:
        Tuple of (is_valid, list of warnings)
    """
    warnings_list = []
    
    # Check for NaN values
    if np.isnan(X).any():
        warnings_list.append("NaN values found in feature matrix")
    
    # Check for infinite values
    if np.isinf(X).any():
        warnings_list.append("Infinite values found in feature matrix")
    
    # Check for constant features
    for i, name in enumerate(feature_names):
        if np.std(X[:, i]) == 0:
            warnings_list.append(f"Constant feature detected: {name}")
    
    # Check for highly correlated features
    corr_matrix = np.corrcoef(X.T)
    for i in range(len(feature_names)):
        for j in range(i+1, len(feature_names)):
            if abs(corr_matrix[i, j]) > 0.95:
                warnings_list.append(f"High correlation between {feature_names[i]} and {feature_names[j]}: {corr_matrix[i, j]:.2f}")
    
    return len(warnings_list) == 0, warnings_list

# =============================================================================
# Early Stopping Class
# =============================================================================
class EarlyStopping:
    def __init__(self, patience: int = 5, min_delta: float = 0.001, mode: str = 'max'):
        """
        Initialize early stopping.
        
        Args:
            patience: Number of epochs to wait before stopping
            min_delta: Minimum change in monitored value to qualify as an improvement
            mode: 'max' for metrics to maximize (e.g., accuracy), 'min' for metrics to minimize
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_value = None
        self.should_stop = False
        
    def __call__(self, current_value: float) -> bool:
        """
        Check if training should stop.
        
        Args:
            current_value: Current value of the monitored metric
            
        Returns:
            True if training should stop, False otherwise
        """
        if self.best_value is None:
            self.best_value = current_value
            return False
            
        if self.mode == 'max':
            if current_value > self.best_value + self.min_delta:
                self.best_value = current_value
                self.counter = 0
            else:
                self.counter += 1
        else:  # mode == 'min'
            if current_value < self.best_value - self.min_delta:
                self.best_value = current_value
                self.counter = 0
            else:
                self.counter += 1
                
        if self.counter >= self.patience:
            self.should_stop = True
            
        return self.should_stop

# =============================================================================
# Modified Main Pipeline
# =============================================================================
def main():
    try:
        # Read input data
        logger.info("Reading vortex SCLK data...")
        vortex_sclk_list = read_vortex_sclk(VORTEX_FILE)
        logger.info(f"Successfully read {len(vortex_sclk_list)} SCLK values")
        
        logger.info("Reading ML data...")
        ml_df = read_ml_data(ML_FILE)
        logger.info(f"Successfully loaded ML data with {len(ml_df)} rows")
        
        # Validate data quality
        logger.info("Validating data quality...")
        is_valid, warnings = validate_data_quality(ml_df)
        if not is_valid:
            logger.warning("Data quality issues detected:")
            for warning in warnings:
                logger.warning(f"- {warning}")
        else:
            logger.info("Data quality validation passed")
        
        # Process vortex events
        logger.info("Processing vortex events...")
        labeled_df = process_all_vortices(vortex_sclk_list, ml_df)
        
        # Prepare features and labels
        logger.info("Preparing features and labels...")
        feature_cols = [
            # Original features
            "pressure_drop_ratio", "z_score", "mean_pressure", "std_pressure",
            "pressure_change", "long_term_mean", "long_term_std", "ema_pressure", "trend",
            
            # Frequency domain features
            "fft_energy", "dominant_freq_1", "dominant_freq_2", "dominant_freq_3",
            "wavelet_energy_1", "wavelet_energy_2", "wavelet_energy_3",
            
            # Temporal features
            "rate_of_change_mean", "rate_of_change_std", "rate_of_change_max",
            "acceleration_mean", "acceleration_std", "acceleration_max",
            
            # Enhanced pressure features
            "pressure_gradient_mean", "pressure_gradient_std",
            "peak_count", "valley_count",
            "peak_height_mean", "valley_height_mean"
        ]
        
        X = labeled_df[feature_cols].values
        y_true = labeled_df["ml_label"].astype(int).values
        
        # Validate feature matrix
        logger.info("Validating feature matrix...")
        is_valid, warnings = validate_feature_matrix(X, feature_cols)
        if not is_valid:
            logger.warning("Feature matrix issues detected:")
            for warning in warnings:
                logger.warning(f"- {warning}")
        else:
            logger.info("Feature matrix validation passed")
        
        # Split data
        logger.info("Splitting data into train/val/test sets...")
        X_train, X_remaining, y_train, y_remaining = train_test_split(X, y_true, test_size=0.4, random_state=42)
        X_val, X_test, y_val, y_test = train_test_split(X_remaining, y_remaining, test_size=0.5, random_state=42)
        
        # Initialize early stopping
        early_stopping = EarlyStopping(patience=5, min_delta=0.001, mode='max')
        
        # Train model with early stopping
        logger.info("Training Random Forest classifier with early stopping...")
        clf = RandomForestClassifier(class_weight='balanced', random_state=42)
        best_score = 0
        best_model = None
        
        for epoch in range(100):  # Maximum 100 epochs
            clf.fit(X_train, y_train)
            val_score = clf.score(X_val, y_val)
            
            logger.info(f"Epoch {epoch + 1}, Validation Score: {val_score:.4f}")
            
            if early_stopping(val_score):
                logger.info("Early stopping triggered")
                break
                
            if val_score > best_score:
                best_score = val_score
                best_model = clf
        
        # Use the best model
        clf = best_model

        # === ADDED: Generate evaluation and plotting images ===
        # Feature importance plot
        importances = clf.feature_importances_
        plot_feature_importance(feature_cols, importances)

        # Cross-validation plot
        tscv = TimeSeriesSplit(n_splits=5)
        cv_scores = cross_val_score(clf, X, y_true, cv=tscv)
        plot_cross_validation(cv_scores)

        # Threshold tuning and metrics plot
        val_prob = clf.predict_proba(X_val)[:, 1]
        best_thresh, best_f1, best_precision, best_recall, thresholds, f1_scores, precisions, recalls = threshold_tuning(y_val, val_prob)
        plot_metrics(thresholds, f1_scores, precisions, recalls, best_thresh)

        # Confusion matrix plot
        test_prob = clf.predict_proba(X_test)[:, 1]
        y_pred_test = (test_prob >= best_thresh).astype(int)
        cm = confusion_matrix(y_test, y_pred_test)
        plot_confusion_matrix(cm, classes=["Negative", "Positive"])

        # Recalculate and log test metrics immediately after predictions
        acc = accuracy_score(y_test, y_pred_test)
        f1 = f1_score(y_test, y_pred_test, zero_division=0)
        prec = precision_score(y_test, y_pred_test, zero_division=0)
        rec = recall_score(y_test, y_pred_test, zero_division=0)
        logger.info(f"Test Accuracy: {acc:.3f}")
        logger.info(f"Test F1 Score: {f1:.3f}")
        logger.info(f"Test Precision: {prec:.3f}")
        logger.info(f"Test Recall: {rec:.3f}")

        # Save test set predictions and metrics
        test_results = pd.DataFrame({
            'y_true': y_test,
            'y_pred': y_pred_test,
            'probability': test_prob
        })
        test_results.to_csv('test_predictions.csv', index=False)
        logger.info('Test set predictions and probabilities saved to test_predictions.csv')

        # === ADDED: Automatically remove low-importance and highly correlated features ===
        # 1. Remove low-importance features
        importance_threshold = 0.01
        importances = clf.feature_importances_
        important_features = [f for f, imp in zip(feature_cols, importances) if imp >= importance_threshold]
        logger.info(f"Selected {len(important_features)} features with importance >= {importance_threshold}")

        # 2. Remove highly correlated features
        def remove_highly_correlated_features(df, feature_list, threshold=0.95):
            corr_matrix = np.corrcoef(df[feature_list].values, rowvar=False)
            to_remove = set()
            for i in range(len(feature_list)):
                for j in range(i+1, len(feature_list)):
                    if abs(corr_matrix[i, j]) > threshold:
                        # Remove the feature with lower importance
                        if importances[i] < importances[j]:
                            to_remove.add(feature_list[i])
                        else:
                            to_remove.add(feature_list[j])
            return [f for f in feature_list if f not in to_remove]

        reduced_features = remove_highly_correlated_features(labeled_df, important_features, threshold=0.95)
        logger.info(f"Selected {len(reduced_features)} features after removing highly correlated ones")

        # Retrain model with reduced feature set
        X_reduced = labeled_df[reduced_features].values
        X_train_r, X_remaining_r, y_train_r, y_remaining_r = train_test_split(X_reduced, y_true, test_size=0.4, random_state=42)
        X_val_r, X_test_r, y_val_r, y_test_r = train_test_split(X_remaining_r, y_remaining_r, test_size=0.5, random_state=42)
        clf_reduced = RandomForestClassifier(class_weight='balanced', random_state=42)
        clf_reduced.fit(X_train_r, y_train_r)
        logger.info("Retrained model with reduced feature set.")

        # Continue with evaluation using reduced feature set
        importances_r = clf_reduced.feature_importances_
        plot_feature_importance(reduced_features, importances_r)
        tscv = TimeSeriesSplit(n_splits=5)
        cv_scores_r = cross_val_score(clf_reduced, X_reduced, y_true, cv=tscv)
        plot_cross_validation(cv_scores_r)
        val_prob_r = clf_reduced.predict_proba(X_val_r)[:, 1]
        best_thresh_r, best_f1_r, best_precision_r, best_recall_r, thresholds_r, f1_scores_r, precisions_r, recalls_r = threshold_tuning(y_val_r, val_prob_r)
        plot_metrics(thresholds_r, f1_scores_r, precisions_r, recalls_r, best_thresh_r)
        test_prob_r = clf_reduced.predict_proba(X_test_r)[:, 1]
        y_pred_test_r = (test_prob_r >= best_thresh_r).astype(int)
        cm_r = confusion_matrix(y_test_r, y_pred_test_r)
        plot_confusion_matrix(cm_r, classes=["Negative", "Positive"])
        acc_r = accuracy_score(y_test_r, y_pred_test_r)
        f1_r = f1_score(y_test_r, y_pred_test_r, zero_division=0)
        prec_r = precision_score(y_test_r, y_pred_test_r, zero_division=0)
        rec_r = recall_score(y_test_r, y_pred_test_r, zero_division=0)
        logger.info(f"Reduced Test Accuracy: {acc_r:.3f}")
        logger.info(f"Reduced Test F1 Score: {f1_r:.3f}")
        logger.info(f"Reduced Test Precision: {prec_r:.3f}")
        logger.info(f"Reduced Test Recall: {rec_r:.3f}")
        # Save reduced test set predictions
        test_results_r = pd.DataFrame({
            'y_true': y_test_r,
            'y_pred': y_pred_test_r,
            'probability': test_prob_r
        })
        test_results_r.to_csv('test_predictions_reduced.csv', index=False)
        logger.info('Reduced test set predictions and probabilities saved to test_predictions_reduced.csv')
        # === END ADDED ===

    except Exception as e:
        logger.error(f"Error in main pipeline: {str(e)}")
        logger.error("Stack trace:", exc_info=True)
        raise

if __name__ == "__main__":
    main()
