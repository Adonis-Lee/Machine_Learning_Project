"""
CENG465 Machine Learning Toolkit
A comprehensive GUI application for training, evaluating, and comparing ML models.
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time
from io import StringIO

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.linear_model import Perceptron
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_curve, auc,
    precision_recall_curve, average_precision_score
)

# Page Configuration
st.set_page_config(
    page_title="CENG465 ML Toolkit",
    layout="wide",
    page_icon="🤖",
    initial_sidebar_state="expanded"
)

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 100


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

@st.cache_data
def load_dataset(uploaded_file):
    """
    Load and cache CSV dataset with error handling.
    
    Args:
        uploaded_file: Uploaded file object from Streamlit
        
    Returns:
        DataFrame: Loaded dataset
    """
    try:
        df = pd.read_csv(uploaded_file)
        return df
    except Exception as e:
        st.error(f"Error loading CSV file: {e}")
        return None


def display_dataset_info(df):
    """
    Display comprehensive dataset information including statistics,
    class distribution, and data types.
    
    Args:
        df: DataFrame to analyze
    """
    st.subheader("📊 Dataset Overview")
    
    # Basic Info
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Rows", f"{df.shape[0]:,}")
    col2.metric("Columns", f"{df.shape[1]:,}")
    col3.metric("Missing Values", f"{df.isnull().sum().sum():,}")
    col4.metric("Memory Usage", f"{df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    
    # Dataset Preview
    with st.expander("📋 Dataset Preview", expanded=True):
        st.dataframe(df.head(10), use_container_width=True)
        st.caption(f"Showing first 10 rows of {df.shape[0]} total rows")
    
    # Data Types
    with st.expander("🔍 Data Types & Missing Values"):
        dtype_df = pd.DataFrame({
            'Column': df.columns,
            'Data Type': df.dtypes.astype(str),
            'Non-Null Count': df.count().values,
            'Null Count': df.isnull().sum().values,
            'Null Percentage': (df.isnull().sum() / len(df) * 100).round(2).values
        })
        st.dataframe(dtype_df, use_container_width=True)
    
    # Numeric Summary
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        with st.expander("📈 Numeric Summary Statistics"):
            st.dataframe(df[numeric_cols].describe(), use_container_width=True)
    
    # Categorical Summary
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns
    if len(categorical_cols) > 0:
        with st.expander("📊 Categorical Feature Counts"):
            for col in categorical_cols:
                st.write(f"**{col}**")
                value_counts = df[col].value_counts()
                st.bar_chart(value_counts)
                st.dataframe(pd.DataFrame({
                    'Value': value_counts.index,
                    'Count': value_counts.values,
                    'Percentage': (value_counts.values / len(df) * 100).round(2)
                }), use_container_width=True)
                st.divider()


def display_class_distribution(y, target_name):
    """
    Display class distribution for target variable.
    
    Args:
        y: Target variable series
        target_name: Name of target column
    """
    with st.expander("🎯 Class Distribution", expanded=True):
        value_counts = pd.Series(y).value_counts()
        col1, col2 = st.columns(2)
        
        with col1:
            st.bar_chart(value_counts)
        
        with col2:
            dist_df = pd.DataFrame({
                'Class': value_counts.index,
                'Count': value_counts.values,
                'Percentage': (value_counts.values / len(y) * 100).round(2)
            })
            st.dataframe(dist_df, use_container_width=True)
            st.metric("Number of Classes", len(value_counts))


def handle_missing_values(X, strategy, numeric_fill='mean', categorical_fill='mode'):
    """
    Handle missing values in the dataset.
    
    Args:
        X: Feature DataFrame
        strategy: 'drop' or 'fill'
        numeric_fill: 'mean' or 'median' for numeric columns
        categorical_fill: 'mode' for categorical columns
        
    Returns:
        DataFrame: Processed DataFrame
    """
    X_processed = X.copy()
    preprocessing_log = []
    
    if strategy == 'drop':
        X_processed = X_processed.dropna()
        preprocessing_log.append("Dropped rows with missing values")
    elif strategy == 'fill':
        # Fill numeric columns
        numeric_cols = X_processed.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            if numeric_fill == 'mean':
                X_processed[numeric_cols] = X_processed[numeric_cols].fillna(
                    X_processed[numeric_cols].mean()
                )
                preprocessing_log.append(f"Filled numeric columns with mean: {list(numeric_cols)}")
            elif numeric_fill == 'median':
                X_processed[numeric_cols] = X_processed[numeric_cols].fillna(
                    X_processed[numeric_cols].median()
                )
                preprocessing_log.append(f"Filled numeric columns with median: {list(numeric_cols)}")
        
        # Fill categorical columns
        categorical_cols = X_processed.select_dtypes(include=['object', 'category']).columns
        if len(categorical_cols) > 0:
            for col in categorical_cols:
                mode_value = X_processed[col].mode()
                if len(mode_value) > 0:
                    X_processed[col] = X_processed[col].fillna(mode_value[0])
                    preprocessing_log.append(f"Filled '{col}' with mode: {mode_value[0]}")
    
    return X_processed, preprocessing_log


def preprocess_data(X, y, scaler_choice, encoding_choice, missing_strategy,
                    numeric_fill, categorical_fill):
    """
    Apply all preprocessing steps to the dataset.
    
    Args:
        X: Feature DataFrame
        y: Target series
        scaler_choice: Normalization method
        encoding_choice: Whether to apply one-hot encoding
        missing_strategy: Missing value handling strategy
        numeric_fill: Fill method for numeric columns
        categorical_fill: Fill method for categorical columns
        
    Returns:
        tuple: (X_processed, y_processed, preprocessing_log, scaler)
    """
    preprocessing_log = []
    scaler = None
    
    # Handle missing values
    X_processed, missing_log = handle_missing_values(
        X, missing_strategy, numeric_fill, categorical_fill
    )
    preprocessing_log.extend(missing_log)
    
    # One-Hot Encoding
    if encoding_choice:
        initial_shape = X_processed.shape
        X_processed = pd.get_dummies(X_processed, drop_first=True)
        preprocessing_log.append(
            f"One-Hot Encoding applied: {initial_shape} → {X_processed.shape}"
        )
    
    # Normalization
    if scaler_choice == "StandardScaler":
        scaler = StandardScaler()
        X_processed = pd.DataFrame(
            scaler.fit_transform(X_processed),
            columns=X_processed.columns,
            index=X_processed.index
        )
        preprocessing_log.append("StandardScaler normalization applied")
    elif scaler_choice == "MinMaxScaler":
        scaler = MinMaxScaler()
        X_processed = pd.DataFrame(
            scaler.fit_transform(X_processed),
            columns=X_processed.columns,
            index=X_processed.index
        )
        preprocessing_log.append("MinMaxScaler normalization applied")
    else:
        preprocessing_log.append("No normalization applied")
    
    # Encode target if needed
    label_encoder = None
    if y.dtype == 'object' or y.dtype.name == 'category':
        label_encoder = LabelEncoder()
        y_processed = pd.Series(label_encoder.fit_transform(y), index=y.index)
        preprocessing_log.append(f"Target encoded: {label_encoder.classes_}")
    else:
        y_processed = y
    
    return X_processed, y_processed, preprocessing_log, scaler, label_encoder


def create_model(model_name, hyperparams):
    """
    Create a model instance with specified hyperparameters.
    
    Args:
        model_name: Name of the model
        hyperparams: Dictionary of hyperparameters
        
    Returns:
        Model instance
    """
        if model_name == "Perceptron":
        return Perceptron(
            penalty=hyperparams.get('penalty', 'l2'),
            alpha=hyperparams.get('alpha', 0.0001),
            max_iter=hyperparams.get('max_iter', 1000),
            fit_intercept=hyperparams.get('fit_intercept', True),
            random_state=42
        )
        elif model_name == "Multilayer Perceptron (Backprop)":
        return MLPClassifier(
            hidden_layer_sizes=hyperparams.get('hidden_layer_sizes', (100,)),
            activation=hyperparams.get('activation', 'relu'),
            learning_rate_init=hyperparams.get('learning_rate_init', 0.001),
            max_iter=hyperparams.get('max_iter', 1000),
            random_state=42
        )
    elif model_name == "Decision Tree":
        return DecisionTreeClassifier(
            criterion=hyperparams.get('criterion', 'gini'),
            max_depth=hyperparams.get('max_depth', None),
            min_samples_split=hyperparams.get('min_samples_split', 2),
            min_samples_leaf=hyperparams.get('min_samples_leaf', 1),
            random_state=42
        )
        else:
        raise ValueError(f"Unknown model: {model_name}")


def train_model(model, X_train, y_train, X_test, y_test, model_name):
    """
    Train a model and compute all metrics.
    
    Args:
        model: Model instance
        X_train: Training features
        y_train: Training labels
        X_test: Test features
        y_test: Test labels
        model_name: Name of the model
        
    Returns:
        dict: Dictionary containing all metrics and predictions
    """
    # Train model and measure time
    start_time = time.time()
            model.fit(X_train, y_train)
    train_time = time.time() - start_time
    
    # Make predictions
            y_pred = model.predict(X_test)

    # Compute metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision_macro = precision_score(y_test, y_pred, average='macro', zero_division=0)
    precision_weighted = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    recall_macro = recall_score(y_test, y_pred, average='macro', zero_division=0)
    recall_weighted = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    f1_macro = f1_score(y_test, y_pred, average='macro', zero_division=0)
    f1_weighted = f1_score(y_test, y_pred, average='weighted', zero_division=0)
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    # Classification report
    report = classification_report(y_test, y_pred, zero_division=0, output_dict=True)
    
    # ROC and PR curves (if predict_proba available)
    y_proba = None
    roc_data = None
    pr_data = None
    
    try:
        if hasattr(model, 'predict_proba'):
            y_proba = model.predict_proba(X_test)
            
            # For multi-class, use One-vs-Rest approach
            if len(np.unique(y_test)) > 2:
                # Use macro-averaged ROC
                fpr = {}
                tpr = {}
                roc_auc = {}
                for i in range(len(np.unique(y_test))):
                    y_test_binary = (y_test == i).astype(int)
                    y_proba_binary = y_proba[:, i]
                    fpr[i], tpr[i], _ = roc_curve(y_test_binary, y_proba_binary)
                    roc_auc[i] = auc(fpr[i], tpr[i])
                roc_data = {'fpr': fpr, 'tpr': tpr, 'auc': roc_auc}
                
                # Precision-Recall curve
                precision_dict = {}
                recall_dict = {}
                ap_dict = {}
                for i in range(len(np.unique(y_test))):
                    y_test_binary = (y_test == i).astype(int)
                    y_proba_binary = y_proba[:, i]
                    precision_dict[i], recall_dict[i], _ = precision_recall_curve(
                        y_test_binary, y_proba_binary
                    )
                    ap_dict[i] = average_precision_score(y_test_binary, y_proba_binary)
                pr_data = {'precision': precision_dict, 'recall': recall_dict, 'ap': ap_dict}
            else:
                # Binary classification
                y_proba_binary = y_proba[:, 1] if y_proba.shape[1] > 1 else y_proba[:, 0]
                fpr, tpr, _ = roc_curve(y_test, y_proba_binary)
                roc_auc = auc(fpr, tpr)
                roc_data = {'fpr': fpr, 'tpr': tpr, 'auc': roc_auc}
                
                precision_curve, recall_curve, _ = precision_recall_curve(y_test, y_proba_binary)
                ap = average_precision_score(y_test, y_proba_binary)
                pr_data = {'precision': precision_curve, 'recall': recall_curve, 'ap': ap}
    except Exception as e:
        st.warning(f"Could not generate probability predictions for {model_name}: {e}")
    
    return {
        'model': model,
        'y_pred': y_pred,
        'y_proba': y_proba,
        'accuracy': accuracy,
        'precision_macro': precision_macro,
        'precision_weighted': precision_weighted,
        'recall_macro': recall_macro,
        'recall_weighted': recall_weighted,
        'f1_macro': f1_macro,
        'f1_weighted': f1_weighted,
        'confusion_matrix': cm,
        'confusion_matrix_normalized': cm_normalized,
        'classification_report': report,
        'train_time': train_time,
        'roc_data': roc_data,
        'pr_data': pr_data
    }


def plot_confusion_matrix(cm, normalized=False, class_names=None):
    """
    Plot confusion matrix heatmap.
    
    Args:
        cm: Confusion matrix array
        normalized: Whether to normalize the matrix
        class_names: Optional list of class names
        
    Returns:
        matplotlib figure
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    
    if normalized:
        fmt = '.2f'
        cm_plot = cm
        title = "Normalized Confusion Matrix"
    else:
        fmt = 'd'
        cm_plot = cm
        title = "Confusion Matrix"
    
    sns.heatmap(
        cm_plot,
        annot=True,
        fmt=fmt,
        cmap="Blues",
        ax=ax,
        xticklabels=class_names if class_names else 'auto',
        yticklabels=class_names if class_names else 'auto'
    )
    ax.set_xlabel('Predicted Label')
    ax.set_ylabel('True Label')
    ax.set_title(title)
    plt.tight_layout()
    return fig


def plot_roc_curve(roc_data, multi_class=False):
    """
    Plot ROC curve(s).
    
    Args:
        roc_data: Dictionary containing ROC curve data
        multi_class: Whether this is multi-class classification
        
    Returns:
        matplotlib figure
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    
    if multi_class:
        for i, (class_idx, auc_score) in enumerate(roc_data['auc'].items()):
            ax.plot(
                roc_data['fpr'][class_idx],
                roc_data['tpr'][class_idx],
                label=f'Class {class_idx} (AUC = {auc_score:.2f})'
            )
    else:
        ax.plot(
            roc_data['fpr'],
            roc_data['tpr'],
            label=f'ROC Curve (AUC = {roc_data["auc"]:.2f})'
        )
    
    ax.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curve')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig


def plot_pr_curve(pr_data, multi_class=False):
    """
    Plot Precision-Recall curve(s).
    
    Args:
        pr_data: Dictionary containing PR curve data
        multi_class: Whether this is multi-class classification
        
    Returns:
        matplotlib figure
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    
    if multi_class:
        for i, (class_idx, ap_score) in enumerate(pr_data['ap'].items()):
            ax.plot(
                pr_data['recall'][class_idx],
                pr_data['precision'][class_idx],
                label=f'Class {class_idx} (AP = {ap_score:.2f})'
            )
    else:
        ax.plot(
            pr_data['recall'],
            pr_data['precision'],
            label=f'PR Curve (AP = {pr_data["ap"]:.2f})'
        )
    
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title('Precision-Recall Curve')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig


def create_confusion_matrix_details(cm, class_names=None):
    """
    Create a detailed confusion matrix breakdown table.
    
    Args:
        cm: Confusion matrix array
        class_names: Optional list of class names
        
    Returns:
        DataFrame: Detailed confusion matrix breakdown
    """
    n_classes = cm.shape[0]
    details = []
    
    for i in range(n_classes):
        for j in range(n_classes):
            true_label = class_names[i] if class_names is not None else f"Class {i}"
            pred_label = class_names[j] if class_names is not None else f"Class {j}"
            count = int(cm[i, j])
            details.append({
                'True Label': true_label,
                'Predicted Label': pred_label,
                'Count': count
            })
    
    return pd.DataFrame(details)


def plot_per_class_metrics(classification_report_dict, metric_name='precision'):
    """
    Plot per-class metrics as a bar chart.
    
    Args:
        classification_report_dict: Dictionary from classification_report
        metric_name: Name of metric to plot ('precision', 'recall', 'f1-score')
        
    Returns:
        matplotlib figure
    """
    # Extract per-class metrics (exclude 'accuracy', 'macro avg', 'weighted avg')
    classes = []
    values = []
    
    for key, value in classification_report_dict.items():
        if key not in ['accuracy', 'macro avg', 'weighted avg'] and isinstance(value, dict):
            if metric_name in value:
                classes.append(key)
                values.append(value[metric_name])
    
    if len(classes) == 0:
        return None
    
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(classes, values, color='steelblue', alpha=0.7)
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}',
                ha='center', va='bottom')
    
    ax.set_xlabel('Class')
    ax.set_ylabel(metric_name.capitalize())
    ax.set_title(f'Per-Class {metric_name.capitalize()}')
    ax.set_ylim([0, 1.1])
    ax.grid(True, alpha=0.3, axis='y')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    return fig


def plot_feature_scatter(X_test, y_test, y_pred, feature1, feature2, class_names=None):
    """
    Plot a scatter plot of two features colored by true and predicted labels.
    
    Args:
        X_test: Test features DataFrame
        y_test: True labels
        y_pred: Predicted labels
        feature1: Name of first feature
        feature2: Name of second feature
        class_names: Optional list of class names
        
    Returns:
        matplotlib figure
    """
    if feature1 not in X_test.columns or feature2 not in X_test.columns:
        return None
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Get unique classes
    unique_classes = np.unique(np.concatenate([y_test, y_pred]))
    colors = plt.cm.Set3(np.linspace(0, 1, len(unique_classes)))
    
    # Plot true labels
    for i, cls in enumerate(unique_classes):
        mask = y_test == cls
        label = class_names[cls] if class_names is not None else f"Class {cls}"
        ax1.scatter(X_test.loc[mask, feature1], X_test.loc[mask, feature2],
                   c=[colors[i]], label=label, alpha=0.6, s=50)
    
    ax1.set_xlabel(feature1)
    ax1.set_ylabel(feature2)
    ax1.set_title('True Labels')
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax1.grid(True, alpha=0.3)
    
    # Plot predicted labels
    for i, cls in enumerate(unique_classes):
        mask = y_pred == cls
        label = class_names[cls] if class_names is not None else f"Class {cls}"
        ax2.scatter(X_test.loc[mask, feature1], X_test.loc[mask, feature2],
                   c=[colors[i]], label=label, alpha=0.6, s=50)
    
    ax2.set_xlabel(feature1)
    ax2.set_ylabel(feature2)
    ax2.set_title('Predicted Labels')
    ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


# ============================================================================
# MAIN APPLICATION
# ============================================================================

def main():
    """Main application function."""
    st.title("🎓 CENG465 - Machine Learning Toolkit")
    st.markdown("""
    A comprehensive GUI application for training, evaluating, and comparing 
    machine learning classification models on CSV datasets.
    """)
    
    # Initialize session state
    if 'results' not in st.session_state:
        st.session_state.results = {}
    if 'preprocessing_log' not in st.session_state:
        st.session_state.preprocessing_log = []
    if 'class_names' not in st.session_state:
        st.session_state.class_names = None
    
    # ========================================================================
    # SIDEBAR: Dataset Upload & Settings
    # ========================================================================
    st.sidebar.header("1. Dataset Upload")
    uploaded_file = st.sidebar.file_uploader(
        "Upload your Dataset (CSV)",
        type=["csv"],
        help="Upload a CSV file containing your dataset"
    )
    
    if uploaded_file is None:
        st.info("👈 Please upload a CSV file from the sidebar to begin.")
        return
    
    # Load dataset
    df = load_dataset(uploaded_file)
    if df is None:
        return
    
    # Display dataset information
    display_dataset_info(df)
    
    # ========================================================================
    # SIDEBAR: Target Selection
    # ========================================================================
    st.sidebar.subheader("2. Target Selection")
    cols = df.columns.tolist()
    target = st.sidebar.selectbox(
        "Select Target Column (Class Label)",
        cols,
        help="Choose the column that contains the class labels"
    )
    
    # Display class distribution
    display_class_distribution(df[target], target)
    
    # ========================================================================
    # SIDEBAR: Preprocessing Settings
    # ========================================================================
    st.sidebar.subheader("3. Preprocessing Settings")
    
    # Missing Value Handling
    missing_strategy = st.sidebar.selectbox(
        "Missing Value Handling",
        ["drop", "fill"],
        help="Choose how to handle missing values"
    )
    
    numeric_fill = None
    categorical_fill = None
    if missing_strategy == "fill":
        numeric_fill = st.sidebar.selectbox(
            "Fill Numeric Columns With",
            ["mean", "median"],
            help="Method to fill missing values in numeric columns"
        )
        categorical_fill = st.sidebar.selectbox(
            "Fill Categorical Columns With",
            ["mode"],
            help="Method to fill missing values in categorical columns"
        )
    
    # Normalization
    scaler_choice = st.sidebar.selectbox(
        "Normalization Method",
        ["None", "StandardScaler", "MinMaxScaler"],
        help="Choose normalization method for numeric features"
    )
    
    # Encoding
    encoding_choice = st.sidebar.checkbox(
        "Apply One-Hot Encoding (Auto-detect categorical)",
        value=True,
        help="Automatically encode categorical columns using one-hot encoding"
    )
    
    # Correlation Matrix Option
    show_correlation = st.sidebar.checkbox(
        "Show Correlation Matrix",
        value=False,
        help="Display correlation matrix heatmap for numeric features"
    )
    
    # ========================================================================
    # MAIN AREA: Preprocessing Summary
    # ========================================================================
    st.divider()
    st.subheader("🔧 Preprocessing Summary")
    
    # Prepare data
    X = df.drop(columns=[target])
    y = df[target]
    
    # Apply preprocessing
    X_processed, y_processed, preprocessing_log, scaler, label_encoder = preprocess_data(
        X, y, scaler_choice, encoding_choice, missing_strategy,
        numeric_fill, categorical_fill
    )
    
    # Store preprocessing log
    st.session_state.preprocessing_log = preprocessing_log
    if label_encoder is not None:
        st.session_state.class_names = label_encoder.classes_
    
    # Display preprocessing log
    with st.expander("📋 Preprocessing Steps Applied", expanded=True):
        for i, step in enumerate(preprocessing_log, 1):
            st.write(f"{i}. {step}")
        st.metric("Final Dataset Shape", f"{X_processed.shape[0]} rows × {X_processed.shape[1]} features")
    
    # Correlation Matrix
    if show_correlation:
        numeric_cols_processed = X_processed.select_dtypes(include=[np.number]).columns
        if len(numeric_cols_processed) > 0:
            with st.expander("📊 Correlation Matrix Heatmap"):
                corr_matrix = X_processed[numeric_cols_processed].corr()
                fig, ax = plt.subplots(figsize=(12, 10))
                sns.heatmap(
                    corr_matrix,
                    annot=False,
                    cmap='coolwarm',
                    center=0,
                    square=True,
                    ax=ax
                )
                ax.set_title("Feature Correlation Matrix")
                plt.tight_layout()
                st.pyplot(fig)
    
    # ========================================================================
    # SIDEBAR: Train/Test Split
    # ========================================================================
    st.sidebar.subheader("4. Train/Test Split")
    test_size = st.sidebar.slider(
        "Test Set Ratio",
        0.1,
        0.5,
        0.3,
        0.05,
        help="Proportion of dataset to use for testing (0.1 to 0.5)"
    )
    
    # Validate split
    if test_size < 0.1 or test_size > 0.5:
        st.sidebar.error("Test size must be between 0.1 and 0.5")
        return
    
    # ========================================================================
    # SIDEBAR: Model Selection & Hyperparameters
    # ========================================================================
    st.sidebar.subheader("5. Model Selection & Hyperparameters")
    
    # Model selection (multiple models)
    st.sidebar.write("**Select Models to Train:**")
    train_perceptron = st.sidebar.checkbox("Perceptron", value=False)
    train_mlp = st.sidebar.checkbox("Multilayer Perceptron (Backprop)", value=False)
    train_dt = st.sidebar.checkbox("Decision Tree", value=False)
    
    selected_models = []
    if train_perceptron:
        selected_models.append("Perceptron")
    if train_mlp:
        selected_models.append("Multilayer Perceptron (Backprop)")
    if train_dt:
        selected_models.append("Decision Tree")
    
    if len(selected_models) == 0:
        st.sidebar.warning("⚠️ Please select at least one model to train")
    
    # Hyperparameters for Perceptron
    perceptron_params = {}
    if train_perceptron:
        with st.sidebar.expander("Perceptron Hyperparameters", expanded=False):
            perceptron_params['penalty'] = st.selectbox(
                "Penalty",
                ['l2', 'l1', 'elasticnet', None],
                index=0,
                key='perceptron_penalty'
            )
            perceptron_params['alpha'] = st.number_input(
                "Alpha",
                min_value=0.0001,
                max_value=1.0,
                value=0.0001,
                step=0.0001,
                format="%.4f",
                key='perceptron_alpha'
            )
            perceptron_params['max_iter'] = st.number_input(
                "Max Iterations",
                min_value=100,
                max_value=10000,
                value=1000,
                step=100,
                key='perceptron_max_iter'
            )
            perceptron_params['fit_intercept'] = st.checkbox(
                "Fit Intercept",
                value=True,
                key='perceptron_fit_intercept'
            )
    
    # Hyperparameters for MLP
    mlp_params = {}
    if train_mlp:
        with st.sidebar.expander("MLP Hyperparameters", expanded=False):
            hidden_layers_str = st.text_input(
                "Hidden Layer Sizes (e.g., 100,50 for two layers)",
                value="100",
                key='mlp_hidden_layers'
            )
            try:
                hidden_layers = tuple(int(x.strip()) for x in hidden_layers_str.split(','))
                mlp_params['hidden_layer_sizes'] = hidden_layers
            except:
                mlp_params['hidden_layer_sizes'] = (100,)
                st.sidebar.warning("Invalid format, using default (100,)")
            
            mlp_params['activation'] = st.selectbox(
                "Activation Function",
                ['relu', 'tanh', 'logistic'],
                index=0,
                key='mlp_activation'
            )
            mlp_params['learning_rate_init'] = st.number_input(
                "Learning Rate",
                min_value=0.0001,
                max_value=1.0,
                value=0.001,
                step=0.0001,
                format="%.4f",
                key='mlp_learning_rate'
            )
            mlp_params['max_iter'] = st.number_input(
                "Max Iterations",
                min_value=100,
                max_value=10000,
                value=1000,
                step=100,
                key='mlp_max_iter'
            )
    
    # Hyperparameters for Decision Tree
    dt_params = {}
    if train_dt:
        with st.sidebar.expander("Decision Tree Hyperparameters", expanded=False):
            dt_params['criterion'] = st.selectbox(
                "Criterion",
                ['gini', 'entropy'],
                index=0,
                key='dt_criterion'
            )
            dt_params['max_depth'] = st.number_input(
                "Max Depth (None for unlimited)",
                min_value=1,
                max_value=50,
                value=None,
                step=1,
                key='dt_max_depth'
            )
            dt_params['min_samples_split'] = st.number_input(
                "Min Samples Split",
                min_value=2,
                max_value=100,
                value=2,
                step=1,
                key='dt_min_samples_split'
            )
            dt_params['min_samples_leaf'] = st.number_input(
                "Min Samples Leaf",
                min_value=1,
                max_value=50,
                value=1,
                step=1,
                key='dt_min_samples_leaf'
            )
    
    # ========================================================================
    # TRAINING BUTTON
    # ========================================================================
    st.sidebar.divider()
    train_button = st.sidebar.button("🚀 Train Selected Models", type="primary")
    
    if train_button and len(selected_models) > 0:
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_processed, y_processed, test_size=test_size, random_state=42, stratify=y_processed
        )
        
        # Store test data in session state for visualization
        st.session_state.X_test = X_test
        st.session_state.y_test = y_test
        st.session_state.X_processed = X_processed
        
        st.session_state.results = {}
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Train each selected model
        for idx, model_name in enumerate(selected_models):
            status_text.text(f"Training {model_name}...")
            
            # Get hyperparameters
            if model_name == "Perceptron":
                hyperparams = perceptron_params
            elif model_name == "Multilayer Perceptron (Backprop)":
                hyperparams = mlp_params
            elif model_name == "Decision Tree":
                hyperparams = dt_params
            
            try:
                # Create and train model
                model = create_model(model_name, hyperparams)
                results = train_model(model, X_train, y_train, X_test, y_test, model_name)
                st.session_state.results[model_name] = results
                
                progress_bar.progress((idx + 1) / len(selected_models))
                status_text.text(f"✓ {model_name} trained successfully!")
                
            except Exception as e:
                st.error(f"Error training {model_name}: {e}")
                st.exception(e)
        
        progress_bar.empty()
        status_text.empty()
    
    # ========================================================================
    # MAIN AREA: Results Display
    # ========================================================================
    if len(st.session_state.results) > 0:
        st.divider()
        st.header("📊 Training Results")
        
        # Model Comparison Dashboard
        if len(st.session_state.results) > 1:
            st.subheader("📈 Model Comparison Dashboard")
            comparison_data = []
            for model_name, results in st.session_state.results.items():
                comparison_data.append({
                    'Model': model_name,
                    'Accuracy': f"{results['accuracy']:.4f}",
                    'Precision (Weighted)': f"{results['precision_weighted']:.4f}",
                    'Recall (Weighted)': f"{results['recall_weighted']:.4f}",
                    'F1 Score (Weighted)': f"{results['f1_weighted']:.4f}",
                    'Train Time (s)': f"{results['train_time']:.4f}"
                })
            
            comparison_df = pd.DataFrame(comparison_data)
            st.dataframe(comparison_df, use_container_width=True, hide_index=True)
            
            # Visual comparison
            fig, axes = plt.subplots(1, 2, figsize=(14, 5))
            
            # Metrics comparison
            models = comparison_df['Model'].values
            metrics = ['Accuracy', 'Precision (Weighted)', 'Recall (Weighted)', 'F1 Score (Weighted)']
            x = np.arange(len(models))
            width = 0.2
            
            for i, metric in enumerate(metrics):
                values = [float(comparison_df[metric].iloc[j]) for j in range(len(models))]
                axes[0].bar(x + i*width, values, width, label=metric)
            
            axes[0].set_xlabel('Model')
            axes[0].set_ylabel('Score')
            axes[0].set_title('Model Performance Comparison')
            axes[0].set_xticks(x + width * 1.5)
            axes[0].set_xticklabels(models, rotation=45, ha='right')
            axes[0].legend()
            axes[0].grid(True, alpha=0.3)
            
            # Training time comparison
            train_times = [float(comparison_df['Train Time (s)'].iloc[i]) for i in range(len(models))]
            axes[1].bar(models, train_times, color='skyblue')
            axes[1].set_xlabel('Model')
            axes[1].set_ylabel('Training Time (seconds)')
            axes[1].set_title('Training Time Comparison')
            axes[1].tick_params(axis='x', rotation=45)
            axes[1].grid(True, alpha=0.3)
            
            plt.tight_layout()
            st.pyplot(fig)
        
        # Individual Model Results
        for model_name, results in st.session_state.results.items():
            st.divider()
            st.subheader(f"🔍 Results for: {model_name}")
            
            # Special enhanced display for Perceptron
            if model_name == "Perceptron":
                # Create enhanced tabs for Perceptron
                tab1, tab2, tab3, tab4 = st.tabs([
                    "📈 Metrics",
                    "🟦 Confusion Matrix",
                    "📊 Charts & Insights",
                    "🔧 Model Details"
                ])
                
                with tab1:
                    st.write("### Key Performance Metrics")
                    
                    # Primary metrics in columns
                    col1, col2, col3, col4 = st.columns(4)
                    col1.metric("Accuracy", f"{results['accuracy']:.4f}")
                    col2.metric("Precision (Weighted)", f"{results['precision_weighted']:.4f}")
                    col3.metric("Recall (Weighted)", f"{results['recall_weighted']:.4f}")
                    col4.metric("F1 Score (Weighted)", f"{results['f1_weighted']:.4f}")
                    
                    st.divider()
                    
                    # Macro-averaged metrics
                    st.write("### Macro-Averaged Metrics")
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Precision (Macro)", f"{results['precision_macro']:.4f}")
                    col2.metric("Recall (Macro)", f"{results['recall_macro']:.4f}")
                    col3.metric("F1 Score (Macro)", f"{results['f1_macro']:.4f}")
                    
                    st.divider()
                    
                    # Full classification report
                    st.write("### Detailed Classification Report")
                    report_df = pd.DataFrame(results['classification_report']).transpose()
                    st.dataframe(report_df, use_container_width=True)
                    
                    st.write("### Classification Report (Text Format)")
                    y_test_viz = st.session_state.get('y_test', None)
                    if y_test_viz is not None:
                        st.code(classification_report(
                            y_test_viz, results['y_pred'], zero_division=0
                        ))
                
                with tab2:
                    st.write("### Confusion Matrix Analysis")
                    
                    # Standard and Normalized confusion matrices side by side
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write("**Standard Confusion Matrix**")
                        fig_cm = plot_confusion_matrix(
                            results['confusion_matrix'],
                            normalized=False,
                            class_names=st.session_state.class_names
                        )
                        st.pyplot(fig_cm)
                    
                    with col2:
                        st.write("**Normalized Confusion Matrix**")
                        fig_cm_norm = plot_confusion_matrix(
                            results['confusion_matrix_normalized'],
                            normalized=True,
                            class_names=st.session_state.class_names
                        )
                        st.pyplot(fig_cm_norm)
                    
                    st.divider()
                    
                    # Confusion Matrix Details Table
                    st.write("### Confusion Matrix Details")
                    st.write("Breakdown of each cell in the confusion matrix:")
                    cm_details = create_confusion_matrix_details(
                        results['confusion_matrix'],
                        class_names=st.session_state.class_names
                    )
                    st.dataframe(cm_details, use_container_width=True, hide_index=True)
                
                with tab3:
                    st.write("### Per-Class Performance Metrics")
                    
                    # Per-class precision
                    st.write("#### Per-Class Precision")
                    fig_precision = plot_per_class_metrics(
                        results['classification_report'],
                        metric_name='precision'
                    )
                    if fig_precision:
                        st.pyplot(fig_precision)
                    
                    st.divider()
                    
                    # Per-class recall
                    st.write("#### Per-Class Recall")
                    fig_recall = plot_per_class_metrics(
                        results['classification_report'],
                        metric_name='recall'
                    )
                    if fig_recall:
                        st.pyplot(fig_recall)
                    
                    st.divider()
                    
                    # Per-class F1 score
                    st.write("#### Per-Class F1 Score")
                    fig_f1 = plot_per_class_metrics(
                        results['classification_report'],
                        metric_name='f1-score'
                    )
                    if fig_f1:
                        st.pyplot(fig_f1)
                    
                    st.divider()
                    
                    # Feature scatter plot (optional)
                    st.write("#### Feature Scatter Plot")
                    st.write("Visualize how the Perceptron separates classes in feature space:")
                    
                    # Get numeric features for scatter plot
                    X_processed_viz = st.session_state.get('X_processed', None)
                    if X_processed_viz is not None:
                        numeric_features = X_processed_viz.select_dtypes(include=[np.number]).columns.tolist()
                    else:
                        numeric_features = []
                    
                    if len(numeric_features) >= 2:
                        col1, col2 = st.columns(2)
                        with col1:
                            feature1 = st.selectbox(
                                "Select First Feature",
                                numeric_features,
                                index=0,
                                key=f'perceptron_feature1_{model_name}'
                            )
                        with col2:
                            # Ensure feature2 is different from feature1
                            feature2_options = [f for f in numeric_features if f != feature1]
                            if len(feature2_options) > 0:
                                feature2 = st.selectbox(
                                    "Select Second Feature",
                                    feature2_options,
                                    index=0,
                                    key=f'perceptron_feature2_{model_name}'
                                )
                                
                                # Create scatter plot
                                X_test_viz = st.session_state.get('X_test', None)
                                y_test_viz = st.session_state.get('y_test', None)
                                
                                if X_test_viz is not None and y_test_viz is not None:
                                    fig_scatter = plot_feature_scatter(
                                        X_test_viz,
                                        y_test_viz,
                                        results['y_pred'],
                                        feature1,
                                        feature2,
                                        class_names=st.session_state.class_names
                                    )
                                    if fig_scatter:
                                        st.pyplot(fig_scatter)
                                else:
                                    st.warning("Test data not available for visualization")
                            else:
                                st.info("Not enough features available for scatter plot")
                    else:
                        st.info("Need at least 2 numeric features for scatter plot visualization")
                
                with tab4:
                    st.write("### Model Configuration")
                    
                    # Model parameters
                    st.write("#### Model Hyperparameters")
                    model_params = results['model'].get_params()
                    st.json(model_params)
                    
            st.divider()
                    
                    # Training information
                    st.write("#### Training Information")
                    
                    # Extract Perceptron-specific attributes
                    n_iter = getattr(results['model'], 'n_iter_', None)
                    converged = getattr(results['model'], 'converged_', None)
                    X_processed_viz = st.session_state.get('X_processed', None)
                    n_features = X_processed_viz.shape[1] if X_processed_viz is not None else 'N/A'
                    y_test_viz = st.session_state.get('y_test', None)
                    n_classes = len(np.unique(y_test_viz)) if y_test_viz is not None else len(np.unique(results['y_pred']))
                    
                    info_data = {
                        'Parameter': [
                            'Training Time (seconds)',
                            'Number of Iterations',
                            'Convergence Status',
                            'Input Feature Count',
                            'Number of Classes',
                            'Test Set Size'
                        ],
                        'Value': [
                            f"{results['train_time']:.4f}",
                            f"{n_iter[0] if n_iter is not None and len(n_iter) > 0 else 'N/A'}",
                            f"{'Yes' if converged is not None and converged else 'N/A'}",
                            f"{n_features}",
                            f"{n_classes}",
                            f"{len(results['y_pred'])}"
                        ]
                    }
                    
                    info_df = pd.DataFrame(info_data)
                    st.dataframe(info_df, use_container_width=True, hide_index=True)
                    
                    # Additional details
                    if n_iter is not None and len(n_iter) > 0:
                        st.info(f"⚠️ The Perceptron required {n_iter[0]} iterations to converge (or reached max_iter limit).")
                    if converged is not None:
                        if converged:
                            st.success("✅ The Perceptron converged successfully.")
                        else:
                            st.warning("⚠️ The Perceptron did not converge within the maximum number of iterations.")
            
            else:
                # Standard display for other models (MLP, Decision Tree)
                # Create tabs for different views
                tab1, tab2, tab3, tab4 = st.tabs([
                    "📈 Metrics",
                    "🟦 Confusion Matrices",
                    "📊 Curves",
                    "🌳 Model Details"
                ])

            with tab1:
                    # Key Metrics
                col1, col2, col3, col4 = st.columns(4)
                    col1.metric("Accuracy", f"{results['accuracy']:.4f}")
                    col2.metric("Precision (Weighted)", f"{results['precision_weighted']:.4f}")
                    col3.metric("Recall (Weighted)", f"{results['recall_weighted']:.4f}")
                    col4.metric("F1 Score (Weighted)", f"{results['f1_weighted']:.4f}")
                    
                    st.divider()
                    
                    # Detailed Metrics
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write("**Macro-Averaged Metrics:**")
                        st.metric("Precision (Macro)", f"{results['precision_macro']:.4f}")
                        st.metric("Recall (Macro)", f"{results['recall_macro']:.4f}")
                        st.metric("F1 Score (Macro)", f"{results['f1_macro']:.4f}")
                    
                    with col2:
                        st.write("**Training Information:**")
                        st.metric("Training Time", f"{results['train_time']:.4f} seconds")
                        st.metric("Test Set Size", f"{len(results['y_pred'])} samples")
                    
                    st.divider()
                    
                    # Classification Report
                    st.write("**Detailed Classification Report:**")
                    report_df = pd.DataFrame(results['classification_report']).transpose()
                    st.dataframe(report_df, use_container_width=True)
                    
                    # Text classification report
                    y_test_viz = st.session_state.get('y_test', None)
                    if y_test_viz is not None:
                        st.code(classification_report(
                            y_test_viz, results['y_pred'], zero_division=0
                        ))

            with tab2:
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write("**Confusion Matrix**")
                        fig_cm = plot_confusion_matrix(
                            results['confusion_matrix'],
                            normalized=False,
                            class_names=st.session_state.class_names
                        )
                        st.pyplot(fig_cm)
                    
                    with col2:
                        st.write("**Normalized Confusion Matrix**")
                        fig_cm_norm = plot_confusion_matrix(
                            results['confusion_matrix_normalized'],
                            normalized=True,
                            class_names=st.session_state.class_names
                        )
                        st.pyplot(fig_cm_norm)

            with tab3:
                    y_test_viz = st.session_state.get('y_test', None)
                    if y_test_viz is not None:
                        if results['roc_data'] is not None:
                            st.write("**ROC Curve**")
                            multi_class = len(np.unique(y_test_viz)) > 2
                            fig_roc = plot_roc_curve(results['roc_data'], multi_class=multi_class)
                            st.pyplot(fig_roc)
                        else:
                            st.info("ROC curve not available (model does not support probability predictions)")
                        
                        if results['pr_data'] is not None:
                            st.write("**Precision-Recall Curve**")
                            multi_class = len(np.unique(y_test_viz)) > 2
                            fig_pr = plot_pr_curve(results['pr_data'], multi_class=multi_class)
                            st.pyplot(fig_pr)
                        else:
                            st.info("Precision-Recall curve not available (model does not support probability predictions)")
                
                with tab4:
                    st.write("**Model Parameters:**")
                    st.json(results['model'].get_params())

                if model_name == "Decision Tree":
                        st.write("**Decision Tree Visualization:**")
                        X_processed_viz = st.session_state.get('X_processed', None)
                        if X_processed_viz is not None:
                    fig_tree, ax_tree = plt.subplots(figsize=(20, 10))
                            plot_tree(
                                results['model'],
                                filled=True,
                                feature_names=X_processed_viz.columns,
                                ax=ax_tree,
                                max_depth=5,
                                fontsize=8
                            )
                    st.pyplot(fig_tree)

                            # Tree statistics
                            st.write("**Tree Statistics:**")
                            n_nodes = results['model'].tree_.node_count
                            depth = results['model'].tree_.max_depth
                            n_leaves = results['model'].tree_.n_leaves
                            col1, col2, col3 = st.columns(3)
                            col1.metric("Number of Nodes", n_nodes)
                            col2.metric("Max Depth", depth)
                            col3.metric("Number of Leaves", n_leaves)


if __name__ == "__main__":
    main()
