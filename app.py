import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.linear_model import Perceptron
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)

# Sayfa Ayarları
st.set_page_config(page_title="CENG465 ML Toolkit", layout="wide", page_icon="🤖")

st.title("🎓 CENG465 - Group ML Toolkit")
st.markdown("This tool allows you to Train, Test and Evaluate ML models on any CSV dataset.")


# ============================================================================
# HELPER FUNCTIONS FOR PERCEPTRON ENHANCEMENTS
# ============================================================================

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

# --- SOL PANEL (Ayarlar) ---
st.sidebar.header("1. Upload & Settings")
uploaded = st.sidebar.file_uploader("Upload your Dataset (CSV)", type=["csv"])

if uploaded:
    df = pd.read_csv(uploaded)

    # --- ORTA ALAN (Veri Önizleme) ---
    st.subheader("📊 Dataset Preview")
    st.dataframe(df.head())

    # --- AYARLAR ---
    st.sidebar.subheader("2. Preprocessing")

    # Hedef Sütun Seçimi
    cols = df.columns.tolist()
    target = st.sidebar.selectbox("Select Target Column (Class Label)", cols)

    # Preprocessing Seçenekleri
    scaler_choice = st.sidebar.selectbox("Normalization Method", ["None", "StandardScaler", "MinMaxScaler"])
    encoding_choice = st.sidebar.checkbox("Apply One-Hot Encoding (Auto-detect categorical)", value=True)

    # Train/Test Split
    st.sidebar.subheader("3. Split & Model")
    test_size = st.sidebar.slider("Test Set Ratio", 0.1, 0.5, 0.3)

    # Model Seçimi
    model_name = st.sidebar.selectbox("Select Classifier", 
                                      ["Perceptron", 
                                       "Multilayer Perceptron (Backprop)", 
                                       "Decision Tree"])

    # --- ÇALIŞTIR BUTONU ---
    if st.sidebar.button("🚀 Train Model"):

        # 1. Veri Hazırlığı
        X = df.drop(columns=[target])
        y = df[target]

        # Store label encoder for class names
        label_encoder = None
        class_names = None

        # Target (y) eğer yazı ise (Örn: Pass/Fail), LabelEncoder ile sayıya (0/1) çeviriyoruz
        if y.dtype == 'object':
            label_encoder = LabelEncoder()
            y = label_encoder.fit_transform(y)
            class_names = label_encoder.classes_
            st.info(f"Target column encoded: {class_names}")

        # One-Hot Encoding (Otomatik - PDF Şartı)
        if encoding_choice:
            # Sadece kategorik (object) sütunları bul ve encode et
            X = pd.get_dummies(X, drop_first=True)
            st.write(f"dataset shape after encoding: {X.shape}")

        # Normalization (PDF Şartı)
        if scaler_choice == "StandardScaler":
            scaler = StandardScaler()
            X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
        elif scaler_choice == "MinMaxScaler":
            scaler = MinMaxScaler()
            X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

        # Split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

        # 2. Model Eğitimi
        if model_name == "Perceptron":
            model = Perceptron()
        elif model_name == "Multilayer Perceptron (Backprop)":
            model = MLPClassifier(hidden_layer_sizes=(100,50), max_iter=1000, random_state=42)
        else:
            model = DecisionTreeClassifier(criterion='entropy', random_state=42)

        try:
            # Train model and measure time
            start_time = time.time()
            model.fit(X_train, y_train)
            train_time = time.time() - start_time
            
            y_pred = model.predict(X_test)

            # --- SONUÇ EKRANI (Tablar ile Düzenli Görünüm) ---
            st.divider()
            st.header(f"Results for: {model_name}")

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
                    col1.metric("Accuracy", f"{accuracy_score(y_test, y_pred):.4f}")
                    col2.metric("Precision (Weighted)", f"{precision_score(y_test, y_pred, average='weighted', zero_division=0):.4f}")
                    col3.metric("Recall (Weighted)", f"{recall_score(y_test, y_pred, average='weighted', zero_division=0):.4f}")
                    col4.metric("F1 Score (Weighted)", f"{f1_score(y_test, y_pred, average='weighted', zero_division=0):.4f}")
                    
                    st.divider()
                    
                    # Macro-averaged metrics
                    st.write("### Macro-Averaged Metrics")
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Precision (Macro)", f"{precision_score(y_test, y_pred, average='macro', zero_division=0):.4f}")
                    col2.metric("Recall (Macro)", f"{recall_score(y_test, y_pred, average='macro', zero_division=0):.4f}")
                    col3.metric("F1 Score (Macro)", f"{f1_score(y_test, y_pred, average='macro', zero_division=0):.4f}")
                    
                    st.divider()
                    
                    # Full classification report
                    st.write("### Detailed Classification Report")
                    report_dict = classification_report(y_test, y_pred, zero_division=0, output_dict=True)
                    report_df = pd.DataFrame(report_dict).transpose()
                    st.dataframe(report_df, use_container_width=True)
                    
                    st.write("### Classification Report (Text Format)")
                    st.code(classification_report(y_test, y_pred, zero_division=0))
                
                with tab2:
                    st.write("### Confusion Matrix Analysis")
                    
                    # Compute confusion matrices
                    cm = confusion_matrix(y_test, y_pred)
                    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
                    
                    # Standard and Normalized confusion matrices side by side
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write("**Standard Confusion Matrix**")
                        fig_cm = plt.figure(figsize=(8, 6))
                        sns.heatmap(
                            cm,
                            annot=True,
                            fmt='d',
                            cmap="Blues",
                            xticklabels=class_names if class_names is not None else 'auto',
                            yticklabels=class_names if class_names is not None else 'auto'
                        )
                        plt.xlabel('Predicted Label')
                        plt.ylabel('True Label')
                        plt.title('Confusion Matrix')
                        plt.tight_layout()
                        st.pyplot(fig_cm)
                    
                    with col2:
                        st.write("**Normalized Confusion Matrix**")
                        fig_cm_norm = plt.figure(figsize=(8, 6))
                        sns.heatmap(
                            cm_normalized,
                            annot=True,
                            fmt='.2f',
                            cmap="Blues",
                            xticklabels=class_names if class_names is not None else 'auto',
                            yticklabels=class_names if class_names is not None else 'auto'
                        )
                        plt.xlabel('Predicted Label')
                        plt.ylabel('True Label')
                        plt.title('Normalized Confusion Matrix')
                        plt.tight_layout()
                        st.pyplot(fig_cm_norm)
                    
                    st.divider()
                    
                    # Confusion Matrix Details Table
                    st.write("### Confusion Matrix Details")
                    st.write("Breakdown of each cell in the confusion matrix:")
                    cm_details = create_confusion_matrix_details(cm, class_names=class_names)
                    st.dataframe(cm_details, use_container_width=True, hide_index=True)
                
                with tab3:
                    st.write("### Per-Class Performance Metrics")
                    
                    # Get classification report as dictionary
                    report_dict = classification_report(y_test, y_pred, zero_division=0, output_dict=True)
                    
                    # Per-class precision
                    st.write("#### Per-Class Precision")
                    fig_precision = plot_per_class_metrics(report_dict, metric_name='precision')
                    if fig_precision:
                        st.pyplot(fig_precision)
                    
                    st.divider()
                    
                    # Per-class recall
                    st.write("#### Per-Class Recall")
                    fig_recall = plot_per_class_metrics(report_dict, metric_name='recall')
                    if fig_recall:
                        st.pyplot(fig_recall)
                    
                    st.divider()
                    
                    # Per-class F1 score
                    st.write("#### Per-Class F1 Score")
                    fig_f1 = plot_per_class_metrics(report_dict, metric_name='f1-score')
                    if fig_f1:
                        st.pyplot(fig_f1)
                    
                    st.divider()
                    
                    # Feature scatter plot (optional)
                    st.write("#### Feature Scatter Plot")
                    st.write("Visualize how the Perceptron separates classes in feature space:")
                    
                    # Get numeric features for scatter plot
                    numeric_features = X_test.select_dtypes(include=[np.number]).columns.tolist()
                    
                    if len(numeric_features) >= 2:
                        col1, col2 = st.columns(2)
                        with col1:
                            feature1 = st.selectbox(
                                "Select First Feature",
                                numeric_features,
                                index=0,
                                key='perceptron_feature1'
                            )
                        with col2:
                            # Ensure feature2 is different from feature1
                            feature2_options = [f for f in numeric_features if f != feature1]
                            if len(feature2_options) > 0:
                                feature2 = st.selectbox(
                                    "Select Second Feature",
                                    feature2_options,
                                    index=0,
                                    key='perceptron_feature2'
                                )
                                
                                # Create scatter plot
                                fig_scatter = plot_feature_scatter(
                                    X_test,
                                    y_test,
                                    y_pred,
                                    feature1,
                                    feature2,
                                    class_names=class_names
                                )
                                if fig_scatter:
                                    st.pyplot(fig_scatter)
                            else:
                                st.info("Not enough features available for scatter plot")
                    else:
                        st.info("Need at least 2 numeric features for scatter plot visualization")
                
                with tab4:
                    st.write("### Model Configuration")
                    
                    # Model parameters
                    st.write("#### Model Hyperparameters")
                    model_params = model.get_params()
                    st.json(model_params)
                    
                    st.divider()
                    
                    # Training information
                    st.write("#### Training Information")
                    
                    # Extract Perceptron-specific attributes
                    n_iter = getattr(model, 'n_iter_', None)
                    converged = getattr(model, 'converged_', None)
                    n_features = X.shape[1]
                    n_classes = len(np.unique(y_test))
                    
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
                            f"{train_time:.4f}",
                            f"{n_iter[0] if n_iter is not None and len(n_iter) > 0 else 'N/A'}",
                            f"{'Yes' if converged is not None and converged else 'N/A'}",
                            f"{n_features}",
                            f"{n_classes}",
                            f"{len(y_pred)}"
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
                tab1, tab2, tab3 = st.tabs(["📈 Metrics", "🟦 Confusion Matrix", "🌳 Model Details"])

                with tab1:
                    col1, col2, col3, col4 = st.columns(4)
                    col1.metric("Accuracy", f"{accuracy_score(y_test, y_pred):.4f}")
                    col2.metric("Precision", f"{precision_score(y_test, y_pred, average='weighted', zero_division=0):.4f}")
                    col3.metric("Recall", f"{recall_score(y_test, y_pred, average='weighted', zero_division=0):.4f}")
                    col4.metric("F1 Score", f"{f1_score(y_test, y_pred, average='weighted', zero_division=0):.4f}")

                    st.text("Detailed Classification Report:")
                    st.code(classification_report(y_test, y_pred, zero_division=0))

                with tab2:
                    st.write("Confusion Matrix Heatmap")
                    cm = confusion_matrix(y_test, y_pred)
                    fig, ax = plt.subplots(figsize=(6, 4))
                    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
                    st.pyplot(fig)

                with tab3:
                    st.write("Model Parameters:")
                    st.json(model.get_params())

                    if model_name == "Decision Tree":
                        st.write("Decision Tree Visualization:")
                        fig_tree, ax_tree = plt.subplots(figsize=(20, 10))
                        plot_tree(model, filled=True, feature_names=X.columns, ax=ax_tree, max_depth=3)
                        st.pyplot(fig_tree)

        except Exception as e:
            st.error(f"An error occurred during training: {e}")
            st.warning("Hint: Check if your dataset contains unprocessable text data.")

else:
    st.info("Please upload a CSV file from the sidebar to begin.")
