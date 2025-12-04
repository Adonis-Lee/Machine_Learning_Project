CENG465 Machine Learning Toolkit
Introduction
This project presents a streamlined Machine Learning GUI Toolkit built using Streamlit. The toolkit provides a functional interface for training and evaluating classification models on CSV datasets. Designed for the CENG465 course, the application simplifies the machine learning pipeline, allowing users to upload data, configure basic preprocessing steps, and visualize model performance metrics without writing code.

The toolkit supports three fundamental classification algorithms:

Perceptron: A linear classifier using standard parameters.

Multilayer Perceptron (MLP): A feedforward neural network with a fixed architecture (100, 50 hidden layers).

Decision Tree: A tree-based classifier using Entropy as the splitting criterion.

Features
1. Dataset Handling
CSV Upload: Simple sidebar interface for uploading datasets.

Dataset Preview: Interactive view of the first 5 rows (head) of the uploaded dataframe.

Target Selection: Dropdown menu to dynamically select the target variable (class label) from the dataset columns.

2. Preprocessing Methods
The application automates several key preprocessing steps:

Label Encoding: Automatically detects if the target column is categorical (text-based) and converts it to numeric values.

Categorical Encoding (One-Hot): Users can opt-in to apply One-Hot Encoding. The system automatically detects categorical features (object types) and creates dummy variables, dropping the first category to prevent multicollinearity.

Normalization: Two normalization options are available via the sidebar:

None: Raw data usage.

StandardScaler: Z-score normalization (mean=0, std=1).

MinMaxScaler: Scaling features to a [0, 1] range.

3. Model Configurations
The toolkit includes three pre-configured classifiers:

Perceptron: Uses Scikit-learn's default implementation.

Multilayer Perceptron (Backpropagation):

Architecture: Two hidden layers (100 neurons, 50 neurons).

Iterations: Fixed at 1000 max iterations to ensure convergence.

Random State: Fixed (42) for reproducibility.

Decision Tree:

Criterion: Entropy (Information Gain).

Depth: Visualized up to depth 3 in results.

4. Train/Test Split
Adjustable Ratio: Users can set the test set proportion via a slider (ranging from 10% to 50%).

Reproducibility: A fixed random_state=42 is used during splitting to ensure consistent results across runs.

5. Metrics & Visualization
Upon training a model, the results are displayed in an organized tabbed interface:

Key Metrics:

Accuracy

Precision (Weighted)

Recall (Weighted)

F1 Score (Weighted)

Detailed Report: A full text-based classification report showing metrics per class.

Confusion Matrix: A heatmap visualization showing the count of true vs. predicted labels.

Model Details:

Displays the JSON parameters of the trained model.

Tree Visualization: Specifically for the Decision Tree model, the application renders the decision tree structure (max depth 3) to aid in interpretability.

Installation & Usage
Prerequisites
Python 3.7+

Libraries: streamlit, pandas, numpy, matplotlib, seaborn, scikit-learn

Steps
Run the application:

Bash

streamlit run app.py
Upload Data: Use the sidebar to upload a CSV file.

Configure:

Select the target column.

Choose a normalization method.

Adjust the Test Set ratio.

Select Model: Choose between Perceptron, MLP, or Decision Tree.

Train: Click the "🚀 Train Model" button.

Code Architecture
The application is built in a single script (app.py) following a linear execution flow:

Layout: Uses st.sidebar for inputs and the main area for outputs.

Data Loading: Reads CSV via Pandas.

Preprocessing Logic:

Separates X (features) and y (target).

Applies LabelEncoder to y if necessary.

Applies pd.get_dummies for categorical X features.

Applies Scaling (Standard or MinMax).

Model Training: Instantiates the selected class from sklearn and calls .fit().

Evaluation: Generates predictions using .predict() and calculates metrics.

Visualization: Uses seaborn for heatmaps and sklearn.tree.plot_tree for tree rendering.

Limitations
Missing Values: The current version presumes the dataset is clean. Users should handle missing values (NaNs) in the CSV before uploading.

Hyperparameter Tuning: Hyperparameters (like learning rate, hidden layers, alpha) are hardcoded for simplicity and ease of use.

Single Model Execution: The tool trains one model at a time rather than comparing multiple models simultaneously.

Authors
Developed as part of CENG465 Machine Learning course project.
