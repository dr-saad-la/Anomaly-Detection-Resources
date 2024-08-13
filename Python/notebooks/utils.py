"""
    Utility functions that will be used in the series of notebooks dedicated to outlier detection algorithms.
"""

import numpy as np
import pandas as pd

from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder
from pyod.models.abod import ABOD
# from pyod.utils import precision_n_scores

from sklearn.metrics import classification_report, roc_auc_score, precision_recall_curve, auc, precision_score
from sklearn.utils.validation import column_or_1d
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

import matplotlib.pyplot as plt
import seaborn as sns

from hyperopt import fmin, tpe, hp, Trials, STATUS_OK
from hyperopt.pyll.base import scope


def generate_data(n_samples=1000, n_features=2, n_inliers=900, n_outliers=100, random_state=42):
    """
    Generate synthetic data for outlier detection algorithms.

    This function creates a dataset with a specified number of inliers and outliers. Inliers 
    are generated using a cluster-based approach, while outliers are uniformly distributed 
    over a larger range.

    Parameters
    ----------
    n_samples : int, optional
        Total number of samples (inliers + outliers). Default is 1000.

    n_features : int, optional
        Number of features for each sample. Default is 2.

    n_inliers : int, optional
        Number of inlier samples. Default is 900.

    n_outliers : int, optional
        Number of outlier samples. Default is 100.

    random_state : int, optional
        Random seed for reproducibility. Default is 42.

    Returns
    -------
    X : ndarray of shape (n_samples, n_features)
        The generated samples, where the first `n_inliers` rows are inliers and the 
        remaining `n_outliers` rows are outliers.

    y : ndarray of shape (n_samples,)
        The labels for the generated samples, where 0 represents inliers and 1 represents 
        outliers.

    Examples
    --------
    >>> X, y = generate_data(n_samples=1000, n_features=2, n_inliers=900, n_outliers=100)
    >>> print(X.shape)
    (1000, 2)
    >>> print(np.unique(y, return_counts=True))
    (array([0., 1.]), array([900, 100]))

    Notes
    -----
    - The inliers are generated using the `make_blobs` function from Scikit-learn, which 
      creates isotropic Gaussian blobs for clustering.
    - Outliers are generated using a uniform distribution over a specified range.
    """
    
    # Generate inliers using make_blobs (clusters)
    X_inliers, _ = make_blobs(n_samples=n_inliers, 
                              n_features=n_features, 
                              centers=3, 
                              cluster_std=1.0, 
                              random_state=random_state)
    
    # Generate outliers randomly distributed over a larger range
    X_outliers = np.random.uniform(low=-10, high=10, size=(n_outliers, n_features))
    
    # Combine inliers and outliers
    X = np.vstack((X_inliers, X_outliers))
    
    # Generate labels (0 for inliers, 1 for outliers)
    y = np.hstack((np.zeros(n_inliers), np.ones(n_outliers)))
    
    return X, y


def visualize_data(X, y, title='Synthetic Data for Outlier Detection'):
    """
    Plot the generated synthetic data to visualize inliers and outliers.

    This function creates a scatter plot of the synthetic data, differentiating 
    between inliers and outliers using distinct colors and markers.

    Parameters
    ----------
    X : ndarray of shape (n_samples, n_features)
        The dataset containing both inliers and outliers. Each row represents a sample, 
        and each column represents a feature.

    y : ndarray of shape (n_samples,)
        The labels for the dataset, where 0 indicates inliers and 1 indicates outliers.

    title : str, optional
        The title of the plot. Default is 'Synthetic Data for Outlier Detection'.

    Returns
    -------
    None

    Examples
    --------
    >>> X, y = generate_data(n_samples=1000, n_features=2, n_inliers=900, n_outliers=100)
    >>> visualize_data(X, y, title='Outlier Detection Visualization')

    Notes
    -----
    - The function assumes that the input data `X` has two features for visualization.
    - Adjust the plot settings if working with higher-dimensional data or different visualization needs.
    """
    plt.figure(figsize=(10, 6))

    plt.scatter(X[y == 0, 0], X[y == 0, 1], c='b', marker='o', label='Inliers', alpha=0.7)
    plt.scatter(X[y == 1, 0], X[y == 1, 1], c='r', marker='x', label='Outliers', alpha=0.7)
    
    plt.title(title)
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    
    plt.legend()
    plt.grid(True)
    plt.show()


def apply_abod(X, y, contamination=0.1):
    """
    Apply the Angle-based Outlier Detector (ABOD) algorithm to the dataset.

    This function applies the ABOD algorithm to identify outliers in the dataset, 
    evaluates its performance using various metrics, and visualizes the results.

    Parameters
    ----------
    X : ndarray of shape (n_samples, n_features)
        The dataset containing both inliers and outliers.

    y : ndarray of shape (n_samples,)
        The labels for the dataset, where 0 indicates inliers and 1 indicates outliers.

    contamination : float, optional
        The proportion of outliers in the dataset. Default is 0.1 (10%).

    Returns
    -------
    None

    Examples
    --------
    >>> X, y = generate_data(n_samples=1000, n_features=2, n_inliers=900, n_outliers=100)
    >>> apply_abod(X, y, contamination=0.1)

    Notes
    -----
    - The function assumes that the input data `X` has been preprocessed and scaled if necessary.
    - The ABOD algorithm is particularly effective in high-dimensional datasets.
    """

    # Scale the features using StandardScaler
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Initialize the ABOD model with specified contamination
    abod_model = ABOD(contamination=contamination)

    # Fit the model to the scaled data
    abod_model.fit(X_scaled)

    # Predict outliers
    y_pred = abod_model.labels_                     # 0 for inliers, 1 for outliers
    outlier_scores = abod_model.decision_scores_

    # Evaluate the model performance if true labels are available
    print("\nClassification Report:")
    print(classification_report(y, y_pred, target_names=['Inlier', 'Outlier']))

    # Calculate ROC AUC and Precision-Recall AUC
    roc_auc = roc_auc_score(y, outlier_scores)
    precision, recall, _ = precision_recall_curve(y, outlier_scores)
    pr_auc = auc(recall, precision)
    print(f"ROC AUC: {roc_auc:.4f}, Precision-Recall AUC: {pr_auc:.4f}")

    # Visualize the outlier scores
    plt.figure(figsize=(10, 6))
    sns.histplot(outlier_scores, bins=50, kde=True)
    plt.title("Distribution of Outlier Scores")
    plt.xlabel("Outlier Score")
    plt.ylabel("Frequency")
    plt.grid(True)
    plt.show()

    # Visualize detected outliers
    plt.figure(figsize=(10, 6))
    plt.scatter(X_scaled[y == 0, 0], X_scaled[y == 0, 1], c='b', marker='o', label='True Inliers', alpha=0.7)
    plt.scatter(X_scaled[y == 1, 0], X_scaled[y == 1, 1], c='r', marker='x', label='True Outliers', alpha=0.7)
    plt.scatter(X_scaled[y_pred == 1, 0], X_scaled[y_pred == 1, 1], facecolors='none', edgecolors='g', label='Detected Outliers', alpha=0.5)
    plt.title("ABOD Outlier Detection Results")
    plt.xlabel("Feature 1 (scaled)")
    plt.ylabel("Feature 2 (scaled)")
    plt.legend()
    plt.grid(True)
    plt.show()


def apply_abod_advanced(X, y, contamination=0.1, scale_data=True, scaler_type='standard', plot_scores=True, 
                        plot_outliers=True, n=None, abod_params=None, plot_params=None, plot_features=None, n_cols=None):
    """
    Apply the Angle-based Outlier Detector (ABOD) algorithm to the dataset.

    This function applies the ABOD algorithm to identify outliers in the dataset,
    evaluates its performance using various metrics, and optionally visualizes the results.

    Parameters
    ----------
    X : ndarray or DataFrame of shape (n_samples, n_features)
        The dataset containing both inliers and outliers.

    y : ndarray or Series of shape (n_samples,)
        The labels for the dataset, where 0 indicates inliers and 1 indicates outliers.

    contamination : float, optional
        The proportion of outliers in the dataset. Default is 0.1 (10%).

    scale_data : bool, optional
        Whether to scale the data before applying the ABOD algorithm. Default is True.

    scaler_type : str, optional
        The type of scaler to use: 'standard' for StandardScaler or 'minmax' for MinMaxScaler. Default is 'standard'.

    plot_scores : bool, optional
        Whether to plot the distribution of outlier scores. Default is True.

    plot_outliers : bool, optional
        Whether to plot the detected outliers. Default is True.

    n : int, optional
        Number of outliers for precision calculation. If None, uses the actual number of outliers.

    abod_params : dict, optional
        Additional parameters to pass to the ABOD model.

    plot_params : dict, optional
        Additional parameters to pass to the plotting functions.

    plot_features : list of int, optional
        Features to plot (main feature + other features). If None, no feature plot will be created.

    n_cols : int, optional
        Number of columns for subplot grid. If None, a reasonable number will be guessed.

    Returns
    -------
    results : dict
        A dictionary containing the following keys:
        - 'y_pred': The predicted labels (0 for inliers, 1 for outliers).
        - 'outlier_scores': The outlier scores for each sample.
        - 'roc_auc': The ROC AUC score for the model.
        - 'pr_auc': The Precision-Recall AUC score for the model.
        - 'classification_report': The classification report as a string.
        - 'precision_at_rank_n': Precision at rank n score.

    Examples
    --------
    >>> X, y = generate_data(n_samples=1000, n_features=2, n_inliers=900, n_outliers=100)
    >>> results = apply_abod_advanced(X, y, contamination=0.1, scale_data=True, scaler_type='standard', 
                                      plot_scores=True, plot_outliers=True, plot_features=[0, 1, 2, 3], n_cols=2)
    """
    if abod_params is None:
        abod_params = {}
    if plot_params is None:
        plot_params = {}

    # Convert X to a NumPy array if it is a DataFrame
    if isinstance(X, pd.DataFrame):
        X = X.values

    # Scale the features if required
    if scale_data:
        if scaler_type == 'standard':
            scaler = StandardScaler()
        elif scaler_type == 'minmax':
            scaler = MinMaxScaler()
        else:
            raise ValueError("Invalid scaler_type. Use 'standard' or 'minmax'.")
        X_scaled = scaler.fit_transform(X)
    else:
        X_scaled = X

    # Initialize the ABOD model with specified contamination and other parameters
    abod_model = ABOD(contamination=contamination, **abod_params)

    # Fit the model to the scaled data
    abod_model.fit(X_scaled)

    # Predict outliers
    y_pred = abod_model.labels_  # 0 for inliers, 1 for outliers
    outlier_scores = abod_model.decision_scores_

    # Ensure y and y_pred are NumPy arrays
    if isinstance(y, pd.Series):
        y = y.values
    if isinstance(y_pred, pd.Series):
        y_pred = y_pred.values

    # Evaluate the model performance if true labels are available
    classification_report_str = classification_report(y, y_pred, target_names=['Inlier', 'Outlier'])
    roc_auc = roc_auc_score(y, outlier_scores)
    precision, recall, _ = precision_recall_curve(y, outlier_scores)
    pr_auc = auc(recall, precision)

    # Calculate precision at rank n
    if n is None:
        n = int(np.sum(y))  # Use the number of true outliers if n is not specified
    precision_at_rank_n = precision_n_scores(y, outlier_scores, n=n)

    # Plot the outlier scores if requested
    if plot_scores:
        plt.figure(figsize=(10, 6))
        sns.histplot(outlier_scores, bins=50, kde=True, **plot_params)
        plt.title("Distribution of Outlier Scores")
        plt.xlabel("Outlier Score")
        plt.ylabel("Frequency")
        plt.grid(True)
        plt.show()

    # Plot the detected outliers if requested
    if plot_outliers:
        if X.shape[1] == 2:
            plt.figure(figsize=(10, 6))
            plt.scatter(X_scaled[y == 0][:, 0], X_scaled[y == 0][:, 1], c='b', marker='o', label='True Inliers', alpha=0.7)
            plt.scatter(X_scaled[y == 1][:, 0], X_scaled[y == 1][:, 1], c='r', marker='x', label='True Outliers', alpha=0.7)
            plt.scatter(X_scaled[y_pred == 1][:, 0], X_scaled[y_pred == 1][:, 1], facecolors='none', edgecolors='g', label='Detected Outliers', alpha=0.5)
            plt.title("ABOD Outlier Detection Results")
            plt.xlabel("Feature 1 (scaled)")
            plt.ylabel("Feature 2 (scaled)")
            plt.legend()
            plt.grid(True)
            plt.show()
        elif X.shape[1] > 2 and plot_features:
            main_feature = plot_features[0]
            other_features = plot_features[1:]

            if n_cols is None:
                n_cols = min(len(other_features), 3)  # Guess a reasonable number of columns

            n_rows = (len(other_features) + n_cols - 1) // n_cols  # Calculate the number of rows needed
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 5 * n_rows))

            if n_rows == 1:
                axes = np.expand_dims(axes, 0)  # Ensure axes is always 2D for consistency

            for idx, (i, feature) in enumerate(zip(range(len(other_features)), other_features)):
                ax = axes[idx // n_cols, idx % n_cols]
                ax.scatter(X_scaled[y == 0, main_feature], X_scaled[y == 0, feature], c='b', marker='o', label='True Inliers', alpha=0.7)
                ax.scatter(X_scaled[y == 1, main_feature], X_scaled[y == 1, feature], c='r', marker='x', label='True Outliers', alpha=0.7)
                ax.scatter(X_scaled[y_pred == 1, main_feature], X_scaled[y_pred == 1, feature], facecolors='none', edgecolors='g', label='Detected Outliers', alpha=0.5)
                ax.set_xlabel(f'Feature {main_feature + 1}')
                ax.set_ylabel(f'Feature {feature + 1}')
                ax.legend()
                ax.grid(True)

            # Remove empty subplots
            for idx in range(len(other_features), n_rows * n_cols):
                fig.delaxes(axes.flatten()[idx])

            plt.suptitle("ABOD Outlier Detection Results", y=1.02)
            plt.tight_layout()
            plt.show()

    # Return results as a dictionary
    results = {
        'y_pred': y_pred,
        'outlier_scores': outlier_scores,
        'roc_auc': roc_auc,
        'pr_auc': pr_auc,
        'classification_report': classification_report_str,
        'precision_at_rank_n': precision_at_rank_n
    }

    return results


def precision_n_scores(y, y_pred, n=None):
    """
    Utility function to calculate precision @ rank n.

    Parameters
    ----------
    y : list or numpy array of shape (n_samples,)
        The ground truth. Binary (0: inliers, 1: outliers).

    y_pred : list or numpy array of shape (n_samples,)
        The raw outlier scores as returned by a fitted model.

    n : int, optional (default=None)
        The number of outliers. If not defined, infer using ground truth.

    Returns
    -------
    precision_at_rank_n : float
        Precision at rank n score.
    """

    # Infer n if not provided and ensure it's an integer
    if n is None:
        n = int(np.sum(y))  # Cast sum to integer

    # Get indices of the top n scores
    top_n_indices = np.argsort(y_pred)[-n:]  # Ensure n is used as an integer

    # Create binary predictions based on top n scores
    y_pred_binary = np.zeros_like(y_pred, dtype=int)
    y_pred_binary[top_n_indices] = 1

    # Enforce formats of y and y_pred_binary
    y = column_or_1d(y)
    y_pred_binary = column_or_1d(y_pred_binary)

    return precision_score(y, y_pred_binary)


def precision_at_rank_n(y_true, y_scores, n=None):
    """
    Calculate precision at rank n for anomaly detection.

    Parameters
    ----------
    y_true : ndarray
        True binary labels for the dataset, where 1 indicates outliers and 0 indicates inliers.
    
    y_scores : ndarray
        Outlier scores for each sample, higher scores indicate higher likelihood of being an outlier.
    
    n : int, optional
        The number of top-ranked samples to consider for calculating precision. If None, it defaults to the number of actual outliers.
    
    Returns
    -------
    precision : float
        Precision at rank n.
    """
    if n is None:
        n = int(np.sum(y_true))  # Default to the number of actual outliers

    # Get indices of the top n scores
    top_n_indices = np.argsort(y_scores)[-n:]

    # Get the labels for the top n scores
    top_n_labels = y_true[top_n_indices]

    # Calculate precision at rank n
    precision = np.sum(top_n_labels) / n
    return precision


# Hyperparameter tuning utility functions
def grid_search_abod(X, y, n_neighbors_options, contamination, n_outliers=None):
    """
    Perform grid search to find the best n_neighbors parameter for ABOD based on precision at rank n.

    Parameters
    ----------
    X : ndarray of shape (n_samples, n_features)
        The dataset containing both inliers and outliers.

    y : ndarray of shape (n_samples,)
        The labels for the dataset, where 0 indicates inliers and 1 indicates outliers.

    n_neighbors_options : list
        A list of n_neighbors values to search over.

    contamination : float
        The proportion of outliers in the dataset.

    n_outliers : int, optional
        The number of top-ranked samples to consider for calculating precision at rank n.

    Returns
    -------
    best_n_neighbors : int
        The best n_neighbors parameter based on precision at rank n.
    
    best_precision_n : float
        The best precision at rank n achieved.
    """
    best_n_neighbors = None
    best_precision_n = 0

    for n_neighbors in n_neighbors_options:
        # Initialize ABOD model with current n_neighbors
        abod = ABOD(n_neighbors=n_neighbors, contamination=contamination)

        # Fit the model to the data
        abod.fit(X)

        # Predict the outlier scores
        outlier_scores = abod.decision_scores_

        # Calculate precision at rank n
        precision_n = precision_at_rank_n(y, outlier_scores, n=n_outliers)

        # Update best parameter if current precision is better
        if precision_n > best_precision_n:
            best_precision_n = precision_n
            best_n_neighbors = n_neighbors

    return best_n_neighbors, best_precision_n


def hyperopt_objective(params, X_scaled, y, n_outliers, contamination):
    """
    Objective function for Hyperopt to optimize the n_neighbors parameter of ABOD.

    Parameters
    ----------
    params : dict
        Dictionary containing the hyperparameter 'n_neighbors'.

    X_scaled : ndarray
        The scaled feature matrix.

    y : ndarray
        The true labels.

    n_outliers : int
        The number of true outliers.

    contamination : float
        The contamination rate (proportion of outliers).

    Returns
    -------
    dict
        Dictionary containing the loss (negative precision at rank n) and the status.
    """
    n_neighbors = params['n_neighbors']

    # Initialize ABOD model with current n_neighbors and contamination rate
    abod = ABOD(n_neighbors=n_neighbors, contamination=contamination)

    # Fit the model to the data
    abod.fit(X_scaled)

    # Predict the outlier scores
    outlier_scores = abod.decision_scores_

    # Calculate precision at rank n
    precision_n = precision_n_scores(y, outlier_scores, n=n_outliers)

    return {'loss': -precision_n, 'status': STATUS_OK}

def plot_outliers_vs_inliers(X, y_pred, main_feature=0, plot_features=None, n_cols=None):
    """
    Plot outliers vs inliers, with one main feature plotted against the provided features.

    Parameters
    ----------
    X : DataFrame or ndarray
        Feature matrix.
    y_pred : ndarray
        Predicted labels (0 for inliers, 1 for outliers).
    main_feature : int, optional
        Index of the main feature to plot against (default is 0).
    plot_features : list of int, optional
        List of feature indices to plot against the main feature. If None, defaults to [1].
    n_cols : int, optional
        Number of columns for the subplot grid. If None, it will be guessed.

    """
    if plot_features is None:
        plot_features = [1]  # Default to plotting against the second feature if none provided

    if len(plot_features) == 1:
        # If only one feature is provided, plot a single scatter plot
        plt.figure(figsize=(10, 6))
        plt.scatter(X.values[y_pred == 0][:, main_feature], X.values[y_pred == 0][:, plot_features[0]], color='blue', label='Inliers', alpha=0.6)
        plt.scatter(X.values[y_pred == 1][:, main_feature], X.values[y_pred == 1][:, plot_features[0]], color='red', label='Outliers', alpha=0.6)
        plt.xlabel(f'Feature {main_feature + 1}')
        plt.ylabel(f'Feature {plot_features[0] + 1}')
        plt.title('Outliers vs Inliers')
        plt.legend()
        plt.grid(True)
        plt.show()
    else:
        # If more than one feature is provided, plot a grid of scatter plots
        if n_cols is None:
            n_cols = min(len(plot_features), 3)  # Guess a reasonable number of columns

        n_rows = (len(plot_features) + n_cols - 1) // n_cols  # Calculate the number of rows needed
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 5 * n_rows))

        if n_rows == 1:
            axes = np.expand_dims(axes, 0)  # Ensure axes is always 2D for consistency

        for idx, feature in enumerate(plot_features):
            ax = axes[idx // n_cols, idx % n_cols]
            ax.scatter(X.values[y_pred == 0][:, main_feature], X.values[y_pred == 0][:, feature], color='blue', label='Inliers', alpha=0.6)
            ax.scatter(X.values[y_pred == 1][:, main_feature], X.values[y_pred == 1][:, feature], color='red', label='Outliers', alpha=0.6)
            ax.set_xlabel(f'Feature {main_feature + 1}')
            ax.set_ylabel(f'Feature {feature + 1}')
            ax.legend()
            ax.grid(True)

        # Remove empty subplots
        for idx in range(len(plot_features), n_rows * n_cols):
            fig.delaxes(axes.flatten()[idx])

        plt.suptitle("Outliers vs Inliers", y=1.02)
        plt.tight_layout()
        plt.show()

def plot_outliers_pairplot(X, y, y_pred, plot_params=None):
    """
    Create a matrix plot for outlier detection results.

    Parameters
    ----------
    X : ndarray or DataFrame of shape (n_samples, n_features)
        The input data.

    y : ndarray or Series of shape (n_samples,)
        The true labels for the data, where 0 indicates inliers and 1 indicates outliers.

    y_pred : ndarray of shape (n_samples,)
        The predicted labels from the outlier detection algorithm.

    plot_params : dict, optional
        Additional keyword arguments for plotting.
    """
    if plot_params is None:
        plot_params = {}

    # Convert to DataFrame for seaborn pairplot
    if isinstance(X, np.ndarray):
        X = pd.DataFrame(X)
    if isinstance(y, np.ndarray):
        y = pd.Series(y)
    if isinstance(y_pred, np.ndarray):
        y_pred = pd.Series(y_pred)

     # Combine X, y, and y_pred into a single DataFrame
    df = X.copy()
    df['True Outlier'] = y
    df['Detected Outlier'] = y_pred

    # Create the pairplot
    sns.pairplot(df, markers="X", hue='Detected Outlier', diag_kind='kde', palette='coolwarm', plot_kws=plot_params)
    plt.suptitle("Matrix Plot of Outlier Detection Results", y=1.02)
    plt.show()


def plot_upper_matrix(X, y, y_pred, plot_params=None):
    """
    Create an upper matrix plot for outlier detection results.

    Parameters
    ----------
    X : ndarray or DataFrame of shape (n_samples, n_features)
        The input data.

    y : ndarray or Series of shape (n_samples,)
        The true labels for the data, where 0 indicates inliers and 1 indicates outliers.

    y_pred : ndarray of shape (n_samples,)
        The predicted labels from the outlier detection algorithm.

    plot_params : dict, optional
        Additional keyword arguments for plotting.
    """
    if plot_params is None:
        plot_params = {}

    # Convert to DataFrame for seaborn pairplot
    if isinstance(X, np.ndarray):
        X = pd.DataFrame(X)
    if isinstance(y, np.ndarray):
        y = pd.Series(y)
    if isinstance(y_pred, np.ndarray):
        y_pred = pd.Series(y_pred)

    # Combine X, y, and y_pred into a single DataFrame
    df = X.copy()
    df['True Outlier'] = y
    df['Detected Outlier'] = y_pred

    # Create the pairplot
    g = sns.pairplot(df, hue='Detected Outlier', diag_kind='kde', palette='coolwarm', plot_kws=plot_params)
    
    # Remove the lower triangle by iterating over the axes
    for i, j in zip(*np.tril_indices_from(g.axes, -1)):
        g.axes[i, j].set_visible(False)
    
    plt.suptitle("Upper Matrix Plot of Outlier Detection Results", y=1.02)
    plt.show()




def preprocess_pipeline(train_data, test_data, numerical_features, categorical_features):
    """
    Preprocess the training and test datasets by applying transformations 
    to numerical and categorical features.

    Parameters
    ----------
    train_data : pandas.DataFrame
        The training dataset containing both numerical and categorical features.

    test_data : pandas.DataFrame
        The test dataset containing both numerical and categorical features.

    numerical_features : list of str
        A list of column names representing numerical features in the dataset.

    categorical_features : list of str
        A list of column names representing categorical features in the dataset.

    Returns
    -------
    X_train_preprocessed : numpy.ndarray
        The preprocessed training data, where numerical features are scaled 
        and categorical features are one-hot encoded.

    X_test_preprocessed : numpy.ndarray
        The preprocessed test data, where numerical features are scaled 
        and categorical features are one-hot encoded.

    Examples
    --------
    >>> train_data = pd.DataFrame({
    ...     'age': [25, 35, 45],
    ...     'income': [50000, 60000, 80000],
    ...     'gender': ['male', 'female', 'female']
    ... })
    >>> test_data = pd.DataFrame({
    ...     'age': [30, 40],
    ...     'income': [55000, 70000],
    ...     'gender': ['female', 'male']
    ... })
    >>> numerical_features = ['age', 'income']
    >>> categorical_features = ['gender']
    >>> X_train_preprocessed, X_test_preprocessed = preprocess_pipeline(train_data, test_data, numerical_features, categorical_features)
    >>> print(X_train_preprocessed)
    >>> print(X_test_preprocessed)
    """
    
    # Preprocessing pipeline for numerical data
    numerical_pipeline = Pipeline([
        ('scaler', StandardScaler())
    ])

    # Preprocessing pipeline for categorical data
    categorical_pipeline = Pipeline([
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    # Combine preprocessing pipelines
    preprocessor = ColumnTransformer([
        ('num', numerical_pipeline, numerical_features),
        ('cat', categorical_pipeline, categorical_features)
    ])

    # Preprocess the data
    X_train_preprocessed = preprocessor.fit_transform(train_data)
    X_test_preprocessed = preprocessor.transform(test_data)

    return X_train_preprocessed, X_test_preprocessed




    