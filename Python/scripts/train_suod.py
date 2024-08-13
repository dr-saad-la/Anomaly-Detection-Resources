#!/usr/bin/env python
# coding: utf-8

# **Copyright**
# **© 2024 Dr. Saad Laouadi. All rights reserved.**

import os
import sys

# Add the path to folder_02 to the system path
sys.path.append(os.path.abspath('../notebooks'))

import time
from collections import Counter

import numpy as np 
import pandas as pd
from scipy.io import arff

from pyod.models.abod import ABOD
from pyod.models.ecod import ECOD
from pyod.utils.data import evaluate_print
from pyod.utils.example import visualize

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.metrics import classification_report, roc_auc_score, precision_recall_curve, auc
from sklearn.metrics import precision_score, recall_score, f1_score

import matplotlib.pyplot as plt
import seaborn as sns

import matplotlib
matplotlib.use('Agg')

# The utils module is not an installable package, it is in the same directory as this notebook
from utils import generate_data, visualize_data, apply_abod, apply_abod_advanced, precision_at_rank_n
from utils import grid_search_abod, hyperopt_objective, plot_outliers_vs_inliers
from utils import preprocess_pipeline


from suod_utils import *

from pyod.models.suod import SUOD
from pyod.models.lof import LOF
from pyod.models.iforest import IForest
from pyod.models.copod import COPOD
from pyod.models.knn import KNN
from pyod.models.mcd import MCD
from pyod.models.ocsvm import OCSVM
from pyod.utils.utility import standardizer
from pyod.utils.data import generate_data, get_outliers_inliers, evaluate_print
from pyod.utils.example import visualize

from hyperopt import fmin, tpe, hp, Trials, STATUS_OK
from hyperopt.pyll.base import scope


def print_banner(sep, nchar, title):
    print(sep * nchar)
    print(title.center(nchar))
    print(sep * nchar)


# Function to load the dataset
def load_arff_data(filepath):
    """
    Load an ARFF file and return it as a pandas DataFrame.

    Parameters
    ----------
    filepath : str
        Path to the ARFF file.

    Returns
    -------
    df : DataFrame
        The loaded dataset.
    """
    data, meta = arff.loadarff(filepath)
    df = pd.DataFrame(data)

    # Convert byte strings to normal strings for categorical variables
    for column in df.select_dtypes([object]).columns:
        df[column] = df[column].str.decode('utf-8')

    return df


# Preprocess the dataset
def preprocess_heart_disease_data(df, target_name, scale=False):
    """
    Preprocess the Heart Disease dataset.

    Parameters
    ----------
    df : DataFrame
        The loaded dataset.
    target_name: str
        The target name.
    scale : bool, optional
        Whether to scale the features (default is False).

    Returns
    -------
    X : ndarray
        Feature matrix.
    y : ndarray
        Target vector (outlier labels).
    """
    X = df.drop(columns=[target_name])
    y = df[target_name].apply(lambda x: 1 if x == 'yes' else 0).values

    if scale:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
    else:
        X_scaled = X
    
    return X_scaled, y


def stratify_split_data(X, y, test_size=0.1, random_state=42):
    """
    Stratify split the data into training and testing sets.

    Parameters
    ----------
    X : ndarray
        Feature matrix.
    y : ndarray
        Target vector.
    test_size : float, optional
        The proportion of the dataset to include in the test split (default is 0.1).
    random_state : int, optional
        Random state for reproducibility (default is 42).

    Returns
    -------
    X_train : ndarray
        Training data.
    X_test : ndarray
        Testing data.
    y_train : ndarray
        Training labels.
    y_test : ndarray
        Testing labels.
    """
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)
    
    # Check class distribution
    print(f"Training set class distribution: {Counter(y_train)}")
    print(f"Test set class distribution: {Counter(y_test)}")
    
    return X_train, X_test, y_train, y_test


def train_suod_with_simulated_data(print_detector_info=False, contamination=0.1, n_train=900, n_test=100, n_jobs=1):
    """
    Train SUOD with simulated data.

    Parameters
    ----------
    print_detector_info : bool, optional
        Set to True to print the detectors parameters (default is False).
    contamination : float, optional
        Percentage of outliers in the dataset (default is 0.1).
    n_train : int, optional
        Number of training points (default is 900).
    n_test : int, optional
        Number of testing points (default is 100).
    n_jobs : int, optional
        Number of parallel jobs to run (default is 1).

    Returns
    -------
    None
    """

    start = time.time()

    # Generate synthetic data
    X_train, X_test, y_train, y_test = generate_sample_data(
        n_train=n_train, n_test=n_test, n_features=2, contamination=contamination, random_state=42
    )

    # Define detector configurations
    detector_configs = [
        {'detector': LOF, 'params': {'n_neighbors': 15}},
        {'detector': LOF, 'params': {'n_neighbors': 20}},
        {'detector': LOF, 'params': {'n_neighbors': 25}},
        {'detector': LOF, 'params': {'n_neighbors': 35}},
        {'detector': COPOD, 'params': {}},
        {'detector': IForest, 'params': {'n_estimators': 100}},
        {'detector': IForest, 'params': {'n_estimators': 200}},
    ]

    # Initialize detectors
    detector_list = initialize_detectors(detector_configs)

    if print_detector_info:
        print(detector_list)

    # Train SUOD
    clf = train_suod(X_train, detector_list, n_jobs=n_jobs)

    # Get predictions
    y_train_pred, y_train_scores, y_test_pred, y_test_scores = get_predictions(clf, X_train, X_test)

    # Evaluate the model
    evaluate_model('SUOD', y_train, y_train_scores, y_test, y_test_scores)

    # Visualize the results
    visualize_results('SUOD', X_train, y_train, X_test, y_test, y_train_pred, y_test_pred)

    end = time.time()

    print(f"Process took {end - start} seconds.")

def main():
    print("*"*72)
    print("Training SUOD with Simulated Data".center(72))
    print("*"*72)
    
    train_suod_with_simulated_data()
    print("*"*72)
    
    DATA_PATH = "../../datasets/HeartDisease/HeartDisease_withoutdupl_norm_44.arff"
    PRINT_DETECTOR_INFO = False
    PRINT_DATA_INFO = False

    start = time.time()

    # Load and preprocess the dataset
    df = load_arff_data(DATA_PATH)
    X, y = preprocess_heart_disease_data(df, 'outlier')

    if PRINT_DATA_INFO:
        print(df.info())

    # Check overall class distribution
    print(f"Overall class distribution: {Counter(y)}")

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = stratify_split_data(X, y, test_size=0.1, random_state=42)

    # Define detector configurations
    detector_configs = [
        {'detector': LOF, 'params': {'n_neighbors': [15, 20, 25, 35]}},
        {'detector': COPOD, 'params': {}},
        {'detector': IForest, 'params': {'n_estimators': [25, 50, 75, 100]}},
        {'detector': ABOD, 'params': {'n_neighbors': 50}},
        {'detector': KNN, 'params': {'n_neighbors': [5, 10]}},
        {'detector': MCD, 'params': {}},
        {'detector': OCSVM, 'params': {}}
    ]

    # Initialize detectors
    detector_list = initialize_detectors(detector_configs)

    if PRINT_DETECTOR_INFO:
        print(detector_list)

    # Train SUOD
    clf, labels = train_suod(X_train, detector_list, n_jobs=1, contamination=0.4444, type="unsupervised")

    # Get predictions
    y_pred = labels                # binary labels (0: inliers, 1: outliers)
    outlier_scores = clf.decision_scores_  # raw outlier scores

    print_banner("*", 72, "Displaying Information")
    print(f"Outliers detected: {sum(y_pred)} out of {len(y_pred)}")

    # Getting the outlier indexes
    outlier_indices = np.where(y_pred == 1)[0]

    # Print the indexes of the outliers
    print("Indices of the outliers:")
    print(outlier_indices)

    # Slice the data to have only the outliers
    outliers_only = X.iloc[outlier_indices, :]
    print(outliers_only.shape[0])

    print_banner("*", 72, "Plotting")

    end = time.time()
    print(f"Process took {end - start} seconds.")


if __name__ == "__main__":
    main()