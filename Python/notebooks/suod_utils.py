__author__ = "Dr. Saad Laouadi"
__verssion__ = "1.0.0"

"""
    Utility functions used to train SUOD System. 
    The objective of these functions is to provide higher API for the system.
"""
from itertools import product

import dill
import joblib
from joblib import Parallel, delayed, parallel_backend

# Set joblib to use dill for serialization
parallel_backend('loky', n_jobs=2)

# Monkey patch joblib to use dill instead of pickle
joblib.externals.loky.backend.reduction.pickle = dill
joblib.externals.loky.backend.reduction.dumps = dill.dumps
joblib.externals.loky.backend.reduction.loads = dill.loads

from pyod.models.lof import LOF
from pyod.models.copod import COPOD
from pyod.models.iforest import IForest
from pyod.models.suod import SUOD
from pyod.utils.data import generate_data
from pyod.utils.data import evaluate_print
from pyod.utils.example import visualize


def generate_sample_data(n_train, n_test, n_features, contamination, random_state):
    """
    Generate sample training and testing data.

    Parameters
    ----------
    n_train : int
        Number of training samples.
    n_test : int
        Number of testing samples.
    n_features : int
        Number of features.
    contamination : float
        Proportion of outliers in the dataset.
    random_state : int
        Seed for the random number generator.

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
    X_train, X_test, y_train, y_test = generate_data(
        n_train=n_train,
        n_test=n_test,
        n_features=n_features,
        contamination=contamination,
        random_state=random_state
    )
    return X_train, X_test, y_train, y_test
    
def initialize_detectors(detector_configs):
    """
    Initialize a list of outlier detectors based on provided configurations.

    Parameters
    ----------
    detector_configs : list of dict
        A list of dictionaries where each dictionary specifies the detector type and its parameters.
        Example:
        [
            {'detector': LOF, 'params': {'n_neighbors': [15, 20, 25, 35]}},
            {'detector': IForest, 'params': {'n_estimators': [25, 50, 100, 750]}}
        ]

    Returns
    -------
    detector_list : list
        List of initialized outlier detectors.
    """
    detector_list = []

    for config in detector_configs:
        detector_class = config['detector']
        param_dict = config.get('params', {})

        # Ensure each parameter value is a list
        for key, value in param_dict.items():
            if not isinstance(value, list):
                param_dict[key] = [value]

        # Generate all combinations of the parameter values
        param_names = list(param_dict.keys())
        param_values = list(param_dict.values())
        param_combinations = list(product(*param_values))

        for combination in param_combinations:
            detector_params = dict(zip(param_names, combination))
            detector_instance = detector_class(**detector_params)
            detector_list.append(detector_instance)
    
    return detector_list


def train_suod(X=None, detector_list=None, contamination=0.1, combination='average', n_jobs=2, verbose=False, type='supervised', **kwargs):
    """
    Train the SUOD model.

    Parameters
    ----------
    X : ndarray, optional
        Input data. Required for all types of training ('supervised', 'unsupervised', or 'semi-supervised').
    detector_list : list
        List of outlier detectors.
    contamination : float, optional
        The amount of contamination in the dataset (default is 0.1).
    combination : str, optional
        Method to combine results from base estimators (default is 'average').
    n_jobs : int, optional
        Number of parallel jobs to run (default is 2).
    verbose : bool, optional
        Whether to print detailed information (default is False).
    type : str, optional
        Type of training: 'supervised' (default), 'unsupervised', or 'semi-supervised'.
    **kwargs : dict, optional
        Additional parameters to pass to the SUOD model.

    Returns
    -------
    clf : SUOD
        Trained SUOD model.
    labels : ndarray, optional
        Predicted labels for unsupervised mode. 1 for outliers, 0 for inliers.
    """
    if X is None:
        raise ValueError("X is required for all types of training.")

    clf = SUOD(
        base_estimators=detector_list,
        contamination=contamination,
        combination=combination,
        n_jobs=n_jobs,
        verbose=verbose,
        **kwargs  # Pass additional parameters here
    )

    # Fit the model for all types
    clf.fit(X)

    if type == 'supervised' or type == 'semi-supervised':
        return clf
    
    elif type == 'unsupervised':
        labels = clf.labels_      # Access the labels_ attribute after fitting
        return clf, labels
    
    else:
        raise ValueError("Invalid type. Choose 'supervised', 'unsupervised', or 'semi-supervised'.")
    

def get_predictions(clf, X_train, X_test):
    """
    Get predictions from the trained model.

    Parameters
    ----------
    clf : SUOD
        Trained SUOD model.
    X_train : ndarray
        Training data.
    X_test : ndarray
        Testing data.

    Returns
    -------
    y_train_pred : ndarray
        Predicted labels for training data.
    y_train_scores : ndarray
        Outlier scores for training data.
    y_test_pred : ndarray
        Predicted labels for testing data.
    y_test_scores : ndarray
        Outlier scores for testing data.
    """
    # Predict on training data
    y_train_pred = clf.predict(X_train)           # outlier labels (0 or 1)
    y_train_scores = clf.decision_function(X_train)  # outlier scores

    # Predict on test data
    y_test_pred = clf.predict(X_test)           # outlier labels (0 or 1)
    y_test_scores = clf.decision_function(X_test)  # outlier scores

    return y_train_pred, y_train_scores, y_test_pred, y_test_scores


# Check if the dataset has only one class
def check_class_presence(y_train, y_test):
    """
    Check which dataset has only one class.

    Parameters
    ----------
    y_train : ndarray
        True labels for training data.
    y_test : ndarray
        True labels for testing data.

    Returns
    -------
    bool, bool
        Two booleans indicating if y_train and y_test have only one class.
    """
    train_has_one_class = len(set(y_train)) == 1
    test_has_one_class = len(set(y_test)) == 1

    if train_has_one_class:
        print(f"y_train has only one class: {set(y_train)}")
    else:
        print(f"y_train has multiple classes: {set(y_train)}")

    if test_has_one_class:
        print(f"y_test has only one class: {set(y_test)}")
    else:
        print(f"y_test has multiple classes: {set(y_test)}")

    return train_has_one_class, test_has_one_class


# Evaluate the model and handle cases with only one class
def evaluate_model(clf_name, y_train, y_train_scores, y_test, y_test_scores):
    """
    Evaluate the model performance.

    Parameters
    ----------
    clf_name : str
        Name of the classifier.
    y_train : ndarray
        True labels for training data.
    y_train_scores : ndarray
        Outlier scores for training data.
    y_test : ndarray
        True labels for testing data.
    y_test_scores : ndarray
        Outlier scores for testing data.
    """
    train_has_one_class, test_has_one_class = check_class_presence(y_train, y_test)
    
    if not train_has_one_class:
        print("\nOn Training Data:")
        evaluate_print(clf_name, y_train, y_train_scores)
    else:
        print("Skipping evaluation on training data due to only one class present.")
    
    if not test_has_one_class:
        print("\nOn Test Data:")
        evaluate_print(clf_name, y_test, y_test_scores)
    else:
        print("Skipping evaluation on test data due to only one class present.")


def visualize_results(clf_name, X_train, y_train, X_test, y_test, y_train_pred, y_test_pred):
    """
    Visualize the results.

    Parameters
    ----------
    clf_name : str
        Name of the classifier.
    X_train : ndarray
        Training data.
    y_train : ndarray
        True labels for training data.
    X_test : ndarray
        Testing data.
    y_test : ndarray
        True labels for testing data.
    y_train_pred : ndarray
        Predicted labels for training data.
    y_test_pred : ndarray
        Predicted labels for testing data.
    """
    visualize(clf_name, X_train, y_train, X_test, y_test, y_train_pred, y_test_pred, show_figure=True, save_figure=False)
