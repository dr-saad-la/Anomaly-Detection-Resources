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
            {'detector': LOF, 'params': {'n_neighbors': 15}},
            {'detector': IForest, 'params': {'n_estimators': 100}}
        ]

    Returns
    -------
    detector_list : list
        List of initialized outlier detectors.
    """
    detector_list = []
    for config in detector_configs:
        detector_class = config['detector']
        detector_params = config.get('params', {})
        detector_instance = detector_class(**detector_params)
        detector_list.append(detector_instance)
    
    return detector_list


def train_suod(X_train, detector_list, n_jobs=2, combination='average', verbose=False):
    """
    Train the SUOD model.

    Parameters
    ----------
    X_train : ndarray
        Training data.
    detector_list : list
        List of outlier detectors.
    n_jobs : int, optional
        Number of parallel jobs to run (default is 4).
    combination : str, optional
        Method to combine results from base estimators (default is 'average').
    verbose : bool, optional
        Whether to print detailed information (default is False).

    Returns
    -------
    clf : SUOD
        Trained SUOD model.
    """
    clf = SUOD(base_estimators=detector_list,
               n_jobs=n_jobs,
               combination=combination,
               verbose=verbose)
    clf.fit(X_train)
    return clf

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
    print("\nOn Training Data:")
    evaluate_print(clf_name, y_train, y_train_scores)
    print("\nOn Test Data:")
    evaluate_print(clf_name, y_test, y_test_scores)


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
