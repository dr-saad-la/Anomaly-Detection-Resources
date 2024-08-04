"""
    A Module to download the bench dataset for anomaly detection
"""

import os
import json
import wget
from tqdm import tqdm
from adbench.myutils import Utils

# Access the project root directory from the environment variable
# Ensure that the ANOMALY_DETECTION_PATH environment variable is set
# if not set here like this
# project_root = "Your/path/to/anomaly-detection-project"   # uncomment this before run unless
                                                            # You set up Your project directory path
                                                            # as an environment variable

project_root = os.getenv('ANOMALY_DETECTION_PATH')

print(f"Project Root: {project_root}")

if project_root is None:
    raise EnvironmentError("The ANOMALY_DETECTION_PATH environment variable is not set.")

# Define the dataset directory 
dataset_dir = os.path.join(project_root, 'datasets')
print(dataset_dir)
print(os.listdir(dataset_dir))


# Ensure the dataset directory exists
os.makedirs(dataset_dir, exist_ok=True)

# Initialize the utility class
utils = Utils()

# List of folders expected to contain datasets
expected_folders = ['CV_by_ResNet18', 'NLP_by_BERT', 'Classical']


# Define the function to check if all datasets are already downloaded
def all_datasets_exist(base_dir, folders):
    """
    Checks if all datasets are present in the specified base directory.

    :param base_dir: The base directory where datasets are stored.
    :param folders: A list of folder names expected to be present in the base directory.
    :param files_dict: Optional dictionary specifying expected files in each folder.
    :return: True if all folders and files are present, False otherwise.
    """
    for folder in folders:
        folder_path = os.path.join(base_dir, folder)
        print(folder_path)
        # Check if folder exists
        if not os.path.exists(folder_path):
            print(f"Missing folder: {folder_path}")
            return False
        # Check if folder is not empty
        if not os.listdir(folder_path):
            print(f"Folder is empty: {folder_path}")
            return False
    return True

# Check if all datasets are already downloaded
if all_datasets_exist(dataset_dir, expected_folders):
    print("All datasets are already downloaded.")
else:
    # Attempt to download datasets
    try:
        utils.download_datasets(repo='jihulab')                    # Use 'github' if not in China
        print("Datasets downloaded successfully using adbench.")
    except Exception as e:
        print(f"An unexpected error occurred during download: {e}")