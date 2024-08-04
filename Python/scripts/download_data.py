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
def check_datasets_exist(base_dir, folders, files_dict=None):
    """Checks if all specified datasets are present in the base directory.

    This function verifies the presence of specified folders within a base directory 
    and checks if the required files within those folders exist.

    Args:
        base_dir (str): The base directory where datasets are expected to be stored.
        folders (list of str): A list of folder names expected to be present in the base directory.
        files_dict (dict, optional): A dictionary mapping folder names to a list of expected filenames.
                                     Default is None, meaning no specific file check is performed.

    Returns:
        bool: True if all specified folders and their required files are present, False otherwise.

    Raises:
        FileNotFoundError: If the base directory does not exist.
    """
    if not os.path.exists(base_dir):
        raise FileNotFoundError(f"The base directory does not exist: {base_dir}")
    
    for folder in folders:
        folder_path = os.path.join(base_dir, folder)

        if not os.path.exists(folder_path):
            print(f"Missing folder: {folder_path}")
            return False
        
        folder_contents = os.listdir(folder_path)
        if not folder_contents:
            print(f"Folder is empty: {folder_path}")
            return False
        
        if files_dict:
            expected_files = files_dict.get(folder, [])
            for expected_file in expected_files:
                file_path = os.path.join(folder_path, expected_file)
                if not os.path.exists(file_path):
                    print(f"Missing file: {file_path}")
                    return False

    return True

# Check if all datasets are already downloaded
if check_datasets_exist(dataset_dir, expected_folders):
    print("All datasets are already downloaded.")
else:
    # Attempt to download datasets
    try:
        utils.download_datasets(repo='jihulab')                    # Use 'github' if not in China
        print("Datasets downloaded successfully using adbench.")
    except Exception as e:
        print(f"An unexpected error occurred during download: {e}")