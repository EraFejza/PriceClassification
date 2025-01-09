import pandas as pd

def data_loader(file_path):
    """
    Load a dataset from a CSV file and return a DataFrame.
    Handles any necessary preprocessing.

    Args:
        file_path (str): Path to the CSV file.

    Returns:
        pd.DataFrame: Loaded dataset as a DataFrame.
    """
    try:
        data = pd.read_csv(file_path)
        return data
    except FileNotFoundError:
        raise Exception(f"File not found: {file_path}")
    except Exception as e:
        raise Exception(f"Error loading file: {e}")
