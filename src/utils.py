import os 

def get_root_directory() ->str:
    """
    Returns the root directory of the project .
    
    """
    # Get the absolute path of the current file
    current_file_path = os.path.abspath(__file__)
    
    # Get the directory containing the current file
    current_dir = os.path.dirname(current_file_path)
    
    root_dir = os.path.dirname(current_dir) 
    
    return root_dir

