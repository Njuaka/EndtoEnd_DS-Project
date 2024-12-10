from utils import *



root_dir = get_root_directory()

# get the path for the different raw data files

PATH_RAIN_FILE = os.path.join(root_dir, "data", "rain.csv")
NEW_RAIN_FILE = os.path.join(root_dir, "data", "modified_rain.csv")
PATH_TEMPERATURE_FILE = os.path.join(root_dir, "data", "temperature.csv")
PATH_PESTICIDE_FILE = os.path.join(root_dir, "data", "pesticides_usage.csv")
PATH_YIELD_FILE  = os.path.join(root_dir, "data", "yield.csv")
FIGURE_PATH = os.path.join(root_dir, "reports/")
