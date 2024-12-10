import csv
import pandas as pd

def read_csv_file(csv_path: str, delimiter: str) -> pd.DataFrame:
    '''
    read csv files for yield and pesticide
    '''
    
    return pd.read_csv(csv_path, sep = delimiter)



def read_temp_file(csv_path: str, encoding) -> pd.DataFrame:
    '''
    read csv file for temperature dataset
    '''
    
    return pd.read_csv(csv_path, encoding=encoding)



def read_rain_file(csv_path: str, new_path)-> pd.DataFrame:
    '''
    read csv file for rain dataset which has additional columns for some rows
    '''
    
    with open(csv_path, 'r') as infile, open(new_path, 'w', newline='') as outfile:
        reader = csv.reader(infile)
        writer = csv.writer(outfile)
        for row in reader:
            # thie enable us join all columns with spaces and then split by the first two columns
            # Assuming the format is consistent with three columns: country, year, rainfall
            if len(row) < 3:
                continue                  # Skip malformed rows
            country = ' '.join(row[:-2])
            year = row[-2]
            average_rain_fall_mm_per_year = row[-1]
            writer.writerow([country, year, average_rain_fall_mm_per_year])
            
            

def read_newrain_file(csv_path: str) -> pd.DataFrame:
    '''
    read csv file for the modified rain dataset
    '''
    return pd.read_csv(csv_path)