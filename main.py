from src.constants import *
from src.writer import *
from src.processor import *
from src.reader import *


def main():
    '''
    runs the all scripts importing data to report building with visuals
    '''
    # reading data
    df_rain = read_rain_file(PATH_RAIN_FILE,NEW_RAIN_FILE)
    df_yield = read_csv_file(PATH_YIELD_FILE , delimiter=";")
    df_temperature = read_temp_file(PATH_TEMPERATURE_FILE, encoding='ISO-8859-1')
    df_rain_2 = read_newrain_file(NEW_RAIN_FILE)
    df_pesticide = read_csv_file(PATH_PESTICIDE_FILE, delimiter=";")
    features_column=["avg_temp (°C)","average_rain_fall (mm/year)", "pest_value (tonnes)"]
    target_column = "yield_value (hg/ha)"
    
    
    # data preprocessing (renaming and combining data)
    
    process_data = InitialPreprocessingData(df_temperature,df_rain_2)
    df_temperature = process_data.replace_value('Country', "CÃ´te D'Ivoire", "Côte D'Ivoire" )
    df_rain_2 = process_data.rename_columns({'country': 'Country', 'year': 'Year'})
 
    df = merge_data(df_pesticide,df_rain_2,['Country', 'Year'])
    df = merge_data(df,df_temperature,['Country', 'Year'])
    df = merge_data(df,df_yield,['Country', 'Year'])
    
    
    # preprocessing combined data
    combined_data = FinalDataColumns(df)
    columns = ['Country', 'Year', 'Item_y', 'avg_temp (°C)','average_rain_fall_mm_per_year', 'Value_x','Value_y']
    final_raw_agric_data = combined_data.select_columns(columns=columns)
    
    renamed_cobined_data = renamedCombinedData(final_raw_agric_data)
    columns_mapping = {'Item_y': 'crop_types','average_rain_fall_mm_per_year': 'average_rain_fall (mm/year)',
    'Value_x': 'pest_value (tonnes)','Value_y': 'yield_value (hg/ha)'
    } 
    final_agric_data = renamed_cobined_data.rename_final_raw_data_columns(columns_mapping)
    

    
    # Data tranformation
    transform_data = TranformRawData(final_agric_data, 'average_rain_fall (mm/year)')
    transform_agric_data = transform_data.map_continent()
    transform_agric_data = transform_data.replace_column_data()
    
    
    # Train and Evaluate Model
    
    model = TrainEvaluateModels(transform_agric_data, features_column, target_column)
    best_model_name, best_model = model.train_and_evaluate()
    
    if best_model_name:
    # Retrieve results for the best model
        best_model_results = model.get_best_model()
        print(f"Best Model: {best_model_results['name']}, MSE: {best_model_results['mse']}")
        
    
    
   
    
    # Generate reports based on user input
    #user_type = input("Enter user type (analyst or breeder): ")
    
    #report = Report(transform_agric_data)
    #report.generate(user_type=user_type)
    
    #plot = ModelPlot(model_results)
    #plot.plot_feature_importance()
   # plot.plot_actual_vs_actual()
        
    
if __name__=="__main__":
    main()