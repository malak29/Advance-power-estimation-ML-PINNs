import pandas as pd

# Replace this with the path to your pickle file
pickle_file_path = 'D:\Building_Modeling_Code\Results\Processed_BuildingSim_Data\ASHRAE_2013_Albuquerque_ApartmentHighRise\Sim_ProcessedData\IDF_OutputVariables_DictDF.pickle'
# pickle_file_path = 'D:\Building_Modeling_Code\Results\Processed_BuildingSim_Data\ASHRAE_2013_Albuquerque_OfficeSmall\Sim_ProcessedData\IDF_OutputVariables_DictDF.pickle'

def inspect_pickle_contents(pickle_file_path):
    # Load the data from the pickle file
    data = pd.read_pickle(pickle_file_path)
    
    # Check if the loaded data is a dictionary
    if isinstance(data, dict):
        print("The pickle file contains a dictionary.")
        print("Keys in the dictionary:", list(data.keys()))
        
        # Optionally, inspect each item to see if any are DataFrames
        for key, value in data.items():
            if isinstance(value, pd.DataFrame):
                print(f"\nKey '{key}' contains a DataFrame:")
                print("-" * 40)
                print("Column Names:", value.columns.tolist())
                print("Number of Rows:", len(value))
                print("Data Types of Columns:", value.dtypes)
                print("\nFirst few rows of the DataFrame under key '{}':".format(key))
                print(value.head())
            else:
                print(f"\nKey '{key}' does not contain a DataFrame.")
                print(f"Type of data under key '{key}':", type(value))
    else:
        print("The loaded data is not a dictionary. It is a:", type(data))

# Call the function
inspect_pickle_contents(pickle_file_path)


