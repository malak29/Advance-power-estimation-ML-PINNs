import os
import pickle
from re import S
import psycopg2
import pandas as pd

pickle_file_path = 'D:\Building_Modeling_Code\Results\Processed_BuildingSim_Data\ASHRAE_2013_Albuquerque_ApartmentHighRise\Sim_ProcessedData\IDF_OutputVariables_DictDF.pickle'

data = pd.read_pickle(pickle_file_path)

print("Keys in the dictionary:", list(data.keys()))

SQL_ColumnNames = []

for key, value in data.items():
    
    if key != 'DateTime_List':          
        CSV_ColumnNames = value.columns.tolist()
        for item in CSV_ColumnNames:
            item_split = item.split(':')
            item = item_split[0]
            item = item.replace(" ", "_")   
            key = key.split('.')[0]
            item = key + '_' + item
            item = item.lower()
            if 'date/time' in item:
                item = item.replace('date/time', 'datetime') # SQL hates the '/'
            if '-' in item:
                item = item.replace('-', '_') # SQL also hates '-'
            SQL_ColumnNames.append(item)
            
print('Number of Columns: ' + str(len(SQL_ColumnNames)))

# Connect to the PostgreSQL database
conn = psycopg2.connect("dbname=Building_Models user=kasey password=OfficeLarge")
cursor = conn.cursor()

# SQL to create table
create_table_query = """
CREATE TABLE timeseriesdata (
    timeseriesdataid SERIAL PRIMARY KEY,
    buildingid INTEGER REFERENCES buildingids(buildingid),  
    zonename TEXT,
    datetime TEXT,
    timeresolution TEXT,
    {}
);
""".format(",\n    ".join(f"{col} FLOAT" for col in SQL_ColumnNames))  # Assuming all other columns are of type FLOAT

# Execute the query
cursor.execute(create_table_query)
conn.commit()

# Close the connection
cursor.close()
conn.close()

print("Table 'timeseriesdata' created successfully.")
