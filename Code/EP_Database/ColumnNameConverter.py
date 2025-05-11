import sys
import os

Current_Filepath = os.path.dirname(__file__)

Input_Filepath = os.path.join(Current_Filepath, 'DatabaseUploaderTemp', 'Surface_Inside_Face_Internal_Gains_Radiation_Heat_Gain_Rate_CSVColumns.txt')
Output_Filepath = os.path.join(Current_Filepath, 'DatabaseUploaderTemp', 'Surface_Inside_Face_Internal_Gains_Radiation_Heat_Gain_Rate_SQLColumns.txt')

with open(Input_Filepath, 'r') as file_read, open(Output_Filepath, 'w') as file_write:

    CSV_ColumnNames = file_read.read()
    
    SQL_ColumnNames = CSV_ColumnNames.replace(', ', '\n')
    
    lines = SQL_ColumnNames.split('\n')
    SQL_ColumnNames = ""
    
    for line in lines: 
        line_split = line.split(':')
        line = line_split[0]
        line = line[1:]
        line = line.replace(" ", "_")        
        line = 'Surface_Inside_Face_Internal_Gains_Radiation_Heat_Gain_Rate' + line
        line = line.lower()
        print(line + ('\n'))
        SQL_ColumnNames = SQL_ColumnNames + line + '\n'
    
    file_write.write(SQL_ColumnNames)
    
    