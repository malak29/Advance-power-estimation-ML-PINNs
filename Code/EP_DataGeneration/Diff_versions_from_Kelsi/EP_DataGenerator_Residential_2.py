# -*- coding: utf-8 -*-
"""
Created on Wed Apr 20 21:11:17 2022

@author: ninad gaikwad 
"""

# This code has been modified to skip Simulations that have lready been done to aid testing. 
# Second simulation is giving issues - see previous code version. 

# This code will be modified to make it easier to exclude certain simulations. 









# =============================================================================
# Import Required Modules
# =============================================================================

# External Modules
import os
import numpy as np
import pandas as pd
import scipy.io
import opyplus as op
import re
import shutil
import datetime
import pickle

import copy

import sys

# Custom Modules

# =============================================================================
# Write Consol Output to txt file
# =============================================================================

class DualOutput:
    def __init__(self, filename):
        self.file = open(filename, "w")
        self.console = sys.stdout

    def write(self, message):
        self.file.write(message)
        self.console.write(message)

    def flush(self):
        self.file.flush()
        self.console.flush()

    def close(self):
        self.file.close()
        self.console.close()

sys.stdout = DualOutput("ConsolOutput.txt")

# =============================================================================
# User Inputs
# =============================================================================


# Select IDF File
''' 
Type of Building:
    
-Commercial_Prototypes
-Manufactured_Prototypes
-Residential_Prototypes

''' 

# Simulation Run Period
Sim_Start_Day = 1

Sim_Start_Month = 1

Sim_End_Day = 31

Sim_End_Month = 12

# IDF File Year
IDF_FileYear = 2013

# Simulation Time Step [in minutes]
Sim_TimeStep = 5

# Simulation Output Variable Reporting Frequency
'''
Types of report frequencies:
    
- timestep
- hourly
- detailed
- daily
- monthly
- runperiod
- environment
- annual
'''

Sim_OutputVariable_ReportingFrequency = 'timestep'

# Simulation People Schedule
'''
People_Schedule_Name_1 = 'Name of Schedule existing IDF'

People_Schedule_Dict_1 = {'field_1': field 1 value/string , ..., 'field_m': field m value/string}

.
.
.

People_Schedule_Name_n = 'Name of Schedule existing IDF'

People_Schedule_Dict_n= {'field_1': field 1 value/string , ..., 'field_m': field m value/string}

Sim_People_Schedules_Dict = {People_Schedule_Name_1: People_Schedule_Dict_1 , ..., People_Schedule_Name_n: People_Schedule_Dict_n }
'''

Sim_People_Schedules_Dict ={}

# Simulation Equipment Schedule
'''
Equipment_Schedule_Name_1 = 'Name of Schedule existing IDF'

Equipment_Schedule_Dict_1 = {'field_1': field 1 value/string , ..., 'field_m': field m value/string}

.
.
.

Equipment_Schedule_Name_n = 'Name of Schedule existing IDF'

Equipment_Schedule_Dict_n= {'field_1': field 1 value/string , ..., 'field_m': field m value/string}

Sim_Equipment_Schedules_Dict = {Equipment_Schedule_Name_1: Equipment_Schedule_Dict_1 , ..., Equipment_Schedule_Name_n: Equipment_Schedule_Dict_n }
'''
Sim_Equipment_Schedules_Dict ={}

# Simulation Light Schedule
'''
Light_Schedule_Name_1 = 'Name of Schedule existing IDF'

Light_Schedule_Dict_1 = {'field_1': field 1 value/string , ..., 'field_m': field m value/string}

.
.
.

Light_Schedule_Name_n = 'Name of Schedule existing IDF'

Light_Schedule_Dict_n= {'field_1': field 1 value/string , ..., 'field_m': field m value/string}

Sim_Light_Schedules_Dict = {Light_Schedule_Name_1: Light_Schedule_Dict_1 , ..., Light_Schedule_Name_n: Light_Schedule_Dict_n }
'''

Sim_Light_Schedules_Dict ={}

# Simulation Exterior Light Schedule
'''
ExteriorLight_Schedule_Name_1 = 'Name of Schedule existing IDF'

ExteriorLight_Schedule_Dict_1 = {'field_1': field 1 value/string , ..., 'field_m': field m value/string}

.
.
.

ExteriorLight_Schedule_Name_n = 'Name of Schedule existing IDF'

ExteriorLight_Schedule_Dict_n= {'field_1': field 1 value/string , ..., 'field_m': field m value/string}

Sim_ExteriorLight_Schedules_Dict = {ExteriorLight_Schedule_Name_1: ExteriorLight_Schedule_Dict_1 , ..., ExteriorLight_Schedule_Name_n: ExteriorLight_Schedule_Dict_n }
'''

Sim_ExteriorLight_Schedules_Dict ={}

# Simulation Heating Setpoint Schedule
'''
HeatingSetPoint_Schedule_Name_1 = 'Name of Schedule existing IDF'

HeatingSetPoint_Schedule_Dict_1 = {'field_1': field 1 value/string , ..., 'field_m': field m value/string}

.
.
.

HeatingSetPoint_Schedule_Name_n = 'Name of Schedule existing IDF'

HeatingSetPoint_Schedule_Dict_n= {'field_1': field 1 value/string , ..., 'field_m': field m value/string}

Sim_HeatingSetPoint_Schedules_Dict = {HeatingSetPoint_Schedule_Name_1: HeatingSetPoint_Schedule_Dict_1 , ..., HeatingSetPoint_Schedule_Name_n: HeatingSetPoint_Schedule_Dict_n }
'''

Sim_HeatingSetPoint_Schedules_Dict ={}

# Simulation Cooling Setpoint Light Schedule
'''
CoolingSetPoint_Schedule_Name_1 = 'Name of Schedule existing IDF'

CoolingSetPoint_Schedule_Dict_1 = {'field_1': field 1 value/string , ..., 'field_m': field m value/string}

.
.
.

CoolingSetPoint_Schedule_Name_n = 'Name of Schedule existing IDF'

CoolingSetPoint_Schedule_Dict_n= {'field_1': field 1 value/string , ..., 'field_m': field m value/string}

Sim_CoolingSetPoint_Schedules_Dict = {CoolingSetPoint_Schedule_Name_1: CoolingSetPoint_Schedule_Dict_1 , ..., CoolingSetPoint_Schedule_Name_n: CoolingSetPoint_Schedule_Dict_n }
'''

Sim_CoolingSetPoint_Schedules_Dict ={}

# Simulation Humidity Setpoint Light Schedule 
'''
HumiditySetPoint_Schedule_Name_1 = 'Name of Schedule existing IDF'

HumiditySetPoint_Schedule_Dict_1 = {'field_1': field 1 value/string , ..., 'field_m': field m value/string}

.
.
.

HumiditySetPoint_Schedule_Name_n = 'Name of Schedule existing IDF'

HumiditySetPoint_Schedule_Dict_n= {'field_1': field 1 value/string , ..., 'field_m': field m value/string}

Sim_HumiditySetPoint_Schedules_Dict = {HumiditySetPoint_Schedule_Name_1: HumiditySetPoint_Schedule_Dict_1 , ..., HumiditySetPoint_Schedule_Name_n: HumiditySetPoint_Schedule_Dict_n }
'''

Sim_HumiditySetPoint_Schedules_Dict ={}

'''
# Creating a List of all modified Schedule Dictionaries
Schedule_Dict_List = [Sim_People_Schedules_Dict, Sim_Equipment_Schedules_Dict, Sim_Light_Schedules_Dict, Sim_ExteriorLight_Schedules_Dict, Sim_HeatingSetPoint_Schedules_Dict, Sim_CoolingSetPoint_Schedules_Dict,Sim_HumiditySetPoint_Schedules_Dict]
'''

Schedule_Dict_List = []

# Simulation Type
SimulationType = 2  # 1 - All variables generated  ; 2 - Required Variables generated

# Required Variables
Required_VariableNames = ['Schedule Value',
                                  'Facility Total HVAC Electric Demand Power',
                                  'Site Diffuse Solar Radiation Rate per Area',
                                  'Site Direct Solar Radiation Rate per Area',
                                  'Site Outdoor Air Drybulb Temperature',
                                  'Site Solar Altitude Angle',
                                  'Surface Inside Face Internal Gains Radiation Heat Gain Rate',
                                  'Surface Inside Face Lights Radiation Heat Gain Rate',
                                  'Surface Inside Face Solar Radiation Heat Gain Rate',
                                  'Surface Inside Face Temperature',
                                  'Zone Windows Total Transmitted Solar Radiation Rate',
                                  'Zone Air Temperature',
                                  'Zone People Convective Heating Rate',
                                  'Zone Lights Convective Heating_Rate',
                                  'Zone Electric Equipment Convective Heating Rate',
                                  'Zone Gas Equipment Convective Heating Rate',
                                  'Zone Other Equipment Convective Heating Rate',
                                  'Zone Hot Water Equipment Convective Heating Rate',
                                  'Zone Steam Equipment Convective Heating Rate',
                                  'Zone People Radiant Heating Rate',
                                  'Zone Lights Radiant Heating Rate',
                                  'Zone Electric Equipment Radiant Heating Rate',
                                  'Zone Gas Equipment Radiant Heating Rate',
                                  'Zone Other Equipment Radiant Heating Rate',
                                  'Zone Hot Water Equipment Radiant Heating Rate',
                                  'Zone Steam Equipment Radiant Heating Rate',
                                  'Zone Lights Visible Radiation Heating Rate',
                                  'Zone Total Internal Convective Heating Rate',
                                  'Zone Total Internal Radiant Heating Rate',
                                  'Zone Total Internal Total Heating Rate',
                                  'Zone Total Internal Visible Radiation Heating Rate',
                                  'Zone Air System Sensible Cooling Rate',
                                  'Zone Air System Sensible Heating Rate',
                                  'System Node Temperature',
                                  'System Node Mass Flow Rate']


Current_FilePath = os.path.dirname(__file__)

# Creating Path to Special IDF File
SpecialIDF_FolderPath = os.path.join(Current_FilePath,  '..',  '..', 'Data')

Residential_FolderPath = os.path.join(Current_FilePath, '..', '..', 'Data', 'Residential_Prototypes')

# User inputs
SkipLocation = ['Miami']
SkipHeatingType = []
SkipFoundationType = []
SkipSimulation = ['MF_CZ1AWH_Miami_elecres_heatedbsmt_IECC_2015', 'MF_CZ1AWH_Miami_elecres_slab_IECC_2015']
#SkipVariable = []
SkipVariable = Required_VariableNames # Not making any CSV's - for debugging purposes. 

for Year_Subfolder in os.listdir(Residential_FolderPath):
    Year_SubfolderPath = os.path.join(Residential_FolderPath, Year_Subfolder)
    # print(Year_SubfolderPath)

    for ClimateZone_Subfolder in os.listdir(Year_SubfolderPath):
        ClimateZone_SubfolderPath = os.path.join(Year_SubfolderPath, ClimateZone_Subfolder)
        # print('---' + ClimateZone_SubfolderPath)

        for FileName in os.listdir(ClimateZone_SubfolderPath):
            if FileName.endswith('.idf'):
                
                 IDF_FilePath = os.path.join(ClimateZone_SubfolderPath, FileName)
                 
                 IDF_FileName_Split = FileName.split('+')
                 Building_Prototype = IDF_FileName_Split[1] # SF or MF
                 ClimateZone = IDF_FileName_Split[2] # Example: CZ1AWH
                 HeatingType = IDF_FileName_Split[3] # Exapmple: elecres
                 FoundationType = IDF_FileName_Split[4] # Example: crawlspace
                 StandardYear = (IDF_FileName_Split[5].split('.'))[0] # Example: IECC_2015
                 
                 if ClimateZone == 'CZ1AWHT':
                     Location = 'Honolulu'
                 elif ClimateZone == 'CZ1AWH':
                     Location = 'Miami'
                 elif ClimateZone == 'CZ2AWH':
                     Location = 'Tampa'
                 elif ClimateZone == 'CZ3A':
                     Location = 'Atlanta'
                 elif ClimateZone == 'CZ3AWH':
                     Location = 'Mongomery'
                 elif ClimateZone == 'CZ3B':
                     Location = 'El.Paso'
                 elif ClimateZone == 'CZ3C':
                     Location = 'SanDiego'
                 elif ClimateZone == 'CZ4A':
                     Location = 'NewYork'
                 elif ClimateZone == 'CZ4B':
                     Location = 'Albuquerque'
                 elif ClimateZone == 'CZ4C':
                     Location = 'Seattle'
                 elif ClimateZone == 'CZ5A':
                     Location = 'Buffalo'
                 elif ClimateZone == 'CZ5B':
                     Location = 'Denver'
                 elif ClimateZone == 'CZ5C':
                     Location = 'Port.Angeles'
                 elif ClimateZone == 'CZ6A':
                     Location = 'Rochester'
                 elif ClimateZone == 'CZ6B':
                     Location = 'Great.Falls'
                 elif ClimateZone == 'CZ7':
                     Location = 'International.Falls'
                 else:
                     Location = 'Fairbanks'
                     
                 SimulationName = Building_Prototype + '_' + ClimateZone + '_' + Location + '_' + HeatingType + '_' + FoundationType + '_' + StandardYear
                 
                 print('Simulation Name: ' + SimulationName + '\n')
                 
                 # get path to weather file 
                 Weather_FolderPath = os.path.join(Residential_FolderPath, '..', 'TMY3_WeatherFiles_Residential')
                 for filename in os.listdir(Weather_FolderPath): 
                     if Location in filename: Weather_FileName = filename
                 Weather_FilePath = os.path.join(Weather_FolderPath, Weather_FileName)
                 
                 #print('Weather File: ' + Weather_FilePath + '\n')      
                 #print('IDF File: ' + IDF_FilePath + '\n')
                 
                 # Copying IDF and Weather Files to Temporary Folder
                 shutil.copy(IDF_FilePath, os.path.join(Current_FilePath, 'TemporaryFolder'))

                 shutil.copy(Weather_FilePath, os.path.join(Current_FilePath, 'TemporaryFolder'))
                 
                 # Getting Temporary IDF/Weather File Paths
                 Temporary_IDF_FilePath = os.path.join(Current_FilePath, 'TemporaryFolder', FileName)

                 Temporary_Weather_FilePath = os.path.join(Current_FilePath, 'TemporaryFolder', Weather_FileName)    
                
                 # =============================================================================
                 # Initial Run of the Building Simulation and Collecting Variable List
                 # =============================================================================      

                 if (SimulationType == 1):  # All required Files to be generated

                    # Initial Building Simulation Run
                    Initial_IDF_Run = op.simulate(Temporary_IDF_FilePath, Temporary_Weather_FilePath, base_dir_path = os.path.join(Current_FilePath,"TemporaryFolder"))

                    # Collecting Simulation Variable List
                    with open(os.path.join(Current_FilePath, 'TemporaryFolder', 'eplusout.rdd')) as f:
                        lines = f.readlines()
                        
                    Simulation_VariableNames = []  

                    Counter_Lines = 0

                    for line in lines:
                        if (Counter_Lines > 1):
                            split_line = line.split(',')
                            Simulation_VariableNames.append(split_line[2].split('[')[0])

                        Counter_Lines = Counter_Lines + 1
                        # Simulation_VariableNames.append(split_line[2])
                        # split_line_unit = split_line[3].split('[')[1]
                        # split_line_unit = split_line_unit[0].split(']')[0]
                        # Simulation_VariableNames.append(split_line_unit)

                    Simulation_VariableNames.sort()

                 elif (SimulationType == 2):  # Required variables generated
                        
                        Simulation_VariableNames = Required_VariableNames

                        Simulation_VariableNames.sort()

                 # =============================================================================
                 # Editing Saving Current IDF File and Saving Weather File in Results Folder
                 # =============================================================================        

                 if Location in SkipLocation or HeatingType in SkipHeatingType or FoundationType in SkipFoundationType: 
                     SkipSimulation.append(SimulationName)
                 
                 if SimulationName in SkipSimulation: 
                     print('Skipping Simulation \n')
                     
                 else:   
                     
                     try:
                         # Loading Current IDF File
                         Current_IDFFile = op.Epm.load(Temporary_IDF_FilePath)

                         # Editing RunPeriod
                         Current_IDF_RunPeriod = Current_IDFFile.RunPeriod.one()

                         Current_IDF_RunPeriod['begin_day_of_month'] = Sim_Start_Day

                         Current_IDF_RunPeriod['begin_month'] = Sim_Start_Month

                         Current_IDF_RunPeriod['end_day_of_month'] = Sim_End_Day

                         Current_IDF_RunPeriod['end_month' ]= Sim_End_Month

                         # Editing TimeStep
                         Current_IDF_TimeStep = Current_IDFFile.TimeStep.one()

                         Current_IDF_TimeStep['number_of_timesteps_per_hour'] = int(60/Sim_TimeStep)

                         # Getting Current Schedule
                         Current_ScheduleCompact = Current_IDFFile.Schedule_Compact

                         Current_ScheduleCompact_Records_Dict = Current_ScheduleCompact._records

                         # FOR LOOP:  For Editing Schedule in Schedule_Dict_List
                         for Schedule_Dicts in Schedule_Dict_List:
                    
                             # FOR LOOP: For Editing Schedules in Schedule_Dicts
                             for Schedule_Dict_Name in Schedule_Dicts:
                        
                                 # Getting Current Schedule Dict
                                 Schedule_Dict = Schedule_Dicts[Schedule_Dict_Name]
                        
                                 # Getting Current Schedule
                                 Current_Schedule = Current_ScheduleCompact_Records_Dict[Schedule_Dict_Name]
                        
                                 # FOR LOOP: For Editing Schedules in Schedule_Dict
                                 for Schedule_Fields in Schedule_Dict:
                            
                                     # Setting Fields in Current Schedule 
                                     Current_Schedule[Schedule_Fields] = Schedule_Dict[Schedule_Fields]

                         # Creating Edited IDF File
                         Edited_IDFFile = Current_IDFFile

                         # Making Additional Folders
                         Processed_BuildingSim_Data_FolderPath = os.path.join(Current_FilePath,  '..',  '..', 'Results', 'Processed_BuildingSim_Data')

                         Sim_IDFWeatherFiles_FolderName = 'Sim_IDFWeatherFiles'

                         Sim_OutputFiles_FolderName = 'Sim_OutputFiles'

                         Sim_IDFProcessedData_FolderName = 'Sim_ProcessedData'

                         # Checking if Folders Exist if not create Folders
                         if (os.path.isdir(os.path.join(Processed_BuildingSim_Data_FolderPath, SimulationName))):

                             # Folders Exist  
                             print('This Simulation has already been completed\n')  
                             z = None
                  
                         else:
                    
                             os.mkdir(os.path.join(Processed_BuildingSim_Data_FolderPath, SimulationName))
                    
                             os.mkdir(os.path.join(Processed_BuildingSim_Data_FolderPath, SimulationName, Sim_IDFWeatherFiles_FolderName))
                    
                             os.mkdir(os.path.join(Processed_BuildingSim_Data_FolderPath, SimulationName, Sim_OutputFiles_FolderName))
                    
                             os.mkdir(os.path.join(Processed_BuildingSim_Data_FolderPath, SimulationName, Sim_IDFProcessedData_FolderName))

                             # Saving Edited IDF and Weather File in Results Folder
                             Edited_IDFFile_FolderPath = os.path.join(Processed_BuildingSim_Data_FolderPath, SimulationName,  Sim_IDFWeatherFiles_FolderName)

                             Edited_IDFFile.save(os.path.join(Edited_IDFFile_FolderPath, "Edited_IDFFile.idf"))

                             shutil.copy(Weather_FilePath, Edited_IDFFile_FolderPath)


                             # =============================================================================
                             # Running Edited IDF File to get Output Variables and saving in Results Folder
                             # =============================================================================

                             # Getting Folder Paths
                             Edited_IDFFile_Path = os.path.join(Edited_IDFFile_FolderPath, "Edited_IDFFile.idf")

                             Special_IDFFile_Path = os.path.join(SpecialIDF_FolderPath, "Special.idf")

                             Edited_WeatherFile_Path = os.path.join(Edited_IDFFile_FolderPath, Weather_FileName)

                             Sim_OutputFiles_FolderPath = os.path.join(Processed_BuildingSim_Data_FolderPath, SimulationName, Sim_OutputFiles_FolderName)

                             Sim_IDFProcessedData_FolderPath = os.path.join(Processed_BuildingSim_Data_FolderPath, SimulationName, Sim_IDFProcessedData_FolderName)

                             # Appending Special IDF File into Edited IDF File
                             IDF_From = open(Special_IDFFile_Path, "r")
                             IDF_To = open(Edited_IDFFile_Path, "a")

                             Data = IDF_From.read()
                             IDF_To.write("\n")
                             IDF_To.write(Data)

                             IDF_From.close()
                             IDF_To.close()

                             # Loading the Edited IDF File
                             epm_Edited_IDFFile = op.Epm.load(Edited_IDFFile_Path)

                             # Loading the Special IDF File
                             epm_Special_IDFFile = op.Epm.load(Special_IDFFile_Path)

                             # Getting Output Variable from Edited IDF File
                             OutputVariable_QuerySet = epm_Edited_IDFFile.Output_Variable.one()               

                             for OutputVariable_Name in Simulation_VariableNames:

                                 #print('----- Output Variable: ' + OutputVariable_Name + '\n')

                                 OutputVariable_QuerySet['key_value'] = '*'

                                 OutputVariable_QuerySet['reporting_frequency'] = Sim_OutputVariable_ReportingFrequency

                                 OutputVariable_QuerySet['variable_name'] = OutputVariable_Name

                                 # Saving Special IDF File
                                 epm_Edited_IDFFile.save(os.path.join(Edited_IDFFile_FolderPath, "Edited_IDFFile.idf"))

                                 # Add conditional here to only simulate if csv has not already been created. 
                                 FileName_List = os.listdir(Sim_IDFProcessedData_FolderPath)
                             
                                 skip_variable = 0
                             
                                 if OutputVariable_Name in SkipVariable: 
                                     skip_variable = 1
                                     #print('----- Skipping Variable: ' + OutputVariable_Name + '\n')

                                 for FileName in FileName_List:
                                     if FileName.endswith('.csv') > 0 and FileName.count(OutputVariable_Name) > 0:
                                          #print('---------- Csv already created - skipped CSV \n')
                                          skip_variable = 1  
                                 
                                 if skip_variable == 0:
                                 
                                     # Running Building Simulation to obtain current output variable
                                     op.simulate(Edited_IDFFile_Path, Edited_WeatherFile_Path, base_dir_path = Sim_OutputFiles_FolderPath)

                                     # Moving Output Variable CSV file to Desired Folder
                                     try:

                                         Current_CSV_FilePath = os.path.join(Sim_OutputFiles_FolderPath, "eplusout.csv")

                                         New_OutputVariable_FileName = OutputVariable_Name.replace(' ','_') + '.csv'
                                        
                                         #print('----- CSV Created\n')

                                         MoveTo_CSV_FilePath = os.path.join(Sim_IDFProcessedData_FolderPath, New_OutputVariable_FileName)

                                         shutil.move(Current_CSV_FilePath, MoveTo_CSV_FilePath)

                                     except:

                                         #print('----- Failed to create CSV\n')
                                        
                                         continue


                             # =============================================================================
                             # Convert and Save Output Variables .csv to.mat in Results Folder
                             # =============================================================================    

                             if skip_variable == 0:
                                 # Getting all .csv Files paths from Sim_IDFProcessedData_FolderPath
                                 FileName_List = os.listdir(Sim_IDFProcessedData_FolderPath)

                                 # Initializing CSV_FileName_List
                                 CSV_FilePath_List = []

                                 # FOR LOOP: For each file in Sim_IDFProcessedData_FolderPath
                                 for file in FileName_List:
                    
                                     # Check only .csv files 
                                     if file.endswith('.csv'):
                        
                                         # Appending .csv file paths to CSV_FilePath_List
                                         CSV_FilePath_List.append(os.path.join(Sim_IDFProcessedData_FolderPath,file))

                                 # Initializing IDF_OutputVariable_Dict
                                 IDF_OutputVariable_Dict = {}

                                 IDF_OutputVariable_Full_Dict = {}

                                 IDF_OutputVariable_Full_DF = pd.DataFrame()

                                 IDF_OutputVariable_ColumnName_List = []

                                 Counter_OutputVariable = 0

                                 # FOR LOOP: For Each .csv File in CSV_FilePath_List
                                 for file_path in CSV_FilePath_List:
                    
                                     # Reading .csv file in dataframe
                                     Current_DF = pd.read_csv(file_path)

                                     # Getting CurrentDF_1
                                     if (Counter_OutputVariable == 0):
                        
                                         # Keeping DateTime Column
                                         Current_DF_1 = Current_DF
                        
                                     else:
                         
                                         # Dropping DateTime Column
                                         Current_DF_1=Current_DF.drop(Current_DF.columns[[0]],axis=1)
                            
                                     # Concatenating IDF_OutputVariable_Full_DF
                                     IDF_OutputVariable_Full_DF = pd.concat([IDF_OutputVariable_Full_DF,Current_DF_1], axis="columns")
                    
                                     # Appending Column Names to IDF_OutputVariable_ColumnName_List
                                     for ColumnName in Current_DF_1.columns:
                        
                                         IDF_OutputVariable_ColumnName_List.append(ColumnName)
                        
                                     # Getting File Name
                                     FileName = file_path.split('\\')[-1].split('_.')[0]
                    
                                     # Storing Current_DF in IDF_OutputVariable_Dict
                                     IDF_OutputVariable_Dict[FileName] = Current_DF
                    
                                     # Incrementing Counter_OutputVariable
                                     Counter_OutputVariable = Counter_OutputVariable + 1

                                 # Creating and saving DateTime to IDF_OutputVariable_Dict
                                 DateTime_List = []

                                 DateTime_Column = Current_DF['Date/Time']

                                 Datetime_counter = 0

                                 for DateTime in DateTime_Column:
                        
                                     Datetime_counter += 1

                                 print("Datetime Column: " + str(Datetime_counter))
                                 print("\n")
                        
                                 DateTime_Split = DateTime.split(' ')

                                 if(len(DateTime_Split) == 4): 
 
                                     Date_Split = DateTime_Split[1].split('/')
                            
                                     Time_Split = DateTime_Split[3].split(':')
                    
                                 elif(len(DateTime_Split) == 3):
 
                                     Date_Split = DateTime_Split[0].split('/')
                        
                                     Time_Split = DateTime_Split[2].split(':')

                                 elif(len(DateTime_Split) == 2):
                            
                                     Date_Split = DateTime_Split[0].split('/')
                        
                                     Time_Split = DateTime_Split[1].split(':')

                                 # Converting all 24th hour to 0th hour as hour must be in 0..23
                                 if int(Time_Split[0]) == 24:
                                     Time_Split[0] = 00
                        
                                 DateTime_List.append(datetime.datetime(IDF_FileYear,int(Date_Split[0]),int(Date_Split[1]),int(Time_Split[0]),int(Time_Split[1]),int(Time_Split[2])))

                                 IDF_OutputVariable_Dict['DateTime_List'] = DateTime_List
 
                                 IDF_OutputVariable_Full_Dict['IDF_OutputVariable_Full_DF'] = IDF_OutputVariable_Full_DF

                                 IDF_OutputVariable_Full_Dict['DateTime_List'] = DateTime_List

                                 pickle.dump(IDF_OutputVariable_Dict, open(os.path.join(Sim_IDFProcessedData_FolderPath,"IDF_OutputVariables_DictDF.pickle"), "wb"))

                                 # Writing and Saving Column Names to a Text FileS
                                 textfile = open(os.path.join(Sim_IDFProcessedData_FolderPath,"IDF_OutputVariable_ColumnName_List.txt"), "w")

                                 for ColumnName in IDF_OutputVariable_ColumnName_List:    
                    
                                     textfile.write(ColumnName + "\n")
                    
                                 textfile.close()

                

                             # =============================================================================
                             # Process .eio Output File and save in Results Folder
                             # =============================================================================    

                             if skip_variable == 0:
                                 # Reading .eio Output File
                                 Eio_OutputFile_Path = os.path.join(Sim_OutputFiles_FolderPath,'eplusout.eio') 

                                 # Initializing Eio_OutputFile_Dict
                                 Eio_OutputFile_Dict = {}
 
                                 with open(Eio_OutputFile_Path) as f:
                                     Eio_OutputFile_Lines = f.readlines()

                                 # Removing Intro Lines
                                 Eio_OutputFile_Lines = Eio_OutputFile_Lines[1:]

                                 Category_Key = ""
                                 Category_Key_List = ["Zone Information", "Zone Internal Gains Nominal", "People Internal Gains Nominal", "Lights Internal Gains Nominal", "ElectricEquipment Internal Gains Nominal", "GasEquipment Internal Gains Nominal", "HotWaterEquipment Internal Gains Nominal", "SteamEquipment Internal Gains Nominal", "OtherEquipment Internal Gains Nominal" ]
                                 Is_Table_Header = 0

                                 #Counting number of EIO file tables
                                 eiotable_count = 0

                                 # FOR LOOP: For each category in .eio File
                                 for Line_1 in Eio_OutputFile_Lines:

                                     #Check if Line contains table Header
                                     for Item in Category_Key_List:
                                         Category_Key = Item
                                         if ((Line_1.find(Item) >= 0) and (Line_1.find('!') >= 0)):
                                             Is_Table_Header = 1
                                             break 
                                         else:
                                             Is_Table_Header = 0
                        
                                     # IF ELSE LOOP: To check category
                                     # Code inside this if/else did not change meaningfully.
                                     if (Is_Table_Header > 0):
                        
                                         #print("EIO Table: ", Category_Key)
                                         #print("\n")
                                         #print(Line_1 + '\n')

                                         eiotable_count += 1
                                         #print("Eio Tables: ", eiotable_count, "/9")
                                         #print("\n")

                                         # Get the Column Names for the .eio File category
                                         DF_ColumnName_List = Line_1.split(',')[1:]

                                         # Removing the '\n From the Last Name
                                         DF_ColumnName_List[-1] = DF_ColumnName_List[-1].split('\n')[0]
 
                                         # Removing Empty Element
                                         if DF_ColumnName_List[-1] == ' ':
                                             DF_ColumnName_List = DF_ColumnName_List[:-1]

                                         # Initializing DF_Index_List
                                         DF_Index_List = []

                                         # Initializing DF_Data_List
                                         DF_Data_List = []

                                         # FOR LOOP: For all elements of current .eio File category
                                         for Line_2 in Eio_OutputFile_Lines:

                                             # IF ELSE LOOP: To check data row belongs to current Category
                                             if ((Line_2.find('!') == -1) and (Line_2.find(Category_Key) >= 0)):

                                                 #print(Line_2 + '\n')

                                                 DF_ColumnName_List_Length = len(DF_ColumnName_List)

                                                 # Split Line_2
                                                 Line_2_Split = Line_2.split(',')

                                                 # Removing the '\n From the Last Data
                                                 Line_2_Split[-1] = Line_2_Split[-1].split('\n')[0]

                                                 # Removing Empty Element
                                                 if Line_2_Split[-1] == ' ':
                                                     Line_2_Split = Line_2_Split[:-1]

                                                 # Getting DF_Index_List element
                                                 DF_Index_List.append(Line_2_Split[0])

                                                 Length_Line2 = len(Line_2_Split[1:])

                                                 Line_2_Split_1 = Line_2_Split[1:]
 
                                                 # Filling up Empty Column
                                                 if Length_Line2 < DF_ColumnName_List_Length:
                                                     Len_Difference = DF_ColumnName_List_Length - Length_Line2

                                                     for ii in range(Len_Difference):
                                                         Line_2_Split_1.append('NA')

                                                     # Getting DF_Data_List element
                                                     DF_Data_List.append(Line_2_Split_1)

                                                 else:
                                                     # Getting DF_Data_List element
                                                     DF_Data_List.append(Line_2_Split[1:])

                                             else:

                                                 continue

                                         # Creating DF_Table
                                         DF_Table = pd.DataFrame(DF_Data_List, index=DF_Index_List, columns=DF_ColumnName_List)

                                         # Adding DF_Table to the Eio_OutputFile_Dict
                                         Eio_OutputFile_Dict[Category_Key] = DF_Table

                                     else:

                                         continue
                    
                                 # Saving Eio_OutputFile_Dict as a .pickle File in Results Folder
                                 pickle.dump(Eio_OutputFile_Dict, open(os.path.join(Sim_IDFProcessedData_FolderPath,"Eio_OutputFile.pickle"), "wb"))
                 
                                 #print('----- Eio Output File Created \n')

                                 # Saving Eio_OutputFile_Dict as a .mat File in Results Folder
                                 # scipy.io.savemat(os.path.join(Sim_IDFProcessedData_FolderPath,"Eio_OutputFile.mat"), Eio_OutputFile_Dict)


                             # =============================================================================
                             # Deleting all files from Temporary Folder
                             # =============================================================================    

                             # Getting Temporary Folder Path
                             Temporary_FolderPath = os.path.join(Current_FilePath, 'TemporaryFolder')

                             # Deleting all files and sub-folders in Temporary Folder
                             for files in os.listdir(Temporary_FolderPath):
                    
                                 path = os.path.join(Temporary_FolderPath, files)
                    
                                 try:
                        
                                     shutil.rmtree(path)
                        
                                 except OSError:
                        
                                     os.remove(path)

                             # =============================================================================
                             # Deleting CSV's
                             # =============================================================================   
                         
                             if skip_variable == 0:
                                 # FOR LOOP: For Each .csv File in CSV_FilePath_List
                                 for file_path in CSV_FilePath_List:
                                     os.remove(file_path)                
                    
                                 print('CSV Files Deleted \n')  
                                 
                     except Exception as e:
                        print('Failed to Simulate: ' + str(e))    
                    
                 
                 


    
    





