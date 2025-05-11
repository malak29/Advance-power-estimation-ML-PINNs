# =============================================================================
# Import Required Modules
# =============================================================================

# External Modules
import os
import pickle
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from tensorflow import keras
from keras import layers
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import joblib
import datetime
import copy
import math
import time

# =============================================================================
# User Inputs
# =============================================================================
Simulation_Name = "test1"

Shoter_ResultsPath = True  # True - Results are stored in shorter path , False - Results are stored in the appropriate directory

Short_ResultFolder = 'Results_1' # Name of the Folder where results will be stored

ANN_Type = 4  # 1 - ANN , 2 - Simple RNN , 3 - LSTM , 4 - GRU

Lag_Number = 1 # Number of lags for input X to predict one step future Y

TimeStepper_ModelType = 2  # 1 - Model Predicts Next State Directly ; 2 - Model Computes Current Derivative

NeuralStateSpace_ModelType = 2 # 1 - Model is not Decomposed into A(x) and B(u) ; 2 - Model is Decomposed into A(x) and B(u)

Learning_Rate = 0.001

Loss_Function = 'mean_squared_error'

Epochs = 2

Batch_Size = 20

Buffer_Input = 100

Validation_Split = 0.2

Early_Stopping_Patience = 5

## User Input: Aggregation Unit Number ##
Complete_Aggregation_Number = 1

Total_Aggregation_Zone_Number = 1

# FeatureType = 0 # 0 - Remove no features, 1 - Remove Internal Heat, 2 - Remove Solar Heat, 3 - Remove Ambient Temp, 4 - Remove HVAC Heat, 5 - Remove Zone Temperature, 6 - Remove all but Zone Temperature, internal heat

# Aggregation Zone NameStem Input
Aggregation_Zone_NameStem = 'Aggregation_Zone'

# Keras extention
kerasExtention = '.keras'

# Percentage Training Data to be used
Training_Data_Control = 1 # 0 = All Data, 1 = Not All Data
Training_Data_Percentage_Used = 0.001 # Values Between 0 and 1
Testing_Points = 100
Training_Points = 1000

if (ANN_Type == 1):

    # Renaming Ann ModelKey
    ANNModel_Key =  "MLP_L_" + str(Lag_Number) + "_TS_" + str(TimeStepper_ModelType) + "_NSS_" + str(NeuralStateSpace_ModelType)

elif (ANN_Type == 2):

    # Renaming Ann ModelKey
    ANNModel_Key =  "SimpleRNN_" + str(Lag_Number) + "_TS_" + str(TimeStepper_ModelType) + "_NSS_" + str(NeuralStateSpace_ModelType)

elif (ANN_Type == 3):

    # Renaming Ann ModelKey
    ANNModel_Key =  "LSTM_" + str(Lag_Number) + "_TS_" + str(TimeStepper_ModelType) + "_NSS_" + str(NeuralStateSpace_ModelType)

elif (ANN_Type == 4):

    # Renaming Ann ModelKey
    ANNModel_Key =  "GRU_" + str(Lag_Number) + "_TS_" + str(TimeStepper_ModelType) + "_NSS_" + str(NeuralStateSpace_ModelType)

# Creating Training Folder Name
Training_FolderName = ANNModel_Key

# =============================================================================
# Initialization
# =============================================================================

PHVAC = np.zeros((1,1))

PHVAC1 = np.zeros((1,1))

# =============================================================================
# Getting Required Data from Sim_ProcessedData
# =============================================================================

# Getting Current File Directory Path
Current_FilePath = os.path.dirname(__file__)

# Getting Folder Path
Sim_ProcessedData_FolderPath_AggregatedTestTrain = os.path.join(Current_FilePath, '..', '..', 'Results',
                                                                'Processed_BuildingSim_Data', Simulation_Name,
                                                                'Sim_TrainingTestingData')
Sim_ProcessedData_FolderPath_Regression = os.path.join(Current_FilePath, '..', '..', 'Results',
                                                       'Processed_BuildingSim_Data', Simulation_Name,
                                                       'Sim_RegressionModelData')

# Initializing Train_X, Train_Y, Test_X, Test_Y
Train_X = copy.deepcopy(pd.DataFrame())
Train_Y = copy.deepcopy(pd.DataFrame())
Test_X = copy.deepcopy(pd.DataFrame())
Test_Y = copy.deepcopy(pd.DataFrame())

Train_X_QHVAC = copy.deepcopy(pd.DataFrame())
Test_X_QHVAC = copy.deepcopy(pd.DataFrame())

Test_X_M_Dot_Ts = copy.deepcopy(pd.DataFrame())

Predict_Actual_Y_DF = copy.deepcopy(pd.DataFrame())
Predict_Actual_Y_DF1 = copy.deepcopy(pd.DataFrame())

PHVAC_Regression_Model_List = []

# LOOP: Output Generation for Each Aggregated Zone

for kk in range(Total_Aggregation_Zone_Number):

    kk = kk + 1

    # Creating Required File Names

    Aggregation_DF_Test_File_Name = 'Aggregation_DF_Test_Aggregation_Dict_' + str(Total_Aggregation_Zone_Number) + 'Zone_' + str(kk) + '.pickle'

    Aggregation_DF_Train_File_Name = 'Aggregation_DF_Train_Aggregation_Dict_' + str(Total_Aggregation_Zone_Number) + 'Zone_' + str(kk) + '.pickle'

    ANN_HeatInput_Test_DF_File_Name = 'ANN_HeatInput_Test_DF_' + str(Total_Aggregation_Zone_Number) + 'Zone_' + str(kk) + '.pickle'

    ANN_HeatInput_Train_DF_File_Name = 'ANN_HeatInput_Train_DF_' + str(Total_Aggregation_Zone_Number) + 'Zone_' + str(kk) + '.pickle'

    # PHVAC_Regression_Model_File_Name = 'QAC' + str(Total_Aggregation_Zone_Number) + 'Zone_' + str(kk)
    PHVAC_Regression_Model_File_Name = 'PHVAC_RegressionModel_' + str(Total_Aggregation_Zone_Number) + 'Zone_' + str(kk)

    # Get Required Files from Sim_AggregatedTestTrainData_FolderPath

    #IF Loop to Control the Amount of Training Data
    if (Training_Data_Control == 1):

        AggregatedTest_Dict_File = open(
            os.path.join(Sim_ProcessedData_FolderPath_AggregatedTestTrain, Aggregation_DF_Test_File_Name), "rb")
        AggregatedTest_DF = pickle.load(AggregatedTest_Dict_File)

        AggregatedTest_DF = AggregatedTest_DF[0:Testing_Points]

        AggregatedTrain_Dict_File = open(
            os.path.join(Sim_ProcessedData_FolderPath_AggregatedTestTrain, Aggregation_DF_Train_File_Name), "rb")
        AggregatedTrain_DF = pickle.load(AggregatedTrain_Dict_File)

        # AggregatedTrain_DF = AggregatedTrain_DF[0:math.floor(Training_Data_Percentage_Used*len(AggregatedTrain_DF))]

        AggregatedTrain_DF = AggregatedTrain_DF[0:Training_Points]

        PHVAC_Regression_Model_File_Path = os.path.join(Sim_ProcessedData_FolderPath_Regression,
                                                        PHVAC_Regression_Model_File_Name)
        # PHVAC_Regression_Model = tf.keras.models.load_model(PHVAC_Regression_Model_File_Path)
        PHVAC_Regression_Model = keras.layers.TFSMLayer(PHVAC_Regression_Model_File_Path, call_endpoint='serving_default')

        PHVAC_Regression_Model_List.append(PHVAC_Regression_Model)

        # Get Required Files from Sim_RegressionModelData_FolderPath
        ANN_HeatInput_Test_DF_File = open(
            os.path.join(Sim_ProcessedData_FolderPath_Regression, ANN_HeatInput_Test_DF_File_Name),
            "rb")
        ANN_HeatInput_Test_DF = pickle.load(ANN_HeatInput_Test_DF_File)

        ANN_HeatInput_Test_DF = ANN_HeatInput_Test_DF[0:Testing_Points]

        ANN_HeatInput_Train_DF_File = open(
            os.path.join(Sim_ProcessedData_FolderPath_Regression, ANN_HeatInput_Train_DF_File_Name), "rb")
        ANN_HeatInput_Train_DF = pickle.load(ANN_HeatInput_Train_DF_File)

        # ANN_HeatInput_Train_DF = ANN_HeatInput_Train_DF[0:math.floor(Training_Data_Percentage_Used*len(ANN_HeatInput_Train_DF))]

        ANN_HeatInput_Train_DF = ANN_HeatInput_Train_DF[0:Training_Points]

    else:

        AggregatedTest_Dict_File = open(
            os.path.join(Sim_ProcessedData_FolderPath_AggregatedTestTrain, Aggregation_DF_Test_File_Name), "rb")
        AggregatedTest_DF = pickle.load(AggregatedTest_Dict_File)

        AggregatedTrain_Dict_File = open(
            os.path.join(Sim_ProcessedData_FolderPath_AggregatedTestTrain, Aggregation_DF_Train_File_Name), "rb")
        AggregatedTrain_DF = pickle.load(AggregatedTrain_Dict_File)

        PHVAC_Regression_Model_File_Path = os.path.join(Sim_ProcessedData_FolderPath_Regression, PHVAC_Regression_Model_File_Name)
        PHVAC_Regression_Model = tf.keras.models.load_model(PHVAC_Regression_Model_File_Path)
        # PHVAC_Regression_Model = keras.layers.TFSMLayer(PHVAC_Regression_Model_File_Path, call_endpoint='serving_default')

        PHVAC_Regression_Model_List.append(PHVAC_Regression_Model)

        # Get Required Files from Sim_RegressionModelData_FolderPath
        ANN_HeatInput_Test_DF_File = open(os.path.join(Sim_ProcessedData_FolderPath_Regression, ANN_HeatInput_Test_DF_File_Name),
                                          "rb")
        ANN_HeatInput_Test_DF = pickle.load(ANN_HeatInput_Test_DF_File)

        ANN_HeatInput_Train_DF_File = open(
            os.path.join(Sim_ProcessedData_FolderPath_Regression, ANN_HeatInput_Train_DF_File_Name), "rb")
        ANN_HeatInput_Train_DF = pickle.load(ANN_HeatInput_Train_DF_File)

    # =============================================================================
    # Creating Sim_ANNModelData Folder
    # =============================================================================

    if (Shoter_ResultsPath == False):

        # Making Additional Folders for storing Aggregated Files
        Processed_BuildingSim_Data_FolderPath = os.path.join(Current_FilePath, '..', '..', 'Results',
                                                            'Processed_BuildingSim_Data')
            
        Sim_ANNModelData_FolderName = 'Sim_ANNModelData'      
    

        # Checking if Folders Exist if not create Folders
        if (
                os.path.isdir(
                    os.path.join(Processed_BuildingSim_Data_FolderPath, Simulation_Name, Sim_ANNModelData_FolderName))):

            # Folders Exist
            z = None

        else:

            os.mkdir(os.path.join(Processed_BuildingSim_Data_FolderPath, Simulation_Name, Sim_ANNModelData_FolderName))

        # Make the Training Folder
        os.mkdir(os.path.join(Processed_BuildingSim_Data_FolderPath, Simulation_Name, Sim_ANNModelData_FolderName, Training_FolderName))

        # Creating Sim_RegressionModelData Folder Path
        Sim_ANNModelData_FolderPath = os.path.join(Processed_BuildingSim_Data_FolderPath, Simulation_Name,
                                                Sim_ANNModelData_FolderName, Training_FolderName) 
    
    elif (Shoter_ResultsPath == True):
    
        ## Shorter Path
        
        Processed_BuildingSim_Data_FolderPath = os.path.join(Current_FilePath, '..', '..')

        Sim_ANNModelData_FolderName = Short_ResultFolder

        # Make the Training Folder
        if(not os.path.exists(os.path.join(Processed_BuildingSim_Data_FolderPath, Sim_ANNModelData_FolderName, Training_FolderName))):
            os.mkdir(os.path.join(Processed_BuildingSim_Data_FolderPath, Sim_ANNModelData_FolderName, Training_FolderName))

        # Creating Sim_RegressionModelData Folder Path
        Sim_ANNModelData_FolderPath = os.path.join(Processed_BuildingSim_Data_FolderPath,
                                                Sim_ANNModelData_FolderName, Training_FolderName)

    # =============================================================================
    # Basic Computation
    # =============================================================================
        
    # Getting DateTime Data
    DateTime_Train = AggregatedTrain_DF['DateTime']
    DateTime_Test = AggregatedTest_DF['DateTime']
    
    # Resetting
    ANN_HeatInput_Train_DF.reset_index(drop=True, inplace=True)
    ANN_HeatInput_Test_DF.reset_index(drop=True, inplace=True)

    # Computing QZic and QZir Train

    # Initialization
    QAC_Test = []
    QAC_Train = []

    # FOR LOOP: Getting Summation
    for ii in range(ANN_HeatInput_Train_DF.shape[0]):

        QAC_Train_1 = AggregatedTrain_DF['QHVAC_X'].iloc[ii]

        QAC_Train.append(QAC_Train_1)

    ANN_HeatInput_Train_DF.insert(2, 'QAC_Corrected'+str(kk), QAC_Train)

    # FOR LOOP: Getting Summation
    for ii in range(ANN_HeatInput_Test_DF.shape[0]):

        QAC_Test_1 = AggregatedTest_DF['QHVAC_X'].iloc[ii]

        QAC_Test.append(QAC_Test_1)

    ANN_HeatInput_Test_DF.insert(2, 'QAC_Corrected'+str(kk), QAC_Test)

    AggregatedTrain_DF.rename({'Zone_Air_Temperature_':'Zone_Air_Temperature_'+str(kk), 'System_Node_Temperature_':'System_Node_Temperature_'+str(kk), 'System_Node_Mass_Flow_Rate_':'System_Node_Mass_Flow_Rate_'+str(kk)}, axis=1, inplace=True)
    AggregatedTest_DF.rename({'Zone_Air_Temperature_': 'Zone_Air_Temperature_' + str(kk), 'System_Node_Temperature_':'System_Node_Temperature_'+str(kk), 'System_Node_Mass_Flow_Rate_':'System_Node_Mass_Flow_Rate_'+str(kk)}, axis=1, inplace=True)

    # Training and Testing X and Y

    ANN_HeatInput_Train_DF.reset_index(drop=True, inplace=True)
    AggregatedTrain_DF.reset_index(drop=True, inplace=True)
    Train_X = pd.concat([Train_X, ANN_HeatInput_Train_DF[['QAC_Corrected'+str(kk)]].iloc[:-1, :],AggregatedTrain_DF[['Zone_Air_Temperature_'+str(kk)]].iloc[:-1, :]], axis=1)
    Train_Y = pd.concat([Train_Y, AggregatedTrain_DF['Zone_Air_Temperature_'+str(kk)].iloc[1:]], axis=1)

    AggregatedTest_DF.reset_index(drop=True, inplace=True)
    ANN_HeatInput_Test_DF.reset_index(drop=True, inplace=True)
    Test_X = pd.concat([Test_X, ANN_HeatInput_Test_DF[['QAC_Corrected'+str(kk)]].iloc[:-1, :],AggregatedTest_DF[['Zone_Air_Temperature_'+str(kk)]].iloc[:-1, :]], axis=1)
    Test_Y = pd.concat([Test_Y, AggregatedTest_DF['Zone_Air_Temperature_'+str(kk)].iloc[1:]], axis=1)

    Train_X_QHVAC = pd.concat([Train_X_QHVAC,AggregatedTrain_DF[['System_Node_Temperature_'+str(kk),'System_Node_Mass_Flow_Rate_'+str(kk)]].iloc[1:]], axis=1)
    Test_X_QHVAC = pd.concat([Test_X_QHVAC,AggregatedTest_DF[['System_Node_Temperature_'+str(kk),'System_Node_Mass_Flow_Rate_'+str(kk)]].iloc[1:]], axis=1)

    # Getting Test_Y for MDot and Ts
    Test_X_M_Dot_Ts = pd.concat([Test_X_M_Dot_Ts,AggregatedTest_DF['System_Node_Mass_Flow_Rate_'+str(kk)].iloc[:-1,], AggregatedTest_DF['System_Node_Temperature_'+str(kk)].iloc[:-1,] ], axis=1)
    
## We add T_am through the Aggregated File
#Train_X = pd.concat([Train_X, AggregatedTrain_DF['Site_Outdoor_Air_Drybulb_Temperature_'].iloc[:-1]], axis=1)
#Test_X = pd.concat([Test_X, AggregatedTest_DF['Site_Outdoor_Air_Drybulb_Temperature_'].iloc[:-1]], axis=1)

# Getting DateTime Data
DateTime_Train = AggregatedTrain_DF['DateTime']
DateTime_Test = AggregatedTest_DF['DateTime']

# =============================================================================
# Augmenting with the Complete Averaged Data
# =============================================================================

# Creating Required File Names

Aggregation_DF_Test_File_Name_Avg = 'Aggregation_DF_Test_Aggregation_Dict_' + str(Complete_Aggregation_Number) + 'Zone_' + str(Complete_Aggregation_Number) + '.pickle'

Aggregation_DF_Train_File_Name_Avg = 'Aggregation_DF_Train_Aggregation_Dict_' + str(Complete_Aggregation_Number) + 'Zone_' + str(Complete_Aggregation_Number) + '.pickle'

ANN_HeatInput_Test_DF_File_Name_Avg = 'ANN_HeatInput_Test_DF_' + str(Complete_Aggregation_Number) + 'Zone_' + str(Complete_Aggregation_Number) + '.pickle'

ANN_HeatInput_Train_DF_File_Name_Avg = 'ANN_HeatInput_Train_DF_' + str(Complete_Aggregation_Number) + 'Zone_' + str(Complete_Aggregation_Number) + '.pickle'

# Get Required Files from Sim_AggregatedTestTrainData_FolderPath
AggregatedTest_Dict_File_Avg = open(
    os.path.join(Sim_ProcessedData_FolderPath_AggregatedTestTrain, Aggregation_DF_Test_File_Name_Avg), "rb")
AggregatedTest_DF_Avg = pickle.load(AggregatedTest_Dict_File_Avg)

AggregatedTrain_Dict_File_Avg = open(
    os.path.join(Sim_ProcessedData_FolderPath_AggregatedTestTrain, Aggregation_DF_Train_File_Name_Avg), "rb")
AggregatedTrain_DF_Avg = pickle.load(AggregatedTrain_Dict_File_Avg)

# Get Required Files from Sim_RegressionModelData_FolderPath
ANN_HeatInput_Test_DF_File_Avg = open(os.path.join(Sim_ProcessedData_FolderPath_Regression, ANN_HeatInput_Test_DF_File_Name_Avg),
                                  "rb")
ANN_HeatInput_Test_DF_Avg = pickle.load(ANN_HeatInput_Test_DF_File_Avg)

ANN_HeatInput_Train_DF_File_Avg = open(
    os.path.join(Sim_ProcessedData_FolderPath_Regression, ANN_HeatInput_Train_DF_File_Name_Avg), "rb")
ANN_HeatInput_Train_DF_Avg = pickle.load(ANN_HeatInput_Train_DF_File_Avg)

if (Training_Data_Control == 1):
    AggregatedTest_DF_Avg = AggregatedTest_DF_Avg[0:Testing_Points]
    AggregatedTrain_DF_Avg = AggregatedTrain_DF_Avg[0:Training_Points]

    ANN_HeatInput_Train_DF_Avg = ANN_HeatInput_Train_DF_Avg[0:Training_Points]
    ANN_HeatInput_Test_DF_Avg = ANN_HeatInput_Test_DF_Avg[0:Testing_Points]

ANN_HeatInput_Train_DF_Avg.reset_index(drop=True, inplace=True)
ANN_HeatInput_Test_DF_Avg.reset_index(drop=True, inplace=True)
AggregatedTest_DF_Avg.reset_index(drop=True, inplace=True)
AggregatedTrain_DF_Avg.reset_index(drop=True, inplace=True)

# Computing QZic and QZir Train

# Initialization
QZic_Train = []
QZir_Train = []
QZic_Test = []
QZir_Test = []
QSol1_Test = []
QSol1_Train = []
QSol2_Test = []
QSol2_Train = []


# FOR LOOP: Getting Summation
for ii in range(ANN_HeatInput_Train_DF_Avg.shape[0]):
    print(str(ii) + '\n')
    QZic_Train_1 = ANN_HeatInput_Train_DF_Avg['QZic_P'][ii][0] + ANN_HeatInput_Train_DF_Avg['QZic_L'][ii][0] + \
                   ANN_HeatInput_Train_DF_Avg['QZic_EE'][ii][0]
    QZir_Train_1 = ANN_HeatInput_Train_DF_Avg['QZir_P'][ii][0] + ANN_HeatInput_Train_DF_Avg['QZir_L'][ii][0] + \
                   ANN_HeatInput_Train_DF_Avg['QZir_EE'][ii][0] + ANN_HeatInput_Train_DF_Avg['QZivr_L'][ii][0]
    QZic_Train.append(QZic_Train_1)
    QZir_Train.append(QZir_Train_1)

    QSol1_Train_1 = ANN_HeatInput_Train_DF_Avg['QSol1'][ii][0]
    QSol2_Train_1 = ANN_HeatInput_Train_DF_Avg['QSol2'][ii][0]

    QSol1_Train.append(QSol1_Train_1)
    QSol2_Train.append(QSol2_Train_1)


ANN_HeatInput_Train_DF_Avg.insert(2, 'QZic', QZic_Train)
ANN_HeatInput_Train_DF_Avg.insert(2, 'QZir', QZir_Train)
ANN_HeatInput_Train_DF_Avg.insert(2, 'QSol1_Corrected', QSol1_Train)
ANN_HeatInput_Train_DF_Avg.insert(2, 'QSol2_Corrected', QSol2_Train)

# FOR LOOP: Getting Summation
for ii in range(ANN_HeatInput_Test_DF_Avg.shape[0]):
    QZic_Test_1 = ANN_HeatInput_Test_DF_Avg['QZic_P'][ii][0] + ANN_HeatInput_Test_DF_Avg['QZic_L'][ii][0] + \
                  ANN_HeatInput_Test_DF_Avg['QZic_EE'][ii][0]
    QZir_Test_1 = ANN_HeatInput_Test_DF_Avg['QZir_P'][ii][0] + ANN_HeatInput_Test_DF_Avg['QZir_L'][ii][0] + \
                  ANN_HeatInput_Test_DF_Avg['QZir_EE'][ii][0] + ANN_HeatInput_Test_DF_Avg['QZivr_L'][ii][0]
    QZic_Test.append(QZic_Test_1)
    QZir_Test.append(QZir_Test_1)

    QSol1_Test_1 = ANN_HeatInput_Test_DF_Avg['QSol1'][ii][0]
    QSol2_Test_1 = ANN_HeatInput_Test_DF_Avg['QSol2'][ii][0]

    QSol1_Test.append(QSol1_Test_1)
    QSol2_Test.append(QSol2_Test_1)

ANN_HeatInput_Test_DF_Avg.insert(2, 'QZic', QZic_Test)
ANN_HeatInput_Test_DF_Avg.insert(2, 'QZir', QZir_Test)
ANN_HeatInput_Test_DF_Avg.insert(2, 'QSol1_Corrected', QSol1_Test)
ANN_HeatInput_Test_DF_Avg.insert(2, 'QSol2_Corrected', QSol2_Test)

# Training and Testing X and Y
Train_X = pd.concat(
    [Train_X, ANN_HeatInput_Train_DF_Avg[['QSol1_Corrected', 'QSol2_Corrected', 'QZic', 'QZir']].iloc[:-1, :],
     AggregatedTrain_DF_Avg[['Site_Outdoor_Air_Drybulb_Temperature_']].iloc[:-1, :]],axis=1)


AggregatedTest_DF_Avg.reset_index(drop=True, inplace=True)
ANN_HeatInput_Test_DF_Avg.reset_index(drop=True, inplace=True)
Test_X = pd.concat(
    [Test_X, ANN_HeatInput_Test_DF_Avg[['QSol1_Corrected', 'QSol2_Corrected', 'QZic', 'QZir']].iloc[:-1, :],
     AggregatedTest_DF_Avg[['Site_Outdoor_Air_Drybulb_Temperature_']].iloc[:-1, :]], axis=1)


# =============================================================================
# ANN Data Formating
# =============================================================================

# Initializing Dataframe for Percentage Accuracy
ANN_PercentageAccuracy_DF = pd.DataFrame(columns=['ANN Model Name', 'Training Mean Error', 'Testing Mean Error Without Simulation', 'Testing Mean Error With Simulation', 'Final Train Error', 'Final Train Loss', 'Final Val Error', 'Final Val Loss', 'Time/Iteration'])

if (ANN_Type == 1):        

    # Converting Train_X and Train_Y into Tensor
    Train_X_TF_WithoutLag = tf.convert_to_tensor(Train_X)
    Train_Y_TF_1 = tf.convert_to_tensor(Train_Y)

    # Converting Text_X into Tensor
    Test_X_TF_WithoutLag = tf.convert_to_tensor(Test_X)
    Test_Y_TF = tf.convert_to_tensor(Test_Y)

    # Converting Test_X_M_Dot_Ts to Tensor
    Test_X_M_Dot_Ts_TF_WithoutLag = tf.convert_to_tensor(Test_X_M_Dot_Ts)

    # Creating Train/Test X's based on lags        
    for ii in range(Train_X_TF_WithoutLag.shape[0]-(Lag_Number-1)):

        for jj in range(Lag_Number):
            if (jj==0):
                Current_TF = tf.reshape(Train_X_TF_WithoutLag[ii+jj,:],[1,Train_X_TF_WithoutLag.shape[1]])
            else:
                Current_TF = tf.concat([Current_TF,tf.reshape(Train_X_TF_WithoutLag[ii+jj,:],[1,Train_X_TF_WithoutLag.shape[1]])],axis=1)

        if (ii==0):
            Train_X_TF_1 = Current_TF
        else:
            Train_X_TF_1 = tf.concat([Train_X_TF_1,Current_TF],axis=0)

    for ii in range(Test_X_TF_WithoutLag.shape[0]-(Lag_Number-1)):
        
        for jj in range(Lag_Number):
            if (jj==0):
                Current_TF = tf.reshape(Test_X_TF_WithoutLag[ii+jj,:],[1,Test_X_TF_WithoutLag.shape[1]])
            else:
                Current_TF = tf.concat([Current_TF,tf.reshape(Test_X_TF_WithoutLag[ii+jj,:],[1,Test_X_TF_WithoutLag.shape[1]])],axis=1)

        if (ii==0):
            Test_X_TF = Current_TF
        else:
            Test_X_TF = tf.concat([Test_X_TF,Current_TF],axis=0)

    for ii in range(Test_X_M_Dot_Ts_TF_WithoutLag.shape[0]-(Lag_Number-1)):
        
        for jj in range(Lag_Number):
            if (jj==0):
                Current_TF = tf.reshape(Test_X_M_Dot_Ts_TF_WithoutLag[ii+jj,:],[1,Test_X_M_Dot_Ts_TF_WithoutLag.shape[1]])
            else:
                Current_TF = tf.concat([Current_TF,tf.reshape(Test_X_M_Dot_Ts_TF_WithoutLag[ii+jj,:],[1,Test_X_M_Dot_Ts_TF_WithoutLag.shape[1]])],axis=1)

        if (ii==0):
            Test_X_M_Dot_Ts_TF = Current_TF
        else:
            Test_X_M_Dot_Ts_TF = tf.concat([Test_X_M_Dot_Ts_TF,Current_TF],axis=0)
    
    # Creating Training and Validation Sets
    Train_X_TF = Train_X_TF_1[0:math.floor(Train_X_TF_1.shape[0] * (1 - Validation_Split)),:]
    Train_Y_TF = Train_Y_TF_1[0:math.floor(Train_Y_TF_1.shape[0] * (1 - Validation_Split))]

    Train_Index = Train_X_TF.shape[0]

    Val_X_TF = Train_X_TF_1[Train_Index:Train_Index + math.floor(Train_X_TF_1.shape[0] * (Validation_Split)),:]
    Val_Y_TF = Train_Y_TF_1[Train_Index:Train_Index + math.floor(Train_Y_TF_1.shape[0] * (Validation_Split))]

    # Getting Approprite Indices for training data for A(x) and B(u)
    ALL_Indices = range(0,Train_X_TF.shape[1],1)

    X_TF_A_Indices = []

    Q_HVAC_A_Indices = []

    Q_HVAC_B_Indices = []

    for ii in range(Lag_Number):

        for jj in range(Total_Aggregation_Zone_Number):
    
            if ((ii == 0) and (jj ==0) ):
        
                X_TF_A_Indices.append(1)

                Q_HVAC_A_Indices.append(0)

                Q_HVAC_B_Indices.append(0)
        
            elif ((ii != 0) and (jj == 0)):
        
                X_TF_A_Indices.append(X_TF_A_Indices[-1]+7)

                Q_HVAC_A_Indices.append(Q_HVAC_A_Indices[-1]+7)

                Q_HVAC_B_Indices.append(Q_HVAC_B_Indices[-1]+6)
        
            else:
        
                X_TF_A_Indices.append(X_TF_A_Indices[-1]+2)

                Q_HVAC_A_Indices.append(Q_HVAC_A_Indices[-1]+2)

                Q_HVAC_B_Indices.append(Q_HVAC_B_Indices[-1]+1)
        
    X_TF_B_Indices = list(set(ALL_Indices).difference(set(X_TF_A_Indices)))

    if (NeuralStateSpace_ModelType == 1):  # Not Nueral State Space

        # Prepare the Training Dataset.
        train_dataset = tf.data.Dataset.from_tensor_slices((Train_X_TF, Train_Y_TF))
        train_dataset = train_dataset.shuffle(buffer_size=Buffer_Input).batch(Batch_Size)

        # Prepare the Validation Dataset.
        val_dataset = tf.data.Dataset.from_tensor_slices((Val_X_TF, Val_Y_TF))
        val_dataset = val_dataset.batch(Batch_Size)

        # Creating Normalization Layer
        Train_X_Array = np.array(Train_X_TF)

        Normalization_Layer = tf.keras.layers.Normalization(axis=-1)
        Normalization_Layer.adapt(Train_X_Array)

    elif (NeuralStateSpace_ModelType == 2):  # Nueral State Space        

        # Creating Train_X_TF_A, Train_X_TF_B
        # Train_X_TF_A = Train_X_TF[:, X_TF_A_Indices]
        Train_X_TF_A = tf.convert_to_tensor(Train_X_TF.numpy()[:, X_TF_A_Indices])
        Train_X_TF_B = tf.convert_to_tensor(Train_X_TF.numpy()[:, X_TF_B_Indices])

        # Creating Test_X_TF_A, Test_X_TF_B
        # Test_X_TF_A = Test_X_TF[:, X_TF_A_Indices]
        Test_X_TF_A = tf.convert_to_tensor(Test_X_TF.numpy()[:, X_TF_A_Indices])
        Test_X_TF_B = tf.convert_to_tensor(Test_X_TF.numpy()[:, X_TF_B_Indices])

        # Creating Val_X_TF_A, Val_X_TF_B
        Val_X_TF_A = tf.convert_to_tensor(Val_X_TF.numpy()[:, X_TF_A_Indices])
        Val_X_TF_B = tf.convert_to_tensor(Val_X_TF.numpy()[:, X_TF_B_Indices])

        # Prepare the Training Dataset for A(x)
        train_dataset_A = tf.data.Dataset.from_tensor_slices((Train_X_TF_A, Train_Y_TF))
        train_dataset_A = train_dataset_A.shuffle(buffer_size=Buffer_Input).batch(Batch_Size)

        # Prepare the Training Dataset for B(u)
        train_dataset_B = tf.data.Dataset.from_tensor_slices((Train_X_TF_B, Train_Y_TF))
        train_dataset_B = train_dataset_B.shuffle(buffer_size=Buffer_Input).batch(Batch_Size)

        # Prepare the Validation Dataset for A(x)
        val_dataset_A = tf.data.Dataset.from_tensor_slices((Val_X_TF_A, Val_Y_TF))
        val_dataset_A = val_dataset_A.batch(Batch_Size)

        # Prepare the Validation Dataset for B(u)
        val_dataset_B = tf.data.Dataset.from_tensor_slices((Val_X_TF_B, Val_Y_TF))
        val_dataset_B = val_dataset_B.batch(Batch_Size)

        # Creating Normalization Layer for A(x)
        Train_X_Array_A = np.array(Train_X_TF_A)

        Normalization_Layer_A = tf.keras.layers.Normalization(axis=-1)
        Normalization_Layer_A.adapt(Train_X_Array_A)  

        # Creating Normalization Layer for B(u)
        Train_X_Array_B = np.array(Train_X_TF_B)

        Normalization_Layer_B = tf.keras.layers.Normalization(axis=-1)
        Normalization_Layer_B.adapt(Train_X_Array_B)            

    

elif ((ANN_Type == 2) or (ANN_Type == 3) or (ANN_Type == 4)):

    # Scaling Data
    scaler_X = StandardScaler()
    scaler_Y = StandardScaler()

    Scaler_Train_X = scaler_X.fit(Train_X.to_numpy())
    Scaler_Train_Y = scaler_Y.fit(Train_Y.to_numpy().reshape(-1,1))   
    
    Train_X_Scaled = Scaler_Train_X.transform(Train_X.to_numpy())
    Train_Y_Scaled = Scaler_Train_Y.transform(Train_Y.to_numpy().reshape(-1,1))

    # Creating Training and Validation Sets
    Train_X1 = Train_X_Scaled[0:math.floor(Train_X.shape[0] * (1 - Validation_Split))]
    Train_Y1 = Train_Y_Scaled[0:math.floor(Train_Y.shape[0] * (1 - Validation_Split))]

    Train_Index = Train_X1.shape[0]

    Val_X1 = Train_X[Train_Index:Train_Index + math.floor(Train_X.shape[0] * (Validation_Split))]
    Val_Y1 = Train_Y[Train_Index:Train_Index + math.floor(Train_Y.shape[0] * (Validation_Split))]

    # Converting Train_X and Train_Y into Tensor
    Train_X_TF = tf.convert_to_tensor(Train_X1)
    Train_Y_TF = tf.convert_to_tensor(Train_Y1)

    # Converting Val_X and Val_Y into Tensor
    Val_X_TF = tf.convert_to_tensor(Val_X1)
    Val_Y_TF = tf.convert_to_tensor(Val_Y1)

    # Converting Text_X into Tensor
    Test_X_TF_1 = tf.convert_to_tensor(Test_X)

    # Converting Test_X_M_Dot_Ts to Tensor
    Test_X_M_Dot_Ts_TF_1 = tf.convert_to_tensor(Test_X_M_Dot_Ts)

    # Getting Approprite Indices for training data for A(x) and B(u)
    ALL_Indices = range(0,Train_X_TF.shape[1],1)

    X_TF_A_Indices = []

    Q_HVAC_A_Indices = []

    for jj in range(Total_Aggregation_Zone_Number):

        if (jj ==0):
    
            X_TF_A_Indices.append(1)

            Q_HVAC_A_Indices.append(0)
    
        else:
    
            X_TF_A_Indices.append(X_TF_A_Indices[-1]+2)

            Q_HVAC_A_Indices.append(Q_HVAC_A_Indices[-1]+2)
        
    X_TF_B_Indices = list(set(ALL_Indices).difference(set(X_TF_A_Indices)))
    

    """ for ii in range(Train_X_TF.shape[0]):

        if (((Batch_Size*Lag_Number)*ii+(Batch_Size*Lag_Number)) <= Train_X_TF.shape[0]):

            Train_X_TF_batch = Train_X_TF[((Batch_Size*Lag_Number)*ii):((Batch_Size*Lag_Number)*ii+(Batch_Size*Lag_Number)),:]

            Train_X_TF_batch = tf.reshape(Train_X_TF_batch, [Batch_Size, Lag_Number, Train_X_TF.shape[1]])

            Train_Y_TF_batch = Train_Y_TF[(Batch_Size*ii):(Batch_Size*ii+Batch_Size)]

            Train_Dataset_List.append((Train_X_TF_batch, Train_Y_TF_batch))

    for ii in range(Val_X_TF.shape[0]):

        if (((Batch_Size*Lag_Number)*ii+(Batch_Size*Lag_Number)) <= Val_X_TF.shape[0]):

            Val_X_TF_batch = Val_X_TF[((Batch_Size*Lag_Number)*ii):((Batch_Size*Lag_Number)*ii+(Batch_Size*Lag_Number)),:]

            Val_X_TF_batch = tf.reshape(Val_X_TF_batch, [Batch_Size, Lag_Number, Val_X_TF.shape[1]])

            Val_Y_TF_batch = Val_Y_TF[(Batch_Size*ii):(Batch_Size*ii+Batch_Size)]

            Val_Dataset_List.append((Val_X_TF_batch, Val_Y_TF_batch))

    for ii in range(Test_X_TF_1.shape[0]):

        if (((Lag_Number)*ii+(Lag_Number)) <= Test_X_TF_1.shape[0]):

            Test_X_TF_batch = Test_X_TF_1[((Lag_Number)*ii):((Lag_Number)*ii+(Lag_Number)),:]

            Test_X_TF_batch = tf.reshape(Test_X_TF_batch, [1, Lag_Number, Test_X_TF_1.shape[1]])

            if (ii == 0):

                Test_X_TF = Test_X_TF_batch

            else:

                Test_X_TF = tf.concat([Test_X_TF, Test_X_TF_batch], axis=0)

    for ii in range(Test_X_M_Dot_Ts_TF_1.shape[0]):

        if (((Lag_Number)*ii+(Lag_Number)) <= Test_X_M_Dot_Ts_TF_1.shape[0]):

            Test_X_M_Dot_Ts_TF_batch = Test_X_M_Dot_Ts_TF_1[((Lag_Number)*ii):((Lag_Number)*ii+(Lag_Number)),:]

            Test_X_M_Dot_Ts_TF_batch = tf.reshape(Test_X_M_Dot_Ts_TF_batch, [1, Lag_Number, Test_X_M_Dot_Ts_TF_1.shape[1]])

            if (ii == 0):

                Test_X_M_Dot_Ts_TF = Test_X_TF_batch

            else:

                Test_X_M_Dot_Ts_TF = tf.concat([Test_X_M_Dot_Ts_TF, Test_X_TF_batch], axis=0) """
    
    if (NeuralStateSpace_ModelType == 1):  # Not Nueral State Space

        # Creating Train_Dataset and Val_Dataset manually
        Train_Dataset_List = []
        Val_Dataset_List = []

        # Creating Training Dataset
        for ii in range(Train_X_TF.shape[0]):

            if ((ii+(Batch_Size*Lag_Number)) <= Train_X_TF.shape[0]):

                Train_X_TF_batch = Train_X_TF[(ii):(ii+(Batch_Size*Lag_Number)),:]

                Train_X_TF_batch = tf.reshape(Train_X_TF_batch, [Batch_Size, Lag_Number, Train_X_TF.shape[1]])

                Train_Y_TF_batch = Train_Y_TF[(ii):(ii+Batch_Size)]

                Train_Dataset_List.append((Train_X_TF_batch, Train_Y_TF_batch))

        # Creating Validation Dataset
        for ii in range(Val_X_TF.shape[0]):

            if ((ii+(Batch_Size*Lag_Number)) <= Val_X_TF.shape[0]):

                Val_X_TF_batch = Val_X_TF[(ii):(ii+(Batch_Size*Lag_Number)),:]

                Val_X_TF_batch = tf.reshape(Val_X_TF_batch, [Batch_Size, Lag_Number, Val_X_TF.shape[1]])

                Val_Y_TF_batch = Val_Y_TF[(ii):(ii+Batch_Size)]

                Val_Dataset_List.append((Val_X_TF_batch, Val_Y_TF_batch))

        # Creating Training X for Batch Testing        
        for ii in range(Train_X_TF.shape[0]):

            if ((ii+(Lag_Number)) <= Train_X_TF.shape[0]):

                Train_X_TF_batch = Train_X_TF[(ii):(ii+(Lag_Number)),:]

                Train_X_TF_batch = tf.reshape(Train_X_TF_batch, [1, Lag_Number, Train_X_TF.shape[1]])

                if (ii == 0):

                    Train_X_TF_11 = Train_X_TF_batch

                else:

                    Train_X_TF_11 = tf.concat([Train_X_TF_11, Train_X_TF_batch], axis=0)


        # Creating Testing X for Batch Testing
        for ii in range(Test_X_TF_1.shape[0]):

            if ((ii+(Lag_Number)) <= Test_X_TF_1.shape[0]):

                Test_X_TF_batch = Test_X_TF_1[(ii):(ii+(Lag_Number)),:]

                Test_X_TF_batch = tf.reshape(Test_X_TF_batch, [1, Lag_Number, Test_X_TF_1.shape[1]])

                if (ii == 0):

                    Test_X_TF = Test_X_TF_batch

                else:

                    Test_X_TF = tf.concat([Test_X_TF, Test_X_TF_batch], axis=0)

        # Creating Testing X M_Dot_Ts for Simulation Testing
        for ii in range(Test_X_M_Dot_Ts_TF_1.shape[0]):

            if ((ii+(Lag_Number)) <= Test_X_M_Dot_Ts_TF_1.shape[0]):

                Test_X_M_Dot_Ts_TF_batch = Test_X_M_Dot_Ts_TF_1[(ii):(ii+(Lag_Number)),:]

                Test_X_M_Dot_Ts_TF_batch = tf.reshape(Test_X_M_Dot_Ts_TF_batch, [1, Lag_Number, Test_X_M_Dot_Ts_TF_1.shape[1]])

                if (ii == 0):

                    Test_X_M_Dot_Ts_TF = Test_X_M_Dot_Ts_TF_batch

                else:

                    Test_X_M_Dot_Ts_TF = tf.concat([Test_X_M_Dot_Ts_TF, Test_X_M_Dot_Ts_TF_batch], axis=0)

        train_dataset = Train_Dataset_List
        val_dataset = Val_Dataset_List

    elif (NeuralStateSpace_ModelType == 2):  # Nueral State Space        

        # Creating Train_Dataset and Val_Dataset manually for A(x)
        Train_Dataset_List_A = []
        Val_Dataset_List_A = []

        ## Datasets for Training , Validating and Testing A(x)

        # Creating Training Dataset for A(x)
        for ii in range(Train_X_TF.shape[0]):

            if ((ii+(Batch_Size*Lag_Number)) <= Train_X_TF.shape[0]):

                Train_X_TF_batch_A = tf.convert_to_tensor(Train_X_TF.numpy()[(ii):(ii+(Batch_Size*Lag_Number)),X_TF_A_Indices], dtype=tf.float32)

                Train_X_TF_batch_A = tf.reshape(Train_X_TF_batch_A, [Batch_Size, Lag_Number, len(X_TF_A_Indices)])

                Train_Y_TF_batch_A = Train_Y_TF[(ii):(ii+Batch_Size)]

                Train_Dataset_List_A.append((Train_X_TF_batch_A, Train_Y_TF_batch_A))

        # Creating Validation Dataset for A(x)
        for ii in range(Val_X_TF.shape[0]):

            if ((ii+(Batch_Size*Lag_Number)) <= Val_X_TF.shape[0]):

                Val_X_TF_batch_A = tf.convert_to_tensor(Val_X_TF.numpy()[(ii):(ii+(Batch_Size*Lag_Number)),X_TF_A_Indices], dtype=tf.float32)

                Val_X_TF_batch_A = tf.reshape(Val_X_TF_batch_A, [Batch_Size, Lag_Number, len(X_TF_A_Indices)])

                Val_Y_TF_batch_A = Val_Y_TF[(ii):(ii+Batch_Size)]

                Val_Dataset_List_A.append((Val_X_TF_batch_A, Val_Y_TF_batch_A))

        # Creating Training X for Batch Testing for A(x)       
        for ii in range(Train_X_TF.shape[0]):

            if ((ii+(Lag_Number)) <= Train_X_TF.shape[0]):

                Train_X_TF_batch_A = tf.convert_to_tensor(Train_X_TF.numpy()[(ii):(ii+(Lag_Number)),X_TF_A_Indices], dtype=tf.float32)

                Train_X_TF_batch_A = tf.reshape(Train_X_TF_batch_A, [1, Lag_Number, len(X_TF_A_Indices)])

                if (ii == 0):

                    Train_X_TF_11_A = Train_X_TF_batch_A

                else:

                    Train_X_TF_11_A = tf.concat([Train_X_TF_11_A, Train_X_TF_batch_A], axis=0)


        # Creating Testing X for Batch Testing for A(x)
        for ii in range(Test_X_TF_1.shape[0]):

            if ((ii+(Lag_Number)) <= Test_X_TF_1.shape[0]):

                Test_X_TF_batch_A = tf.convert_to_tensor(Test_X_TF_1.numpy()[(ii):(ii+(Lag_Number)),X_TF_A_Indices], dtype=tf.float32)

                Test_X_TF_batch_A = tf.reshape(Test_X_TF_batch_A, [1, Lag_Number, len(X_TF_A_Indices)])

                if (ii == 0):

                    Test_X_TF_A = Test_X_TF_batch_A

                else:

                    Test_X_TF_A = tf.concat([Test_X_TF_A, Test_X_TF_batch_A], axis=0)

        # Creating Train_Dataset and Val_Dataset manually for B(u)
        Train_Dataset_List_B = []
        Val_Dataset_List_B = []

        ## Datasets for Training , Validating and Testing B(u)

        # Creating Training Dataset for B(u)
        for ii in range(Train_X_TF.shape[0]):

            if ((ii+(Batch_Size*Lag_Number)) <= Train_X_TF.shape[0]):

                Train_X_TF_batch_B = tf.convert_to_tensor(Train_X_TF.numpy()[(ii):(ii+(Batch_Size*Lag_Number)),X_TF_B_Indices], dtype=tf.float32)

                Train_X_TF_batch_B = tf.reshape(Train_X_TF_batch_B, [Batch_Size, Lag_Number, len(X_TF_B_Indices)])

                Train_Y_TF_batch_B = Train_Y_TF[(ii):(ii+Batch_Size)]

                Train_Dataset_List_B.append((Train_X_TF_batch_B, Train_Y_TF_batch_B))

        # Creating Validation Dataset for B(u)
        for ii in range(Val_X_TF.shape[0]):

            if ((ii+(Batch_Size*Lag_Number)) <= Val_X_TF.shape[0]):

                Val_X_TF_batch_B = tf.convert_to_tensor(Val_X_TF.numpy()[(ii):(ii+(Batch_Size*Lag_Number)),X_TF_B_Indices], dtype=tf.float32)

                Val_X_TF_batch_B = tf.reshape(Val_X_TF_batch_B, [Batch_Size, Lag_Number, len(X_TF_B_Indices)])

                Val_Y_TF_batch_B = Val_Y_TF[(ii):(ii+Batch_Size)]

                Val_Dataset_List_B.append((Val_X_TF_batch_B, Val_Y_TF_batch_B))

        # Creating Training X for Batch Testing for B(u)       
        for ii in range(Train_X_TF.shape[0]):

            if ((ii+(Lag_Number)) <= Train_X_TF.shape[0]):

                Train_X_TF_batch_B = tf.convert_to_tensor(Train_X_TF.numpy()[(ii):(ii+(Lag_Number)),X_TF_B_Indices], dtype=tf.float32)

                Train_X_TF_batch_B = tf.reshape(Train_X_TF_batch_B, [1, Lag_Number, len(X_TF_B_Indices)])

                if (ii == 0):

                    Train_X_TF_11_B = Train_X_TF_batch_B

                else:

                    Train_X_TF_11_B = tf.concat([Train_X_TF_11_B, Train_X_TF_batch_B], axis=0)


        # Creating Testing X for Batch Testing for B(u)
        for ii in range(Test_X_TF_1.shape[0]):

            if ((ii+(Lag_Number)) <= Test_X_TF_1.shape[0]):

                Test_X_TF_batch_B = tf.convert_to_tensor(Test_X_TF_1.numpy()[(ii):(ii+(Lag_Number)),X_TF_B_Indices], dtype=tf.float32)

                Test_X_TF_batch_B = tf.reshape(Test_X_TF_batch_B, [1, Lag_Number, len(X_TF_B_Indices)])

                if (ii == 0):

                    Test_X_TF_B = Test_X_TF_batch_B

                else:

                    Test_X_TF_B = tf.concat([Test_X_TF_B, Test_X_TF_batch_B], axis=0)

        # Creating Testing X M_Dot_Ts for Simulation Testing
        for ii in range(Test_X_M_Dot_Ts_TF_1.shape[0]):

            if ((ii+(Lag_Number)) <= Test_X_M_Dot_Ts_TF_1.shape[0]):

                Test_X_M_Dot_Ts_TF_batch = Test_X_M_Dot_Ts_TF_1[(ii):(ii+(Lag_Number)),:]

                Test_X_M_Dot_Ts_TF_batch = tf.reshape(Test_X_M_Dot_Ts_TF_batch, [1, Lag_Number, Test_X_M_Dot_Ts_TF_1.shape[1]])

                if (ii == 0):

                    Test_X_M_Dot_Ts_TF = Test_X_M_Dot_Ts_TF_batch

                else:

                    Test_X_M_Dot_Ts_TF = tf.concat([Test_X_M_Dot_Ts_TF, Test_X_M_Dot_Ts_TF_batch], axis=0)

        train_dataset_A = Train_Dataset_List_A
        val_dataset_A = Val_Dataset_List_A

        train_dataset_B = Train_Dataset_List_B
        val_dataset_B = Val_Dataset_List_B


# =============================================================================
# ANN Modelling
# =============================================================================

# Defining Regularizer
tf.keras.regularizers.L2(l2=0.01)

# Creating ANN Model
if (ANN_Type == 1): # MLP

    # Creating Model Based on if it is either Nueral State Space or Not
    if (NeuralStateSpace_ModelType == 1):  # Not Nueral State Space
        
        # Creating Model
        Current_ANNModel = tf.keras.Sequential([
            Normalization_Layer,
            # layers.Dense(units=1)
            layers.Dense(100, activation='relu', kernel_regularizer='l2'),
            # layers.Dense(20, activation='relu', kernel_regularizer='l2'),
            # layers.Dense(2, activation='relu', kernel_regularizer='l2'),
            # layers.Dense(10, activation='relu', kernel_regularizer='l2'),
            # layers.Dense(30, activation='relu'),
            # layers.Dense(30, activation='relu', kernel_regularizer='l2'),
            # layers.LSTM(10, return_sequences=True, return_state=True),
            layers.Dense(Total_Aggregation_Zone_Number)
        ])

    elif (NeuralStateSpace_ModelType == 2):  # Neural State Space
        
        # Creating Model A(x)
        Current_ANNModel_A = tf.keras.Sequential([
            Normalization_Layer_A,
            # layers.Dense(units=1)
            layers.Dense(100, activation='relu', kernel_regularizer='l2'),
            # layers.Dense(20, activation='relu', kernel_regularizer='l2'),
            # layers.Dense(2, activation='relu', kernel_regularizer='l2'),
            # layers.Dense(10, activation='relu', kernel_regularizer='l2'),
            # layers.Dense(30, activation='relu'),
            # layers.Dense(30, activation='relu', kernel_regularizer='l2'),
            # layers.LSTM(10, return_sequences=True, return_state=True),
            layers.Dense(Total_Aggregation_Zone_Number)
        ])

        # Creating Model B(u)
        Current_ANNModel_B = tf.keras.Sequential([
            Normalization_Layer_B,
            # layers.Dense(units=1)
            layers.Dense(100, activation='relu', kernel_regularizer='l2'),
            # layers.Dense(20, activation='relu', kernel_regularizer='l2'),
            # layers.Dense(2, activation='relu', kernel_regularizer='l2'),
            # layers.Dense(10, activation='relu', kernel_regularizer='l2'),
            # layers.Dense(30, activation='relu'),
            # layers.Dense(30, activation='relu', kernel_regularizer='l2'),
            # layers.LSTM(10, return_sequences=True, return_state=True),
            layers.Dense(Total_Aggregation_Zone_Number)
        ])


elif (ANN_Type == 2): # Simple RNN

    # Creating Model Based on if it is either Nueral State Space or Not
    if (NeuralStateSpace_ModelType == 1):  # Not Nueral State Space

        # Creating Model
        Current_ANNModel = tf.keras.Sequential([
            # Normalization_Layer,
            # layers.Dense(units=1)
            # layers.Dense(100, activation='relu', kernel_regularizer='l2'),
            # layers.Dense(5, activation='relu', kernel_regularizer='l2'),
            # layers.Dense(10, activation='relu', kernel_regularizer='l2'),
            # layers.Dense(30, activation='relu'),
            # layers.Dense(30, activation='relu'),
            # layers.Dense(30, activation='relu', kernel_regularizer='l2'),
            layers.SimpleRNN((5+Total_Aggregation_Zone_Number*2)*Lag_Number, input_shape=(Lag_Number,Train_X_TF.shape[1]), return_sequences=False),
            #layers.Dense(7, activation='relu', kernel_regularizer='l2'),
            layers.Dense(units=Total_Aggregation_Zone_Number, activation='linear')
        ])

    elif (NeuralStateSpace_ModelType == 2):  # Neural State Space

        # Creating Model A(x)
        Current_ANNModel_A = tf.keras.Sequential([
            # Normalization_Layer,
            # layers.Dense(units=1)
            # layers.Dense(100, activation='relu', kernel_regularizer='l2'),
            # layers.Dense(5, activation='relu', kernel_regularizer='l2'),
            # layers.Dense(10, activation='relu', kernel_regularizer='l2'),
            # layers.Dense(30, activation='relu'),
            # layers.Dense(30, activation='relu'),
            # layers.Dense(30, activation='relu', kernel_regularizer='l2'),
            layers.SimpleRNN((5+Total_Aggregation_Zone_Number*2)*Lag_Number, input_shape=(Lag_Number,Train_X_TF_11_A.shape[2]), return_sequences=False),
            #layers.Dense(7, activation='relu', kernel_regularizer='l2'),
            layers.Dense(units=Total_Aggregation_Zone_Number, activation='linear')
        ])

        # Creating Model B(u)
        Current_ANNModel_B = tf.keras.Sequential([
            # Normalization_Layer,
            # layers.Dense(units=1)
            # layers.Dense(100, activation='relu', kernel_regularizer='l2'),
            # layers.Dense(5, activation='relu', kernel_regularizer='l2'),
            # layers.Dense(10, activation='relu', kernel_regularizer='l2'),
            # layers.Dense(30, activation='relu'),
            # layers.Dense(30, activation='relu'),
            # layers.Dense(30, activation='relu', kernel_regularizer='l2'),
            layers.SimpleRNN((5+Total_Aggregation_Zone_Number*2)*Lag_Number, input_shape=(Lag_Number,Train_X_TF_11_B.shape[2]), return_sequences=False),
            #layers.Dense(7, activation='relu', kernel_regularizer='l2'),
            layers.Dense(units=Total_Aggregation_Zone_Number, activation='linear')
        ])

elif (ANN_Type == 3): # LSTM

    # Creating Model Based on if it is either Nueral State Space or Not
    if (NeuralStateSpace_ModelType == 1):  # Not Nueral State Space

        # Creating Model
        Current_ANNModel = tf.keras.Sequential([
            # Normalization_Layer,
            # layers.Dense(units=1)
            # layers.Dense(100, activation='relu', kernel_regularizer='l2'),
            # layers.Dense(5, activation='relu', kernel_regularizer='l2'),
            # layers.Dense(10, activation='relu', kernel_regularizer='l2'),
            # layers.Dense(30, activation='relu'),
            # layers.Dense(30, activation='relu'),
            # layers.Dense(30, activation='relu', kernel_regularizer='l2'),
            layers.LSTM((5+Total_Aggregation_Zone_Number*2)*Lag_Number, input_shape=(Lag_Number,Train_X_TF.shape[1]), return_sequences=False),
            layers.Dense(units=Total_Aggregation_Zone_Number, activation='linear')
        ])

    elif (NeuralStateSpace_ModelType == 2):  # Neural State Space

        # Creating Model A(x)
        Current_ANNModel_A = tf.keras.Sequential([
            # Normalization_Layer,
            # layers.Dense(units=1)
            # layers.Dense(100, activation='relu', kernel_regularizer='l2'),
            # layers.Dense(5, activation='relu', kernel_regularizer='l2'),
            # layers.Dense(10, activation='relu', kernel_regularizer='l2'),
            # layers.Dense(30, activation='relu'),
            # layers.Dense(30, activation='relu'),
            # layers.Dense(30, activation='relu', kernel_regularizer='l2'),
            layers.LSTM((5+Total_Aggregation_Zone_Number*2)*Lag_Number, input_shape=(Lag_Number,Train_X_TF_11_A.shape[2]), return_sequences=False),
            layers.Dense(units=Total_Aggregation_Zone_Number, activation='linear')
        ])

        # Creating Model B(u)
        Current_ANNModel_B = tf.keras.Sequential([
            # Normalization_Layer,
            # layers.Dense(units=1)
            # layers.Dense(100, activation='relu', kernel_regularizer='l2'),
            # layers.Dense(5, activation='relu', kernel_regularizer='l2'),
            # layers.Dense(10, activation='relu', kernel_regularizer='l2'),
            # layers.Dense(30, activation='relu'),
            # layers.Dense(30, activation='relu'),
            # layers.Dense(30, activation='relu', kernel_regularizer='l2'),
            layers.LSTM((5+Total_Aggregation_Zone_Number*2)*Lag_Number, input_shape=(Lag_Number,Train_X_TF_11_B.shape[2]), return_sequences=False),
            layers.Dense(units=Total_Aggregation_Zone_Number, activation='linear')
        ])

elif (ANN_Type == 4): # GRU

    # Creating Model Based on if it is either Nueral State Space or Not
    if (NeuralStateSpace_ModelType == 1):  # Not Nueral State Space

        # Creating Model
        Current_ANNModel = tf.keras.Sequential([
            # Normalization_Layer,
            # layers.Dense(units=1)
            # layers.Dense(100, activation='relu', kernel_regularizer='l2'),
            # layers.Dense(5, activation='relu', kernel_regularizer='l2'),
            # layers.Dense(10, activation='relu', kernel_regularizer='l2'),
            # layers.Dense(30, activation='relu'),
            # layers.Dense(30, activation='relu'),
            # layers.Dense(30, activation='relu', kernel_regularizer='l2'),
            # layers.GRU(21, input_shape=(1,Train_X_TF.shape[1]), return_sequences=True, kernel_regularizer='l2'),
            # layers.GRU(14, input_shape=(1,Train_X_TF.shape[1]), return_sequences=True, kernel_regularizer='l2'),
            layers.GRU((5+Total_Aggregation_Zone_Number*2)*Lag_Number, input_shape=(Lag_Number,Train_X_TF.shape[1]), return_sequences=False),
            layers.Dense(units=Total_Aggregation_Zone_Number, activation='linear')
        ])

    elif (NeuralStateSpace_ModelType == 2):  # Neural State Space

        # Creating Model A(x)
        Current_ANNModel_A = tf.keras.Sequential([
            # Normalization_Layer,
            # layers.Dense(units=1)
            # layers.Dense(100, activation='relu', kernel_regularizer='l2'),
            # layers.Dense(5, activation='relu', kernel_regularizer='l2'),
            # layers.Dense(10, activation='relu', kernel_regularizer='l2'),
            # layers.Dense(30, activation='relu'),
            # layers.Dense(30, activation='relu'),
            # layers.Dense(30, activation='relu', kernel_regularizer='l2'),
            # layers.GRU(21, input_shape=(1,Train_X_TF.shape[1]), return_sequences=True, kernel_regularizer='l2'),
            # layers.GRU(14, input_shape=(1,Train_X_TF.shape[1]), return_sequences=True, kernel_regularizer='l2'),
            layers.GRU((5+Total_Aggregation_Zone_Number*2)*Lag_Number, input_shape=(Lag_Number,Train_X_TF_11_A.shape[2]), return_sequences=False),
            layers.Dense(units=Total_Aggregation_Zone_Number, activation='linear')
        ])

        # Creating Model B(u)
        Current_ANNModel_B = tf.keras.Sequential([
            # Normalization_Layer,
            # layers.Dense(units=1)
            # layers.Dense(100, activation='relu', kernel_regularizer='l2'),
            # layers.Dense(5, activation='relu', kernel_regularizer='l2'),
            # layers.Dense(10, activation='relu', kernel_regularizer='l2'),
            # layers.Dense(30, activation='relu'),
            # layers.Dense(30, activation='relu'),
            # layers.Dense(30, activation='relu', kernel_regularizer='l2'),
            # layers.GRU(21, input_shape=(1,Train_X_TF.shape[1]), return_sequences=True, kernel_regularizer='l2'),
            # layers.GRU(14, input_shape=(1,Train_X_TF.shape[1]), return_sequences=True, kernel_regularizer='l2'),
            layers.GRU((5+Total_Aggregation_Zone_Number*2)*Lag_Number, input_shape=(Lag_Number,Train_X_TF_11_B.shape[2]), return_sequences=False),
            layers.Dense(units=Total_Aggregation_Zone_Number, activation='linear')
        ])


# =============================================================================
# ANN Training
# =============================================================================

# Instantiate an Optimizer
optimizer = keras.optimizers.SGD(learning_rate=Learning_Rate)

# Instantiate a Loss Function.
loss_fn = tf.keras.losses.MeanSquaredError()

# Printing out the Summary of Regression Model
if (NeuralStateSpace_ModelType == 1):
    Current_ANNModel.summary()

elif (NeuralStateSpace_ModelType == 2):
    Current_ANNModel_A.summary()
    Current_ANNModel_B.summary()

# Prepare the Metrics.
train_acc_metric = keras.metrics.MeanSquaredError()
val_acc_metric = keras.metrics.MeanSquaredError()

Training_Loss_Value_Set = []
Val_Loss_Value_Set = []
Training_Error_Value_Set = []
Val_Error_Value_Set = []

# These setting are for early stopping the training
wait = 0
best = float('inf')

for epoch in range(Epochs):

    print("\nStart of epoch %d" % (epoch,))
    start_time = time.time()

    ## Training Model Based on if it is either Nueral State Space or Not

    if (NeuralStateSpace_ModelType == 1):  # Not Nueral State Space

        # Iterate over the batches of the dataset.
        #for step, (x_batch_train, y_batch_train) in enumerate(train_dataset):
        for step, (x_batch_train, y_batch_train) in enumerate(train_dataset):

            with tf.GradientTape() as tape:

                if (TimeStepper_ModelType == 1): # Model Predicts Next State Directly

                    # Computing Model Output
                    logits = Current_ANNModel(x_batch_train, training=True)

                elif(TimeStepper_ModelType == 2):  # Model Computes Current Derivative

                    if (ANN_Type == 1): # ANN

                        # Getting Tz_Previous
                        # x_batch_train_Tz_Prev = tf.cast(tf.reshape(x_batch_train[:,X_TF_A_Indices[-Total_Aggregation_Zone_Number:]],[x_batch_train.shape[0],Total_Aggregation_Zone_Number]), tf.float32)

                        x_batch_train_Tz_Prev_desired = [x_batch_train[:,i] for i in X_TF_A_Indices[-Total_Aggregation_Zone_Number:]]
                        x_batch_train_Tz_Prev = tf.cast(tf.reshape(x_batch_train_Tz_Prev_desired[0],[x_batch_train.shape[0],Total_Aggregation_Zone_Number]), tf.float32)
                        
                        # Computing Model Output
                        logits = x_batch_train_Tz_Prev + Current_ANNModel(x_batch_train, training=True)

                    elif ((ANN_Type == 2) or (ANN_Type == 3) or (ANN_Type == 4)):  # LSTM, RNN, GRU

                        # Getting Tz_Previous
                        x_batch_train_Tz_Prev = tf.cast(tf.reshape(x_batch_train.numpy()[:,Lag_Number-1,X_TF_A_Indices],[x_batch_train.shape[0],len(X_TF_A_Indices)]), tf.float32)

                        # Computing Model Output
                        logits = x_batch_train_Tz_Prev + Current_ANNModel(x_batch_train, training=True)
                
                # Compute Loss
                loss_value = loss_fn(y_batch_train, logits)

            grads = tape.gradient(loss_value, Current_ANNModel.trainable_weights)
            optimizer.apply_gradients(zip(grads, Current_ANNModel.trainable_weights))

            # Update training metric.
            train_acc_metric.update_state(y_batch_train, logits)

            # Log every 200 batches.
            # if step % 200 == 0:
            # print("Training loss (for one batch) at step %d: %.4f"% (step, float(loss_value)))
            # print("Seen so far: %d samples" % ((step + 1) * Batch_Size))

        # training_loss_value[counter] = loss_value
        Training_Loss_Value_Set = tf.experimental.numpy.append(Training_Loss_Value_Set, loss_value)

        # Display metrics at the end of each epoch.
        train_acc = train_acc_metric.result()
        # print("Training acc over epoch: %.4f" % (float(train_acc),))

        # training_error_value[counter] = train_acc
        Training_Error_Value_Set = tf.experimental.numpy.append(Training_Error_Value_Set, train_acc)

        # Reset training metrics at the end of each epoch
        train_acc_metric.reset_state()

        # Run a validation loop at the end of each epoch.
        #for x_batch_val, y_batch_val in val_dataset:
        for x_batch_val, y_batch_val in val_dataset:
            
            if (TimeStepper_ModelType == 1): # Model Predicts Next State Directly

                # Computing Model Output
                val_logits = Current_ANNModel(x_batch_val, training=False) 

            elif(TimeStepper_ModelType == 2):  # Model Computes Current Derivative

                if (ANN_Type == 1): # ANN

                    # Getting Tz_Previous
                    # x_batch_val_Tz_Prev = tf.cast(tf.reshape(x_batch_val[:,X_TF_A_Indices[-Total_Aggregation_Zone_Number:]],[x_batch_val.shape[0],Total_Aggregation_Zone_Number]), tf.float32)
                    
                    x_batch_val_Tz_Prev_desired = [x_batch_val[:,i] for i in X_TF_A_Indices[-Total_Aggregation_Zone_Number:]]
                    x_batch_val_Tz_Prev = tf.cast(tf.reshape(x_batch_val_Tz_Prev_desired[0],[x_batch_val.shape[0],Total_Aggregation_Zone_Number]), tf.float32)
                    # Computing Model Output
                    val_logits = x_batch_val_Tz_Prev + Current_ANNModel(x_batch_val, training=False) 

                elif ((ANN_Type == 2) or (ANN_Type == 3) or (ANN_Type == 4)):  # LSTM, RNN, GRU

                    # Getting Tz_Previous
                    x_batch_val_Tz_Prev = tf.cast(tf.reshape(x_batch_val.numpy()[:,Lag_Number-1,X_TF_A_Indices],[x_batch_val.shape[0],len(X_TF_A_Indices)]), tf.float32)

                    # Computing Model Output
                    val_logits = x_batch_val_Tz_Prev + Current_ANNModel(x_batch_val, training=False) 

            val_loss_value = loss_fn(y_batch_val, val_logits)
            # Update val metrics
            val_acc_metric.update_state(y_batch_val, val_logits)

        val_acc = val_acc_metric.result()
        val_acc_metric.reset_state()
        # print("Validation acc: %.4f" % (float(val_acc),))
        # print("Time taken: %.2fs" % (time.time() - start_time))

        Val_Loss_Value_Set = tf.experimental.numpy.append(Val_Loss_Value_Set, val_loss_value)
        Val_Error_Value_Set = tf.experimental.numpy.append(Val_Error_Value_Set, val_acc)

    elif (NeuralStateSpace_ModelType == 2):  # Nueral State Space

        # Iterate over the batches of the dataset.
        #for step, (x_batch_train, y_batch_train) in enumerate(train_dataset):
        for ((x_batch_train_A, y_batch_train_A), (x_batch_train_B, y_batch_train_B)) in zip(train_dataset_A, train_dataset_B):

            with tf.GradientTape(persistent=True) as tape:

                if (TimeStepper_ModelType == 1): # Model Predicts Next State Directly

                    # Computing Model Output
                    logits = Current_ANNModel_A(x_batch_train_A, training=True) + Current_ANNModel_B(x_batch_train_B, training=True)

                elif(TimeStepper_ModelType == 2):  # Model Computes Current Derivative

                    if (ANN_Type == 1): # ANN

                        # Getting Tz_Previous
                        x_batch_train_Tz_Prev = tf.cast(tf.reshape(x_batch_train_A[:,(Total_Aggregation_Zone_Number*(Lag_Number-1)):(Total_Aggregation_Zone_Number*Lag_Number)],[x_batch_train_A.shape[0],Total_Aggregation_Zone_Number]), tf.float32)

                        # Computing Model Output
                        logits = x_batch_train_Tz_Prev + Current_ANNModel_A(x_batch_train_A, training=True) + Current_ANNModel_B(x_batch_train_B, training=True)

                    elif ((ANN_Type == 2) or (ANN_Type == 3) or (ANN_Type == 4)):  # LSTM, RNN, GRU

                        # Getting Tz_Previous
                        x_batch_train_Tz_Prev = tf.cast(tf.reshape(x_batch_train_A[:,Lag_Number-1,:],[x_batch_train_A.shape[0],x_batch_train_A.shape[2]]), tf.float32)

                        # Computing Model Output
                        logits = x_batch_train_Tz_Prev + Current_ANNModel_A(x_batch_train_A, training=True) + Current_ANNModel_B(x_batch_train_B, training=True)
                
                # Compute Loss
                loss_value = loss_fn(y_batch_train_A, logits)

            A_grads = tape.gradient(loss_value, Current_ANNModel_A.trainable_weights)
            B_grads = tape.gradient(loss_value, Current_ANNModel_B.trainable_weights)

            # optimizer.apply_gradients([zip(A_grads, Current_ANNModel_A.trainable_weights),
            #                             zip(B_grads, Current_ANNModel_B.trainable_weights)])

            # Zip gradients and model variables for both models
            A_pairs = zip(A_grads, Current_ANNModel_A.trainable_weights)
            B_pairs = zip(B_grads, Current_ANNModel_B.trainable_weights)

            # Combine the pairs into a single list
            all_pairs = list(A_pairs) + list(B_pairs)

            # Apply the gradients using the optimizer
            optimizer.apply_gradients(all_pairs)

            # Update training metric.
            train_acc_metric.update_state(y_batch_train_A, logits)

            # Log every 200 batches.
            # if step % 200 == 0:
            # print("Training loss (for one batch) at step %d: %.4f"% (step, float(loss_value)))
            # print("Seen so far: %d samples" % ((step + 1) * Batch_Size))

        # training_loss_value[counter] = loss_value
        Training_Loss_Value_Set = tf.experimental.numpy.append(Training_Loss_Value_Set, loss_value)

        # Display metrics at the end of each epoch.
        train_acc = train_acc_metric.result()
        # print("Training acc over epoch: %.4f" % (float(train_acc),))

        # training_error_value[counter] = train_acc
        Training_Error_Value_Set = tf.experimental.numpy.append(Training_Error_Value_Set, train_acc)

        # Reset training metrics at the end of each epoch
        train_acc_metric.reset_state()

        # Run a validation loop at the end of each epoch.
        #for x_batch_val, y_batch_val in val_dataset:
        for ((x_batch_val_A, y_batch_val_A), (x_batch_val_B, y_batch_val_B)) in zip(val_dataset_A, val_dataset_B):                

            if (TimeStepper_ModelType == 1): # Model Predicts Next State Directly

                # Computing Model Output
                val_logits = Current_ANNModel_A(x_batch_val_A, training=False) + Current_ANNModel_B(x_batch_val_B, training=False)

            elif(TimeStepper_ModelType == 2):  # Model Computes Current Derivative

                if (ANN_Type == 1): # ANN

                    # Getting Tz_Previous
                    x_batch_val_Tz_Prev = tf.cast(tf.reshape(x_batch_val_A[:,(Total_Aggregation_Zone_Number*(Lag_Number-1)):(Total_Aggregation_Zone_Number*Lag_Number)],[x_batch_val_A.shape[0],Total_Aggregation_Zone_Number]), tf.float32)

                    # Computing Model Output
                    val_logits = x_batch_val_Tz_Prev + Current_ANNModel_A(x_batch_val_A, training=False) + Current_ANNModel_B(x_batch_val_B, training=False)

                elif ((ANN_Type == 2) or (ANN_Type == 3) or (ANN_Type == 4)):  # LSTM, RNN, GRU

                    # Getting Tz_Previous
                    x_batch_val_Tz_Prev = tf.cast(tf.reshape(x_batch_val_A[:,Lag_Number-1,:],[x_batch_val_A.shape[0],x_batch_val_A.shape[2]]), tf.float32)

                    # Computing Model Output
                    val_logits = x_batch_val_Tz_Prev + Current_ANNModel_A(x_batch_val_A, training=False) + Current_ANNModel_B(x_batch_val_B, training=False)

            val_loss_value = loss_fn(y_batch_val_A, val_logits)
            # Update val metrics
            val_acc_metric.update_state(y_batch_val_A, val_logits)

        val_acc = val_acc_metric.result()
        val_acc_metric.reset_state()
        # print("Validation acc: %.4f" % (float(val_acc),))
        # print("Time taken: %.2fs" % (time.time() - start_time))

        Val_Loss_Value_Set = tf.experimental.numpy.append(Val_Loss_Value_Set, val_loss_value)
        Val_Error_Value_Set = tf.experimental.numpy.append(Val_Error_Value_Set, val_acc)

    # The early stopping strategy: stop the training if `val_loss` does not decrease over a certain number of epochs.
    wait += 1
    if val_loss_value < best:
        best = val_loss_value
        wait = 0
    if wait >= Early_Stopping_Patience:
        break

# Getting Final Train/Val Loss/Error
Final_Train_Loss = Training_Loss_Value_Set.numpy()[-1]
Final_Train_Error = Training_Error_Value_Set.numpy()[-1]
Final_Val_Loss = Val_Loss_Value_Set.numpy()[-1]
Final_Val_Error = Val_Error_Value_Set.numpy()[-1]


# =============================================================================
# ANN Testing
# =============================================================================

## Testing the Trained Model Based on if it is either Nueral State Space or Not

if (NeuralStateSpace_ModelType == 1):  # Not Nueral State Space

    if (ANN_Type == 1):
    
        # Converting Test_X tensors to numpy arrays
        Test_X_TF_Array = Test_X_TF.numpy()
        Test_X_TF_Array = np.reshape(Test_X_TF_Array, (Test_X_TF_Array.shape[0], Test_X_TF_Array.shape[1]))

        Test_X_M_Dot_Ts_TF_Array = Test_X_M_Dot_Ts_TF.numpy()
        Test_X_M_Dot_Ts_TF_Array = np.reshape(Test_X_M_Dot_Ts_TF_Array, (Test_X_M_Dot_Ts_TF_Array.shape[0], Test_X_M_Dot_Ts_TF_Array.shape[1]))

        Test_Y_Predict = np.zeros((Test_X_TF_Array.shape[0],Total_Aggregation_Zone_Number))
        # Test_Y_Predict = np.zeros((Test_X_TF_Array.shape[0]))

        # Initializing the Simulation
        PHVAC_Current = np.zeros((Test_X_TF_Array.shape[0], Total_Aggregation_Zone_Number))
        # PHVAC_Current = np.zeros((Test_X_TF_Array.shape[0],1))

        # Getting Appropriate Indices
        Tz_Indices = X_TF_A_Indices
        QHVAC_Indices = Q_HVAC_A_Indices
        M_Dot_Indices = list(range(0, Test_X_M_Dot_Ts_TF_Array.shape[1], 2))
        Ts_Indices = list(range(1, Test_X_M_Dot_Ts_TF_Array.shape[1]+1, 2))

        # Initializing Initial Temp and Control Signals
        Tz_Current = np.reshape(Test_X_TF_Array[0,Tz_Indices], (1, len(Tz_Indices)))
        M_Dot = Test_X_M_Dot_Ts_TF_Array[:,M_Dot_Indices]
        Ts = Test_X_M_Dot_Ts_TF_Array[:,Ts_Indices]

        Ca = 1.004    

        # Timing Simulation
        Sim_StartTime = time.time()

        # LOOP: For Loop Simulation
        for jj in range(Test_X_TF_Array.shape[0]):

            # Computing QHVAC from Predicted Temperature
            QHVAC_Computed = Ca * M_Dot[jj,:] * (Ts[jj,:] - Tz_Current[0,:])

            # Creating X for Current Timestep
            Test_X_TF_Array[jj,QHVAC_Indices] = QHVAC_Computed
            Test_X_TF_Array[jj,Tz_Indices] = Tz_Current

            # Predicting Next Timestep Temperature
            if (TimeStepper_ModelType == 1): # Model Predicts Next State Directly

                Test_Y_Predict[jj,:] = Current_ANNModel(np.reshape(Test_X_TF_Array[jj,:], (1, Test_X_TF_Array.shape[1])))

            elif(TimeStepper_ModelType == 2):  # Model Computes Current Derivative

                Test_Y_Predict[jj,:] = Test_X_TF_Array[jj,Tz_Indices[-Total_Aggregation_Zone_Number:]] + Current_ANNModel(np.reshape(Test_X_TF_Array[jj,:], (1, Test_X_TF_Array.shape[1])))

            # Current QHVAC_Computed
            QHVAC_Computed_Current = QHVAC_Computed[-Total_Aggregation_Zone_Number:]   

            # Computing PHVAC Current
            for kk in range(Total_Aggregation_Zone_Number):

                QHVAC_Computed_Array = np.abs(np.reshape(QHVAC_Computed_Current[kk],(1,1)))
                PHVAC_Regression_Model = PHVAC_Regression_Model_List[kk]
                PHVAC_Dict = PHVAC_Regression_Model(QHVAC_Computed_Array, training=False)
                PHVAC_Current[jj, kk] = list(PHVAC_Dict.values())[0]

            # Feedback Step
            Tz_Current[0,:-Total_Aggregation_Zone_Number] = Tz_Current[0,Total_Aggregation_Zone_Number:]
            Tz_Current[0,-Total_Aggregation_Zone_Number:] = Test_Y_Predict[jj,:]

        # Timing Simulation
        Sim_EndTime = time.time()

        SimTime = (Sim_EndTime - Sim_StartTime)/(Test_X_TF_Array.shape[0])

        # Computing PHVAC
        PHVAC = PHVAC_Current.sum(axis=1)

        # Predicting on Training and Testing Set Using Trained Model without Simulation
        if (TimeStepper_ModelType == 1): # Model Predicts Next State Directly

            Test_Y_Predict1 = Current_ANNModel.predict(Test_X_TF_Array)
            Train_Y_Predict1 = Current_ANNModel.predict(Train_X_TF)

        elif (TimeStepper_ModelType == 2):  # Model Computes Current Derivative
            
            Test_Y_Predict1 = tf.reshape(Test_X_TF_Array[:,Tz_Indices[-Total_Aggregation_Zone_Number:]],[Test_X_TF_Array[:,Tz_Indices[-Total_Aggregation_Zone_Number:]].shape[0],Total_Aggregation_Zone_Number]) + Current_ANNModel.predict(Test_X_TF_Array)
            Train_Y_Predict1 =  tf.reshape(Train_X_TF.numpy()[:,Tz_Indices[-Total_Aggregation_Zone_Number:]],[Train_X_TF.numpy()[:,Tz_Indices[-Total_Aggregation_Zone_Number:]].shape[0],Total_Aggregation_Zone_Number]) + Current_ANNModel.predict(Train_X_TF)

        for kk in range(Total_Aggregation_Zone_Number):

            kk1 = kk + 1

            PHVAC_Regression_Model = PHVAC_Regression_Model_List[kk]
            PHVAC1_Dict = PHVAC_Regression_Model(np.abs(np.reshape(np.array([Test_X['QAC_Corrected'+ str(kk1)].iloc[Lag_Number-1:]]), (Test_X['QAC_Corrected'+ str(kk1)].iloc[Lag_Number-1:].shape[0],1) )))
            PHVAC1_Current = list(PHVAC1_Dict.values())[0]

            PHVAC1 = PHVAC1 + PHVAC1_Current

    elif ((ANN_Type == 2) or (ANN_Type == 3) or (ANN_Type == 4)):

        # Converting Test_X tensors to numpy arrays
        Test_X_TF_Array = Test_X_TF.numpy()
        Test_X_TF_Array = np.reshape(Test_X_TF_Array, (Test_X_TF_Array.shape[0], Test_X_TF_Array.shape[1], Test_X_TF_Array.shape[2]))

        Test_X_M_Dot_Ts_TF_Array = Test_X_M_Dot_Ts_TF.numpy()
        Test_X_M_Dot_Ts_TF_Array = np.reshape(Test_X_M_Dot_Ts_TF_Array, (Test_X_M_Dot_Ts_TF_Array.shape[0], Test_X_M_Dot_Ts_TF_Array.shape[1], Test_X_M_Dot_Ts_TF_Array.shape[2]))

        Test_Y_Predict = np.zeros((Test_X_TF_Array.shape[0],Total_Aggregation_Zone_Number))
        # Test_Y_Predict = np.zeros((Test_X_TF_Array.shape[0]))

        # Initializing the Simulation
        PHVAC_Current = np.zeros((Test_X_TF_Array.shape[0], Total_Aggregation_Zone_Number))
        # PHVAC_Current = np.zeros((Test_X_TF_Array.shape[0],1))

        # Getting Appropriate Indices
        Tz_Indices = X_TF_A_Indices
        QHVAC_Indices = Q_HVAC_A_Indices
        M_Dot_Indices = list(range(0, Test_X_M_Dot_Ts_TF_Array.shape[2], 2))
        Ts_Indices = list(range(1, Test_X_M_Dot_Ts_TF_Array.shape[2]+1, 2))

        # Initializing Initial Temp and Control Signals
        Tz_Current = np.reshape(Test_X_TF_Array[0,:,Tz_Indices], (1, Test_X_TF_Array.shape[1], len(Tz_Indices)))
        M_Dot = Test_X_M_Dot_Ts_TF_Array[:,:,M_Dot_Indices]
        Ts = Test_X_M_Dot_Ts_TF_Array[:,:,Ts_Indices]

        Ca = 1.004

        # Timing Simulation
        Sim_StartTime = time.time()

        # LOOP: For Loop Simulation
        for jj in range(Test_X_TF_Array.shape[0]):

            # Computing QHVAC from Predicted Temperature
            QHVAC_Computed = Ca * M_Dot[jj,:,:] * (Ts[jj,:,:] - Tz_Current[0,:,:])
            QHVAC_Computed = np.reshape(QHVAC_Computed, (1, Lag_Number, Total_Aggregation_Zone_Number))

            # Creating X for Current Timestep
            Test_X_TF_Array[jj,:,QHVAC_Indices] = QHVAC_Computed
            Test_X_TF_Array[jj,:,Tz_Indices] = Tz_Current

            # Scaling of Input
            Test_X_Scaled = Scaler_Train_X.transform(Test_X_TF_Array[jj,:,:].reshape([Test_X_TF_Array.shape[1], Test_X_TF_Array.shape[2]]))

            # Reshaping Input for RNN
            Test_X_TF_Array_Reshaped = tf.reshape(Test_X_Scaled, [1,Test_X_Scaled.shape[0],Test_X_Scaled.shape[1]])

            # Predicting Next Timestep Temperature
            if (TimeStepper_ModelType == 1): # Model Predicts Next State Directly

                Test_Y_Predict[jj,:] = Current_ANNModel(Test_X_TF_Array_Reshaped)

            elif(TimeStepper_ModelType == 2):  # Model Computes Current Derivative

                Test_Y_Predict[jj,:] = tf.cast(tf.reshape(Test_X_TF_Array_Reshaped.numpy()[0,Lag_Number-1,Tz_Indices],[1,Total_Aggregation_Zone_Number]), tf.float32) + Current_ANNModel(Test_X_TF_Array_Reshaped)

            # Undo Scaling of Output
            if (Total_Aggregation_Zone_Number == 1):

                Test_Y_Predict[jj,:] = Scaler_Train_Y.inverse_transform(np.array(Test_Y_Predict[jj,:]).reshape(1,-1))

            else:

                Test_Y_Predict[jj,:] = Scaler_Train_Y.inverse_transform(np.array(Test_Y_Predict[jj,:]).reshape(1,Test_Y_Predict.shape[1]))   

            # Current QHVAC_Computed
            QHVAC_Computed_Current = QHVAC_Computed[0,-1,:] 

            # Computing PHVAC Current
            for kk in range(Total_Aggregation_Zone_Number):

                QHVAC_Computed_Array = np.abs(np.reshape(QHVAC_Computed[0,0,kk],(1,1)))
                PHVAC_Regression_Model = PHVAC_Regression_Model_List[kk]
                PHVAC_Dict = PHVAC_Regression_Model(QHVAC_Computed_Array, training=False)
                PHVAC_Current[jj, kk] = list(PHVAC_Dict.values())[0]

            # Feedback Step
            Tz_Current[0,:-1,:] = Tz_Current[0,1:,:]
            Tz_Current[0,-1,:] = tf.reshape(Test_Y_Predict[jj,:],[1,1,Total_Aggregation_Zone_Number])

        # Timing Simulation
        Sim_EndTime = time.time()

        SimTime = (Sim_EndTime - Sim_StartTime)/(Test_X_TF_Array.shape[0])

        PHVAC = PHVAC_Current.sum(axis=1)

        # Predicting on Training and Testing Set Using Trained Model without Simulation

        #Test_Y_Predict1 = Current_ANNModel.predict(Test_X)
        #Train_Y_Predict1 = Current_ANNModel.predict(Train_X)
        for kk in range(Test_X_TF.shape[0]):

            Test_X_TF_Scaled_batch = Scaler_Train_X.transform(Test_X_TF[kk,:,:])

            if (kk == 0):

                Test_X_TF_reshaped= tf.reshape(Test_X_TF_Scaled_batch, [1,Lag_Number,Test_X_TF.shape[2]])

            else:

                Test_X_TF_reshaped = tf.concat([Test_X_TF_reshaped, tf.reshape(Test_X_TF_Scaled_batch, [1,Lag_Number,Test_X_TF.shape[2]])], axis=0)
        
        if (TimeStepper_ModelType == 1): # Model Predicts Next State Directly

            Test_Y_Predict1 = Current_ANNModel.predict(Test_X_TF_reshaped)           
            Train_Y_Predict1 = Current_ANNModel.predict(Train_X_TF_11)

            # Undo Scaling of Output
            if (Total_Aggregation_Zone_Number == 1):

                Test_Y_Predict1 = Scaler_Train_Y.inverse_transform(Test_Y_Predict1.reshape(-1,1))
                Train_Y_Predict1 = Scaler_Train_Y.inverse_transform(Train_Y_Predict1.reshape(-1,1))

            else:

                Test_Y_Predict1 = Scaler_Train_Y.inverse_transform(Test_Y_Predict1.reshape(-1,Test_Y_Predict1.shape[1]))
                Train_Y_Predict1 = Scaler_Train_Y.inverse_transform(Train_Y_Predict1.reshape(-1,Train_Y_Predict1.shape[1]))
            

        elif (TimeStepper_ModelType == 2):  # Model Computes Current Derivative            

            Test_Y_Predict1 = tf.reshape(Test_X_TF_reshaped.numpy()[:,Lag_Number-1,Tz_Indices],[Test_X_TF_reshaped.numpy()[:,Lag_Number-1,Tz_Indices].shape[0],Total_Aggregation_Zone_Number]) + Current_ANNModel.predict(Test_X_TF_reshaped)           
            Train_Y_Predict1 = tf.reshape(Train_X_TF_11.numpy()[:,Lag_Number-1,Tz_Indices],[Train_X_TF_11.numpy()[:,Lag_Number-1,Tz_Indices].shape[0],Total_Aggregation_Zone_Number])  + Current_ANNModel.predict(Train_X_TF_11)

            # Undo Scaling of Output
            if (Total_Aggregation_Zone_Number == 1):

                Test_Y_Predict1 = Scaler_Train_Y.inverse_transform(Test_Y_Predict1.numpy().reshape(-1,1))
                Train_Y_Predict1 = Scaler_Train_Y.inverse_transform(Train_Y_Predict1.numpy().reshape(-1,1))

            else:

                Test_Y_Predict1 = Scaler_Train_Y.inverse_transform(Test_Y_Predict1.numpy().reshape(-1,Test_Y_Predict1.shape[1]))
                Train_Y_Predict1 = Scaler_Train_Y.inverse_transform(Train_Y_Predict1.numpy().reshape(-1,Train_Y_Predict1.shape[1]))
          

        for kk in range(Total_Aggregation_Zone_Number):

            kk1 = kk + 1

            PHVAC_Regression_Model = PHVAC_Regression_Model_List[kk]
            PHVAC1_Dict = PHVAC_Regression_Model(np.abs(np.reshape(np.array([Test_X['QAC_Corrected'+ str(kk1)].iloc[Lag_Number-1:]]), (Test_X['QAC_Corrected'+ str(kk1)].iloc[Lag_Number-1:].shape[0],1) )))
            PHVAC1_Current = list(PHVAC1_Dict.values())[0]

            PHVAC1 = PHVAC1 + PHVAC1_Current

elif (NeuralStateSpace_ModelType == 2):  # Nueral State Space

    if (ANN_Type == 1):
    
        # Converting Test_X tensors to numpy arrays
        Test_X_TF_Array_A = Test_X_TF_A.numpy()
        Test_X_TF_Array_A = np.reshape(Test_X_TF_Array_A, (Test_X_TF_Array_A.shape[0], Test_X_TF_Array_A.shape[1]))
        
        Test_X_TF_Array_B = Test_X_TF_B.numpy()
        Test_X_TF_Array_B = np.reshape(Test_X_TF_Array_B, (Test_X_TF_Array_B.shape[0], Test_X_TF_Array_B.shape[1]))

        Test_X_M_Dot_Ts_TF_Array = Test_X_M_Dot_Ts_TF.numpy()
        Test_X_M_Dot_Ts_TF_Array = np.reshape(Test_X_M_Dot_Ts_TF_Array, (Test_X_M_Dot_Ts_TF_Array.shape[0], Test_X_M_Dot_Ts_TF_Array.shape[1]))

        Test_Y_Predict = np.zeros((Test_X_TF_Array_A.shape[0],Total_Aggregation_Zone_Number))
        # Test_Y_Predict = np.zeros((Test_X_TF_Array.shape[0]))

        # Initializing the Simulation
        PHVAC_Current = np.zeros((Test_X_TF_Array_A.shape[0], Total_Aggregation_Zone_Number))
        # PHVAC_Current = np.zeros((Test_X_TF_Array.shape[0],1))

        # Getting Appropriate Indices
        Tz_Indices = list(range(0, Test_X_TF_Array_A.shape[1], 1))
        QHVAC_Indices = Q_HVAC_B_Indices
        
        M_Dot_Indices = list(range(0, Test_X_M_Dot_Ts_TF_Array.shape[1], 2))
        Ts_Indices = list(range(1, Test_X_M_Dot_Ts_TF_Array.shape[1]+1, 2))

        # Initializing Initial Temp and Control Signals
        # Tz_Current = Test_X_TF_Array_A[0,Tz_Indices]
        # M_Dot = Test_X_M_Dot_Ts_TF_Array[:,M_Dot_Indices]
        # Ts = Test_X_M_Dot_Ts_TF_Array[:,Ts_Indices]
        Tz_Current = np.reshape(Test_X_TF_Array_A[0,Tz_Indices], (1, len(Tz_Indices)))
        M_Dot = np.reshape(Test_X_M_Dot_Ts_TF_Array[:,M_Dot_Indices], (Test_X_M_Dot_Ts_TF_Array.shape[0], len(M_Dot_Indices)))
        Ts = np.reshape(Test_X_M_Dot_Ts_TF_Array[:,Ts_Indices], (Test_X_M_Dot_Ts_TF_Array.shape[0], len(Ts_Indices)))

        Ca = 1.004    

        # Timing Simulation
        Sim_StartTime = time.time()

        # LOOP: For Loop Simulation
        for jj in range(Test_X_TF_Array_A.shape[0]):

            # Computing QHVAC from Predicted Temperature
            QHVAC_Computed = Ca * M_Dot[jj,:] * (Ts[jj,:] - Tz_Current[0,:])

            # Creating X for Current Timestep
            Test_X_TF_Array_B[jj,QHVAC_Indices] = QHVAC_Computed
            Test_X_TF_Array_A[jj,Tz_Indices] = Tz_Current

            # Predicting Next Timestep Temperature
            if (TimeStepper_ModelType == 1): # Model Predicts Next State Directly

                # Test_Y_Predict[jj,:] = Current_ANNModel_A(Test_X_TF_Array_A[jj,:]) + Current_ANNModel_B(Test_X_TF_Array_B[jj,:])
                Test_Y_Predict[jj,:] = Current_ANNModel_A(tf.reshape(Test_X_TF_Array_A[jj,:], (1, Test_X_TF_Array_A[jj,:].shape[0]))) + Current_ANNModel_B(tf.reshape(Test_X_TF_Array_B[jj,:], (1, Test_X_TF_Array_B[jj,:].shape[0])))

            elif(TimeStepper_ModelType == 2):  # Model Computes Current Derivative

                Test_Y_Predict[jj,:] = Test_X_TF_Array_A[jj,Tz_Indices[-Total_Aggregation_Zone_Number:]] +  Current_ANNModel_A(tf.reshape(Test_X_TF_Array_A[jj,:], (1, Test_X_TF_Array_A[jj,:].shape[0]))) + Current_ANNModel_B(tf.reshape(Test_X_TF_Array_B[jj,:], (1, Test_X_TF_Array_B[jj,:].shape[0])))

            # Current QHVAC_Computed
            QHVAC_Computed_Current = QHVAC_Computed[-Total_Aggregation_Zone_Number:]   

            # Computing PHVAC Current
            for kk in range(Total_Aggregation_Zone_Number):

                QHVAC_Computed_Array = np.abs(np.reshape(QHVAC_Computed_Current[kk],(1,1)))
                PHVAC_Regression_Model = PHVAC_Regression_Model_List[kk]
                PHVAC_Dict = PHVAC_Regression_Model(QHVAC_Computed_Array, training=False)
                PHVAC_Current[jj, kk] = list(PHVAC_Dict.values())[0]

            # Feedback Step
            Tz_Current[0,:-Total_Aggregation_Zone_Number] = Tz_Current[0,Total_Aggregation_Zone_Number:]
            Tz_Current[0,-Total_Aggregation_Zone_Number:] = Test_Y_Predict[jj,:]

        # Timing Simulation
        Sim_EndTime = time.time()

        SimTime = (Sim_EndTime - Sim_StartTime)/(Test_X_TF_Array_A.shape[0])

        # Computing PHVAC
        PHVAC = PHVAC_Current.sum(axis=1)

        # Predicting on Training and Testing Set Using Trained Model without Simulation
        if (TimeStepper_ModelType == 1): # Model Predicts Next State Directly

            Test_Y_Predict1 = Current_ANNModel_A.predict(Test_X_TF_Array_A) + Current_ANNModel_B.predict(Test_X_TF_Array_B)
            Train_Y_Predict1 = Current_ANNModel_A.predict(Train_X_TF_A) + Current_ANNModel_B.predict(Train_X_TF_B)

        elif (TimeStepper_ModelType == 2):  # Model Computes Current Derivative
            
            Test_Y_Predict1 = tf.reshape(Test_X_TF_Array_A[:,Tz_Indices[-Total_Aggregation_Zone_Number:]],[Test_X_TF_Array_A[:,Tz_Indices[-Total_Aggregation_Zone_Number:]].shape[0],Total_Aggregation_Zone_Number]) + Current_ANNModel_A.predict(Test_X_TF_Array_A) + Current_ANNModel_B.predict(Test_X_TF_Array_B)
            Train_Y_Predict1 =  tf.reshape(Train_X_TF_A.numpy()[:,Tz_Indices[-Total_Aggregation_Zone_Number:]],[Train_X_TF_A.numpy()[:,Tz_Indices[-Total_Aggregation_Zone_Number:]].shape[0],Total_Aggregation_Zone_Number]) + Current_ANNModel_A.predict(Train_X_TF_A) + Current_ANNModel_B.predict(Train_X_TF_B)

        for kk in range(Total_Aggregation_Zone_Number):

            kk1 = kk + 1

            PHVAC_Regression_Model = PHVAC_Regression_Model_List[kk]
            PHVAC1_Dict = PHVAC_Regression_Model(np.abs(np.reshape(np.array([Test_X['QAC_Corrected'+ str(kk1)].iloc[Lag_Number-1:]]), (Test_X['QAC_Corrected'+ str(kk1)].iloc[Lag_Number-1:].shape[0],1) )))
            PHVAC1_Current = list(PHVAC1_Dict.values())[0]

            PHVAC1 = PHVAC1 + PHVAC1_Current

    elif ((ANN_Type == 2) or (ANN_Type == 3) or (ANN_Type == 4)):

        # Converting Test_X tensors to numpy arrays
        Test_X_TF_Array_A = Test_X_TF_A.numpy()
        Test_X_TF_Array_A = np.reshape(Test_X_TF_Array_A, (Test_X_TF_Array_A.shape[0], Test_X_TF_Array_A.shape[1], Test_X_TF_Array_A.shape[2]))
        
        Test_X_TF_Array_B = Test_X_TF_B.numpy()
        Test_X_TF_Array_B = np.reshape(Test_X_TF_Array_B, (Test_X_TF_Array_B.shape[0], Test_X_TF_Array_B.shape[1], Test_X_TF_Array_B.shape[2]))

        Test_X_M_Dot_Ts_TF_Array = Test_X_M_Dot_Ts_TF.numpy()
        Test_X_M_Dot_Ts_TF_Array = np.reshape(Test_X_M_Dot_Ts_TF_Array, (Test_X_M_Dot_Ts_TF_Array.shape[0], Test_X_M_Dot_Ts_TF_Array.shape[1], Test_X_M_Dot_Ts_TF_Array.shape[2]))

        Test_Y_Predict = np.zeros((Test_X_TF_Array_A.shape[0],Total_Aggregation_Zone_Number))
        # Test_Y_Predict = np.zeros((Test_X_TF_Array.shape[0]))

        # Initializing the Simulation
        PHVAC_Current = np.zeros((Test_X_TF_Array_A.shape[0], Total_Aggregation_Zone_Number))
        # PHVAC_Current = np.zeros((Test_X_TF_Array.shape[0],1))

        # Getting Appropriate Indices
        Tz_Indices = list(range(0, Test_X_TF_Array_A.shape[2], 1))
        QHVAC_Indices = list(range(0, Test_X_TF_Array_B.shape[2]-5, 1))
        
        M_Dot_Indices = list(range(0, Test_X_M_Dot_Ts_TF_Array.shape[2], 2))
        Ts_Indices = list(range(1, Test_X_M_Dot_Ts_TF_Array.shape[2]+1, 2))

        # Initializing Initial Temp and Control Signals
        Tz_Current = np.reshape(Test_X_TF_Array_A[0,:,Tz_Indices], (1, Test_X_TF_Array_A.shape[1], len(Tz_Indices)))
        M_Dot = Test_X_M_Dot_Ts_TF_Array[:,:,M_Dot_Indices]
        Ts = Test_X_M_Dot_Ts_TF_Array[:,:,Ts_Indices]

        Ca = 1.004

        # Timing Simulation
        Sim_StartTime = time.time()

        # LOOP: For Loop Simulation
        for jj in range(Test_X_TF_Array_A.shape[0]):

            # Computing QHVAC from Predicted Temperature
            QHVAC_Computed = Ca * M_Dot[jj,:,:] * (Ts[jj,:,:] - Tz_Current[0,:,:])
            QHVAC_Computed = np.reshape(QHVAC_Computed, (1, Test_X_TF_Array_A.shape[1], len(Tz_Indices)))
            
            # Creating X for Current Timestep
            Test_X_TF_Array_B[jj,:,QHVAC_Indices] = QHVAC_Computed
            Test_X_TF_Array_A[jj,:,Tz_Indices] = Tz_Current
            
            # Combining Test_X_TF_Array_B and Test_X_TF_Array_A for Scaling Opration [Check if Right!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!] is Right
            Test_X_TF_Array = np.zeros((1,Lag_Number,Test_X_TF_Array_A.shape[2]+Test_X_TF_Array_B.shape[2]))
            
            Test_X_TF_Array[0,:,X_TF_A_Indices] = Test_X_TF_Array_A[jj,:,:]
            Test_X_TF_Array[0,:,X_TF_B_Indices] = Test_X_TF_Array_B[jj,:,:].transpose()

            # Scaling of Input
            Test_X_Scaled = Scaler_Train_X.transform(Test_X_TF_Array.reshape([Test_X_TF_Array.shape[1], Test_X_TF_Array.shape[2]]))

            # Reshaping Input for RNN
            Test_X_TF_Array_Reshaped = tf.reshape(Test_X_Scaled, [1,Lag_Number,Test_X_TF_Array_A.shape[2]+Test_X_TF_Array_B.shape[2]])
            
            # Disaggregating Test_X_TF_Array_Reshaped into Test_X_TF_Array_Reshaped_A and Test_X_TF_Array_Reshaped_B
            Test_X_TF_Array_Reshaped_A = tf.convert_to_tensor(Test_X_TF_Array_Reshaped.numpy()[:,:,X_TF_A_Indices], dtype=tf.float32)
            Test_X_TF_Array_Reshaped_B = tf.convert_to_tensor(Test_X_TF_Array_Reshaped.numpy()[:,:,X_TF_B_Indices], dtype=tf.float32)
            

            # Predicting Next Timestep Temperature
            if (TimeStepper_ModelType == 1): # Model Predicts Next State Directly

                Test_Y_Predict[jj,:] = Current_ANNModel_A(Test_X_TF_Array_Reshaped_A) + Current_ANNModel_B(Test_X_TF_Array_Reshaped_B)

            elif(TimeStepper_ModelType == 2):  # Model Computes Current Derivative

                Test_Y_Predict[jj,:] = tf.cast(tf.reshape(tf.convert_to_tensor(Test_X_TF_Array_Reshaped_A.numpy()[0,Lag_Number-1,Tz_Indices]),[1,Total_Aggregation_Zone_Number]), tf.float32) + Current_ANNModel_A(Test_X_TF_Array_Reshaped_A) + Current_ANNModel_B(Test_X_TF_Array_Reshaped_B)

            # Undo Scaling of Output
            if (Total_Aggregation_Zone_Number == 1):

                Test_Y_Predict[jj,:] = Scaler_Train_Y.inverse_transform(np.array(Test_Y_Predict[jj,:]).reshape(1,-1))

            else:

                Test_Y_Predict[jj,:] = Scaler_Train_Y.inverse_transform(np.array(Test_Y_Predict[jj,:]).reshape(1,Test_Y_Predict.shape[1]))   

            # Current QHVAC_Computed
            QHVAC_Computed_Current = QHVAC_Computed[0,-1,:] 

            # Computing PHVAC Current
            for kk in range(Total_Aggregation_Zone_Number):

                QHVAC_Computed_Array = np.abs(np.reshape(QHVAC_Computed[0,0,kk],(1,1)))
                PHVAC_Regression_Model = PHVAC_Regression_Model_List[kk]
                PHVAC_Dict = PHVAC_Regression_Model(QHVAC_Computed_Array, training=False)
                PHVAC_Current[jj, kk] = list(PHVAC_Dict.values())[0]

            # Feedback Step
            Tz_Current[0,:-1,:] = Tz_Current[0,1:,:]
            Tz_Current[0,-1,:] = tf.reshape(Test_Y_Predict[jj,:],[1,1,Total_Aggregation_Zone_Number])

        # Timing Simulation
        Sim_EndTime = time.time()

        SimTime = (Sim_EndTime - Sim_StartTime)/(Test_X_TF_Array_A.shape[0])

        PHVAC = PHVAC_Current.sum(axis=1)

        # Predicting on Training and Testing Set Using Trained Model without Simulation

        #Test_Y_Predict1 = Current_ANNModel.predict(Test_X)
        #Train_Y_Predict1 = Current_ANNModel.predict(Train_X)
        
        Test_X_TF_1 = np.zeros((Test_X_TF_Array_A.shape[0],Lag_Number,Test_X_TF_Array_A.shape[2]+Test_X_TF_Array_B.shape[2]))
        
        Test_X_TF_1[:,:,X_TF_A_Indices] = Test_X_TF_Array_A[:,:,:]
        Test_X_TF_1[:,:,X_TF_B_Indices] = Test_X_TF_Array_B[:,:,:]
        
        for kk in range(Test_X_TF_1.shape[0]):

            Test_X_TF_Scaled_batch = Scaler_Train_X.transform(Test_X_TF_1[kk,:,:])

            if (kk == 0):

                Test_X_TF_reshaped= tf.reshape(Test_X_TF_Scaled_batch, [1,Lag_Number,Test_X_TF_1.shape[2]])

            else:

                Test_X_TF_reshaped = tf.concat([Test_X_TF_reshaped, tf.reshape(Test_X_TF_Scaled_batch, [1,Lag_Number,Test_X_TF_1.shape[2]])], axis=0)
                
        # Disaggregating Test_X_TF_reshaped into Test_X_TF_reshaped_A and Test_X_TF_reshaped_B
        Test_X_TF_reshaped_A = tf.convert_to_tensor(Test_X_TF_reshaped.numpy()[:,:,X_TF_A_Indices], dtype=tf.float32)
        Test_X_TF_reshaped_B = tf.convert_to_tensor(Test_X_TF_reshaped.numpy()[:,:,X_TF_B_Indices], dtype=tf.float32)
        
        if (TimeStepper_ModelType == 1): # Model Predicts Next State Directly

            Test_Y_Predict1 = Current_ANNModel_A.predict(Test_X_TF_reshaped_A) + Current_ANNModel_B.predict(Test_X_TF_reshaped_B)         
            Train_Y_Predict1 = Current_ANNModel_A.predict(Train_X_TF_11_A) + Current_ANNModel_B.predict(Train_X_TF_11_B)

            # Undo Scaling of Output
            if (Total_Aggregation_Zone_Number == 1):

                Test_Y_Predict1 = Scaler_Train_Y.inverse_transform(Test_Y_Predict1.reshape(-1,1))
                Train_Y_Predict1 = Scaler_Train_Y.inverse_transform(Train_Y_Predict1.reshape(-1,1))

            else:

                Test_Y_Predict1 = Scaler_Train_Y.inverse_transform(Test_Y_Predict1.reshape(-1,Test_Y_Predict1.shape[1]))
                Train_Y_Predict1 = Scaler_Train_Y.inverse_transform(Train_Y_Predict1.reshape(-1,Train_Y_Predict1.shape[1]))
            

        elif (TimeStepper_ModelType == 2):  # Model Computes Current Derivative            

            # Test_Y_Predict1 = tf.reshape(Test_X_TF_reshaped_A[:,Lag_Number-1,Tz_Indices],[Test_X_TF_reshaped_A[:,Lag_Number-1,Tz_Indices].shape[0],Total_Aggregation_Zone_Number]) + Current_ANNModel_A.predict(Test_X_TF_reshaped_A) + Current_ANNModel_B.predict(Test_X_TF_reshaped_B)           
            # Train_Y_Predict1 = tf.reshape(Train_X_TF_11_A[:,Lag_Number-1,Tz_Indices],[Train_X_TF_11_A[:,Lag_Number-1,Tz_Indices].shape[0],Total_Aggregation_Zone_Number])  + Current_ANNModel_A.predict(Train_X_TF_11_A) + Current_ANNModel_B.predict(Train_X_TF_11_B)
 
            Test_Y_Predict1 = tf.reshape(tf.convert_to_tensor(Test_X_TF_reshaped_A.numpy()[:,Lag_Number-1,Tz_Indices]),[Test_X_TF_reshaped_A.numpy()[:,Lag_Number-1,Tz_Indices].shape[0],Total_Aggregation_Zone_Number]) + Current_ANNModel_A.predict(Test_X_TF_reshaped_A) + Current_ANNModel_B.predict(Test_X_TF_reshaped_B)           
            Train_Y_Predict1 = tf.reshape(tf.convert_to_tensor(Train_X_TF_11_A.numpy()[:,Lag_Number-1,Tz_Indices]),[Train_X_TF_11_A.numpy()[:,Lag_Number-1,Tz_Indices].shape[0],Total_Aggregation_Zone_Number]) + Current_ANNModel_A.predict(Train_X_TF_11_A) + Current_ANNModel_B.predict(Train_X_TF_11_B)

            # Undo Scaling of Output
            if (Total_Aggregation_Zone_Number == 1):

                Test_Y_Predict1 = Scaler_Train_Y.inverse_transform(Test_Y_Predict1.numpy().reshape(-1,1))
                Train_Y_Predict1 = Scaler_Train_Y.inverse_transform(Train_Y_Predict1.numpy().reshape(-1,1))

            else:

                Test_Y_Predict1 = Scaler_Train_Y.inverse_transform(Test_Y_Predict1.numpy().reshape(-1,Test_Y_Predict1.shape[1]))
                Train_Y_Predict1 = Scaler_Train_Y.inverse_transform(Train_Y_Predict1.numpy().reshape(-1,Train_Y_Predict1.shape[1]))
          

        for kk in range(Total_Aggregation_Zone_Number):

            kk1 = kk + 1

            PHVAC_Regression_Model = PHVAC_Regression_Model_List[kk]
            PHVAC1_Dict = PHVAC_Regression_Model(np.abs(np.reshape(np.array([Test_X['QAC_Corrected'+ str(kk1)].iloc[Lag_Number-1:]]), (Test_X['QAC_Corrected'+ str(kk1)].iloc[Lag_Number-1:].shape[0],1) )))
            PHVAC1_Current = list(PHVAC1_Dict.values())[0]

            PHVAC1 = PHVAC1 + PHVAC1_Current


# =============================================================================
# Computing Percentage Accuracy of the Trained Model on Train/Test Data
# =============================================================================  

# LOOP: For Loop for Each Zone Type to Compute Results
for Zone_Number in range(Total_Aggregation_Zone_Number):

    Zone_Number1 = Zone_Number + 1

    # Computing Percentage Accuracy of the Model without Simulation
    Train_PercentageAccuracy = np.mean(((np.absolute(np.reshape(Train_Y_Predict1[:, Zone_Number], (Train_Y_Predict1.shape[0],1))-np.reshape(Train_Y_TF[:, Zone_Number],(Train_Y_Predict1.shape[0],1))))/(np.reshape(Train_Y_TF[:, Zone_Number],(Train_Y_Predict1.shape[0],1)))) *100)

    Test_PercentageAccuracy = np.mean(((np.absolute(np.reshape(Test_Y_Predict1[:, Zone_Number], (Test_Y_Predict1.shape[0],1))-np.reshape(Test_Y.to_numpy()[:, Zone_Number],(Test_Y_Predict1.shape[0],1))))/(np.reshape(Test_Y.to_numpy()[:, Zone_Number],(Test_Y_Predict1.shape[0],1)))) *100)


    # Computing Percentage Accuracy of the Model with Simulation
    # Test_Y_Predict = np.reshape(Test_Y_Predict,(Test_Y_Predict.shape[0],1))
    Test_PercentageAccuracy_Sim = np.mean(((np.absolute(np.reshape(Test_Y_Predict[:, Zone_Number], (Test_Y_Predict.shape[0],1))-np.reshape(Test_Y.to_numpy()[:, Zone_Number],(Test_Y_Predict.shape[0],1))))/(np.reshape(Test_Y.to_numpy()[:, Zone_Number],(Test_Y_Predict.shape[0],1)))) *100)

    Train_PercentageAccuracy = Train_PercentageAccuracy.tolist()
    Test_PercentageAccuracy = Test_PercentageAccuracy.tolist()
    Test_PercentageAccuracy_Sim = Test_PercentageAccuracy_Sim.tolist()

    # Appending Percentage Accuracy into Table
    ANN_PercentageAccuracy_Current_DF = pd.DataFrame([[ANNModel_Key + '_Zone_' + str(Zone_Number1), Train_PercentageAccuracy, Test_PercentageAccuracy, Test_PercentageAccuracy_Sim, Final_Train_Error, Final_Train_Loss, Final_Val_Error, Final_Val_Loss, SimTime]],columns=['ANN Model Name', 'Training Mean Error', 'Testing Mean Error Without Simulation', 'Testing Mean Error With Simulation', 'Final Train Error', 'Final Train Loss', 'Final Val Error', 'Final Val Loss', 'Time/Iteration'])
    ANN_PercentageAccuracy_DF = pd.concat([ANN_PercentageAccuracy_DF, ANN_PercentageAccuracy_Current_DF],ignore_index=True)


    ## Plotting ##

    # Prediction Plot without Simulation
    plt.figure()
    plt.plot(Test_Y_Predict1[0:12096,Zone_Number], color='g', label='Predicted Temp')
    plt.plot(Test_Y.iloc[0:12096,Zone_Number], color='b', label='Actual Temp', linestyle='dashed')
    # plt.plot(Test_Y_Predict1[0:2016,Zone_Number], color='g', label='Predicted Temp')
   # plt.plot(Test_Y.iloc[0:2016,Zone_Number], color='b', label='Actual Temp', linestyle='dashed')
    plt.xlabel('Time')
    plt.ylabel('Zone Temperature', labelpad=15)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(Sim_ANNModelData_FolderPath,ANNModel_Key + '_Dep2_' + str(Total_Aggregation_Zone_Number) + 'Zone_' + str(Zone_Number1) + '_PredictionPlot_withoutSim' + '.png'))
    # plt.show()
    plt.close()

    # Prediction Plot with Simulation
    plt.figure()
    plt.plot(Test_Y_Predict[0:12096, Zone_Number], color='g', label='Predicted Temp accounting Bias')
    plt.plot(Test_Y.iloc[0:12096, Zone_Number], color='b', label='Actual Temp', linestyle='dashed')
    # plt.plot(Test_Y_Predict[0:2016, Zone_Number], color='g', label='Predicted Temp accounting Bias')
    # plt.plot(Test_Y.iloc[0:2016, Zone_Number], color='b', label='Actual Temp', linestyle='dashed')
    plt.xlabel('Time')
    plt.ylabel('Zone Temperature', labelpad=15)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(Sim_ANNModelData_FolderPath,
                             ANNModel_Key + '_Dep2_' + str(Total_Aggregation_Zone_Number) + 'Zone_' + str(
                                 Zone_Number1) + '_PredictionPlot_withSim' + '.png'))
    # plt.show()

    plt.close()

'''
# Pair Plots for Training Data
sns.pairplot(Train_X, diag_kind='kde')
plt.gcf().set_size_inches(10, 10)
plt.tight_layout()
plt.savefig(os.path.join(Sim_ANNModelData_FolderPath, ANNModel_Key + '_' + str(Total_Aggregation_Zone_Number) + 'Zone' + '_PairPlot' + '.png'), dpi=300)
# plt.show()
plt.close()
'''

# Training Plot
plt.figure()
plt.plot(Training_Loss_Value_Set, label='Loss')
plt.plot(Val_Loss_Value_Set, label='Val_Loss')
plt.title('Loss Plot: ' + ANNModel_Key + ' Model')
plt.xlabel('Epoch')
plt.ylabel('Loss', labelpad=15)
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(Sim_ANNModelData_FolderPath, ANNModel_Key + '_Dep2_' + str(Total_Aggregation_Zone_Number) + 'Zone' + '_LossPlot' + '.png'))
# plt.show()
plt.close()

# Error Plot
plt.figure()
plt.plot(Training_Error_Value_Set, label='Loss')
plt.plot(Val_Error_Value_Set, label='Val_Loss')
plt.title('Error Plot: ' + ANNModel_Key + ' Model')
plt.xlabel('Epoch')
plt.ylabel('Error', labelpad=15)
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(Sim_ANNModelData_FolderPath, ANNModel_Key + '_Dep2_' + str(Total_Aggregation_Zone_Number) + 'Zone' + '_ErrorPlot' + '.png'))
# plt.show()
plt.close()

# =============================================================================
# Creating ANN Model Output Data in Sim_ANNModelData Folder
# =============================================================================

# Saving Output Data with Simulation
Predict_Y_List = np.transpose(np.reshape(Test_Y_Predict, (np.shape(Test_Y_Predict)[0],Total_Aggregation_Zone_Number))).tolist()
Actual_Y_DF = copy.deepcopy(Test_Y)

for Number in range(Total_Aggregation_Zone_Number):

    Key_Now = 'Predict_Y' + str(Number+1)

    Predict_Y_Dict_Current = {Key_Now : Predict_Y_List[Number]}

    Predict_Actual_Y_DF = pd.concat([Predict_Actual_Y_DF,pd.DataFrame(Predict_Y_Dict_Current)])
    

Actual_Y_DF.reset_index(drop=True, inplace=True)
Predict_Actual_Y_DF.reset_index(drop=True, inplace=True)

Predict_Actual_Y_DF = pd.concat([pd.DataFrame({'DateTime': DateTime_Test.tolist()[1:]}), Actual_Y_DF, Predict_Actual_Y_DF],axis=1)

Predict_Actual_Y_DF_File_Name = ANNModel_Key + '_Predict_Actual_Y_DF' + '_Dep2_' + str(Total_Aggregation_Zone_Number) + 'Zone' + '_withSim' + '.pickle'

# Saving Output Data without Simulation
Predict_Y_List1 = np.transpose(np.reshape(Test_Y_Predict1, (np.shape(Test_Y_Predict1)[0], Total_Aggregation_Zone_Number))).tolist()
Actual_Y_DF1 = copy.deepcopy(Test_Y)

for Number in range(Total_Aggregation_Zone_Number):

    Key_Now = 'Predict_Y' + str(Number + 1)

    Predict_Y_Dict_Current = {Key_Now: Predict_Y_List1[Number]}

    Predict_Actual_Y_DF1 = pd.concat([Predict_Actual_Y_DF1, pd.DataFrame(Predict_Y_Dict_Current)])

Actual_Y_DF1.reset_index(drop=True, inplace=True)
Predict_Actual_Y_DF1.reset_index(drop=True, inplace=True)

Predict_Actual_Y_DF1 = pd.concat([pd.DataFrame({'DateTime': DateTime_Test.tolist()[1:]}), Actual_Y_DF1, Predict_Actual_Y_DF1], axis=1)

Predict_Actual_Y_DF_File_Name1 = ANNModel_Key + '_Predict_Actual_Y_DF' + '_Dep2_' + str(Total_Aggregation_Zone_Number) + 'Zone' + '_withoutSim' + '.pickle'

# =============================================================================
# Storing ANN Model Data in Sim_ANNModelData Folder
# =============================================================================

# Saving the Accuracy Table
ANN_Model_Accuracy_File_Name = ANNModel_Key + 'ANN_Model_Accuracy' + '_Dep2_' + str(Total_Aggregation_Zone_Number) + 'Zone' + '.csv'
ANN_PercentageAccuracy_DF.to_csv(os.path.join(Sim_ANNModelData_FolderPath, ANN_Model_Accuracy_File_Name), index=False)

# Saving ANN Model Output Data as a .pickle File in Results Folder
pickle.dump(Predict_Actual_Y_DF,open(os.path.join(Sim_ANNModelData_FolderPath, Predict_Actual_Y_DF_File_Name), "wb"))
pickle.dump(Predict_Actual_Y_DF1,open(os.path.join(Sim_ANNModelData_FolderPath, Predict_Actual_Y_DF_File_Name1), "wb"))


# Saving Trained ANN Model
if (NeuralStateSpace_ModelType == 1):
    ANNModel_FileName = ANNModel_Key + '_Dep2_' + str(Total_Aggregation_Zone_Number) + 'Zone' + kerasExtention
    Current_ANNModel.save(os.path.join(Sim_ANNModelData_FolderPath, ANNModel_FileName))

elif (NeuralStateSpace_ModelType == 2):
    ANNModel_FileName_A = ANNModel_Key + '_Dep2_' + str(Total_Aggregation_Zone_Number) + 'Zone' + '_A' + kerasExtention
    Current_ANNModel_A.save(os.path.join(Sim_ANNModelData_FolderPath, ANNModel_FileName_A))
    
    ANNModel_FileName_B = ANNModel_Key + '_Dep2_' + str(Total_Aggregation_Zone_Number) + 'Zone' + '_B' + kerasExtention
    Current_ANNModel_B.save(os.path.join(Sim_ANNModelData_FolderPath, ANNModel_FileName_B))

if ((ANN_Type == 2) or (ANN_Type == 3) or (ANN_Type == 4)):

    # Saving Data Scaler
    TrainDataX_Scaler_FileName = ANNModel_Key + '_TrainDataX_Scaler' + '_Dep2_' + str(Total_Aggregation_Zone_Number) + 'Zone'  + '.save'
    joblib.dump(Scaler_Train_X , os.path.join(Sim_ANNModelData_FolderPath, TrainDataX_Scaler_FileName)) 
    
    TrainDataY_Scaler_FileName = ANNModel_Key + '_TrainDataY_Scaler' + '_Dep2_' + str(Total_Aggregation_Zone_Number) + 'Zone' + '.save'
    joblib.dump(Scaler_Train_Y , os.path.join(Sim_ANNModelData_FolderPath, TrainDataY_Scaler_FileName)) 
    

# =============================================================================
# PHVAC without Simulation: PHVAC1
# =============================================================================

AggregatedTest_DF.reset_index(drop=True, inplace=True)

# PHVAC Plot
plt.figure()
plt.plot(PHVAC1[0:2016], label='PHVAC Computed without Sim')
plt.plot(AggregatedTest_DF['Facility_Total_HVAC_Electric_Demand_Power_'].iloc[0:2016], label='PHVAC Actual')
plt.title('PHVAC Plot: ' + ANNModel_Key + ' Model')
plt.xlabel('Time')
plt.ylabel('PHVAC', labelpad=15)
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(Sim_ANNModelData_FolderPath, ANNModel_Key + '_Dep2_' + str(Total_Aggregation_Zone_Number) + 'Zone_' + 'PHVACPlot_withoutSim' + '.png'))
# plt.show()
plt.close()


# Computing Percentage Accuracy of the PHVAC1 (without Simulation)
PHVAC1_PercentageAccuracy = np.mean(((np.absolute(PHVAC1-np.reshape(AggregatedTest_DF['Facility_Total_HVAC_Electric_Demand_Power_'].iloc[0:-1].to_numpy(),(PHVAC1.shape[0],1))))/(np.reshape(AggregatedTest_DF['Facility_Total_HVAC_Electric_Demand_Power_'].iloc[0:-1].to_numpy(),(PHVAC1.shape[0],1)))) *100)
PHVAC1_PercentageAccuracy = PHVAC1_PercentageAccuracy.tolist()

# Appending Percentage Accuracy into Table
PHVAC1_PercentageAccuracy_DF = pd.DataFrame([[ANNModel_Key, PHVAC1_PercentageAccuracy]],columns=['ANN Model Name', 'PHVAC1_Percentage_Mean_Error'])


Predict_Y_List = np.reshape(PHVAC1, (np.shape(PHVAC1)[0])).tolist()
Actual_Y_List = AggregatedTest_DF['Facility_Total_HVAC_Electric_Demand_Power_'].tolist()

Predict_Actual_PHVAC1_Dict = {'DateTime': DateTime_Test.tolist()[:-1], 'Predict_Y': Predict_Y_List, 'Actual_Y': Actual_Y_List[:-1]}
Predict_Actual_PHVAC1_DF = pd.DataFrame(Predict_Actual_PHVAC1_Dict)

Predict_Actual_PHVAC1_DF_File_Name = ANNModel_Key + '_Predict_Actual_PHVAC_withoutSim' + '_Dep2_' + str(Total_Aggregation_Zone_Number) + 'Zone' + '.pickle'

# Saving PHVAC Model Output Data as a .pickle File in Results Folder
pickle.dump(Predict_Actual_PHVAC1_DF,open(os.path.join(Sim_ANNModelData_FolderPath, Predict_Actual_PHVAC1_DF_File_Name), "wb"))

# Saving the Accuracy Table as a .csv File in Results Folder
PHVAC1_Accuracy_File_Name = ANNModel_Key + '_PHVAC_Model_Mean_Error_withoutSim' + '_Dep2_' + str(Total_Aggregation_Zone_Number) + 'Zone' + '.csv'
PHVAC1_PercentageAccuracy_DF.to_csv(os.path.join(Sim_ANNModelData_FolderPath, PHVAC1_Accuracy_File_Name), index=False)


# =============================================================================
# PHVAC from Simulation
# =============================================================================

# PHVAC Plot
plt.figure()
plt.plot(PHVAC[0:2016], label='PHVAC Computed with Sim - Bias accounted')
plt.plot(AggregatedTest_DF['Facility_Total_HVAC_Electric_Demand_Power_'].iloc[0:2016], label='PHVAC Actual')
plt.title('PHVAC Plot: ' + ANNModel_Key + ' Model')
plt.xlabel('Time')
plt.ylabel('PHVAC', labelpad=15)
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(Sim_ANNModelData_FolderPath, ANNModel_Key + '_Dep2_' + str(Total_Aggregation_Zone_Number) + 'Zone_' + 'PHVACPlot_withSim' + '.png'))
# plt.show()
plt.close()


# Computing Percentage Accuracy of the PHVAC
PHVAC_PercentageAccuracy = np.mean(((np.absolute(PHVAC-np.reshape(AggregatedTest_DF['Facility_Total_HVAC_Electric_Demand_Power_'].iloc[0:-1].to_numpy(),(PHVAC.shape[0],1))))/(np.reshape(AggregatedTest_DF['Facility_Total_HVAC_Electric_Demand_Power_'].iloc[0:-1].to_numpy(),(PHVAC.shape[0],1)))) *100)
PHVAC_PercentageAccuracy = PHVAC_PercentageAccuracy.tolist()

# Appending Percentage Accuracy into Table
PHVAC_PercentageAccuracy_DF = pd.DataFrame([[ANNModel_Key, PHVAC_PercentageAccuracy]],columns=['ANN Model Name', 'PHVAC_Percentage_Mean_Error'])

Predict_Y_List = np.reshape(PHVAC, (np.shape(PHVAC)[0])).tolist()
Actual_Y_List = AggregatedTest_DF['Facility_Total_HVAC_Electric_Demand_Power_'].tolist()

Predict_Actual_PHVAC_Dict = {'DateTime': DateTime_Test.tolist()[:-1], 'Predict_Y': Predict_Y_List, 'Actual_Y': Actual_Y_List[:-1]}
Predict_Actual_PHVAC_DF = pd.DataFrame(Predict_Actual_PHVAC_Dict)

Predict_Actual_PHVAC_DF_File_Name = ANNModel_Key + '_Predict_Actual_PHVAC_withSim' + '_Dep2_' + str(Total_Aggregation_Zone_Number) + 'Zone' + '.pickle'

# Saving PHVAC Model Output Data as a .pickle File in Results Folder
pickle.dump(Predict_Actual_PHVAC_DF,open(os.path.join(Sim_ANNModelData_FolderPath, Predict_Actual_PHVAC_DF_File_Name), "wb"))

# Saving the Accuracy Table as a .csv File in Results Folder
PHVAC_Accuracy_File_Name = ANNModel_Key + '_PHVAC_Model_Mean_Error_withSim' + '_Dep2_'  + str(Total_Aggregation_Zone_Number) + 'Zone' + '.csv'
PHVAC_PercentageAccuracy_DF.to_csv(os.path.join(Sim_ANNModelData_FolderPath, PHVAC_Accuracy_File_Name), index=False)

