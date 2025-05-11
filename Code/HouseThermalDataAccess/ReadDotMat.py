from scipy import io

# Load the .mat file data
data = io.loadmat('HouseData_Gainesville_Baseline_3Months_SC_PVBat1_Bat0_PV0_None0.mat')

# Print the keys to see available variables
print(data.keys())

# Print the particular variable 'House_ThermalModel_History'
desired_data = data['House_ThermalModel_History']
print(desired_data)

