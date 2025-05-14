# Advanced Power Estimation using Traditional ML and Physics-Informed Neural Networks

## Overview

This project presents a comprehensive framework for advanced power estimation in building and power systems by integrating traditional machine learning (ML) techniques and Physics-Informed Neural Networks (PINNs). The approach leverages both data-driven methods and physical laws to deliver accurate, robust, and generalizable power and temperature predictions, even in complex or partially observed environments.

## Key Features

- **Hybrid Modeling Workflow:** Combines regression-based ML models with neural networks informed by domain-specific physics for superior prediction accuracy.
- **Physics-Informed Neural Networks (PINNs):** Incorporates physical constraints (e.g., power flow equations, thermal dynamics) directly into the neural network architecture and loss function, enhancing generalization and reducing the need for large training datasets[1][2][6].
- **Traditional ML Baselines:** Implements and benchmarks classic regression and time series models for power and temperature forecasting[3].
- **Data Aggregation and Preprocessing:** Automated pipeline for simulation data aggregation, feature engineering, and train/test split.
- **Modular Codebase:** Easily extensible for different power system scenarios, including building energy management, grid state estimation, and converter parameter identification[5][6].

![image](https://github.com/user-attachments/assets/8bf787b9-9d2e-40d7-857a-b4d23f84aae6)

## Methodology

![image](https://github.com/user-attachments/assets/246de6f4-5200-4fbf-958d-3ca372855133)

### 1. Data Aggregation & Preprocessing

- Aggregate simulation or measurement data by zone or system component.
- Engineer features such as weather, occupancy, equipment, solar gains, and HVAC operation for building systems, or voltage, current, and power flows for grid systems.
- Split data into training and testing sets for robust model evaluation.

![image](https://github.com/user-attachments/assets/d8fdb0d1-2cee-4293-b4bc-3a3b00b6dc5f)

### 2. Traditional ML Modeling
   ![image](https://github.com/user-attachments/assets/b30d9b67-86b9-4ab7-9bbe-3053376811fe)
   ![image](https://github.com/user-attachments/assets/b5cb0c10-f402-4d28-8754-f36c151f050e)

- Apply regression models (e.g., linear regression, random forest) and time series methods (e.g., LSTM) to establish performance baselines for power and temperature estimation[3].

### 3. Physics-Informed Neural Networks

- Design neural networks that embed physical laws (e.g., conservation of energy, power flow equations) into the loss function and architecture[1][2][4][6].
- Train PINNs to minimize both data-driven and physics-based loss terms, improving accuracy and robustness, especially with limited or noisy data.

### 4. Evaluation

- Compare model predictions with ground truth and assess using metrics such as RMSE, MAE, and relative error.
- Visualize results and analyze model performance under varying data availability and system scenarios.
![image](https://github.com/user-attachments/assets/c051fdaa-7ce6-4ea7-b292-beb512de8df2)
![LSTM_1_TS_2_NSS_2_Ind_1Zone_PHVACPlot_withoutSim](https://github.com/user-attachments/assets/c2cc84f5-9c77-49be-be22-2a0c3600844b)
![GRU_1_TS_2_NSS_2_Ind_1Zone_PHVACPlot_withoutSim](https://github.com/user-attachments/assets/60d9da72-50ed-4d30-b80b-762818591cde)
MLP_L_1_TS_2_NSS_2_Ind_1Zone_PHVACPlot_withoutSim
## Example Use Cases

- **Building Energy Management:** Predict zone temperature and power consumption using hybrid ML and PINN models.
- **Power Grid State Estimation:** Estimate grid states (voltages, angles) with limited sensor data by leveraging physical grid constraints[6].
- **Power Electronics Parameter Estimation:** Identify system parameters (e.g., inertia, damping) in converters using physics-informed ML[5].

## Getting Started

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/advanced-power-estimation.git
   cd advanced-power-estimation
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Prepare your data in the `data/raw/` directory.
4. Run data aggregation and preprocessing scripts:
   ```bash
   python src/data_aggregation.py
   ```
5. Train traditional ML models:
   ```bash
   python src/regression.py
   ```
6. Train and evaluate PINNs:
   ```bash
   python src/pinn.py
   ```

## References

- [Physics-Informed Neural Networks for Power Systems][1]
- [Parameter Estimation of Power Electronic Converters with Physics-informed Machine Learning][5]
- [Physics-Informed Deep Neural Network Method for Limited Observability State Estimation of a Distribution Power Grid][6]
- [Energy Consumption Prediction using Machine Learning][3]

## License

This project is licensed under the MIT License.

---

> *For questions, contributions, or to report issues, please open an issue or submit a pull request.*

---[1][2][3][4][5][6]

Citations:
[1] https://github.com/gmisy/Physics-Informed-Neural-Networks-for-Power-Systems
[2] https://upcommons.upc.edu/bitstream/handle/2117/428167/Physics-Informed_Neural_Networks_for_Power_Systems_Warm-Start_Optimization.pdf?sequence=1&isAllowed=y
[3] https://github.com/MohamadNach/Machine-Learning-to-Predict-Energy-Consumption
[4] https://github.com/xbeat/Machine-Learning/blob/main/Physics%20Informed%20Neural%20Networks%20with%20Python.md
[5] https://github.com/ms140429/PIML_Converter
[6] https://github.com/kostyanoob/Power
[7] https://www.youtube.com/watch?v=1AyAia_NZhQ
[8] https://github.com/MonishaMCA/electricity-demand-prediction-using-machine-learning
[9] https://github.com/idrl-lab/PINNpapers
[10] https://www.mdpi.com/2079-9292/13/11/2208
[11] https://www.linkedin.com/pulse/building-your-first-machine-learning-project-beginners-sahin-ahmed-zxrpc
[12] https://github.com/thunil/Physics-Based-Deep-Learning
[13] https://repository.kaust.edu.sa/bitstreams/03f1349d-62ea-4e31-9872-23cee6639f30/download
[14] https://atomai.readthedocs.io/en/latest/README.html
[15] https://github.com/SiddeshSambasivam/Physics-Informed-Neural-Networks/blob/main/README.md
[16] http://www.chatziva.com/presentations/Chatzivasileiadis_NeuralNetVerification_PIML_PES_Webinar.pdf
[17] https://cran.r-project.org/web/packages/aifeducation/readme/README.html
[18] https://github.com/tsotchke/PINN
[19] https://github.com/CompPhysics/AdvancedMachineLearning
[20] https://github.com/qiaosun22/AwesomePhysicsInformedLLMs
