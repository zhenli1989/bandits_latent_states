# *A Direct Approach for Handling Contextual Bandits with Latent State Dynamics* (Code for Numerical Simulations)
***

## Introduction
This is an instruction to reproduce the Numerical Simulation Results in of the article 
**A Direct Approach for Handling Contextual Bandits with Latent State Dynamics**. 

---

## Step 0: Prepare the Environment and Download the Underlying Data
1. Unzip the *numerical_simulation.zip* and use the unzipped folder *numerical\_simulation* as **root directory**

2. Setup the Python environment:
    - Python Version: 3.11
    - Dependence Package: In *requirements.txt*

---

## Step 1: Prepare the dataset
- Download the underlying dataset from 
[UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/default+of+credit+card+clients)
and save it as *./data/default of credit card clients.xls*


- Open the script *1\_Prepare\_Data.py* and verify/modify the section **AREA OF INPUT PARAMETERS** if needed 
(see code comments for the explaination of each paramters) as following:

~~~
PATH = "." # Root directory, should be the same path this "README.md" file locates
PATH_DATA = f"{PATH}/data" # Path for data
PATH_MODELS = f"{PATH}/models" # Path for models

retrain_model = True # Default False. If retrain_model is True, it will retrain the PD model. If False, it will load the pretrained model
random_seed = 1989 # Random seed, used when retrain_model is True. To reproduce the PD model, set as 1989
~~~

- Run the script *1\_Prepare\_Data.py*

---

## Step 2-6: Run the script step 2 to step 6 with default parameters and with random seed from 1951 to 2051
- Run the script *2\_Simulation\_Benchmark\_Policy.py* 
- Run the script *3\_Linear\_Bandits\_Simulations.py*
- Run the script *4\_Belief\_estimation\_spectral\_method.py*
- Run the script *5\_BoxA\_Simulations.py*
- Run the script *6\_BoxB\_Simulations.py*
---

## Step 7: Collect Simulation Results and Generate the Graphs/Tables
- Run the script *7\_Export\_Performance.py* and get the **Tables** in *./tables* and **graphs** in 
*./pics*.