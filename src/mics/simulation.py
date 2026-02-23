# Import all necessary coding libaries 
import polars as pl 
import pandas as pd
import numpy as np 
import seaborn as sns
import seaborn as sns

# Import model creation libaries 

#### MODEL 1 - Pre-treatment Simulation & Training ####

# STEP 1: Create a simulated population of 100k patinets 

# STEP 2: Simulate patinet characteristics using literature based numbers 
def generate_patients(n:int, random_seed:int)-> pl.DataFrame:
    """Generate as simulated patient dataset containing variables from the QRISK3 algorithm, with n records.
    
    Args:
    n(int):Number of patinet records to generate. 
    random_seed(int):Seed for random number generator for reproducibility

    Returns:
    pl.DataFrame: A Polas DataFrame containing simulated patient data.

    Current columns in the simulated dataset: 
    -age(int):Age of the patient, randomly generated between 20 and 80 
    - sex(str):Sex of the pateint, randomly assigned Male or Female 
    -
    """

    # DEMOGRAPHICS: 
    ages = np.random.randint(20, 80, size=n)
    sex = np.random.choice(["Male","Female"], size=n)

    # Ethnicity: 
    # 9 ethnic groups outlined by the QRISK3 algorithm
    # p is the proportion the simulation model should assign each ethnicity
    ethnicity = np.random.choice(
        ["White","Black African","Black Caribbean",
        "Indian","Pakistani","Bangladeshi","Chinese","Other Asian", "Other Ethic Group"],
        size=n,
        p=[0.850, 0.025, 0.015, 0.010, 0.010, 0.015, 0.015, 0.010, 0.050]
        )


    # Polas dataframe showing the QRISK3 variables
    return pl.DataFrame({
        # Demographics
        "Age": ages,
        "Sex": sex,
        "Ethnicity": ethnicity})