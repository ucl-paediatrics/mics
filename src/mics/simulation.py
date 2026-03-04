"""Generate a simulated patient dataset containing variables from the QRISK3 algorithm."""
# Import all necessary coding libaries
import numpy as np
import polars as pl

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

    np.random.seed(random_seed)
    # DEMOGRAPHICS:
    ages = np.random.randint(20, 80, size=n)
    sex = np.random.choice(["Male","Female"], size=n)

    # Ethnicity:
    # 9 ethnic groups outlined by the QRISK3 algorithm
    # p is the proportion the simulation model should assign each ethnicity
    uk_ethnicities = {
        "White": 0.850,
        "Black African": 0.025,
        "Black Caribbean": 0.015,
        "Indian": 0.010,
        "Pakistani": 0.010,
        "Bangladeshi": 0.015,
        "Chinese": 0.015,
        "Other Asian": 0.010,
        "Other Ethic Group": 0.050
    }

    ethnicity = np.random.choice(
        list(uk_ethnicities.keys()),
        p=list(uk_ethnicities.values()),
        size=n
    )


    # Polas dataframe showing the QRISK3 variables
    return pl.DataFrame({
        # Demographics
        "Age": ages,
        "Sex": sex,
        "Ethnicity": ethnicity})

def calculate_qrisk3(X: pl.DataFrame) -> np.ndarray:
    """Calculate the QRISK3 score for each patient in the dataset.
    
    Args:
    X(pl.DataFrame): A Polas DataFrame containing simulated patient data.

    Returns:
    np.ndarray: An array containing the QRISK3 score for each patient. This is the 10 year risk of an event. 
    """
    # For simplicity, we will use a very basic formula to calculate the QRISK3 score.
    # In reality, the QRISK3 algorithm is much more complex and takes into account many more variables.
    # This is just for demonstration purposes.

    # Base risk is 0.05 (5% risk of a CVD event in 10 years)
    base_risk = 0.05

    # Age increases risk by 1% per year over 30
    age_risk = (X["Age"].to_numpy() - 30) * 0.01

    # Combine all risk factors
    risk = base_risk + age_risk

    return risk
