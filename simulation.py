import numpy as np
import polars as pl

def generate_patients(n: int, random_seed: int) -> pl.DataFrame:
    """Generate a simulated patient dataset with n records.
    
    Args:
        n (int): Number of patient records to generate.
        random_seed (int): Seed for random number generator for reproducibility.
    
    Returns:
        pl.DataFrame: A Polars DataFrame containing simulated patient data. 

    Current columns in the simulated dataset:
        - age (int): Age of the patient, randomly generated between 20 and 80
    
    """

    np.random.seed(random_seed)
    ages = np.random.randint(20, 80, size=n)

    return pl.DataFrame({
        "age": ages
    })