"""Generate a simulated patient dataset containing variables from the QRISK3 algorithm."""
# Import all necessary coding libaries
# Import all necessary coding libaries 
import numpy as np
import polars as pl

#### MODEL 1 - Pre-treatment Simulation & Training ####

# STEP 1: Create a simulated population of 100k patinets 

# STEP 2: Simulate patinet characteristics using literature based numbers 
# Yes = 1, No = 0

def generate_patients(n:int, random_seed:int)-> pl.DataFrame:
    """Generate as simulated patient dataset containing variables from the QRISK3 algorithm, with n records.
    
    Args:
    n(int):Number of patinet records to generate
    random_seed(int):Seed for random number generator for reproducibility
    p: Probabilities for each of the categories within the variables -  this gives a weighted random selection

    Returns:
    pl.DataFrame: A Polas DataFrame containing simulated patient data.

    Current columns in the simulated dataset: 
    - age(int):Age of the patient, randomly generated between 20 and 80 
    - sex(str):Sex of the pateint, randomly assigned Male or Female 
    - ethnicity(str): Ethnicity of the patient, randomly assigned from a provided selection
    - 
    """
    # SET RANDOM SEED:
    np.random.seed(random_seed)
    # DEMOGRAPHICS: 
    ages = np.random.randint(20, 80, size=n)
    sex = np.random.choice(["Male","Female"], size=n)

    # Ethnicity: 
    # 9 ethnic groups outlined by the QRISK3 algorithm 
    # - need to find repuatble data sources/links for each of the risk factors. 
    # - need to adjust the proportions based on the census. 
    uk_ethnicities = {
        "White":  0.850,
        "Indian": 0.010,
        "Pakistani": 0.010,
        "Bangladeshi": 0.015,
        "Chinese": 0.015,
        "Other Asian": 0.010,
        "Black Caribbean": 0.015,
        "Black African": 0.025,
        "Other Ethnic Group": 0.050

    }

    ethnicity = np.random.choice(
        list(uk_ethnicities.keys()),
        p=list(uk_ethnicities.values()),
        size=n
    )

    # CLINICAL MEASUREMENTS: 

    # BMI:
    height_m = np.random.uniform(1.50, 2.00, size=n) # Use 'uniform' for float numbers 
    weight_kg = np.random.uniform(45.00, 120.00, size=n)
    bmi_calculated = np.round((weight_kg / height_m **2), 2)
    bmi = bmi_calculated.clip(15,50) # Setting the realistic min and max values for final BMI score

    # Systolic Blood Pressure: UK ideal blood pressure is 120 mmHg
    # Average blood pressure can increase with age
    # Here there is a 0.5 mmHg increase in sbp with each year of age 

    # Using a minimum of 3 sbp readings to calculate the SD 
    # Literatures states 2 or more readings are needed
    # scale=10 allows for sbp values vary by 10 from the patients baseline with each different reading
    bp_reading1 = np.random.normal(120 + (0.5 * ages), scale=10, size=n).clip(90,220) 
    bp_reading2 = np.random.normal(120 + (0.5 * ages), scale=10, size=n).clip(90,220)
    bp_reading3 = np.random.normal(120 + (0.5 * ages), scale=10, size=n).clip(90,220)

    # Stack the columns so that each pateint's sbp SD can be called across the readings
    collated_readings = np.column_stack([bp_reading1, bp_reading2, bp_reading3])
    # Shows one value for the sbp reading in the dataframe - the most current reading, rounded to 2 d.p
    systolic_bp_mmHg = np.round(bp_reading1, 2)

    # Standard Deviation: 
    # This provides a 'history' of sbp values for the patinets in order summarise their variabilities
    # Axis = 1, reading across the array (rows) of sbp readings to calculate the final SD value per patinet 
    sd_systolic_bp_mmHg = np.round(np.std(collated_readings, axis= 1), 2)

    # Total Cholesterol / HDL ("good" cholesterol) - Ratio
    # UK average TC:HDL normal range is between 3.5-5
    # Here an average is taken as 4
    cholesterol_hdl_ratio = np.round(np.random.normal(4, 1, size=n).clip(1.5, 8), 2)

    #LIFESTYLE:

    # Smoking status split into 5 categories
    uk_smoking_status = {
        "Non-Smoker": 0.60,
        "Ex Smoker": 0.20,
        "Light Smoker": 0.10,
        "Moderate Smoker": 0.05,
        "Heavy Smoker": 0.05
        
    }

    smoking_status = np.random.choice(
        list(uk_smoking_status.keys()),
        p=list(uk_smoking_status.values()),
        size=n
    )

    # MEDICAL CONDITIONS:

    # Type 2 Diabetes: About 8% of adults in the UK are diagnosed with it.
    # the probability of develping it increases with age. 
    # Type 1 Diabetes: Is a rare occurance, about 1% of adults in the UK are diagnosed. It is independent of age.
    # No one can have both Type 1 and Type 2
    diabetes_status = []

    for age in ages: 
        if age < 40:
            type2_prob = 0.08 * 0.60 # This reduces the probability to 4.8% in younger patients
        elif age >= 40 and age < 60:
            type2_prob = 0.08
        elif age >= 60: 
            type2_prob = 0.08 * 1.5 # This incrases the probability to 12% in older patients
    
        type1_prob = 0.01
        none_prob = 1 - type1_prob - type2_prob # All the probability values must equal to 1 
 
        status = np.random.choice(
        ["None", "Type 1", "Type 2"], 
        p=[none_prob, type1_prob, type2_prob],
        size=n
        )

    diabetes_status.append(status) 
    # need to ensure that the complete for loop for type2 diabetes runs in all the patients

    # Chronic Kidney Disease (Stages 3-5): About 10% of adults in the UK have this
    # Risk increases and chances becomes greater in adults aged 35+ 
    
    # np.where - selects elements based on conditions, it allows you to filter and transform data in an array 
    # 10% of the population have a chance of devloping ckd 
    # the 'Yes' for the condition >=35 ->  need to find what percentage that chances increase over 35 

    # ckd_prob = np.where(ages >= 35, 0.10)

    # This ensures the above probabilties map onto patients over 35 who can radomly have 'Yes'=10% and 'No'=90%
    # p=[p, 1-p] is doing in the loop — it automatically calculates the "No" probability
    # based on whichever age group the patient falls into.
    chronic_kidney_disease = np.random.choice([1, 0], size=n, p=[0.10, 0.90])

    # Atrial Fibrillation: About 2.5% of the population have this, rises sharply with age
    # Need to account for age! 
    atrial_fibrillation = np.random.choice([1, 0], size=n, p=[0.025, 0.975])

    # Rheumatoid arthritis: 1% prevalence in UK population
    rheumatoid_arthritis = np.random.choice([1, 0], size=n, p=[0.01, 0.99])

    # Systemic Lupus Erythematosus (SLE): 0.1% prevalence in the UK population
    systemic_lupus_erythematosus = np.random.choice([1, 0], size=n, p=[0.001, 0.999])

    # Severe Mental Illness: About 2% of adults in the UK
    severe_mental_illness = np.random.choice([1, 0], size=n, p=[0.02, 0.98])

    # Migraine: Higher prevalence in women (25%) than men (13%)
    migraine= np.where(
        sex == "Female",
        np.random.choice([1, 0], size=n, p=[0.25, 0.75]),
        np.random.choice([1, 0], size=n, p=[0.13, 0.87])
    )

    # Erectile Disfunction: Around 50% of men experience this at some point 
    # It is more prevelant in older men aged 40-70
    # It is not clinically possible for females to get such issues 
    erectile_disfunction = np.where(
        (sex == "Male") & ages >= 40,
        np.random.choice([1, 0], size=n, p=[0.50, 0.50])
    )
    
    # MEDICATIONS:

    # Corticosteroids: Around 1% of adults in the UK are currently using them
    on_corticosteroids = np.random.choice([1, 0], size=n, p=[0.01, 0.99])

    # Atypical Antipsychotics: Around 1.5% of adults in the UK currently using them ong term
    on_atypical_antipsychotics = np.random.choice([1, 0], size=n, p=[0.015, 0.985])

    # BP Medication: Around 21.9% of people in UK are on medication for high bp
    on_bp_medication = np.random.choice([1, 0], size=n, p=[0.219, 0.781])

    # FAMILY HISTORY:

    # Family history of CHD: Approximately 37% of people have a close relative who had
    # a heart attack or angina before age 60
    family_history_chd =  np.random.choice([1, 0], size=n, p=[0.37, 0.63])


    # Polas dataframe showing the QRISK3 variables
    return pl.DataFrame({
        # Demographics
        "Age": ages,
        "Sex": sex,
        "Ethnicity": ethnicity,
        # Clinical Measures
        "BMI":  bmi,
        "Systolic_Blood_Pressure": systolic_bp_mmHg,
        "SD_Systolic_Blood_Pressure": sd_systolic_bp_mmHg,
        "Cholesterol_HDL_Ratio": cholesterol_hdl_ratio,
        # Lifestyle
        "Smoking_Status": smoking_status,
        # Medical Conditions
        "Diabetes_Status": diabetes_status,
        "Systemic_Lupus_Erythematosus":systemic_lupus_erythematosus,
        "Severe_Mental_Illness":severe_mental_illness,
        "CKD": chronic_kidney_disease,
        "Atrial_Fibrillation": atrial_fibrillation,
        "Rheumatoid_Arthritis": rheumatoid_arthritis,
        "Migraine": migraine,
        "Erectile_Disfunction": erectile_disfunction,
        # Medications
        "Corticosteroids": on_corticosteroids,
        "Atypical_Antipsychotics": on_atypical_antipsychotics,
        "Blood_Pressure_Medication": on_bp_medication,
        # Family History
        "Family_History_CHD": family_history_chd
        })

# STEP 3: Generate the risk function 
def calculate_qrisk3(X: pl.DataFrame) -> np.ndarray:
    """Calculate the QRISK3 score for each patient in the dataset.
    
    Args:
    X(pl.DataFrame): A Polas DataFrame containing simulated patient data.

    Returns:
    np.ndarray: An array containing the QRISK3 score for each patient. This is the 10 year risk of an event. 
    """

    ## 1. Extract all the variables from the model

    # Continuous Variables
    age_risk = X["Age"].to_numpy() 
    sex_risk = X["Sex"].to_numpy()
    bmi_risk = X["BMI"].to_numpy()
    sbp_risk = X["Systolic_Blood_Pressure"].to_numpy()
    sd_sbp_risk = X["SD_Systolic_Blood_Pressure"].to_numpy()
    chdl_risk = X["Cholesterol_HDL_Ratio"].to_numpy()

    # Categorical Variables
    # Need to account for these variables with their  covariates to ensure that a numerical relationship isn't assummed 
    ethnicity_coefficients = {
        "White": 0.0000, # Reference 
        "Indian": 0.2804,
        "Pakistani": 0.5630,
        "Bangladeshi": 0.2959,
        "Other Asian": 0.0728,
        "Black Caribbean": -0.1707,
        "Black African": -0.3937,
        "Chinese": -0.3263,
        "Other Ethnic Group": -0.1713
    }
    
    smoking_status_coefficients =  {
        "Non-Smoker": 0.0000,
        "Ex Smoker": 0.1339,
        "Light Smoker": 0.5620,
        "Moderate Smoker": 0.6675,
        "Heavy Smoker": 0.8495
        
    }

    ethnicity_risk = X["Ethnicity"].replace(ethnicity_coefficients).to_numpy()
    smoking_risk = X["Smoking_Status"].replace(smoking_status_coefficients).to_numpy()

    # Binaray Conditions (Y=1, N=0) 
    diabetes_risk = (X["Diabetes_Status"] == "Type 1").to_numpy() * 1.727 \
        + (X["Diabetes_Status"] == "Type 2").to_numpy() * 1.069
    sle_risk = X["Systemic_Lupus_Erythematosus"].to_numpy()
    mental_risk = X["Severe_Mental_Illness"].to_numpy()
    ckd_risk = X["CKD"].to_numpy()
    af_risk = X["Atrial_Fibrillation"].to_numpy()
    ra_risk = X["Rheumatoid_Arthritis"].to_numpy()
    migraine_risk = X["Migraine"].to_numpy()
    ed_risk = X["Erectile_Disfunction"].to_numpy()
    cortico_risk = X["Corticosteroids"].to_numpy()
    aa_risk = X["Atypical_Antipsychotics"].to_numpy()
    bp_meds_risk = X["Blood_Pressure_Medication"].to_numpy()
    fh_risk = X["Family_History_CHD"].to_numpy()
        
    ## 2. Multiply each risk factor by its corresponding QRISK3 weight and sum into 'weighted_sum'
    #   weighted_sum is a single number that represents the combined weighted risk for eacj pateint 

    weighted_sum = (
         # Continuous Variables
        age_risk * 0.08
        + sex_risk * -0.40
        + bmi_risk * 0.029
        + sbp_risk * 0.013
        + sd_sbp_risk * 0.008
        + chdl_risk * 0.153

         # Categorical Variables
        + ethnicity_risk 
        + smoking_risk

        # Binaray Conditions 
        + diabetes_risk
        + sle_risk * 0.759
        + mental_risk * 0.126
        + ckd_risk * 0.652
        + af_risk * 1.592
        + ra_risk * 0.214
        + migraine_risk * 0.301
        + ed_risk * 0.222
        + cortico_risk * 0.595
        + aa_risk * 0.252
        + bp_meds_risk * 0.509
        + fh_risk * 0.454
    )

    ## 3. Covert the weighted_sum into 'final risk' for each patinet
    #   Using the logistic regression sigmoid function to gnerate a probability value between 0 and 1 
    #   𝑝(𝐱) = 1 / (1 + exp(−𝑓(𝐱))
    #   𝑝(𝐱) is often interpreted as the predicted probability that the output for a given 𝐱 is equal to 1. 

    risk = 1 / (1 + np.exp(-weighted_sum))
 
    return risk
