"""Module for testing the QRISK3 score calculation."""
import polars as pl

from mics import qrisk3


def test_smoking_risk():
    """Test the smoking risk calculation function."""
    # Create a sample DataFrame
    data = pl.DataFrame({
        "Sex": ["Male", "Female"],
        "Smoking Status": ["Light Smoker", "Ex Smoker"]
    })
    # Calculate the smoking risk
    smoking_risk = qrisk3.smoking_risk(data)
    # Check the values of the smoking risk
    assert smoking_risk.to_list() == [
        0.55241588192645552, # Male Light Smoker
        0.13386833786546262  # Female Ex Smoker
    ]

def test_ethnicity_risk():
    """Test the ethnicity risk calculation function."""
    # Create a sample DataFrame
    data = pl.DataFrame({
        "Sex": ["Male", "Female"],
        "Ethnicity": ["Indian", "Black African"]
    })
    # Calculate the ethnicity risk
    ethnicity_risk = qrisk3.ethnicity_risk(data)
    # Check the values of the ethnicity risk
    assert ethnicity_risk.to_list() == [
        0.27719248760308279, # Male Indian
        -0.39371043314874971  # Female Black African
    ]

def test_calculate_qrisk3():
    """Test the overall QRISK3 calculation function."""
    # Create a sample DataFrame
    data = pl.DataFrame({
        "Sex": ["Male", "Male", "Female", "Female", "Female"],
        "Age": [70, 45, 60, 55, 78],
        "BMI": [28, 26, 29, 24, 31],
        "Smoking Status": ["Non-Smoker", "Light Smoker", "Ex Smoker", "Non-Smoker", "Heavy Smoker"],
        "Ethnicity": ["White", "Indian", "Bangladeshi", "Chinese", "Black Caribbean"],
        "Cholesterol_HDL_Ratio": [4.0, 3.0, 5.0, 4.2, 6.0],
        "Systolic_Blood_Pressure": [120, 130, 140, 135, 165],
        "SD_Systolic_Blood_Pressure": [10, 5, 8, 6, 12],
        "Townsend_Score": [0, 0, 1, -1, 3], 
        "Atrial_Fibrillation": [0, 0, 1, 0, 1],
        "Atypical_Antipsychotics": [0, 0, 0, 0, 0],
        "Corticosteroids": [0, 0, 0, 0, 1],
        "Erectile_Disfunction": [0, 0, 0, 0, 0],
        "Migraine": [0, 0, 0, 1, 0],
        "Rheumatoid_Arthritis": [0, 0, 0, 0, 0],
        "CKD": [0, 0, 0, 0, 1],
        "Severe Mental Illness": [0, 0, 0, 0, 0],
        "SLE": [0, 0, 0, 0, 0],
        "Treated Hypertension": [0, 0, 0, 1, 1],
        "Type 1 Diabetes": [0, 0, 0, 0, 0],
        "Type 2 Diabetes": [0, 1, 1, 0, 1],
        "Family History of CVD": [0, 0, 0, 1, 1]
    })

    # Matches the five-patient R example in src/mics/qrisk3.R.
    reference_risks = [17.6, 10.9, 48.8, 5.9, 91.9]
    # From our QRISK3 implementation
    calculated_risks = qrisk3.calculate_qrisk3(data).round(1).to_list()
    assert calculated_risks == reference_risks
