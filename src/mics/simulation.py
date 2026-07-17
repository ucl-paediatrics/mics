"""Generate a simulated patient dataset containing variables from the QRISK3 algorithm."""
# Import all necessary coding libaries
import numpy as np
import polars as pl
from sklearn.metrics import roc_curve

#### MODEL 1 - Pre-treatment Simulation & Training ####

# STEP 1: Create a simulated population of 100k patinets

# STEP 2: Simulate patient characteristics using literature based numbers
# Yes = 1, No = 0
# TODO: Need to input the data sources/links for each of the risk factors values.
# TODO: Need to consider how to simulate the realistic and relavent interdependencies between the risk factors?
# TODO: Need to consider how to simulate the correlaations between the risk factors?
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
    """
    # SET RANDOM SEED:
    np.random.seed(random_seed)

    # DEMOGRAPHICS:
    ages = np.random.randint(20, 85, size=n)
    sex = np.random.choice(["Male","Female"], size=n)

    # Ethnicity:
    # Source: ONS 2021 Census, England & Wales (TS021 / detailed 19-group breakdown).
    # Categories collapsed onto QRISK3's 9-group scheme.
    uk_ethnicities = {
        "White":  0.817, # Whie British (74.4%) + White Other (7.3%)
        "Indian": 0.031,
        "Pakistani": 0.027,
        "Bangladeshi": 0.011,
        "Chinese": 0.007,
        "Other Asian": 0.016,
        "Black Caribbean": 0.010,
        "Black African": 0.030, # Black African (2.5%) + Black Other (0.5%)
        "Other Ethnic Group": 0.051 # Mixed (2.9%) + Arab (0.6%) + Any Other (1.6%)

    }
    ethnicity = np.random.choice(
        list(uk_ethnicities.keys()),
        p=list(uk_ethnicities.values()),
        size=n
    )

    # Townsend Deprivation Index: A measure of UK socioeconomic deprivation at the postcode/area level
    # Calculated from four census variables: unempolymeny, non-car ownership, non-home ownership, household over-crowding
    # Standardised so the national mean is 0, negative scores = less deprived, positive scores = more deprived
    # It is sampled independently here with a mean of 0 and SD of 3.3, representative of real UK scores.
    # Real Townsend is right-skewed and correlates with smoking, ethnicity, BMI.
    # TODO: revisit once population-correlation strategy is agreed with supervisor.
    townsend_score = np.random.normal(0, 3.3, size=n).clip(-7, 11)

    # CLINICAL MEASUREMENTS:

    # BMI:
    # Generate normally distributed BMI values - allows for realistic BMI variation across the population
    # with most values clustering around the mean 
    # and fewer values being derived from potentially extreme/unrealistic w/h combinations
    bmi = np.round(np.random.normal(27, 3, size=n),2) # mean =27, sd =3
    
    # Systolic Blood Pressure: UK ideal blood pressure is 120 mmHg
    # Average blood pressure can increase with age
    # Here there is a 0.2 mmHg increase in sbp with each year of age

    # Using a minimum of 3 sbp readings to calculate the SD_SBP
    # Literatures states 2 or more readings are needed
    # scale=10 allows for sbp values vary by 10 from the patients baseline with each different reading
    underlying_sbp = np.random.normal(120 + (0.2 * ages), scale=10, size=n)
    noise = np.random.normal(0, 5, size=(n,3)) # 3 readings per patient
    # Add noise to the underlying sbp to get the three readings
    collated_readings = underlying_sbp[:, np.newaxis] + noise

    # Shows one value for the sbp reading in the dataframe - the most current reading, rounded to 2 d.p
    systolic_bp_mmHg = np.round(collated_readings[:, 0], 2) #pylint: disable=invalid-name

    # Standard Deviation:
    # This provides a 'history' of sbp values for the patinets in order summarise their variabilities
    # Axis = 1, reading across the array (rows) of sbp readings to calculate the final SD value per patient
    sd_systolic_bp_mmHg = np.round(np.std(collated_readings, axis= 1), 2) #pylint: disable=invalid-name

    # Total Cholesterol / HDL ("good" cholesterol) - Ratio
    # UK average TC:HDL normal range is between 3.5-5 
    # https://www.bupa.co.uk/health-information/heart-blood-circulation/high-cholesterol
    # https://www.bhf.org.uk/informationsupport/risk-factors/high-cholesterol/understanding-your-cholesterol-levels
    # Here an average is taken as 4
    cholesterol_hdl_ratio = np.round(np.random.normal(4, 1, size=n).clip(1.5, 8), 2)

    #LIFESTYLE:
    # Source: ONS Annual Population Survey 2024 (smoking status),
    # combined with Health Survey for England trends in cigarettes/day for the intensity split.
    # TODO: Add source link.
    
    # Smoking status split into QRISK3's 5 categories 
    # QRISK3 intensity definitions: light <10/day, moderate 10-19/day, heavy >=20/day.
    # TODO: Age and sex dependencies 
    uk_smoking_status = {
        "Non-Smoker": 0.635,
        "Ex Smoker": 0.260,
        "Light Smoker": 0.052,
        "Moderate Smoker": 0.036,
        "Heavy Smoker": 0.017

    }
    # smoking_prob = np.where()
    smoking_status = np.random.choice(
        list(uk_smoking_status.keys()),
        p=list(uk_smoking_status.values()),
        size=n
    )

    # MEDICAL CONDITIONS:

    # Type 2 Diabetes: About 8% of adults in the UK are diagnosed with it.
    # the probability of developing it increases with age and is dependent on BMI
    # Type 1 Diabetes: Is a rare occurance, about 1% of adults in the UK are diagnosed. It is independent of age.
    # No one can have both Type 1 and Type 2

    base_type2_prob = np.where(ages < 40 , 0.08 * 0.60, # Reduces the probability to 4.8% in younger patients
                               np.where(ages < 60, 0.08, 0.08 * 1.5)) 
    # Increases the probability to 12% in older patients
    
    # 1% increase in the probability of type 2 diabetes per unit BMI above 25
    bmi_adjustment = 0.01 * np.maximum(0, bmi - 25)
    type2_prob = np.minimum(base_type2_prob * (1 + bmi_adjustment), 0.99)
    
    type1_prob = 0.01

    # Vectorised operation for Type 1 and Type 2 diabetes probabilities - guarantees no patient can have both conditions
    # np.random.random (comparison apporach) is used for the different probabilities for each patient, 
    # which are calculated based on their age and BMI
    # type1_diabetes is masked when type2 is already generated - shows mutual exclusivity 
    type2_diabetes = (np.random.random(n) < type2_prob).astype(int)
    type1_diabetes = (np.random.random(n) < type1_prob).astype(int) * (1 - type2_diabetes)

    # Chronic Kidney Disease (Stages 3-5): About 6% of adults in the UK have this, 
    # with prevalence higher in women and increasing with age.
    # Risk becomes greater in adults aged 35+, ~1% for those aged 35-44, rising to over 36% in those aged 75 and over.
    chronic_kidney_disease = []
    
    for age in ages:
        if age < 35:
            ckd_prob = 0.01
        elif age < 45:
            ckd_prob =  0.02
        elif age < 55:
            ckd_prob = 0.03
        elif age < 65:
            ckd_prob = 0.05
        elif age < 75:
            ckd_prob = 0.135
        else:  # age >= 75
            ckd_prob = 0.36

        status = np.random.choice([1, 0], p=[ckd_prob, 1 - ckd_prob], size=1)[0]
        chronic_kidney_disease.append(status)
    # p=[p, 1-p] is doing in the loop — it automatically calculates the "No" probability
    # based on whichever age group the patient falls into.
    
    # Atrial Fibrillation: About 2.5% of the population have this, with increasing risk with age 
    # AF prevalence rises from roughly 0.3% in those age 40 to over 10% in those over 80 
    atrial_fibrillation = []

    for age in ages:
        if age < 40:
            af_prob = 0.001
        elif age < 55:
            af_prob = 0.003     
        elif age < 65:
            af_prob = 0.005      
        elif age < 75:
            af_prob = 0.017
        elif age < 80:
            af_prob = 0.055
        else:  # age >= 80
            af_prob = 0.10

        status = np.random.choice([1, 0], p=[af_prob, 1 - af_prob], size=1)[0]
        atrial_fibrillation.append(status)

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
        (sex == "Male") & (ages >= 40),
        np.random.choice([1, 0], size=n, p=[0.50, 0.50]),
        np.zeros(n, dtype=int)
    )

    # MEDICATIONS:

    # Corticosteroids: Around 1% of adults in the UK are currently using them
    on_corticosteroids = np.random.choice([1, 0], size=n, p=[0.01, 0.99])

    # Atypical Antipsychotics: Around 1.5% of adults in the UK currently using them ong term
    on_atypical_antipsychotics = np.random.choice([1, 0], size=n, p=[0.015, 0.985])

    # BP Medication: Around 21.9% of people in UK are on medication for high bp
    # TODO: Probability of being on medication increases with systolic blood pressure; realistic relationships between input features 
    on_bp_medication = np.random.choice([1, 0], size=n, p=[0.219, 0.781])

    # FAMILY HISTORY:

    # Family history of CVD: Approximately 37% of people have a close relative who had
    # a heart attack or angina before age 60
    family_history_cvd =  np.random.choice([1, 0], size=n, p=[0.37, 0.63])


    # Polas dataframe showing the QRISK3 variables
    return pl.DataFrame({
        # Demographics
        "Age": ages,
        "Sex": sex,
        "Ethnicity": ethnicity,
        "Townsend_Score": townsend_score,
        # Clinical Measures
        "BMI": bmi,
        "Systolic_Blood_Pressure": systolic_bp_mmHg,
        "SD_Systolic_Blood_Pressure": sd_systolic_bp_mmHg,
        "Cholesterol_HDL_Ratio": cholesterol_hdl_ratio,
        # Lifestyle
        "Smoking Status": smoking_status,
        # Medical Conditions
        "Type 1 Diabetes": type1_diabetes,
        "Type 2 Diabetes": type2_diabetes,
        "SLE":systemic_lupus_erythematosus,
        "Severe Mental Illness":severe_mental_illness,
        "CKD": chronic_kidney_disease,
        "Atrial_Fibrillation": atrial_fibrillation,
        "Rheumatoid_Arthritis": rheumatoid_arthritis,
        "Migraine": migraine,
        "Erectile_Disfunction": erectile_disfunction,
        # Medications
        "Corticosteroids": on_corticosteroids,
        "Atypical_Antipsychotics": on_atypical_antipsychotics,
        "Treated Hypertension": on_bp_medication,
        # Family History
        "Family History of CVD": family_history_cvd
        })

# STEP 3: Simulate CVD  Outcomes 
# y = np.random.uniform(0, 1, n_patients) < (risk/100) - simple clean version
def risk_to_event(risk_pct: np.ndarray, random_seed: int):
    """
    Convert 10-year QRISK3 risk percentages to binary CVD event outcomes using Bernoulli sampling.
    For each patient, sample y ~ Bernoulli(p) where p is the patient's risk probability (risk_pct / 100).
    
    Args:
    - risk_pct: An array of shape (n_patients) containing QRISK3 risk percentages (0-100)
    - random_seed: An integer, with a set seed for reproducibility when generating random samples

    Returns:
    - y: An array containing the binary outcomes (0/1), same shape as risk_pct
    """

    # Sanity check: the risk_pct should be between 0 and 100
    if not np.all((risk_pct >= 0) & (risk_pct <= 100)):
        raise ValueError("Risk percentages should be between 0 and 100.")

    # Need to compare the generated QRISK percentages against probabilities (0-1)
    # Divide by 100 to convert risk to a probability
    risk_prob = risk_pct / 100
    
    # Set seed so the results are reproduced with each cell run 
    np.random.seed(random_seed)
    
    # Draw a uniform random number between 0 and 1 for each patient
    # Then compare to their risk probability to determine if they have an event (y=1) or not (y=0)
    uniform_draw = np.random.uniform(0, 1, size=len(risk_pct)) # size of output should match the number of patients
    
    # An event occurs (y=1) if the uniform draw is less than the patient's risk probability, otherwise no event (y=0)
    # This gives a TRUE/FALSE array which is converted to integers (1/0)
    y = (uniform_draw < risk_prob).astype(int) 
    
    return y

 # STEP 4: Generate Feature Matrix for Model 1
def build_feature_matrix(patients):
    """Build a one hot encoded feature matric for the logistic regression models.
    
    Args:
    patients: polars dataframe from generate_patinets

    Returns:
    X: An array of shape (n_patients, n_features)
    feature_names: list of column names in the same order as X's columns
    """
# Clone the patients dataframe to avoid modifying the original data structure
    matrix = patients.clone()

# One-hot encode the categorical variables 
# It turns each category into its own separate binary column (0/1)
    matrix = matrix.to_dummies(columns=["Sex", "Ethnicity", "Smoking Status"])

# Convert into numpy array for model fitting
    X = matrix.to_numpy()
    feature_names = matrix.columns
    return X, feature_names

# STEP 5: Fit Model 1 (pre-treatment)

def fit_model(X, y):
    """Fit a logistic regression model on the patient features and outcomes.
    
    Args:
    X: An array of patient features, shape (n_patients, n_features).
    y: An array of binary outcomes, shape (n_patients,).

    Returns:
    model: A fitted LogisticRegression model.
    """
    model_1 = LogisticRegression(max_iter=2000).fit(X, y)
    return model_1

# STEP 6: Produce Statin Intervention
def apply_statin_intervention(patients_2, model_1, *, threshold=0.10, rrr=0.25, random_seed=None):
    """Deploy M1 to predict the risk of CVD events in this new population and simulate the effect of statin intervention
    
    Args:
    patients_2: A Polars DataFrame of the new population cohort.
    model_1: A trained pre-intervention scikit-learn model.
    threshold: The decision line for clinical intervention (default 0.10 for NICE).
    rrr: Relative Risk Reduction from treatment (0.25 = 25% reduction).
    random_seed: The seed for outcome sampling reproducibility.

    Returns:
    model: A fitted LogisticRegression model.
    """
    # Build a new feature matrix for the new population
    X_2, feature_names = build_feature_matrix(patients_2)

    # M1 predicts the risk of the new population
    predicted_risk_b = model_1.predict_proba(X_2)[:, 1]

    # NICE CG181 threshold: CVD predicted risk >= 10% triggers GP statin prescribing.
    # M1 predicted_risk is on the 0-1 scale, so the threshold is 0.10
    on_statins = predicted_risk_b >= threshold

    # Calculate the true underlying risk for the new population using QRISK3.
    # This generates the ground truth risk; the underlying biological risk for each patient
    # It can be used to evaluate how well M1 performed in this new population and is independent og M1's prediction
    from mics import qrisk3
    true_risk_b = qrisk3.calculate_qrisk3(patients_2).to_numpy()

    # Statins have a CVD relative risk reduction (RRR) of approximately 25%
    # They biologically lower the true probability of a CVD event in patients who are prescribed them

    true_risk_post_b = true_risk_b.copy() # True risk modified for treated patients only
    true_risk_post_b[on_statins] = true_risk_post_b[on_statins] * (1.0 - rrr) 

    # Generate post-intervention ouctomes 
    y_2 = risk_to_event(true_risk_post_b, random_seed=random_seed)

    return X_2, y_2, on_statins, true_risk_b, true_risk_post_b

# STEP 7: Compare Model 1 and Model 2 Coefficients (MICS Signal)
def coefficients_comparison(model_1, model_2, feature_names, features=None):
    """Compare the coefficients of the two logistic regression models across their features.
    
    Returns a polars DataFrame with one row per feature, showing M1's coefficient,
    M2's coefficient, and the difference (M2 - M1). A negative difference on a
    strong positive M1 coefficient is the signature of MICS: As M2 has learned a
    weaker association between that feature and CVD events.

    Args:
    model_1: A fitted LogisticRegression model (pre-intervention).
    model_2: A fitted LogisticRegression model (post-intervention).
    feature_names: List of feature name strings, corresponding to the columns in the feature matrix.
    features: Optional list of specific features names to include in the comparison. If None, all features are included.

    Returns:
    pl.DataFrame: A Polars DataFrame containing the features, the coefficients from both models and their differences.
    """
    # Extract the coefficients from both models
    coef_1 = model_1.coef_[0]
    coef_2 = model_2.coef_[0]

    # Create a DataFrame for comparison
    comparison_df = pl.DataFrame({
        "Feature": feature_names,
        "Model 1 Coefficient": coef_1,
        "Model 2 Coefficient": coef_2,
        "Difference (M2 - M1)": coef_2 - coef_1
    })

    # If certian features are specified, filter the DataFrame to include only those features
    if features is not None:
        comparison_df = comparison_df.filter(pl.col("Feature").is_in(features))

    return comparison_df

# STEP 8: Generate ROC & AUC curves for boths models on Populatiopn C to compare discrimination 
def plot_roc_curve(y_true, y_prob, label=None):
    """Plot ROC curves for both models and calculate AUC scores.
    
    Args:
    y_true: An arry of true binary outcomes for the population.
    y_prob: An array of predicted probabilities from the model.
    
    Returns:
    None
    """
    import matplotlib.pyplot as plt

    # Generate the ROC curves for both models on population C based on the probabilities predicted by each model
    fpr, tpr, roc_thresholds = roc_curve(y_true, y_prob)

    # Produce a precision-recall curve for both models on population C based on the probabilities predicted by each model
    precision, recall, pr_thresholds = precision_recall_curve(y_true, y_prob)

    # Generate the ßAUC score for both models on population C based on the probabilities predicted by each model
    auc_roc = roc_auc_score(y_true, y_prob)
    auc_pr = average_precision_score(y_true, y_prob)

    # Plot the ROC curve
    curve_label = f"{label} (AUC = {auc_roc:.4f})" if label else f"AUC = {auc_roc:.4f}"
    plt.plot(fpr, tpr, label=curve_label)
    plt.xlabel('False Positive Rate (FPR)')
    plt.ylabel('True Positive Rate (TPR)')

    # Print AUC and ROC curve values for both models
    # print(f"{label} AUC: {auc_roc:.4f}" if label else f"AUC: {auc_roc:.4f}")
    print(f"ROC-AUC: {auc_roc:.4f} | PR-AUC (AP): {auc_pr:.4f}")
    print(f"{label} ROC Curve: FPR = {fpr}, TPR = {tpr}" if label else f"ROC Curve: FPR = {fpr}, TPR = {tpr}")