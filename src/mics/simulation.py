"""Generate a simulated patient dataset containing variables from the QRISK3 algorithm."""
# Import all necessary coding libaries
import numpy as np
import polars as pl
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix, f1_score
from sklearn.metrics import precision_recall_curve, average_precision_score
import matplotlib.pyplot as plt
from mics import qrisk3
                           
#### MODEL 1 - Pre-treatment Simulation & Training ####

# STEP 1: Create a simulated population of 100k patinets

# STEP 2: Simulate patient characteristics using literature based numbers
# Yes = 1, No = 0
# TODO: Need to input the data sources/links for each of the risk factors values.

# LIMITATION: Risk factors are generated independently, apart from the explicit age dependencies below. 
# Real primary care populations have substantial correlation between BMI, blood pressure, diabetes and deprivation. 
# This does not bias the MICS comparison, since M1 and M2 face populations with identical
# structure, but it does limit external realism. Modelling these interdependencies offers scope for future work 
# on how correlation between risk factors affects the MICS effect.

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
    # This has been left independent by design as per the limitations.
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

    # Rheumatoid Arthritis: 1% prevalence in UK population
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
    # TODO: Probability of being on medication should increase with systolic blood pressure; 
    # requires a realistic relationship between the input features, as it is currently independent

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

# STEP 3: Simulate CVD Outcomes 
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

# Drop one reference level per categorical variable to avoid multicollinearity
# Polars keeps all k levels, which makes the dummy block collinear with the intercept and 
# the coefficients remain uninterpretable. 
# The reference categorie are the largest group in each variable, so every coefficient now reads 
# as the difference from that baseline; matching the QRISK3's own reference groups.
    reference_columns = ["Sex_Female", "Ethnicity_White", "Smoking Status_Non-Smoker"]

# Guard against silently missing reference columns in the feature matrix, which would indicate a problem with the one-hot encoding or the input data.
    missing = []
    for col in reference_columns:
        if col not in matrix.columns:
            missing.append(col)
        if missing:
         raise ValueError(f"Expected reference columns missing from feature matrix: {missing}")
    
    matrix = matrix.drop(reference_columns)

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
    # As sklearn defaults to L2 regularisation, which shrinks by an amount specific to each model's data and breaks M1 $ M2  coeficinent comparability
    # Explicitly set C=np.inf as it gives ab unpenalised fit. 
    # Sklean regularises by default, which would shrink the M1 and M2 coefficients by different amounts, making them not directly comparable.
    
    model_1 = LogisticRegression(C=np.inf, max_iter=2000).fit(X, y)
    return model_1

# STEP 6: Produce Statin Intervention
def apply_statin_intervention(patients_2, model_1, *, threshold=0.10, rrr=0.25, uptake=1.0, statin_covariate=False, random_seed: int, uptake_seed: int):
    """Deploy M1 to predict the risk of CVD events in this new population and simulate the effect of statin intervention
    
    Args:
    patients_2: A Polars DataFrame of the new population cohort.
    model_1: A trained pre-intervention scikit-learn model.
    threshold: The decision line for clinical intervention (default 0.10 for NICE).
    rrr: Relative Risk Reduction from treatment (0.25 = 25% reduction).
    random_seed: The seed for outcome sampling reproducibility.
    uptake: Proportion of eligible patients who accept treatment 
    (1.0 = full adherence, mathcing the derterministic allocation used in Experiment 1)
    uptake_seed: seed for the acceptance draw, keot seperate from the random_seed,
    so the uptake and outcome draws don't share the same random stream
    statin_covariate: This makes the treatmnet a covariate which should partially restore
    accuracy by conditioning predictions on intervention exposure. 

    Returns:
    model: A fitted LogisticRegression model.
    """
    # Build a new feature matrix for the new population
    X_2, feature_names = build_feature_matrix(patients_2)

    # M1 predicts the risk of the new population
    predicted_risk_b = model_1.predict_proba(X_2)[:, 1]

    # NICE CG181 threshold: CVD predicted risk >= 10% triggers GP statin prescribing.
    # M1 predicted_risk is on the 0-1 scale, so the threshold is 0.10
    eligible = predicted_risk_b >= threshold

    # Generate the per-patient uptake array 
    if isinstance (uptake, dict):
        # Map the original ethnicity column through the dict using replace_strict
        uptake_per_patient = patients_2["Ethnicity"].replace_strict(uptake).to_numpy()
    else:
        # Broadcast the scalar float to match the population size
        uptake_per_patient = np.full(len(predicted_risk_b), float(uptake))

    # Bernoulli acceptance draw using the per-patinet uptake array.
    # A patient is treated only if eligible and accepting, so 1.0 reproduces full adherence to treatment
    rng = np.random.default_rng(uptake_seed)
    accepts_statins = rng.random(len(predicted_risk_b)) < uptake_per_patient
    on_statins = eligible & accepts_statins

    # Rebuild the feature matrix with treatmnet status included as a feature 
    # Must be after on_statins so that M1 can predict risk and the threshold can be applied 
    # This builds a wider matrix that only M2 uses as M1 was trained on a different number for features
    if statin_covariate:
        patients_2 = patients_2.with_columns(
            pl.Series("On_Statins", on_statins.astype(int))
        )
        X_2, feature_names = build_feature_matrix(patients_2)
        
    # Calculate the true underlying risk for the new population using QRISK3.
    # This generates the ground truth risk; the underlying biological risk for each patient
    # It can be used to evaluate how well M1 performed in this new population and is independent og M1's prediction
    true_risk_b = qrisk3.calculate_qrisk3(patients_2).to_numpy()

    # Statins have a CVD relative risk reduction (RRR) of approximately 25%
    # They biologically lower the true probability of a CVD event in patients who are prescribed them
    true_risk_post_b = true_risk_b.copy() # True risk modified for treated patients only
    true_risk_post_b[on_statins] = true_risk_post_b[on_statins] * (1.0 - rrr) 

    # Generate post-intervention ouctomes 
    y_2 = risk_to_event(true_risk_post_b, random_seed=random_seed)

    return X_2, y_2, on_statins, true_risk_b, true_risk_post_b, feature_names, uptake_per_patient

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

# STEP 8: Generate ROC & AUC curve for boths models on Populatiopn C to compare discrimination 
def plot_roc_curve(y_true, y_prob, label=None):
    """Plot ROC curves for both models and calculate AUC scores.
    
    Args:
    y_true: An arry of true binary outcomes for the population.
    y_prob: An array of predicted probabilities from the model.
    
    Returns:
    None
    """

    # Generate the ROC curves for both models on population C based on the probabilities predicted by each model
    fpr, tpr, roc_thresholds = roc_curve(y_true, y_prob)

    # Generate the AUC score for both models on population C based on the probabilities predicted by each model
    auc  = roc_auc_score(y_true, y_prob)
    
    # Plot the ROC curve
    curve_label = f"{label} (AUC = {auc:.4f})" if label else f"AUC = {auc:.4f}"
    plt.plot(fpr, tpr, label=curve_label)
    plt.xlabel('False Positive Rate (FPR)')
    plt.ylabel('True Positive Rate (TPR)')

    # Print AUC and ROC curve values for both models
    # print(f"{label} AUC: {auc_roc:.4f}" if label else f"AUC: {auc_roc:.4f}")
    print(f"{label} ROC-AUC: {auc:.4f}" if label else f"ROC-AUC: {auc:.4f}")
    
# STEP 9: Generate Precision-Recall Curve for both models on Population C; which focuses on the minority class (CVD events)
def plot_pr_curve(y_true, y_prob, label=None):
    """Plot a precision-recall curve for both models.
    
    Args:
    y_true: An arry of true binary outcomes for the population.
    y_prob: An array of predicted probabilities from the model.
    
    Returns:
    None
    """

    # Produce a precision-recall curve for both models on population C based on the probabilities predicted by each model
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    pr_auc = average_precision_score(y_true, y_prob)

    # Plot the precision-recall curve
    curve_label = f"{label} (PR-AUC = {pr_auc:.4f})" if label else f"PR-AUC = {pr_auc:.4f}"
    plt.plot(recall, precision, label=curve_label)
    plt.xlabel("Recall (Sensitivity)")
    plt.ylabel("Precision (PPV)")

    print(f"{label} PR-AUC: {pr_auc:.4f}" if label else f"PR-AUC: {pr_auc:.4f}")

# STEP 10: Evaluate a model's prediction on Population C
def evaluate_predictions(y_true, predicted_risk, true_risk_pct, prefix, *, threshold=0.10):
    """Calculate performance metrics for a model's predictions on Population C.
    
    Args:
    y_true: The true binary outcomes for the evaluation population
    predicted_risk: The predicted probabilities from one model
    true_risk_pct: True QEISK3 risk for the same patients
    prefix: A string added to each key to start the metric names
    threshold: The predicted risk at which the statins were allocated

    Returns:
    dict: Conytains the performance metrics prefixed by the model with the float values
    """
    metrics = {}

    # Guard against division by zero, which can occur at higher thresholds if a model has very few patients
    #The safe divide is a nested function to avoid division by zero errors when calculating metrics
    # It returns NaN which helps stop one odd cell form killing the entire experiment loop
    def safe_divide(numerator, denominator):
            return float(numerator / denominator) if denominator != 0 else float("nan")
    

    # Discrimination: ROC-AUC and PR-AUC
    # These measure ranking rather than absolute risk, 
    # so they use the continuous probabilities rather than binary predictions 
    metrics[f"{prefix}_auc"] = float(roc_auc_score(y_true, predicted_risk))
    metrics[f"{prefix}_pr_auc"] = float(average_precision_score(y_true, predicted_risk))

    # Threshold-based metrics: Sensitivity, Specificity, PPV, NPV, F1 Score
    # Convert the probabilities into binary predictions based on the clinical threshold
    y_pred = (predicted_risk >= threshold).astype(int)

    # Confusion matrix retuns [[tn, fp], [fn, tp]]
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    
    metrics[f"{prefix}_sensitivity"] = safe_divide(tp, tp + fn)
    metrics[f"{prefix}_specificity"] = safe_divide(tn, tn + fp)
    metrics[f"{prefix}_ppv"] = safe_divide(tp, tp + fp)
    metrics[f"{prefix}_npv"] = safe_divide(tn, tn + fn)
    metrics[f"{prefix}_f1"] = float(f1_score(y_true, y_pred, zero_division=0))

    # Number of patients flagged for statin allocation based on the threshold
    metrics[f"{prefix}_n_flagged"] = int(tp + fp)

    # Underprediction among genuinely high-risk patinets 
    # Patinets are selected based on thier true QRISK3 risk, rather than the model's prediction
    # So both models are compared on the same people (boolean indexing)
    truely_high_risk = true_risk_pct >= (threshold * 100)  # Converts the threshold to percentage for comparison

    metrics[f"{prefix}_n_truely_high_risk"] = int(truely_high_risk.sum())
    metrics[f"{prefix}_mean_pred_high_risk"] = float(predicted_risk[truely_high_risk].mean())

    return metrics

# Step 11: Generate the three populations, fit both models , predict on C 
def run_pipeline(seed, n=100_000, *, threshold=0.10, rrr=0.25, uptake=1.0, statin_covariate=False, covariate_value="zeros"):
    """Run the MICS simulation pipeline and return the models and per-patient arrays.
    
    Args:
    seed: All six random draws are derived from this single seed, so a run is reproducible
    from one number
    n: Number of patients in each population
    threshold: The predicted risk at which statins are allocated (NICE CG181 = 0.10)
    rrr: Relative risk reduction from statins (0.25 = 25%)
    covariate_value: This provides the covariate value if statins have been allocated 
    
    Returns:
    dict: Both fitted models, feature names, treatment allocation, 
    and the outcomes Run settings, true risks and predictions for population C 
        
    """
    # Derive independent seeds from the single input seed so the whole run is
    # reproducible from a single number. 
    # Outcome sampling needs its own seeds too, since risk_to_event resets the global numpy seed.
    seed_pop_a, seed_pop_b, seed_pop_c = seed, seed + 1000, seed + 2000
    seed_out_a, seed_outs_b, seed_out_c = seed + 3000, seed + 4000, seed + 5000

    # A distinct seed offset for the non-colliding uptake draw 
    seed_uptake = seed + 6000

    # Population A: Train M1 on pre-intervention outcomes 
    patients_a = generate_patients(n, random_seed=seed_pop_a)
    true_risk_a = qrisk3.calculate_qrisk3(patients_a).to_numpy()
    y_a = risk_to_event(true_risk_a, random_seed=seed_out_a)
    X_a, feature_names = build_feature_matrix(patients_a)
    model_1 = fit_model(X_a, y_a)

    # Population B: Allocate statins, train M2 on post-intervention data 
    # apply_statin_intervention which builds the matrix, deploys M1, allocates statins above the threshold,
    # which reduces the treated patinets true risk by the RRR 
    patients_b = generate_patients(n, random_seed=seed_pop_b)
    X_b, y_b, on_statins, true_risk_b, true_risk_post_b, feature_names_b, uptake_per_patient = apply_statin_intervention(
        patients_b, model_1, threshold=threshold, rrr=rrr, random_seed=seed_outs_b, 
        uptake=uptake, uptake_seed=seed_uptake, statin_covariate=statin_covariate,
        )
    model_2 = fit_model(X_b, y_b)

    # Population C: Evaluate both models on the untreated outcomes - no intervention and no model is fitted
    patients_c = generate_patients(n, random_seed=seed_pop_c)

    # M1 was trained without a statin column, so it always predicts on this
    # narrower matrix regardless of the covariate setting
    X_c_m1, feature_names_c = build_feature_matrix(patients_c)

    if statin_covariate: 
        if covariate_value == "zeros":
            # No-one in population C has been allocated, so setting the covariate 
            # to zero causes M2 to generate each patinet's untreated risk 
            statin_column = np.zeros(n, dtype=int)
        else:
            raise ValueError(f"Unknown covariate_value: {covariate_value}")
        patients_c_with_statins = patients_c.with_columns(pl.Series("On_Statins", statin_column))

        # Build a new feature matrix that seperates both the M1 and M2 dataframes
        # M2's matrix is now wider by the On_Statins column. M1 was trained on the
        # narrower one and cannot predict on a matrix with more features.
        X_c_m2, _ = build_feature_matrix(patients_c_with_statins)
    else: 
        X_c_m2 = X_c_m1
    

    if feature_names_c != feature_names:
        raise ValueError("Feature names differ between populations A and C.")

    true_risk_c = qrisk3.calculate_qrisk3(patients_c).to_numpy()
    y_c = risk_to_event(true_risk_c, random_seed=seed_out_c)

    return{
        "model_1": model_1,
        "model_2": model_2,
        "feature_names": feature_names,
        "feature_names_b": feature_names_b,
        "on_statins": on_statins,
        "true_risk_c": true_risk_c,
        "y_c": y_c,
        "uptake_per_patient": uptake_per_patient,
        "ethnicity_b": patients_b["Ethnicity"].to_numpy(),
        "ethnicity_c": patients_c["Ethnicity"].to_numpy(),
        "m1_predicted_risk_c":model_1.predict_proba(X_c_m1)[:, 1],
        "m2_predicted_risk_c": model_2.predict_proba(X_c_m2)[:, 1],

    }

# Step 12: Run one complete simulation and return a flat dictionary of results
def run_simulation(seed, n=100_000, *, threshold=0.10, rrr=0.25, uptake=1.0, statin_covariate=False, covariate_value="zeros"):
    """Run one complete MICS simulation and return the results as a dictionary.

    M1 is trained on an untreated population (A), then deployed on a second
    population (B) which is used to allocate statins, 
    which lowers the true risk of treated patients.
    M2 is trained on those post-intervention outcomes. Both models
    are evaluated on an untreated third population (C), so any difference
    between them comes from the intervention rather than from the patients.

    Args:
    seed: All six random draws are derived from this single seed, so a run is reproducible
    from one number
    n: Number of patients in each population
    threshold: The predicted risk at which statins are allocated (NICE CG181 = 0.10)
    rrr: Relative risk reduction from statins (0.25 = 25%)

    Returns:
    dict: Contains the run settings, proportion treated, actual event rate in population C,
    both models' mean predicted risk, the under-prediction gaps in
    percentage points, both models' coefficients, and the metrics
    """

    # run_pipeline generates the three populations and fits both models. 
    # It is kept seperate so the calibratin analysis can reuse the per-patient arrays,
    # which this function onnly needs in a summarised form
    pipeline_output = run_pipeline(seed, n, threshold=threshold, rrr=rrr, uptake=uptake, 
                                   statin_covariate=statin_covariate, covariate_value=covariate_value,
                                )

    # Unpack each variable into local names so the results block can be read simply 
    model_1 = pipeline_output["model_1"]
    model_2 = pipeline_output["model_2"]
    feature_names = pipeline_output["feature_names"]
    feature_names_b = pipeline_output["feature_names_b"]
    on_statins = pipeline_output["on_statins"]
    true_risk_c = pipeline_output["true_risk_c"]
    y_c = pipeline_output["y_c"]
    m1_predicted_risk_c = pipeline_output["m1_predicted_risk_c"]
    m2_predicted_risk_c = pipeline_output["m2_predicted_risk_c"]
    uptake_per_patient = pipeline_output["uptake_per_patient"]
    ethnicity_b = pipeline_output["ethnicity_b"]
    ethnicity_c = pipeline_output["ethnicity_c"]

    # The metrics are calculated seperately for each model on the same untreated population,
    # so any difference comes from the models rather that the patients 
    m1_metrics = evaluate_predictions(
        y_c, m1_predicted_risk_c, true_risk_c, "m1", threshold=threshold
    )
    m2_metrics = evaluate_predictions(
        y_c, m2_predicted_risk_c, true_risk_c, "m2", threshold=threshold
    )

    # Results:
    # QRISK3 returns risk scores as percentages (0-100), predict_proba returns proportions(0-1), 
    # so divide the true risks by 100 to keep one consistent scale throughout
    actual_c = float(y_c.mean())
    m1_mean_c = float(m1_predicted_risk_c.mean())
    m2_mean_c = float(m2_predicted_risk_c.mean())

    # Multiply by 100 to express the gaps in percentage points
    underprediction_vs_truth_pp = (actual_c - m2_mean_c) * 100
    m1_m2_pp_gap = (m1_mean_c - m2_mean_c) * 100

    # One key per feature per model.
    # The feature names and coefficients share the same column order, 
    # so they are zipped together into pairs so that each feature has a corresponding 
    # coefficient for easy interpretation.
    m1_coefficients = {f"m1_coefficient_{name}": float(coef)
                       for name, coef in zip(feature_names, model_1.coef_[0])}
    m2_coefficients = {f"m2_coefficient_{name}": float(coef) 
                       for name, coef in zip(feature_names_b, model_2.coef_[0])}

    # Experiment 4 per-group results. Ethnicity is one-hot encoded in the feature matrix, 
    # so the raw column is carried through from run_pipeline to allow the patients to be grouped
    subgroup_results = {}
    
    for group in np.unique(ethnicity_c): # Returns the nine ethnicity names present in the population
        # Populations B and C contain different patients, so each needs its
        # own selection built from its own ethnicity array
        group_patients_b = ethnicity_b == group
        group_patients_c = ethnicity_c == group

        subgroup_results[f"n_{group}"] = int(group_patients_c.sum())
        subgroup_results[f"treated_{group}"] = float(on_statins[group_patients_b].mean())
        subgroup_results[f"m1_mean_pred_{group}"] = float(m1_predicted_risk_c[group_patients_c].mean())
        subgroup_results[f"m2_mean_pred_{group}"] = float(m2_predicted_risk_c[group_patients_c].mean())

    results = {"seed": seed, "n": n, "threshold": threshold, "rrr": rrr, "uptake_type": "differential" if isinstance(uptake, dict) else "uniform",
    "uptake_mean": float(uptake_per_patient.mean()) if isinstance(uptake, dict) else float(uptake),
        "proportion_treated": float(on_statins.mean()),
        "event_rate_c": actual_c,
        "m1_mean_c": m1_mean_c, 
        "m2_mean_c": m2_mean_c,
        "underprediction_vs_truth_pp": underprediction_vs_truth_pp,
        "m1_m2_pp_gap": m1_m2_pp_gap,
        # M2's under-prediction is partly due to a baseline shift, not only model attenuataion. 
        # Part of the shift lives in the intercept rather than in the coefficients, so it is stored alongside the coefficients
        # so the shift can be separated from the attenuation.
        "m1_intercept": float(model_1.intercept_[0]),
        "m2_intercept": float(model_2.intercept_[0]),      
 
    }
    # Add the coefficients keys into to the results dictionary, keeping them flat and easily accessible for analysis.
    results.update(m1_coefficients)
    results.update(m2_coefficients)
    results.update(m1_metrics)
    results.update(m2_metrics)
    results.update(subgroup_results)

    return results

# STEP 13: Generate the caliabration coodrinates for a population C plot
def calibration_points(y_true, predicted_risk, n_bins=10):
    """Return the mean predicted risk and observed event rate per deciles

    Patients are sorted by thier predicted risk and placed into equal-sized bins.
    Equal-sized bins are used rather than equal-width because predicted risk
    has a heavily skewed distribution, which would leave the highest-risk bins nearly empty.
    
    Args: 
    y_true: The binary outcomes for the evaluation population
    predicted_risk: The predicted probabilities from one model 
    n_bins: Number of bins, 10 produces deciles 

    Returns:
    mean_predicted: Mean predicted risk in each bin
    observed_rate: Proportion of patients with an event in each bin
    """

    # Sort pateints by thier predicted risk
    sort_patients = np.argsort(predicted_risk)
    predicted_sorted = predicted_risk[sort_patients]
    y_sorted = y_true[sort_patients]

    # Cut the predicted patients into 10 equal-sized bins
    predicted_bins = np.array_split(predicted_sorted, n_bins)
    y_bins = np.array_split(y_sorted, n_bins)

    # For each bin generate the mean predicted rosk and the actual proportion of patients with events 
    mean_predicted = np.array([bin.mean() for bin in predicted_bins])
    observed_rate = np.array([bin.mean() for bin in y_bins])

    return mean_predicted, observed_rate

