# Run this script using: `Rscript src/mics/qrisk3.R 2>/dev/null`
if (!requireNamespace("QRISK3", quietly = TRUE)) {
  install.packages("QRISK3", repos = "https://cloud.r-project.org")
}
library(QRISK3)

# Patient data matching qrisk3_test.py :: test_calculate_qrisk3
# gender: 0 = men, 1 = women
# ethrisk: 1 = White, 2 = Indian
# smoke_cat: 1 = non-smoker, 3 = light smoker
# weight back-calculated from BMI assuming height 175 cm

test_data <- data.frame(
  ID                = c(1,    2,    3,    4,    5),
  gender            = c(0,    0,    1,    1,    1),
  age               = c(70,   45,   60,   55,   78),
  b_AF              = c(0,    0,    1,    0,    1),
  b_atypicalantipsy = c(0,    0,    0,    0,    0),
  b_corticosteroids = c(0,    0,    0,    0,    1),
  b_impotence2      = c(0,    0,    0,    0,    0),
  b_migraine        = c(0,    0,    0,    1,    0),
  b_ra              = c(0,    0,    0,    0,    0),
  b_renal           = c(0,    0,    0,    0,    1),
  b_semi            = c(0,    0,    0,    0,    0),
  b_sle             = c(0,    0,    0,    0,    0),
  b_treatedhyp      = c(0,    0,    0,    1,    1),
  b_type1           = c(0,    0,    0,    0,    0),
  b_type2           = c(0,    1,    1,    0,    1),
  weight            = c(28,   26,   29,   24,   31) * (175 / 100)^2,
  height            = c(175,  175,  175,  175,  175),
  ethrisk           = c(1,    2,    4,    8,    6),
  fh_cvd            = c(0,    0,    0,    1,    1),
  rati              = c(4.0,  3.0,  5.0,  4.2,  6.0),
  sbp               = c(120,  130,  140,  135,  165),
  sbps5             = c(10,   5,    8,    6,    12),
  smoke_cat         = c(1,    3,    2,    1,    5),
  town              = c(0,    0,    1,    -1,   3)
)

result <- QRISK3_2017(
  data                        = test_data,
  patid                       = "ID",
  gender                      = "gender",
  age                         = "age",
  atrial_fibrillation         = "b_AF",
  atypical_antipsy            = "b_atypicalantipsy",
  regular_steroid_tablets     = "b_corticosteroids",
  erectile_disfunction        = "b_impotence2",
  migraine                    = "b_migraine",
  rheumatoid_arthritis        = "b_ra",
  chronic_kidney_disease      = "b_renal",
  severe_mental_illness       = "b_semi",
  systemic_lupus_erythematosis= "b_sle",
  blood_pressure_treatment    = "b_treatedhyp",
  diabetes1                   = "b_type1",
  diabetes2                   = "b_type2",
  weight                      = "weight",
  height                      = "height",
  ethiniciy                   = "ethrisk",
  heart_attack_relative       = "fh_cvd",
  cholesterol_HDL_ratio       = "rati",
  systolic_blood_pressure     = "sbp",
  std_systolic_blood_pressure = "sbps5",
  smoke                       = "smoke_cat",
  townsend                    = "town"
)

print(result[, c("ID", "QRISK3_2017", "QRISK3_2017_1digit")])
