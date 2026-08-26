# MICS Results Notes

This serves as a running log of all the findings, decisions and headline numbers observed throughout the course of the project. 

## Status 
Last updated: 25 Augsted 2026 
Completed: Experiments 1-4. The code runner, the metrics layer, figures and tables 
Outstanding: 
- Experiment 4's interaction terms 
- Multi-seed calibration figure, for consistency (Experiment 1)
- Experiment 3 option B - M1 implied statin covariate value 
- Source links for prevalence values in generate_patients

## Design decisions 
- No train/test split inside run_simulation because the populations B and C are independent, this is a stronger evelaution than holding out 20% of the same poulation of testing. A split would also mean that M1 would be trained on 80_000 patinets while M2 is trained on the full 100_000, meaning part of the coeffiecient differences would come from the sample size rather than the statin intervention. (Methods)
- Population C receives no intervention as it the untreated counterfactual and the whole MICS argument depends on it being unaffected by treatment decisions. (Methods)
- Reference categories: Sex_Female, Ethnicity_White, Smoking Status_Non-Smoker. 
Were each chosen becasue they are the three largest groups in each category, this helps to minimise the standard error for each contrast feature measured againist them. They also  match QRISK3's own baselines, which is what made the internal validatin check possible. (Methods)
- Seed offsets of 1000, not +1/+2/+3 beacause adjacent seeds would collide cross the 20 runs - seed 1's population C offset would equal seed 4's population A. 
- All risks are stored on the 0–1 scale beacuse QRISK3 returns 0–100, while predict_proba returns 0–1; mixing them would cause an 100× error in the headline number.
- The seeds are required, not defaulted, so no run can be unreproducible by accident.

## Pipeline fixes
Two bugs affected coefficient comparability and were fixed on 4 August:
1. `build_feature_matrix` kept all categorical levels, making the design matrix
collinear with the intercept.
2. `fit_model` used sklearn's default L2 penalty, shrinking M1 and M2 by different amounts.

Reference categories have now been dropped and fitting is unpenalised (`C=np.inf`). 
All results were regenerated afterwards.

## Experiment 1 - Baseline MICS

Internal validation: M1 recovers the QRISK3 structure
M1 is fitted only on simulated binary outcomes, never on the real QRISK3 coefficients, yet it was able to reproduce the published patterns:
- Smoking gradient monotonic vs non-smokers: Ex 0.147 < Light 0.455 < Moderate 0.460 < Heavy 0.700
- Ethnicity: South Asian groups were elevated (Pakistani 0.726, Bangladeshi 0.548, Indian 0.312); Black African −0.475, Black Caribbean −0.339, Chinese −0.331 relative to White
- T1D 1.333 > T2D 0.660; AF 0.669; Male 0.393
This serves as evidence that the pipeline recovers the data generating mechanism of QRISK3; supporting the underlying assumption of the whole MICS argument. 
* This was done at a single seed 42, n=100_000 - it still needs to be regenerated from a multiseed table before qouting the final results

This matters as the entire project design relies on M1 being able to faithfully stand in for a real clinical model. If M1 failed to recover the true risk structure, then any difference between M1 and M2 could be due to a poorly specified baseline rather than the effect of the intervention. We would be unable to conclude the origin of MICS or weather the effect is correctly attributed to treatment.

M1 reproducing QRISK3's published risk patterns (the smoking gradient, the ethnicity ordering, T1D above T2D), despite only seeing binary outcomes shows that it recovered the data generating mechanism well. M1 is therefore a valid baseline, and any subsequent divergenece seen in M2 can be attributed to the intervention rather than a broken starting point. 

(Results opening)

Coefficient attenuation:
Feature,	M1,	Attenuation (M2 − M1),	Interval excludes zero
Type 1 Diabetes	1.257	−0.198	yes
Atrial Fibrillation	0.613	−0.102	yes
Type 2 Diabetes	0.664	−0.095	yes
Heavy Smoker	0.563	−0.081	yes
Family History CVD	0.283	−0.036	yes
Cholesterol/HDL	0.195	−0.033	yes
CKD	0.087	−0.014	no
Treated Hypertension	0.209	−0.011	no
SLE	0.266	+0.102	yes — wrong direction

- Age's −0.011 is a ~16% proportional drop, comparable to T1D's 16%, but the raw number looks trivial because Age is measured per-year across a 65-year span.
- The CKD / Treated Hypertension contrast is the mechanism. Both have substantial M1 coefficients but neither attenuates. 
Why? 
Attenuation is driven by how much a feature influences who receives treatment, not by the size of its coefficient. A feature must be both common enough within the population and strong enough on the log-odds scale to push a patient's predicted risk across the 10% threshold, altering the treatment decision. SLE has a large QRISK3 coefficient but affects roughly 100 patients per 100,000, so it barely moves aggregate allocation. Treated Hypertension is common at 21.9% but is too weak to push patients over on its own. Neither shows any attenuation. The proposal's prediction assumed coefficient magnitude was sufficient; it is not, both properties must hold at once.

CKD also fails to attenuate, though its M1 coefficient is low (0.087) partly because CKD generation is steeply age-dependent and Age is in the model, so age absorbs most of its signal. This makes CKD a less clean illustration of the mechanism than SLE or Treated Hypertension.

Attenuation follows the number of treatment decisions a feature is responsible
for, which is both prevalence and decisiveness together.

- Ethnicity as a further check: negatively-signed groups move toward zero, which is the attenuation in the correct direction for a protective coefficient. 
Attenuation means the model learns a weaker association. For a negative coefficient, weaker means closer to zero rather than more negative. Black African moving from −0.475 to −0.396 is the same phenomenon as T1D dropping from 1.257 to 1.060, it's just mirrored.
- Heavy smoker: went from an inexplicable negative coefficient (the encoding bug) to a real, interval-backed attenuation. Heavy smokers cluster above the 10% threshold, so they are heavily treated. 

(Results main finidng, Discussion as a mechanims finding)

Peformance metrics 
Metric,	M1,	M2,	Difference [95%]
AUC	0.8267	0.8266	−0.0002 [−0.0007, +0.0003]
PR-AUC	0.3744	0.3734	−0.0009 [−0.0034, +0.0008]
Sensitivity	0.835	0.788	−0.047 [−0.055, −0.040]
Specificity	0.670	0.718	+0.048 [+0.042, +0.054]
PPV	0.259	0.278	+0.019 [+0.017, +0.022]
NPV	0.967	0.961	−0.006 [−0.007, −0.005]
F1	0.395	0.411	+0.016 [+0.014, +0.018]
N flagged	39,123	34,360	−4,762 [−5,398, −4,197]

* Discrimination is unaffected
AUC: M1 0.8267, M2 0.8266. Difference −0.0002 [−0.0007, +0.0003].
PR-AUC: M1 0.3744, M2 0.3734. Difference −0.0009 [−0.0034, +0.0008].

Both intervals straddle zero, so both discrimination metrics are statistically indistinguishable between the two models.

AUC measures ordering. It only asks whether high-risk pateints (event) rank above lowe-risk ones (no event). Flattening compresses the scale in which patients are ranked but leaves the order intact. If M1 ranks patient A above patient B, M2 does too. Every patient's predicted risk has been squashed toward the middle, but nobody has swapped places, so AUC is unchanged.
Calibration measures whether a model's predicted probabilities match the observed frequencies - does the predicted 20% CVD events actually correspond to 20% of those patients having events. That depends on the absolute numbers, and the absolute numbers are exactly what compression breaks.

Hypothesis Test: a single-seed run suggested PR-AUC dropped ~4.7% and might be more
sensitive to MICS than AUC. The 20-seed intervals do not support this.
Both metrics are equally blind.
 
Underprediction 
- Population-wide: 2.53pp [2.33, 2.71]
- Among patients with true risk >= 10%: M1 predicts 24.94%, M2 predicts 19.05% — a 5.89pp gap, 2.3× the population average
- The underprediction_vs_truth_pp (2.5272) and m1_m2_pp_gap (2.5295) agree to within 0.002, which gives evidence M1 tracks the data generating mechanism closely

Flattening, rather than uniform shift
M2's coefficients compress while its intercept rises, so M2 predicts lower for high-risk patients and possibly slightly higher for low-risk ones.
- Why?: A logistic regression model's prediction is intercept + Σ(coefficient × feature).
M2's coefficients are all smaller, so every adjustment is weaker. E.g a patient with T1D, AF and heavy smoking gets raised on the log-odds scale less than what M1 would; which lowers the predicted risk for the patient. However, the intercept rises in the experiment works opposite, in that M2's is −8.24 versus M1's −8.97, so everyone starts about 0.73 higher.
The effect which takes place is determined by the number of risk factors a patient has. 
Patients with few risk factors have very few adjustments added to the starting point, so M2's higher intercept dominates and it may predict slightly higher than M1. Patients with many risk factors have a large number of adjustments, so M2's weakened coefficients cost more than the raised intercept gives bac; so M2 predicts substantially less. 
- This results in compression as the high-risk end of the distribution is pulled down further than the low-risk end is lifted - converging to the middle 
- Seen in the calibration figure: M2's top decile averages 0.33 predicted against M1's 044, on the same patients with the same observed event rate, so its predictions are compressed at the high end. The high-risk gap of 5.89pp [5.52, 6.25] against 2.53pp [2.3615, 2.7452] population-wide shows the error concentrating at the top. The intervals don't overlap so the difference is unambiguous. (Both 20 seeds; calibration figure is seed 42.)

Patients with high predicted risk sit near the 10% threshold. Compression pushes
those patients below it, so they are not allocated statins. This leads to 4,762
fewer patients flagged per 100,000 and a 4.7pp drop in sensitivity. M2 therefore
misses patients whose true risk exceeds 10%; the exact group the original model
was built to identify.

Calibration Figure (single seed 42)
- M2 sits **above** the diagonal, which indicates under-prediction. This reads counterintuitively: the y-axis is what actually happened and the x-axis is
what the model predicted, so a point above the line means the obsrved events exceeded predicted risk. Below the line would be over-prediction.
- Endpoints: M1's top decile averages 0.44 predicted, M2's 0.33 - each have the same
patients and same observed event rate. The predictions compressed at the high end.
- Divergence pattern: both curves track the diagonal below ~0.04, then M2
diverges and the models' gap widens with risk. Under-prediction is negligible for
low-risk patients and severe for high-risk ones; this is the same finding as the
5.89pp high-risk gap against 2.53pp population-wide, shown visually rather than stated.
- M1 is also mildly above the diagonal in the mid-range: suggesting some under-prediction or even pre-intervention, it is probably due to logistic regression trying to approximate to a Cox-derived risk. (Touch point to unpick in the discussion) 

Why this figure matters most (along side AUC result):  Both models rank patients almost identically (0.8267 vs 0.8266), which is visible as both curves are monotonic and well-ordered. Yet M2's absolute risk estimates are systematically wrong. Discrimination remains intact but calibration is broken — Van Calster's "calibration is the Achilles heel" argument is now demonstrated in a controlled setting where MICS is the only source of degradation.

(Results, Discussion: Discrimination/calibration contrast) 

Central Arguement:
Coefficients attenuate → predictions flatten → high-risk patients fall below the threshold → 12.2% fewer statin allocations. AUC stays at 0.827 throughout, so conventional monitoring sees nothing.

## Experiment 2 — Sensitivity to RRR and uptake
20 seeds per combination, 240 runs total 

**Under-prediction ≈ 10.13 × RRR × uptake** 
- Slope 10.1283, intercept −0.00049, R² = 0.999973
- All 12 combination values collapse onto one line regardless of which RRR they came from - this provides evidence that the two parameters act only through their product and don't matter individually. Any combination giving the same product produces the same under-prediction; for example RRR 0.40 with uptake 0.25 and RRR 0.25 with
uptake 0.40 both give an intensity of 0.10. Hence why all 12 points fall on the line.
  - This allows MICS to be described MICS with the single quantatity, effective treatment intensity, rather than two seperate parameters. 
  - The higher the intensity, the more MICS occurs. 
- The intercept is effectively zero (-0.005pp). At zero intensity either the staians don't do anything (RRR = 0) or nobody takes them (Uptake = 0). So either way there is no treatment effect as no events are prevented, so nothing distorts M2's training data, so there should be no under-prediction. This matters beacuse a line can fit well over a limited range without any furher meaning. Passing through the origin is what the mechanism predicts, so this confirms the relationship is mechanistic rather than a local approximation. 
- The residuals are between −0.009 and +0.008pp, against measurements ranging from 0.38 to 4.04pp, so the fit is off by less than a hundredth of a percentage point everywhere. The residuals scatter randomly around zero with no curve arc and no widening (variability increasing with intensity), meaning a straight line is the right model and nothing systematic is left unexplained.

Sense check: The line fit predicts 2.527pp baseline (0.25 × 1.0), and Experiment 1 measured 2.527pp, highlighting the exact agreement between the two independent routes in the pipeline. 

Why this works mechanically?: M2 is trained on the outcomes which have cardiovascular events did not happen because statins prevented them. It reads those absent events as evidence that the risk factors are weaker than they really are, so the model's coefficients attenuate and it under-predicts CVD risk. The number of events prevented is the number of patients treated multiplied by how much each one's risk fell = uptake × RRR. So the MICS magnitude tracks this product because the distortion *is* the prevented events, which are the product.

(Results, Discussion: Multiplicative claim)

## Experiment 3 — Statin status as a covariate

20 seeds, paired by seed.

At full uptake:
Under-prediction 2.5272 → 1.6536, a reduction of 0.8736pp (34.6%), SE 0.0816
On_Statins coefficient −0.1458 ± 0.0133 against a true value of log(0.75) = −0.2877. M2 recovers about half the statin effect

Superseded: the single-seed (seed 0) figure of 18.7% is not representative.

Feature-level recovery:
Feature	Baseline gap	Covariate gap	Recovery	SE	%
Type 1 Diabetes	−0.198	−0.158	0.040	0.007	20%
Type 2 Diabetes	−0.095	−0.075	0.020	0.002	21%
Heavy Smoker	−0.081	−0.070	0.012	0.004	14%
Family History	−0.036	−0.026	0.010	0.001	27%
Cholesterol/HDL	−0.033	−0.027	0.006	0.001	19%
Atrial Fibrillation	−0.102	−0.103	−0.001	0.003	0%

So the improvement is partly structural (coefficients recover ~20%) and partly a level shift. Coefficient attenuation persists; the covariate compensates for the consequence of MICS rather than repairing the model.

MICS is not removed. Across the strong allocation drivers, only 14–27% of the
baseline attenuation is recovered, so roughly 80% remains. The model is still learning weaker associations than the truth. The covariate reduces the under-prediction without repairing the attenuated coefficients that cause it. Report as a reduction in under-prediction, not as MICS being corrected.

Uptake sweep 
Uptake	On_Statins	Under-prediction (with covariate)	Reduction vs no covariate
0.25	−0.375	0.04pp	~94%
0.50	−0.369	0.11pp	~92%
0.75	−0.336	0.33pp	~83%
1.00	−0.146	1.65pp	~35%

The reduction column compares across two experiments, so it carries no interval
of its own.

The coefficient sits near −0.37 for three of four levels, then collapses at full uptake. SD also widens (0.025 → 0.060): the estimate becomes both wrong and unstable.

Observed mechanism: Statin allocation is determined by patient predicted risk. At full uptake, treatment status and predicted risk are nearly the same variable (collinear), so the covariate carries almost no independent signal. At lower uptake they decouple, and the covariate recovers the statin effect fully; it actually overshoots. Reaching -0.375 againts the true -0.288 at uptake 0.25. The overshoot is consistent across the 20 seeds (SD 0.025) so it is not noise, but it has no clear established explaination.

Practical reading: Real-world statin adherence is well below 75%, so in realistic conditions this mitigation would work. Two separate things make the mitigation work at lower uptake:
1. There is less to correct. MICS scales with treatment volume, so when fewer patients take statins, fewer events are prevented and less distortion enters M2's training data.

2. The covariate can actually learn the treatment effect. It needs some eligible patients to go untreated, otherwise on statins and high predicted risk describe the same people and the model cannot tell them apart. At full uptake everyone eligible is treated, so there is no separation. That is the pathological case, and it does not occur in practice.

(Results, Discussion: Mechanism and practical reading) 

## Experiment 4 — Differential uptake by ethnicity

20 seeds per condition, paired by seed. 

Equal uptake set to 0.7294, which is the population-weighted mean of the differential condition, so both uptakes treat the same total number of patients with only the distribution differing. The uptake rates are stylised, illustrative rather than empirical. Tables 8 and 9 are in notebook 02.

Table 8 - under-prediction by group
Every group except White has a smaller under-prediction gap under differential uptake, and the size of that reduction grows as the group's uptake falls. (Bangladeshi at 0.55 uptake: −0.684pp; White at 0.75: +0.054pp). Only Chinese (n=698) has an interval spanning zero.

Table 9 - attenuation of ethnicity coefficients  
Each group's coefficient attenuates less under differential uptake, and the size of that difference rises as the group's uptake falls. From +0.018 for Indian at 0.70 uptake to 
+0.073 for Bangladeshi at 0.55.
Only Pakistani and Black Caribbean swap order. All eight intervals exclude zero.
White is the reference category so has no coefficient.

Mechanism:
Experiment 2 established that MICS scales with treatment volumne (RRR x uptake), this same logic applies within a subgroup. Taking Bangladeshi patients as an example: cutting thier uptake cut from 0.7294 to 0.55, menas fewer patients accept the stains they are offered. Therefore, fewer patients are treated and fewer of thier CVD events are prevented. Population B's outcome for this group stays closer to true underlying risk, meaning less distortion in the training data for M2. M2 now learns a more accurate association for those patients, so thier ethnicity coeffcient attenutes less. 

Uptake here refers to treatment acceptance, not prescribing. Eligibility is identical
across both conditions; only the acceptance rates differ. Lower acceptance points to variable adherence, access, follow-up and trust in health services rather than to
clinician prescribing behaviour.

Equity Framing: If read naively it could be said that MICS harms White
patients the most and Bangladeshi the least. However this understanding is wrong because White patients have a higher uptake, meaning more treatment taken, more distortion occurring and therefore an increase in attenuation. Table 9 shows the smallest `change` values for the higher upatke groups, which means thier attenuation stayed high in both conditions. Any protection White patients recieve comes from the statins themselves, not from the model. As white patients get more treatment, they also gain more clinical benefit (CVD event risk down 25%). This validates that MICS is a side-effect of treatment. 

- White is the reference category in the feature matrix, so it has no ethnicity coefficient of its own. Table 9 measures every other group relative to White, and the under-prediction figures in Table 8 are what allow White patients to be compared directly.

Lower attenuation ('less MICS harm') for Bangladeshi patients isn't a benefit, because it exists only as a consequence of undertreatment, so fewer patients gain the protective benefits of statins. The patients accumulate less model distortion because they received less of the treatment causing it; less treatment means more CVD events. The model error is smaller but the health outcome is actually worse.

MICS does not offset the underlying inequity. It sits on top of it: the group taking up less of the treatment they are offered appears less harmed by the model precisely because they are receiving less treatment, while still carrying the greater burden of cardiovascular events.

Comparison with the proposal: 
The proposal predicted that groups with higher uptake would show greater coefficient attenuation. This is confirmed directionally, but there is a caveat as to what can be measured per group.
Coefficient attenuation is a property of the whole model, M2 has one T1D coefficient learned from all 100,000 patients in Population B, not a separate coefficient per ethnic group. The closest available per-group measure is the ethnicity coefficients themselves, which Table 9 uses. Each group's coefficient is compared between the equal and differential conditions. Table 8 reports under-prediction, which can be measured per group directly.

**Outstanding:** Interaction terms (ethnicity × treatment) as the proposed
mitigation. It is expected to be unstable given Experiment 3's collinearity and small size of groups.

## Open questions
- SLE reverses. +0.102, the interval excludes zero, contradicting the proposal's prediction of large attenuation. There is currently no established mechanism behind this. M1's coefficient is barely identified (percentile range 0.067–0.502) at ~0.1% prevalence, but the positive sign is consistent across all 20 seeds so it is not simple noise. *Ask Dr Brown*
  - TDOD: Run a diagnostic count on the SLE patients per population and the proportion treated. If essentially all are above threshold, that is a different regime from partially-treated features.
- Atrial fibrillation recovers nothing in Experiment 3 (−0.001 ± 0.003) while T1D and T2D recover ~20%. It is the one strong allocation driver that gets no benefit? No current explanation. *Ask Dr Brown*
- `On_Statins` overshoots at low uptake, −0.375 against a true −0.2877, about 30% too strong, at 20 seeds with SD 0.025 so it is not noise. It could be possibly absorbing residual attenuation from other predictors, since it is the only term free to move. *Ask Dr Brown*
- TODO: Experiment 3 option B hasn't been implemented: M1-implied covariate value at deployment. Clinicians would know who is already on statins, so this is arguably the more realistic assumption than the current mapping.

## Resolved questions
- Rheumatoid arthritis. Logged as ~0.003 and probably noise from a single seed. At 20 seeds it is 0.2419 ± 0.0199 (a clear positive), it is a real risk factor consistent with QRISK3. The single-seed value was an unrepresentative draw. This result is the agrugment for multi-seed simulation: two single-seed observations looked equally like noise, one was and one was not. Acts as a methods justification
- CKD. 0.0870 ± 0.0057, it is precisely estimated and genuinely low for a condition that QRISK3 treats as substantial. The age-collinearity explanation currently stands; CKD generation is steeply age-dependent and Age is in the model.

## Limitations
- Independent risk factor generation. The risk factors are generated independently;there is currently no correlation structure apart from explicit age dependencies. 
Clearest example: patients are assigned BP medication at a flat 21.9% regardless of systolic bp, when in practice treatment is prescribed in response to it. This doesn't bias the MICS comparison, since M1 and M2 both have populations with identical structure.
The missing correlation is shared rather than confounding. This limits external realism rather than internal validity.
  - Possible consequence: Treated Hypertension shows minimal attenuation (−0.011), which may partly be due to it being independent of blood pressure, so carries a lower allocation signal than it should.
  - Future work: modelling these interdependencies may also amplify MICS not just add realism, since correlated risk factors would produce a more sharply defined treated group.
- The CKD signal is absorbed by age. M1's coefficient of 0.087 is low for a condition that QRISK3 treats as subtantial. CKD generation is steeply age-dependent, however age is present within the model, so it absorbs much of CKD's predictive signal.
- QRISK3 implementation is validated against the R package, not the original ClinRisk C source, which has not been publicly available since April 2026.
- Single intervention type. Real CVD management involves multiple simultaneous interventions. MICS is demonstrated here for statins, but whether it behaves the same way in other interventions is untested.
- Experiment 4 uptake rates are stylised, not empirical estimates. This only tests the mechanism behind ethnicity based upptake but not the real-world magnitudes.
- Small ethnic subgroups (Chinese 698, Black Caribbean 998) give wide intervals. *Subgroup MICS effects are hard to detect at realsitic population sizes*
- The intervals describe Monte Carlo variability i.e. how much the result moves across different random draws, we are not sampling uncertainty about a real population, since the data is generated rather than sampled. It can be described as empirical 95% intervals across 20 replicates, not confidence intervals. With 20 values the tails (2.5, 97.5) are essentially the min and max; 100 seeds would be needed to stabilise them.
- The calibration figure is generated on a single-seed, since it needs per-patient predictions rather than summary statistics.

(Disscussion)

## Discussion pile
Ordered roughly by strength of finding

# Must be discussed
1. Monitoring metrics fail to detect MICS. AUC and PR-AUC were unchanged. Specificity, PPV and F1 all improve, and each improvement is interval-backed. A retrained model that detects 4.7pp fewer genuine CVD risk cases looks better on four of the six conventional metrics and identical on the other two. Only sensitivity and NPV reveal the problem, and those are the ones most easily rationalised as reduced overtreatment. This is stronger than the proposal predicted: it claimed that stable AUC would mask degradation but several metrics actively improve. This can be explained by flattening. Compression preserves the ordering that AUC measures while breaking the absolute scale that calibration measures, the harm now concentrates at the treatment threshold. 
2. Attenuation tracks allocation influence, not coefficient magnitude. A feature must be both common enough and decisive enough to chnage treatment decisions. This explains why the SLE prediction was wrong and why CKD and treated hypertension stay flat.
3. Differential uptake produces heterogeneous MICS effects, but the equity reading is inverted. Lower-uptake groups accumulate less distortion, only because they receive less of the treatment causing it; a small model error causes a worse health outcome. MICS sits on top of the existing inequity rather than offsetting it.

# Need to mention
4. MICS is continuous and multiplicative, only the product of RRR and uptake matters. Meaning it can be described with a single quantity. 
5. The covariate mitigation works at realistic uptake values and fails at total concordance, which is the opposite of the usual theory vs in practice story.
6. Risk factor correlation as an area for future work, it may amplify MICS rather than merely add realism.
7. Multi-seed replication is the correct methodology, as illustrated by the RA episode.