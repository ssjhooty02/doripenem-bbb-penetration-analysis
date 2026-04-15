# Doripenem Blood-Brain Barrier Penetration — Statistical Analysis

**Author:** Sartaj Jhooty, 2026

Statistical analysis of doripenem pharmacokinetics across the intact blood-brain barrier (BBB), extending the preliminary findings of Margetis et al. (2011). Implemented in both **Python** and **R**.

## Paper Reference

> Margetis K, Dimaraki E, Charkoftaki G, et al. *Penetration of Intact Blood-Brain Barrier by Doripenem.* Antimicrob Agents Chemother. 2011;55(7):3637-3638.
> https://pmc.ncbi.nlm.nih.gov/articles/PMC3122383/

## Dataset

The original study (Margetis et al., 2011) reported pharmacokinetic data on 5 neurosurgical patients, measuring weight, sampling time post-infusion, plasma concentration, and CSF concentration. A synthetic dataset of **n = 200 patients** was generated with pharmacokinetic profiles consistent with the published data. To enable multivariate regression modeling, clinically relevant covariates not reported in the original study — age and creatinine clearance — were added to the simulated cohort. CSF/plasma ratio was derived as a measure of BBB penetration efficiency.

Whereas the original paper established preliminary evidence that doripenem penetrates the intact BBB, the present analysis extends those findings by evaluating whether patient-level clinical variables can predict the degree of CSF penetration (multiple linear regression on CSF/plasma ratio) and the probability of achieving therapeutic CSF concentrations (logistic regression on CSF ≥ 0.25 µg/mL).

> **Disclaimer:** This dataset is entirely simulated and all findings are derived from synthetic data. The results presented herein are not validated against clinical observations and must not be used to inform clinical decision-making in any capacity. The purpose of this project is to demonstrate proficiency in applying statistical analysis methodologies (correlation, regression, ROC analysis) to neurosurgical pharmacokinetic datasets using Python and R.

**Variables:** age, weight, creatinine clearance, sampling time post-infusion, CSF/plasma ratio. Plasma concentration (µg/mL) and CSF concentration (µg/mL) were used to derive the CSF/plasma ratio and the binary therapeutic threshold (CSF ≥ 0.25 µg/mL).

## Analyses

| Analysis | Description | Figures |
|---|---|---|
| Correlation Analysis | Pearson & Spearman bivariate relationships among clinical covariates and CSF/plasma ratio | 1 |
| Multiple Linear Regression (OLS) | Multivariate predictors of CSF/plasma ratio with coefficients and 95% CIs (R² = 0.47) | 3 |
| Logistic Regression & ROC Analysis | Binary classification of therapeutic CSF achievement (≥ 0.25 µg/mL) with odds ratios, forest plot, and model discrimination (AUC = 0.954) | 4, 5 |

> Figure 2 provides a comprehensive p-value summary spanning all analyses above.

## Key Results

**Multiple Linear Regression — CSF/Plasma Ratio (Figures 1, 2, 3)**

Sampling time post-infusion was the strongest predictor of CSF/plasma ratio (p < 0.001), consistent with the expected pharmacokinetics of a renally-cleared carbapenem undergoing time-dependent passive diffusion across the intact BBB. Weight was also a significant predictor (p = 0.023), in agreement with established dosing principles whereby body weight influences volume of distribution and, consequently, achievable drug concentrations at the target site. Age and creatinine clearance did not reach statistical significance in the multivariate model.

The model yielded an R² of 0.47, indicating that 47% of the variance in CSF/plasma ratio is attributable to the four predictor variables (age, weight, creatinine clearance, and sampling time). The residual 53% of unexplained variance likely reflects unmeasured sources of inter-individual biological variability, including differences in BBB permeability, regional cerebral blood flow, meningeal inflammatory status, CSF turnover rate, and plasma protein binding.

**Logistic Regression — Therapeutic Threshold (Figures 2, 4, 5)**

To assess predictors of clinically meaningful BBB penetration, a logistic regression was performed with the binary outcome of CSF concentration ≥ 0.25 µg/mL — the MIC threshold for susceptible organisms — as opposed to the continuous CSF/plasma ratio modeled in the linear regression. Sampling time (OR = 1.07, p < 0.001) and weight (OR = 0.93, p = 0.043) were the significant predictors, mirroring the linear regression findings and affirming the consistent influence of these two covariates across both analytical approaches. An OR of 1.07 corresponds to a 7% increase in the odds of therapeutic achievement per additional minute post-infusion; an OR of 0.93 corresponds to a 7% decrease in odds per additional kilogram of body weight.

The model achieved an AUC of 0.954 on ROC analysis, indicating excellent discriminative performance. In clinical classification contexts, an AUC exceeding 0.9 is considered outstanding (reference range: 0.5 = no discrimination, 1.0 = perfect discrimination), demonstrating that the logistic model reliably distinguishes patients who achieve therapeutic CSF concentrations from those who do not based on readily available clinical parameters.

### Figures

**Figure 1 — Correlation Analysis (Python)** — Pearson correlation heatmap and CSF/plasma ratio vs. sampling time:
![Correlation Analysis](figure1_correlation_analysis.png)

**Figure 2 — Comprehensive P-Value Summary (Python)** — All statistical tests and their p-values across Spearman correlations, OLS regression, and logistic regression:
![P-Value Summary](figure2_pvalue_summary.png)

**Figure 3 — OLS Regression Summary (R)** — Coefficients, standard errors, t-statistics, and confidence intervals (R² = 0.47):
![OLS Summary](figure3_ols_summary.png)

**Figure 4 — Odds Ratios with Forest Plot (Python)** — Odds ratios with 95% confidence intervals for predictors of therapeutic threshold achievement:
![Odds Ratios](figure4_odds_ratios.png)

**Figure 5 — ROC Curve & Predicted Probability (R)** — Model discrimination performance and predicted therapeutic achievement by sampling time:
![Logistic Regression](figure5_logistic_regression.png)

## How to Run

### Prerequisites

**Python 3.8+**
```bash
pip install pandas numpy matplotlib seaborn scipy statsmodels
```

**R 4.0+**
```r
install.packages(c("pROC", "car"))
```

### Generate Data

```bash
python generate_synthetic_data.py
```
This creates `doripenem_bbb_synthetic_data.csv`.

### Run Analysis

**Python:**
```bash
python analysis.py
```

**R:**
```bash
Rscript analysis.R
```

Both produce console output with full statistical results. Figures are split between the two scripts — each figure is generated by only one language (see Output Files table).

## Output Files

| File | Description | Generated by |
|---|---|---|
| `doripenem_bbb_synthetic_data.csv` | Synthetic dataset (n=200) | Python |
| `figure1_correlation_analysis.png` | Pearson heatmap + CSF/plasma ratio vs. time scatter | Python |
| `figure2_pvalue_summary.png` | Comprehensive p-value table across all tests | Python |
| `figure3_ols_summary.png` | Full OLS regression summary table | R |
| `figure4_odds_ratios.png` | Odds ratios with forest plot | Python |
| `figure5_logistic_regression.png` | ROC curve + predicted probability plot | R |
