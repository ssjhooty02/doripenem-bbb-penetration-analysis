"""
Statistical Analysis: Doripenem BBB Penetration Study
======================================================
Based on: Margetis et al. (2011) "Penetration of Intact Blood-Brain Barrier by Doripenem"
          Antimicrobial Agents and Chemotherapy, 55(7):3637-3638
          https://pmc.ncbi.nlm.nih.gov/articles/PMC3122383/

Analyses performed (NOT in the original paper):
  Option A — Correlation & Multiple Linear Regression
    - Pearson & Spearman correlations between PK variables and covariates
    - Multiple linear regression: CSF/plasma ratio ~ age + weight + CrCl + sampling_time + sex
  Option B — Logistic Regression for Therapeutic Threshold
    - Binary outcome: CSF concentration >= 0.25 ug/ml (MIC threshold)
    - ROC curve with AUC
    - Classification report

Dataset: Synthetic (n=55), generated to be consistent with published PK profiles.
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, auc, classification_report, confusion_matrix
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler

import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor

import warnings
warnings.filterwarnings("ignore")

# ============================================================
# 0. LOAD DATA
# ============================================================
df = pd.read_csv("doripenem_bbb_synthetic_data.csv")
print("=" * 70)
print("DORIPENEM BBB PENETRATION — STATISTICAL ANALYSIS (Python)")
print("=" * 70)
print(f"\nDataset: n = {len(df)} patients")
print(f"Therapeutic threshold met (CSF >= 0.25 ug/ml): "
      f"{df['therapeutic_threshold_met'].sum()}/{len(df)} "
      f"({100 * df['therapeutic_threshold_met'].mean():.1f}%)\n")

# ============================================================
# PART A: CORRELATION & MULTIPLE LINEAR REGRESSION
# ============================================================
print("\n" + "=" * 70)
print("PART A: CORRELATION ANALYSIS & MULTIPLE LINEAR REGRESSION")
print("=" * 70)

# --- A1: Correlation Matrix ---
print("\n--- A1: Pearson Correlation Matrix (key variables) ---\n")
corr_vars = [
    "age_years", "weight_kg", "bmi", "creatinine_clearance_ml_min",
    "sampling_time_min", "plasma_concentration_ug_ml",
    "csf_concentration_ug_ml", "csf_plasma_ratio"
]
corr_matrix = df[corr_vars].corr(method="pearson")
print(corr_matrix.round(3).to_string())

# --- A2: Key Spearman correlations ---
print("\n--- A2: Spearman Rank Correlations (CSF concentration vs. covariates) ---\n")
spearman_targets = [
    "sampling_time_min", "weight_kg", "age_years",
    "creatinine_clearance_ml_min", "bmi", "plasma_concentration_ug_ml"
]
spearman_results = []
for var in spearman_targets:
    rho, p = stats.spearmanr(df[var], df["csf_concentration_ug_ml"])
    sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
    print(f"  CSF conc vs {var:>35s}: rho = {rho:+.3f}, p = {p:.4f} {sig}")
    spearman_results.append({"Variable": var, "Rho": rho, "P_Value": p, "Significance": sig})

# --- A3: Multiple Linear Regression ---
print("\n--- A3: Multiple Linear Regression ---")
print("    Outcome: CSF/Plasma Ratio")
print("    Predictors: age, weight, CrCl, sampling_time, sex\n")

df["sex_numeric"] = (df["sex"] == "M").astype(int)
predictors = ["age_years", "weight_kg", "creatinine_clearance_ml_min",
              "sampling_time_min", "sex_numeric"]

X = df[predictors].copy()
y = df["csf_plasma_ratio"].copy()

# Add constant for statsmodels
X_sm = sm.add_constant(X)
ols_model = sm.OLS(y, X_sm).fit()
print(ols_model.summary())

# VIF for multicollinearity check
print("\n--- Variance Inflation Factors (VIF) ---")
vif_results = []
for i, col in enumerate(predictors):
    vif = variance_inflation_factor(X.values, i)
    flag = " *** HIGH" if vif > 5 else ""
    print(f"  {col:>35s}: VIF = {vif:.2f}{flag}")
    vif_results.append({"Variable": col, "VIF": vif})

# --- A4: Correlation Heatmap ---
fig, axes = plt.subplots(1, 2, figsize=(18, 7))

# Heatmap
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="RdBu_r", center=0,
            square=True, linewidths=0.5, ax=axes[0],
            cbar_kws={"shrink": 0.8})
axes[0].set_title("Pearson Correlation Matrix", fontsize=14, fontweight="bold")
axes[0].tick_params(axis="x", rotation=45)

# Scatter: sampling time vs CSF concentration
axes[1].scatter(df["sampling_time_min"], df["csf_concentration_ug_ml"],
                c=df["therapeutic_threshold_met"], cmap="RdYlGn",
                edgecolors="black", alpha=0.7, s=60)
axes[1].axhline(y=0.25, color="red", linestyle="--", linewidth=1.5,
                label="MIC Threshold (0.25 µg/ml)")
axes[1].set_xlabel("Sampling Time (min post-infusion)", fontsize=12)
axes[1].set_ylabel("CSF Doripenem Concentration (µg/ml)", fontsize=12)
axes[1].set_title("CSF Concentration vs. Sampling Time", fontsize=14, fontweight="bold")
axes[1].legend(fontsize=11)

plt.tight_layout()
plt.savefig("figure1_correlation_analysis.png", dpi=150, bbox_inches="tight")
plt.close()
print("\nSaved: figure1_correlation_analysis.png")

# ============================================================
# PART B: LOGISTIC REGRESSION — THERAPEUTIC THRESHOLD
# ============================================================
print("\n\n" + "=" * 70)
print("PART B: LOGISTIC REGRESSION — THERAPEUTIC THRESHOLD ACHIEVEMENT")
print("=" * 70)
print("\n    Outcome: CSF >= 0.25 µg/ml (binary)")
print("    Predictors: age, weight, CrCl, sampling_time, sex\n")

# --- B1: Logistic Regression (statsmodels for inference) ---
X_logit = sm.add_constant(X)
logit_model = sm.Logit(df["therapeutic_threshold_met"], X_logit).fit(disp=0)
print(logit_model.summary())

# Odds ratios
print("\n--- Odds Ratios with 95% CI ---")
odds_ratios = np.exp(logit_model.params)
ci = np.exp(logit_model.conf_int())
ci.columns = ["OR_lower_95", "OR_upper_95"]
or_table = pd.DataFrame({
    "Odds_Ratio": odds_ratios.round(4),
    "OR_lower_95": ci["OR_lower_95"].round(4),
    "OR_upper_95": ci["OR_upper_95"].round(4),
    "p_value": logit_model.pvalues.round(4),
})
print(or_table.to_string())

# --- B2: sklearn Logistic Regression (for ROC/AUC and CV) ---
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

lr_sklearn = LogisticRegression(random_state=42, max_iter=1000)
lr_sklearn.fit(X_scaled, df["therapeutic_threshold_met"])

y_pred = lr_sklearn.predict(X_scaled)
y_prob = lr_sklearn.predict_proba(X_scaled)[:, 1]

print("\n--- Classification Report ---")
class_report_dict = classification_report(df["therapeutic_threshold_met"], y_pred,
                                          target_names=["Below MIC", "Above MIC"],
                                          output_dict=True)
print(classification_report(df["therapeutic_threshold_met"], y_pred,
                            target_names=["Below MIC", "Above MIC"]))

print("--- Confusion Matrix ---")
cm = confusion_matrix(df["therapeutic_threshold_met"], y_pred)
print(f"  TN={cm[0,0]}  FP={cm[0,1]}")
print(f"  FN={cm[1,0]}  TP={cm[1,1]}")

# Cross-validation
cv_scores = cross_val_score(lr_sklearn, X_scaled, df["therapeutic_threshold_met"],
                            cv=5, scoring="accuracy")
print(f"\n--- 5-Fold Cross-Validation ---")
print(f"  Accuracy: {cv_scores.mean():.3f} (+/- {cv_scores.std():.3f})")

cv_auc = cross_val_score(lr_sklearn, X_scaled, df["therapeutic_threshold_met"],
                         cv=5, scoring="roc_auc")
print(f"  AUC:      {cv_auc.mean():.3f} (+/- {cv_auc.std():.3f})")

# NOTE: Figure 5 (ROC curve + probability plot) is generated by analysis.R

# ============================================================
# ADDITIONAL TABLE FIGURES
# ============================================================

# --- Figure 2: Comprehensive P-Value Summary ---
pval_rows = []
for r in spearman_results:
    pval_rows.append(["Spearman Correlation", r["Variable"], f"{r['Rho']:+.3f} (rho)", f"{r['P_Value']:.4f}", r["Significance"]])
for var in ols_model.pvalues.index:
    if var == "const":
        continue
    pval_rows.append(["OLS Regression", var, f"{ols_model.params[var]:.4f} (coef)", f"{ols_model.pvalues[var]:.4f}",
                       "***" if ols_model.pvalues[var] < 0.001 else "**" if ols_model.pvalues[var] < 0.01 else "*" if ols_model.pvalues[var] < 0.05 else "ns"])
for var in logit_model.pvalues.index:
    if var == "const":
        continue
    pval_rows.append(["Logistic Regression", var, f"{logit_model.params[var]:.4f} (coef)", f"{logit_model.pvalues[var]:.4f}",
                       "***" if logit_model.pvalues[var] < 0.001 else "**" if logit_model.pvalues[var] < 0.01 else "*" if logit_model.pvalues[var] < 0.05 else "ns"])

fig3, ax3 = plt.subplots(figsize=(14, max(6, len(pval_rows) * 0.4 + 2)))
ax3.axis("off")
ax3.set_title("Comprehensive P-Value Summary\n*** p<0.001  ** p<0.01  * p<0.05  ns = not significant",
              fontsize=14, fontweight="bold", pad=20)
col_labels = ["Test Type", "Variable", "Statistic", "P-Value", "Sig."]
colors = []
for row in pval_rows:
    p = float(row[3])
    if p < 0.001:
        colors.append(["#c6efce"] * 5)
    elif p < 0.01:
        colors.append(["#d4edda"] * 5)
    elif p < 0.05:
        colors.append(["#fff3cd"] * 5)
    else:
        colors.append(["#f8f9fa"] * 5)
table3 = ax3.table(cellText=pval_rows, colLabels=col_labels, cellColours=colors,
                   colColours=["#4472c4"] * 5, loc="center", cellLoc="center")
table3.auto_set_font_size(False)
table3.set_fontsize(10)
table3.scale(1, 1.6)
for (row, col), cell in table3.get_celld().items():
    if row == 0:
        cell.set_text_props(color="white", fontweight="bold")
    cell.set_edgecolor("#dddddd")
fig3.text(0.5, 0.01,
          "Dependent variables — Spearman: CSF concentration (µg/mL)  |  OLS Regression: CSF/plasma ratio  |  "
          "Logistic Regression: therapeutic threshold (CSF ≥ 0.25 µg/mL)",
          ha="center", fontsize=9, fontstyle="italic", color="#555555")
plt.savefig("figure2_pvalue_summary.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: figure2_pvalue_summary.png")

# NOTE: Figure 3 (OLS regression summary table) is generated by analysis.R

# --- Figure 4: Odds Ratios with Forest Plot ---
or_plot_data = or_table.drop("const", errors="ignore")
# Exclude sex from forest plot (wide CI distorts scale; not significant p=0.80)
or_forest_data = or_plot_data.drop("sex_numeric", errors="ignore")
fig5, (ax5a, ax5b) = plt.subplots(1, 2, figsize=(16, max(4, len(or_plot_data) * 0.8 + 2)),
                                   gridspec_kw={"width_ratios": [1, 1.2]})

# Forest plot (without sex)
y_pos = range(len(or_forest_data))
ax5a.errorbar(or_forest_data["Odds_Ratio"], y_pos,
              xerr=[or_forest_data["Odds_Ratio"] - or_forest_data["OR_lower_95"],
                    or_forest_data["OR_upper_95"] - or_forest_data["Odds_Ratio"]],
              fmt="o", color="#4472c4", ecolor="#999999", elinewidth=2, capsize=5, markersize=8)
ax5a.axvline(x=1.0, color="red", linestyle="--", linewidth=1.5, label="OR = 1.0 (no effect)")
ax5a.set_yticks(list(y_pos))
ax5a.set_yticklabels(or_forest_data.index, fontsize=11)
ax5a.set_xlabel("Odds Ratio (95% CI)", fontsize=12)
ax5a.set_title("Forest Plot — Odds Ratios", fontsize=13, fontweight="bold")
ax5a.legend(fontsize=10)
ax5a.grid(axis="x", alpha=0.3)

# Table
ax5b.axis("off")
or_rows = []
for var in or_plot_data.index:
    or_rows.append([var, f"{or_plot_data.loc[var, 'Odds_Ratio']:.4f}",
                    f"{or_plot_data.loc[var, 'OR_lower_95']:.4f}",
                    f"{or_plot_data.loc[var, 'OR_upper_95']:.4f}",
                    f"{or_plot_data.loc[var, 'p_value']:.4f}",
                    "***" if or_plot_data.loc[var, 'p_value'] < 0.001 else
                    "**" if or_plot_data.loc[var, 'p_value'] < 0.01 else
                    "*" if or_plot_data.loc[var, 'p_value'] < 0.05 else "ns"])
or_col_labels = ["Variable", "OR", "CI Lower", "CI Upper", "P-Value", "Sig."]
or_colors = []
for row in or_rows:
    p = float(row[4])
    if p < 0.001:
        or_colors.append(["#c6efce"] * 6)
    elif p < 0.05:
        or_colors.append(["#fff3cd"] * 6)
    else:
        or_colors.append(["#f8f9fa"] * 6)
table5 = ax5b.table(cellText=or_rows, colLabels=or_col_labels, cellColours=or_colors,
                    colColours=["#4472c4"] * 6, loc="center", cellLoc="center")
table5.auto_set_font_size(False)
table5.set_fontsize(10)
table5.scale(1, 1.8)
for (row, col), cell in table5.get_celld().items():
    if row == 0:
        cell.set_text_props(color="white", fontweight="bold")
    cell.set_edgecolor("#dddddd")
ax5b.set_title("Odds Ratios — 95% Confidence Intervals", fontsize=13, fontweight="bold")

plt.tight_layout()
fig5.text(0.5, 0.02,
          "Dependent variable: therapeutic threshold achievement (CSF ≥ 0.25 µg/mL, binary)",
          ha="center", fontsize=9, fontstyle="italic", color="#555555")
fig5.text(0.5, -0.01,
          "Note: sex excluded from forest plot due to wide CI distorting scale (not significant, p = 0.80); included in table.",
          ha="center", fontsize=8, fontstyle="italic", color="#888888")
plt.savefig("figure4_odds_ratios.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: figure4_odds_ratios.png")

# ============================================================
# SUMMARY
# ============================================================
print("\n\n" + "=" * 70)
print("ANALYSIS COMPLETE")
print("=" * 70)
print("""
Files generated (Python):
  - figure1_correlation_analysis.png   (heatmap + scatter plot)
  - figure2_pvalue_summary.png         (comprehensive p-value table)
  - figure4_odds_ratios.png            (odds ratios + forest plot)

Files generated by analysis.R:
  - figure3_ols_summary.png            (full OLS regression table)
  - figure5_logistic_regression.png    (ROC curve + probability plot)
Analyses performed:
  Part A: Pearson & Spearman correlations, Multiple Linear Regression (OLS)
          with VIF multicollinearity diagnostics
  Part B: Logistic Regression with odds ratios, ROC/AUC, confusion matrix,
          5-fold cross-validation
""")
