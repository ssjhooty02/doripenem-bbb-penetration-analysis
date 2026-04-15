"""
Statistical Analysis: Doripenem BBB Penetration Study
======================================================
Based on: Margetis et al. (2011) "Penetration of Intact Blood-Brain Barrier by Doripenem"
          Antimicrobial Agents and Chemotherapy, 55(7):3637-3638
          https://pmc.ncbi.nlm.nih.gov/articles/PMC3122383/

Analyses performed (NOT in the original paper):
  Part A — Correlation & Multiple Linear Regression
    - Pearson & Spearman correlations between PK variables and covariates
    - Multiple linear regression: CSF/plasma ratio ~ age + weight + CrCl + sampling_time
  Part B — Logistic Regression for Therapeutic Threshold
    - Binary outcome: CSF concentration >= 0.25 ug/ml (MIC threshold)
    - Odds ratios with 95% CI
    - ROC curve with AUC

Variables analysed: age, weight, creatinine clearance, sampling time, CSF/plasma ratio.
Dataset: Synthetic (n=200), generated to be consistent with published PK profiles.
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

import statsmodels.api as sm

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
# DEFINE ANALYSIS VARIABLES
# ============================================================
# Display labels for clean figure output
VAR_LABELS = {
    "age_years": "Age (yrs)",
    "weight_kg": "Weight (kg)",
    "creatinine_clearance_ml_min": "CrCl (mL/min)",
    "sampling_time_min": "Sampling Time (min)",
    "csf_plasma_ratio": "CSF/Plasma Ratio",
}

# Five core variables used consistently across all analyses
analysis_vars = [
    "age_years", "weight_kg", "creatinine_clearance_ml_min",
    "sampling_time_min", "csf_plasma_ratio"
]
# Predictors for regression models (all except the outcome)
predictors = ["age_years", "weight_kg", "creatinine_clearance_ml_min",
              "sampling_time_min"]

X = df[predictors].copy()

# ============================================================
# PART A: CORRELATION & MULTIPLE LINEAR REGRESSION
# ============================================================
print("\n" + "=" * 70)
print("PART A: CORRELATION ANALYSIS & MULTIPLE LINEAR REGRESSION")
print("=" * 70)

# --- A1: Correlation Matrix ---
print("\n--- A1: Pearson Correlation Matrix ---\n")
corr_matrix = df[analysis_vars].corr(method="pearson")
print(corr_matrix.round(3).to_string())

# --- A2: Key Spearman correlations ---
print("\n--- A2: Spearman Rank Correlations (CSF/plasma ratio vs. covariates) ---\n")
spearman_results = []
for var in predictors:
    rho, p = stats.spearmanr(df[var], df["csf_plasma_ratio"])
    sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
    print(f"  CSF/plasma ratio vs {var:>35s}: rho = {rho:+.3f}, p = {p:.4f} {sig}")
    spearman_results.append({"Variable": var, "Rho": rho, "P_Value": p, "Significance": sig})

# --- A3: Multiple Linear Regression ---
print("\n--- A3: Multiple Linear Regression ---")
print("    Outcome: CSF/Plasma Ratio")
print("    Predictors: age, weight, CrCl, sampling_time\n")

y = df["csf_plasma_ratio"].copy()
X_sm = sm.add_constant(X)
ols_model = sm.OLS(y, X_sm).fit()
print(ols_model.summary())

# --- Figure 1: Correlation Heatmap + Scatter ---
fig, axes = plt.subplots(1, 2, figsize=(16, 7))

corr_display = corr_matrix.copy()
corr_display.index = [VAR_LABELS.get(v, v) for v in corr_display.index]
corr_display.columns = [VAR_LABELS.get(v, v) for v in corr_display.columns]
sns.heatmap(corr_display, annot=True, fmt=".2f", cmap="RdBu_r", center=0,
            square=True, linewidths=0.5, ax=axes[0],
            cbar_kws={"shrink": 0.8})
axes[0].set_title("Pearson Correlation Matrix", fontsize=14, fontweight="bold")
axes[0].tick_params(axis="x", rotation=45)

axes[1].scatter(df["sampling_time_min"], df["csf_plasma_ratio"],
                c=df["therapeutic_threshold_met"], cmap="RdYlGn",
                edgecolors="black", alpha=0.7, s=60)
axes[1].set_xlabel("Sampling Time (min post-infusion)", fontsize=12)
axes[1].set_ylabel("CSF/Plasma Ratio", fontsize=12)
axes[1].set_title("CSF/Plasma Ratio vs. Sampling Time", fontsize=14, fontweight="bold")

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
print("    Predictors: age, weight, CrCl, sampling_time\n")

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

# ============================================================
# TABLE FIGURES
# ============================================================

# --- Figure 2: Comprehensive P-Value Summary ---
pval_rows = []
for r in spearman_results:
    pval_rows.append(["Spearman Correlation", VAR_LABELS.get(r["Variable"], r["Variable"]), f"{r['Rho']:+.3f} (rho)", f"{r['P_Value']:.4f}", r["Significance"]])
for var in ols_model.pvalues.index:
    if var == "const":
        continue
    pval_rows.append(["OLS Regression", VAR_LABELS.get(var, var), f"{ols_model.params[var]:.4f} (coef)", f"{ols_model.pvalues[var]:.4f}",
                       "***" if ols_model.pvalues[var] < 0.001 else "**" if ols_model.pvalues[var] < 0.01 else "*" if ols_model.pvalues[var] < 0.05 else "ns"])
for var in logit_model.pvalues.index:
    if var == "const":
        continue
    pval_rows.append(["Logistic Regression", VAR_LABELS.get(var, var), f"{logit_model.params[var]:.4f} (coef)", f"{logit_model.pvalues[var]:.4f}",
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
          "Dependent variables — Spearman: CSF/plasma ratio  |  OLS Regression: CSF/plasma ratio  |  "
          "Logistic Regression: therapeutic threshold (CSF ≥ 0.25 µg/mL)",
          ha="center", fontsize=9, fontstyle="italic", color="#555555")
plt.savefig("figure2_pvalue_summary.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: figure2_pvalue_summary.png")

# NOTE: Figure 3 (OLS regression summary table) is generated by analysis.R

# --- Figure 4: Odds Ratios with Forest Plot ---
or_plot_data = or_table.drop("const", errors="ignore")
or_plot_data.index = [VAR_LABELS.get(v, v) for v in or_plot_data.index]
fig5, (ax5a, ax5b) = plt.subplots(1, 2, figsize=(16, max(4, len(or_plot_data) * 0.8 + 2)),
                                   gridspec_kw={"width_ratios": [1, 1.2]})

# Forest plot
y_pos = range(len(or_plot_data))
ax5a.errorbar(or_plot_data["Odds_Ratio"], y_pos,
              xerr=[or_plot_data["Odds_Ratio"] - or_plot_data["OR_lower_95"],
                    or_plot_data["OR_upper_95"] - or_plot_data["Odds_Ratio"]],
              fmt="o", color="#4472c4", ecolor="#999999", elinewidth=2, capsize=5, markersize=8)
ax5a.axvline(x=1.0, color="red", linestyle="--", linewidth=1.5, label="OR = 1.0 (no effect)")
ax5a.set_yticks(list(y_pos))
ax5a.set_yticklabels(or_plot_data.index, fontsize=11)
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
fig5.subplots_adjust(bottom=0.18)
fig5.text(0.5, 0.09,
          "Dependent variable: therapeutic threshold achievement (CSF ≥ 0.25 µg/mL, binary). "
          "Coefficients (log-odds) exponentiated to odds ratios for clinical interpretability.",
          ha="center", fontsize=8.5, fontstyle="italic", color="#555555")
fig5.text(0.5, 0.03,
          "Interpretation: OR > 1 indicates increased odds per unit increase in predictor; OR < 1 indicates decreased odds "
          "(e.g., OR = 1.07 for Sampling Time = 7% increase in odds per additional minute post-infusion).",
          ha="center", fontsize=8, fontstyle="italic", color="#888888")
plt.savefig("figure4_odds_ratios.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: figure4_odds_ratios.png")

# NOTE: Figure 5 (ROC curve + probability plot) is generated by analysis.R

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
  Part B: Logistic Regression with odds ratios, ROC/AUC
""")
