"""
Synthetic Data Generator for Doripenem BBB Penetration Study
=============================================================
Based on: Margetis et al. (2011) "Penetration of Intact Blood-Brain Barrier by Doripenem"
          Antimicrobial Agents and Chemotherapy, 55(7):3637-3638
          https://pmc.ncbi.nlm.nih.gov/articles/PMC3122383/

NOTE: This dataset is SIMULATED. The original study had n=5 patients.
      This script generates n=200 synthetic patients with pharmacokinetic
      profiles consistent with the published data for statistical analysis
      demonstration purposes.

Original Table 1 data (Margetis et al.):
  Infusion(min) | SamplingTime(min) | CSF(ug/ml) | Plasma(ug/ml) | Weight(kg)
  33            | 10                | 0.11       | 32.8          | 75
  38            | 17                | 0.14       | 18.7          | 65
  41            | 22                | 0.21       | 14.9          | 70
  27            | 71                | 0.38       | 12.9          | 75
  34            | 176               | 0.48       | 9.5           | 75
"""

import numpy as np
import pandas as pd

np.random.seed(42)

N = 200

# --- Demographics ---
age = np.random.normal(loc=52, scale=12, size=N).clip(25, 80).round(0).astype(int)
sex = np.random.choice(["M", "F"], size=N, p=[0.58, 0.42])
weight_kg = np.random.normal(loc=72, scale=10, size=N).clip(50, 110).round(1)
height_cm = np.where(
    sex == "M",
    np.random.normal(175, 7, N),
    np.random.normal(163, 6, N),
).clip(148, 198).round(1)
bmi = (weight_kg / (height_cm / 100) ** 2).round(1)

# Creatinine clearance (mL/min) - Cockcroft-Gault influenced by age/weight/sex
crcl_base = ((140 - age) * weight_kg) / (72 * np.random.normal(0.9, 0.15, N).clip(0.5, 1.5))
crcl = np.where(sex == "F", crcl_base * 0.85, crcl_base).clip(40, 180).round(1)

# --- Infusion & Sampling ---
infusion_duration_min = np.random.normal(loc=34, scale=5, size=N).clip(20, 50).round(0).astype(int)
# Sampling times: spread across 10-200 min post-infusion to capture PK curve
sampling_time_min = np.random.uniform(8, 200, size=N).round(0).astype(int)

# --- Pharmacokinetic Modeling ---
# Plasma: mono-exponential decay from ~35 ug/ml at t=0
# Based on paper: plasma drops from 32.8 (t=10) to 9.5 (t=176)
# Approx half-life ~60 min => k_elim ~ 0.0115
plasma_c0 = np.random.normal(38, 5, N).clip(25, 55)
k_elim = np.random.normal(0.0115, 0.002, N).clip(0.005, 0.02)
# Weight and renal function affect elimination
k_elim_adj = k_elim * (1 - 0.002 * (weight_kg - 72)) * (crcl / 100) ** 0.3
plasma_conc = (plasma_c0 * np.exp(-k_elim_adj * sampling_time_min)).round(2)
plasma_conc = plasma_conc.clip(0.5, 60)

# CSF: slow penetration with lag, modeled as rising curve approaching plateau
# Based on paper: CSF rises from 0.11 (t=10) to 0.48 (t=176)
# CSF(t) = CSF_max * (1 - exp(-k_pen * t))
csf_max = np.random.normal(0.55, 0.1, N).clip(0.3, 0.9)
k_pen = np.random.normal(0.012, 0.003, N).clip(0.004, 0.025)
# BBB penetration influenced by weight (inversely) and age
k_pen_adj = k_pen * (72 / weight_kg) ** 0.5 * (1 + 0.003 * (age - 52))
csf_conc = (csf_max * (1 - np.exp(-k_pen_adj * sampling_time_min))).round(3)
csf_conc = csf_conc.clip(0.02, 1.2)

# Add realistic noise
csf_conc += np.random.normal(0, 0.02, N)
csf_conc = csf_conc.clip(0.02, 1.2).round(3)
plasma_conc += np.random.normal(0, 1.0, N)
plasma_conc = plasma_conc.clip(0.5, 60).round(2)

# --- Derived Variables ---
csf_plasma_ratio = (csf_conc / plasma_conc).round(4)
# Therapeutic threshold: CSF >= 0.25 ug/ml (MIC cutoff from paper)
therapeutic_threshold_met = (csf_conc >= 0.25).astype(int)

# --- Assemble DataFrame ---
df = pd.DataFrame({
    "patient_id": range(1, N + 1),
    "age_years": age,
    "sex": sex,
    "weight_kg": weight_kg,
    "height_cm": height_cm,
    "bmi": bmi,
    "creatinine_clearance_ml_min": crcl,
    "infusion_duration_min": infusion_duration_min,
    "sampling_time_min": sampling_time_min,
    "plasma_concentration_ug_ml": plasma_conc,
    "csf_concentration_ug_ml": csf_conc,
    "csf_plasma_ratio": csf_plasma_ratio,
    "therapeutic_threshold_met": therapeutic_threshold_met,
})

# Save to CSV
output_path = "doripenem_bbb_synthetic_data.csv"
df.to_csv(output_path, index=False)

print(f"Generated synthetic dataset: {output_path}")
print(f"  N = {len(df)} patients")
print(f"  Therapeutic threshold met: {therapeutic_threshold_met.sum()}/{N} "
      f"({100*therapeutic_threshold_met.mean():.1f}%)")
print(f"\nSummary statistics:")
print(df.describe().round(3).to_string())
print(f"\nFirst 10 rows:")
print(df.head(10).to_string(index=False))
