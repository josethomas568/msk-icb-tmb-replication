"""
01_build_dataset.py
Build the analysis dataset for the replication of Samstein et al. (Nat Genet 2019)
from the public cBioPortal study `tmb_mskcc_2018`.

Inputs : data/raw/data_clinical_patient.txt, data/raw/data_clinical_sample.txt
Outputs: data/analysis.csv, data/data_dictionary.csv, output/qc_report.txt
Run    : python 01_build_dataset.py   (exits non-zero if any hard QC check fails)
"""
from pathlib import Path
import sys
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
RAW = ROOT / "data" / "raw"
OUT = ROOT / "output"
OUT.mkdir(exist_ok=True)

pat = pd.read_csv(RAW / "data_clinical_patient.txt", sep="\t", comment="#")
smp = pd.read_csv(RAW / "data_clinical_sample.txt", sep="\t", comment="#")

qc_lines, failures = [], []


def check(name, ok, detail="", hard=True):
    status = "PASS" if ok else ("FAIL" if hard else "WARN")
    qc_lines.append(f"[{status}] {name}" + (f" -- {detail}" if detail else ""))
    if not ok and hard:
        failures.append(name)


# ---------- raw-file checks ----------
check("patient file: one row per patient", pat.PATIENT_ID.is_unique, f"n={len(pat)}")
check("sample file: one sample per patient", smp.PATIENT_ID.is_unique, f"n={len(smp)}")
check("sample IDs unique", smp.SAMPLE_ID.is_unique)
check("every sample maps to a patient", smp.PATIENT_ID.isin(pat.PATIENT_ID).all())
check("every patient has a sample", pat.PATIENT_ID.isin(smp.PATIENT_ID).all())
check("all samples are tumor", (smp.SAMPLE_CLASS == "Tumor").all())
check("all samples from MSKCC", (smp.INSTITUTE == "MSKCC").all())

df = pat.merge(smp, on="PATIENT_ID", how="inner", validate="one_to_one")

# ---------- derive analysis variables ----------
AGE_LEVELS = ["<30", "31-50", "50-60", "61-70", ">71"]
DRUG_LEVELS = ["PD-1/PDL-1", "CTLA4", "Combo"]
PANEL_LEVELS = ["IMPACT341", "IMPACT410", "IMPACT468"]

a = pd.DataFrame({
    "patient_id": df.PATIENT_ID,
    "sample_id": df.SAMPLE_ID,
    "cancer_type": df.CANCER_TYPE,
    "cancer_type_detailed": df.CANCER_TYPE_DETAILED,
    "oncotree_code": df.ONCOTREE_CODE,
    "sex": df.SEX,
    "age_group": pd.Categorical(df.AGE_GROUP, categories=AGE_LEVELS, ordered=True),
    "age_years": pd.to_numeric(df.AGE_AT_SEQ_REPORT, errors="coerce"),
    "drug_class": pd.Categorical(df.DRUG_TYPE, categories=DRUG_LEVELS),
    "os_months": pd.to_numeric(df.OS_MONTHS, errors="coerce"),
    "os_event": df.OS_STATUS.str.startswith("1:").astype(int),
    "tmb": pd.to_numeric(df.TMB_NONSYNONYMOUS, errors="coerce"),
    "panel": pd.Categorical(df.GENE_PANEL, categories=PANEL_LEVELS, ordered=True),
    "sample_type": df.SAMPLE_TYPE,
    "matched_normal": (df.SOMATIC_STATUS == "Matched").astype(int),
    "tumor_purity": pd.to_numeric(df.TUMOR_PURITY, errors="coerce"),
    "coverage": pd.to_numeric(df.SAMPLE_COVERAGE, errors="coerce"),
})

# Histologies with <10 patients cannot support a within-histology top-20% cut.
counts = a.cancer_type.value_counts()
a["in_histology_analysis"] = a.cancer_type.map(counts >= 10).astype(int)

a["log2_tmb1"] = np.log2(a.tmb + 1)

# TMB-high (primary): top 20% within histology, i.e. TMB >= that histology's 80th percentile.
# Percentile uses the 'linear' (type-7) definition; ties at the cut are counted high.
q80 = a[a.in_histology_analysis == 1].groupby("cancer_type").tmb.quantile(0.80)
a["tmb_cut80"] = a.cancer_type.map(q80)
a["tmb_high20"] = np.where(a.in_histology_analysis == 1, (a.tmb >= a.tmb_cut80).astype(float), np.nan)

# Sensitivity cuts
q90 = a[a.in_histology_analysis == 1].groupby("cancer_type").tmb.quantile(0.90)
a["tmb_high10pct"] = np.where(a.in_histology_analysis == 1, (a.tmb >= a.cancer_type.map(q90)).astype(float), np.nan)
a["tmb_ge10"] = (a.tmb >= 10).astype(int)

# ---------- analysis-dataset QC ----------
check("OS months non-missing", a.os_months.notna().all())
check("OS months >= 0", (a.os_months >= 0).all(), f"range {a.os_months.min()}-{a.os_months.max()}")
check("OS months <= 80 (study max follow-up)", (a.os_months <= 80).all())
check("OS status coded for all", df.OS_STATUS.isin(["0:LIVING", "1:DECEASED"]).all())
check("TMB non-missing", a.tmb.notna().all())
check("TMB >= 0", (a.tmb >= 0).all(), f"range {a.tmb.min():.2f}-{a.tmb.max():.2f}")
check("TMB plausible (< 300 mut/Mb)", (a.tmb < 300).all())
check("age group levels valid", a.age_group.notna().all())
check("drug class levels valid", a.drug_class.notna().all())
check("panel levels valid", a.panel.notna().all())
check("sex coded", a.sex.isin(["Male", "Female"]).all())

# age_years is labelled 'Days' in cBioPortal metadata but values are years:
# confirm it agrees with age_group for nearly everyone.
bins = pd.cut(a.age_years, [0, 30, 50, 60, 70, 200], labels=AGE_LEVELS)
agree = (bins.astype(str) == a.age_group.astype(str))[a.age_years.notna()].mean()
check("age_years is in years (agrees with age group)", agree > 0.9, f"agreement {agree:.1%}")
check("age_years missingness", a.age_years.isna().sum() <= 5,
      f"{a.age_years.isna().sum()} missing", hard=False)

# Top-20% definition: within-histology high fraction should be ~20%
frac = a[a.in_histology_analysis == 1].groupby("cancer_type").tmb_high20.mean()
check("TMB-high fraction 18-26% in every histology", frac.between(0.18, 0.26).all(),
      "; ".join(f"{k}: {v:.1%}" for k, v in frac.items()))
check("histologies excluded from per-histology analysis",
      True, ", ".join(f"{k} (n={v})" for k, v in counts[counts < 10].items()), hard=False)
check("zero-month OS times present (handled as-is)", True,
      f"{(a.os_months == 0).sum()} patients", hard=False)
check("unmatched-normal samples (TMB may be inflated)", True,
      f"{(a.matched_normal == 0).sum()} patients", hard=False)
check("deaths / censored", True, f"{a.os_event.sum()} deaths, {(1 - a.os_event).sum()} censored", hard=False)

a.to_csv(ROOT / "data" / "analysis.csv", index=False)

# ---------- data dictionary ----------
dd = [
    ("patient_id", "string", "MSK patient identifier", "PATIENT_ID (patient file)", ""),
    ("sample_id", "string", "MSK-IMPACT tumor sample identifier", "SAMPLE_ID", "One sample per patient"),
    ("cancer_type", "category", "Histology (OncoTree cancer type)", "CANCER_TYPE", "11 levels; 10 with n>=10"),
    ("cancer_type_detailed", "category", "Detailed histology", "CANCER_TYPE_DETAILED", ""),
    ("oncotree_code", "category", "OncoTree code", "ONCOTREE_CODE", ""),
    ("sex", "category", "Sex", "SEX", "Male / Female"),
    ("age_group", "ordered category", "Age group at diagnosis (years)", "AGE_GROUP",
     "<30, 31-50, 50-60, 61-70, >71 (source labels; boundaries overlap at 50 as published)"),
    ("age_years", "numeric (years)", "Age when MSK-IMPACT report issued", "AGE_AT_SEQ_REPORT",
     "Metadata says days; values are years (QC-verified against age_group)"),
    ("drug_class", "category", "Checkpoint inhibitor class", "DRUG_TYPE",
     "PD-1/PDL-1 (reference), CTLA4, Combo (anti-CTLA4 + anti-PD-1/PD-L1)"),
    ("os_months", "numeric (months)", "Overall survival from first ICI dose to death or last follow-up", "OS_MONTHS",
     "Integer months in public release"),
    ("os_event", "0/1", "Death indicator", "OS_STATUS", "1 = 1:DECEASED, 0 = 0:LIVING (censored)"),
    ("tmb", "numeric (mut/Mb)", "Nonsynonymous somatic mutations per Mb of panel sequenced", "TMB_NONSYNONYMOUS", ""),
    ("log2_tmb1", "numeric", "log2(TMB + 1)", "derived", "Used for continuous / spline models"),
    ("panel", "ordered category", "MSK-IMPACT panel version", "GENE_PANEL",
     "IMPACT341 < 410 < 468; used as proxy for treatment era (year not in public data)"),
    ("sample_type", "category", "Primary or metastasis", "SAMPLE_TYPE", ""),
    ("matched_normal", "0/1", "Matched normal used for somatic calling", "SOMATIC_STATUS", ""),
    ("tumor_purity", "numeric (%)", "Pathologist-estimated tumor purity", "TUMOR_PURITY", "Some missing"),
    ("coverage", "numeric", "Median sequencing coverage", "SAMPLE_COVERAGE", ""),
    ("in_histology_analysis", "0/1", "Histology has >=10 patients", "derived",
     "Excludes Skin Cancer, Non-Melanoma (n=1)"),
    ("tmb_cut80", "numeric (mut/Mb)", "80th percentile of TMB within the patient's histology", "derived", ""),
    ("tmb_high20", "0/1", "PRIMARY exposure: TMB >= histology-specific 80th percentile", "derived",
     "Missing if in_histology_analysis = 0"),
    ("tmb_high10pct", "0/1", "Sensitivity: TMB >= histology-specific 90th percentile", "derived", ""),
    ("tmb_ge10", "0/1", "Sensitivity: TMB >= 10 mut/Mb (fixed, pan-cancer)", "derived", ""),
]
pd.DataFrame(dd, columns=["variable", "type", "definition", "source", "notes"]).to_csv(
    ROOT / "data" / "data_dictionary.csv", index=False)

cut_tbl = pd.DataFrame({"n": counts[counts >= 10], "tmb_80th_pct": q80.round(2), "tmb_90th_pct": q90.round(2)})
cut_tbl.to_csv(OUT / "tmb_cutoffs_by_histology.csv")

summary = [
    f"Analysis dataset: {len(a)} patients ({a.in_histology_analysis.sum()} in per-histology analyses)",
    f"Hard checks failed: {len(failures)}", "",
] + qc_lines + ["", "Histology-specific TMB cut-offs (mut/Mb):", cut_tbl.to_string()]
(OUT / "qc_report.txt").write_text("\n".join(summary) + "\n")
print("\n".join(summary))
sys.exit(1 if failures else 0)
