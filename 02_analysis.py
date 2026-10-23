"""
02_analysis.py
All analyses for the Samstein et al. (2019) replication.
Run after 01_build_dataset.py.  Writes tables to output/ and figures to figures/,
plus output/results.json with every number quoted in the report.
"""
from pathlib import Path
import json
import warnings
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.formula.api as smf
from lifelines import KaplanMeierFitter, CoxPHFitter
from lifelines.statistics import logrank_test, proportional_hazard_test
from patsy import dmatrix, build_design_matrices
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent
OUT, FIG = ROOT / "output", ROOT / "figures"
OUT.mkdir(exist_ok=True); FIG.mkdir(exist_ok=True)
RNG = np.random.default_rng(20190114)

# Chart palette (validated reference categorical slots 1-2; text stays in ink tokens)
C_HIGH, C_LOW = "#2a78d6", "#eb6834"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({"font.size": 8, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
                     "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
                     "font.family": "DejaVu Sans", "savefig.dpi": 200})

R = {}  # results for the report
full = pd.read_csv(ROOT / "data" / "analysis.csv")
d = full[full.in_histology_analysis == 1].copy()
AGE = ["<30", "31-50", "50-60", "61-70", ">71"]
for c, lv in [("age_group", AGE), ("drug_class", ["PD-1/PDL-1", "CTLA4", "Combo"]),
              ("panel", ["IMPACT341", "IMPACT410", "IMPACT468"])]:
    d[c] = pd.Categorical(d[c], categories=lv)
HIST = d.cancer_type.value_counts().index.tolist()  # ordered by n
SHORT = {"Non-Small Cell Lung Cancer": "NSCLC", "Melanoma": "Melanoma", "Bladder Cancer": "Bladder",
         "Renal Cell Carcinoma": "Renal cell", "Head and Neck Cancer": "Head & neck",
         "Esophagogastric Cancer": "Esophagogastric", "Glioma": "Glioma", "Colorectal Cancer": "Colorectal",
         "Cancer of Unknown Primary": "Unknown primary", "Breast Cancer": "Breast"}
d["hist"] = d.cancer_type.map(SHORT)
R["n_analysis"] = len(d); R["n_full"] = len(full)
R["deaths"] = int(d.os_event.sum())


def fmt_p(p):
    return "<0.001" if p < 0.001 else f"{p:.3f}"


def med_iqr(x, dp=1):
    q = np.nanpercentile(x, [50, 25, 75])
    return f"{q[0]:.{dp}f} ({q[1]:.{dp}f}–{q[2]:.{dp}f})"


def km_median(t, e):
    k = KaplanMeierFitter().fit(t, e)
    m = k.median_survival_time_
    return "NR" if np.isinf(m) else f"{m:.0f}"


def followup(t, e):  # reverse Kaplan-Meier median follow-up
    k = KaplanMeierFitter().fit(t, 1 - e)
    return k.median_survival_time_


# ------------------------------------------------------------------ Table 1
def col(g):
    n = len(g)
    pct = lambda m: f"{int(m.sum())} ({100 * m.mean():.0f}%)"
    row = {"N": str(n),
           "Age, years, median (IQR)": med_iqr(g.age_years, 0),
           "Age >70 (group >71), n (%)": pct(g.age_group == ">71"),
           "Female, n (%)": pct(g.sex == "Female"),
           "Anti-PD-1/PD-L1 alone, n (%)": pct(g.drug_class == "PD-1/PDL-1"),
           "Anti-CTLA-4 alone, n (%)": pct(g.drug_class == "CTLA4"),
           "Combination, n (%)": pct(g.drug_class == "Combo"),
           "Metastatic sample, n (%)": pct(g.sample_type == "Metastasis"),
           "IMPACT341 / 410 / 468, %": " / ".join(f"{100 * (g.panel == p).mean():.0f}" for p in
                                                  ["IMPACT341", "IMPACT410", "IMPACT468"]),
           "TMB, mut/Mb, median (IQR)": med_iqr(g.tmb),
           "TMB-high cut-off (80th pct), mut/Mb": f"{g.tmb.quantile(.8):.1f}",
           "TMB ≥10 mut/Mb, n (%)": pct(g.tmb >= 10),
           "Deaths, n (%)": pct(g.os_event == 1),
           "Median OS, months (KM)": km_median(g.os_months, g.os_event),
           "Median follow-up, months (reverse KM)": f"{followup(g.os_months, g.os_event):.0f}"}
    return row


t1 = pd.DataFrame({SHORT[h]: col(d[d.cancer_type == h]) for h in HIST})
t1["All"] = pd.Series(col(d))
t1.loc["TMB-high cut-off (80th pct), mut/Mb", "All"] = "histology-specific"
t1.to_csv(OUT / "table1_by_cancer_type.csv")
R["median_followup"] = float(followup(d.os_months, d.os_event))
R["median_os_all"] = km_median(d.os_months, d.os_event)
R["table1"] = t1.to_dict()

# ------------------------------------------------------------------ Categorical
cat = {}
# (a) Top-20% definition: prevalence is fixed at ~20% by construction -> test is a design check
ct20 = pd.crosstab(d.cancer_type, d.tmb_high20)
chi20 = stats.chi2_contingency(ct20)
cat["top20_chisq"] = {"chi2": chi20[0], "df": chi20[2], "p": chi20[1]}
# (b) Substantive question: fixed 10 mut/Mb cut
ct10 = pd.crosstab(d.cancer_type, d.tmb_ge10)
chi10 = stats.chi2_contingency(ct10)
cat["ge10_chisq"] = {"chi2": chi10[0], "df": chi10[2], "p": chi10[1],
                     "min_expected": float(chi10[3].min()),
                     "cells_expected_lt5": int((chi10[3] < 5).sum())}
# Fisher-Freeman-Halton exact test for r x 2 via Monte Carlo (B=20000)
obs_stat = chi10[0]
lab = pd.factorize(d.cancer_type)[0]; y = d.tmb_ge10.values
k = lab.max() + 1; nrow = np.bincount(lab, minlength=k); p1 = y.mean()
E1, E0 = nrow * p1, nrow * (1 - p1)
B = 20000
perm = np.array([RNG.permutation(y) for _ in range(B)])          # B x n
O1 = np.stack([np.bincount(lab, weights=row, minlength=k) for row in perm])
chis = ((O1 - E1) ** 2 / E1 + ((nrow - O1) - E0) ** 2 / E0).sum(1)
ge = int((chis >= obs_stat - 1e-9).sum())
cat["ge10_mc_exact_p"] = (ge + 1) / (B + 1)
prev = d.groupby(d["hist"]).agg(n=("tmb_ge10", "size"), n_high=("tmb_ge10", "sum"))
prev["pct_ge10"] = 100 * prev.n_high / prev.n
prev["pct_top20"] = 100 * d.groupby(d["hist"]).tmb_high20.mean()
prev.sort_values("pct_ge10", ascending=False).round(1).to_csv(OUT / "tmb_prevalence_by_cancer_type.csv")
cat["prevalence"] = prev.round(1).to_dict(orient="index")

# Logistic model: P(TMB >= 10) ~ histology + age + sex + sample type + panel + matched normal
d["cancer_type_ref"] = pd.Categorical(d["hist"], categories=["Renal cell"] + [SHORT[h] for h in HIST if SHORT[h] != "Renal cell"])
lg = smf.logit("tmb_ge10 ~ C(cancer_type_ref) + C(age_group, Treatment('61-70')) + C(sex) + C(sample_type)"
               " + C(panel) + matched_normal", data=d).fit(disp=0)
lgt = pd.DataFrame({"OR": np.exp(lg.params), "lo": np.exp(lg.conf_int()[0]), "hi": np.exp(lg.conf_int()[1]),
                    "p": lg.pvalues}).round(4)
lgt.to_csv(OUT / "logistic_tmb_ge10.csv")
lg0 = smf.logit("tmb_ge10 ~ C(age_group, Treatment('61-70')) + C(sex) + C(sample_type) + C(panel) + matched_normal",
                data=d).fit(disp=0)
cat["logistic_lrt_histology"] = {"chi2": 2 * (lg.llf - lg0.llf), "df": int(lg.df_model - lg0.df_model),
                                 "p": stats.chi2.sf(2 * (lg.llf - lg0.llf), lg.df_model - lg0.df_model)}
cat["logistic_n"] = int(lg.nobs)
cat["logistic_pseudo_r2"] = lg.prsquared
cat["logistic_or"] = {k: [round(v, 2) for v in r[["OR", "lo", "hi"]]] + [r.p] for k, r in lgt.iterrows()}
R["categorical"] = cat

# ------------------------------------------------------------------ Survival helpers
def design(df, extra=()):
    X = pd.get_dummies(df[["age_group", "drug_class", "panel"]], drop_first=True, dtype=float)
    X = X.drop(columns=[c for c in X if c == "age_group_61-70"], errors="ignore")
    out = pd.concat([df[["os_months", "os_event", "cancer_type"] + list(extra)].reset_index(drop=True),
                     X.reset_index(drop=True)], axis=1)
    return out


def adjusted_age_dummies(df):
    # reference age group = 61-70 (largest)
    X = pd.get_dummies(pd.Categorical(df.age_group, categories=["61-70", "<30", "31-50", "50-60", ">71"]),
                       prefix="age", drop_first=True, dtype=float)
    return X


def cox_df(df, expo_cols):
    X = pd.concat([df[["os_months", "os_event", "cancer_type"] + expo_cols].reset_index(drop=True),
                   adjusted_age_dummies(df).reset_index(drop=True),
                   pd.get_dummies(df.drug_class, prefix="drug", drop_first=True, dtype=float).reset_index(drop=True),
                   pd.get_dummies(df.panel, prefix="panel", drop_first=True, dtype=float).reset_index(drop=True)],
                  axis=1)
    return X


def fit_strat(X, penal=0.0):
    return CoxPHFitter(penalizer=penal).fit(X, "os_months", "os_event", strata=["cancer_type"])


def hr_row(cph, var):
    s = cph.summary.loc[var]
    return {"HR": float(s["exp(coef)"]), "lo": float(s["exp(coef) lower 95%"]),
            "hi": float(s["exp(coef) upper 95%"]), "p": float(s["p"])}


# ------------------------------------------------------------------ KM + log-rank
surv = {}
lr_pool = logrank_test(d.os_months[d.tmb_high20 == 1], d.os_months[d.tmb_high20 == 0],
                       d.os_event[d.tmb_high20 == 1], d.os_event[d.tmb_high20 == 0])
surv["logrank_pooled_unstratified"] = {"chi2": lr_pool.test_statistic, "p": lr_pool.p_value}
# stratified log-rank = score test of stratified Cox with tmb only
cs = fit_strat(d[["os_months", "os_event", "cancer_type", "tmb_high20"]])
surv["logrank_stratified"] = {"chi2": float(cs.log_likelihood_ratio_test().test_statistic),
                              "p": float(cs.log_likelihood_ratio_test().p_value)}
surv["unadjusted_stratified_hr"] = hr_row(cs, "tmb_high20")

per = []
for h in HIST:
    g = d[d.cancer_type == h]
    hi, lo = g[g.tmb_high20 == 1], g[g.tmb_high20 == 0]
    lr = logrank_test(hi.os_months, lo.os_months, hi.os_event, lo.os_event)
    per.append({"hist": SHORT[h], "n": len(g), "n_high": len(hi), "events": int(g.os_event.sum()),
                "logrank_p": lr.p_value, "median_os_high": km_median(hi.os_months, hi.os_event),
                "median_os_low": km_median(lo.os_months, lo.os_event)})
per = pd.DataFrame(per).set_index("hist")


def km_panel(ax, g, title, show_risk=True):
    kms = []
    for val, colr, name in [(1, C_HIGH, "TMB-high"), (0, C_LOW, "TMB-low")]:
        s = g[g.tmb_high20 == val]
        k = KaplanMeierFitter(label=f"{name} (n={len(s)})").fit(s.os_months, s.os_event)
        k.plot_survival_function(ax=ax, ci_show=True, color=colr, lw=1.6, ci_alpha=0.12)
        kms.append(k)
    ax.set_title(title, color=INK, fontsize=8.5, loc="left")
    ax.set_ylim(0, 1.02); ax.set_xlim(0, 60); ax.set_xticks([0, 12, 24, 36, 48, 60])
    ax.grid(axis="y", color=GRID, lw=0.6); ax.set_xlabel(""); ax.legend(frameon=False, fontsize=7)
    return kms


# Figure 1: pooled KM with risk table
fig, ax = plt.subplots(figsize=(4.6, 3.4))
kms = km_panel(ax, d, "")
ax.set_xlabel("Months from first ICI dose"); ax.set_ylabel("Overall survival")
from lifelines.plotting import add_at_risk_counts
add_at_risk_counts(*kms, ax=ax, rows_to_show=["At risk"], xticks=[0, 12, 24, 36, 48, 60])
hr = None  # filled in caption after Cox
fig.tight_layout(); fig.savefig(FIG / "fig1_km_pooled.png"); plt.close(fig)

# Supplementary: KM by histology
fig, axes = plt.subplots(2, 5, figsize=(11, 5.0), sharey=True)
for ax, h in zip(axes.flat, HIST):
    g = d[d.cancer_type == h]
    km_panel(ax, g, f"{SHORT[h]}\nlog-rank p={fmt_p(per.loc[SHORT[h], 'logrank_p'])}")
    ax.legend(frameon=False, fontsize=6, loc="upper right")
for ax in axes[1]: ax.set_xlabel("Months")
for ax in axes[:, 0]: ax.set_ylabel("Overall survival")
fig.tight_layout(); fig.savefig(FIG / "figS1_km_by_histology.png"); plt.close(fig)

# ------------------------------------------------------------------ Cox models
X = cox_df(d, ["tmb_high20"])
cox_main = fit_strat(X)
cox_main.summary.to_csv(OUT / "cox_primary_stratified.csv")
surv["cox_primary"] = hr_row(cox_main, "tmb_high20")
surv["cox_primary_terms"] = {k: hr_row(cox_main, k) for k in cox_main.summary.index}
surv["cox_primary_concordance"] = cox_main.concordance_index_

# Alternative: cancer type as covariate (unstratified) — closer to the paper's description
Xa = pd.concat([X.drop(columns="cancer_type"),
                pd.get_dummies(d.cancer_type, prefix="ct", drop_first=True, dtype=float).reset_index(drop=True)], axis=1)
cox_cov = CoxPHFitter().fit(Xa, "os_months", "os_event")
surv["cox_cancer_type_as_covariate"] = hr_row(cox_cov, "tmb_high20")

# Interaction: TMB-high x cancer type (stratified on cancer type)
Xi = X.copy()
for h in HIST[1:]:
    Xi[f"int_{SHORT[h]}"] = Xi.tmb_high20 * (d.cancer_type.values == h)
cox_int = fit_strat(Xi)
lrt = 2 * (cox_int.log_likelihood_ - cox_main.log_likelihood_)
surv["interaction_lrt"] = {"chi2": lrt, "df": len(HIST) - 1, "p": stats.chi2.sf(lrt, len(HIST) - 1)}
# Interaction test excluding glioma
d_ng = d[d.cancer_type != "Glioma"]
Xng = cox_df(d_ng, ["tmb_high20"]); m0 = fit_strat(Xng)
Xng_i = Xng.copy(); hist_ng = [h for h in HIST if h != "Glioma"]
for h in hist_ng[1:]:
    Xng_i[f"int_{SHORT[h]}"] = Xng_i.tmb_high20 * (d_ng.cancer_type.values == h)
m1 = fit_strat(Xng_i); lrt2 = 2 * (m1.log_likelihood_ - m0.log_likelihood_)
surv["interaction_lrt_excl_glioma"] = {"chi2": lrt2, "df": len(hist_ng) - 1, "p": stats.chi2.sf(lrt2, len(hist_ng) - 1)}
surv["cox_primary_excl_glioma"] = hr_row(m0, "tmb_high20")

# Per-histology adjusted Cox (forest plot). Within a histology the covariate set is reduced
# to keep events-per-parameter reasonable: age (years, 1 missing imputed to group median),
# any anti-CTLA-4 exposure, panel as ordinal 0/1/2.
fr = []
for h in HIST:
    g = d[d.cancer_type == h].copy()
    g["age_c"] = g.age_years.fillna(g.groupby("age_group").age_years.transform("median")) / 10
    g["ctla4_any"] = g.drug_class.isin(["CTLA4", "Combo"]).astype(float)
    g["panel_ord"] = g.panel.cat.codes.astype(float)
    nct = g.ctla4_any.sum()
    cols = ["tmb_high20", "age_c", "panel_ord"] + (["ctla4_any"] if 5 <= nct <= len(g) - 5 else [])
    m = CoxPHFitter().fit(g[["os_months", "os_event"] + cols], "os_months", "os_event")
    r = hr_row(m, "tmb_high20"); r.update(hist=SHORT[h], n=len(g), events=int(g.os_event.sum()),
                                          n_high=int(g.tmb_high20.sum()), adj=",".join(cols[1:]))
    try:
        r["ph_p"] = float(proportional_hazard_test(m, g[["os_months", "os_event"] + cols],
                                                   time_transform="rank").summary.loc["tmb_high20", "p"])
    except Exception:
        r["ph_p"] = np.nan
    fr.append(r)
fr = pd.DataFrame(fr).set_index("hist")
fr = fr.join(per[["logrank_p", "median_os_high", "median_os_low"]])
fr.round(4).to_csv(OUT / "cox_by_histology.csv")

surv["by_histology"] = fr.round(4).to_dict(orient="index")

# Forest plot
fig, ax = plt.subplots(figsize=(6.4, 3.9))
rows = fr.sort_values("HR", ascending=True)
labels = list(rows.index) + ["All (stratified, adjusted)"]
ys = np.arange(len(labels))[::-1]
for y, (name, r) in zip(ys[:-1], rows.iterrows()):
    ax.plot([r.lo, r.hi], [y, y], color=INK2, lw=1.2, solid_capstyle="round")
    ax.plot(r.HR, y, "s", color=C_HIGH, ms=3 + 5 * np.sqrt(r.events / fr.events.max()), mec="white", mew=0.8)
pr = surv["cox_primary"]
yd = ys[-1]
ax.fill([pr["lo"], pr["HR"], pr["hi"], pr["HR"]], [yd, yd + 0.28, yd, yd - 0.28], color=INK, lw=0)
ax.axvline(1, color=INK2, lw=0.8, ls="--")
ax.set_xscale("log"); ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator()); ax.set_xticks([0.25, 0.5, 1, 2, 4]); ax.set_xticklabels(["0.25", "0.5", "1", "2", "4"])
ax.set_xlim(0.08, 6)
ax.set_yticks(ys); ax.set_yticklabels([f"{l}" for l in labels], color=INK)
for y, (name, r) in zip(ys[:-1], rows.iterrows()):
    ax.text(6.6, y, f"{r.HR:.2f} ({r.lo:.2f}–{r.hi:.2f})  n={r.n:.0f}", va="center", fontsize=7, color=INK)
ax.text(6.6, yd, f"{pr['HR']:.2f} ({pr['lo']:.2f}–{pr['hi']:.2f})  n={len(d)}", va="center", fontsize=7,
        color=INK, fontweight="bold")
ax.set_xlabel("Hazard ratio for death, TMB-high vs TMB-low (log scale)")
ax.text(0.9, ys[0] + 0.75, "← favours TMB-high", fontsize=7, color=INK2, ha="right")
ax.text(1.11, ys[0] + 0.75, "favours TMB-low →", fontsize=7, color=INK2, ha="left")
ax.set_ylim(-0.8, len(labels) - 0.1)
ax.grid(axis="x", color=GRID, lw=0.5)
fig.tight_layout(rect=(0, 0, 0.72, 1)); fig.savefig(FIG / "fig2_forest_by_histology.png", bbox_inches="tight"); plt.close(fig)

# ------------------------------------------------------------------ Proportional hazards
ph = proportional_hazard_test(cox_main, X, time_transform="rank")
ph.summary.to_csv(OUT / "schoenfeld_tests_primary.csv")
ph_km = proportional_hazard_test(cox_main, X, time_transform="km")
surv["ph_tests"] = {k: {"rank_p": float(ph.summary.loc[k, "p"]), "km_p": float(ph_km.summary.loc[k, "p"])}
                    for k in ph.summary.index}
# Global test (sum of chi-squares is not exact; use the lifelines per-term + report min p)
sr = cox_main.compute_residuals(X, "scaled_schoenfeld")
tvals = X.loc[sr.index, "os_months"]
fig, ax = plt.subplots(figsize=(4.4, 2.9))
beta = cox_main.params_["tmb_high20"]
yv = sr["tmb_high20"]
ax.scatter(tvals, yv, s=4, color=INK2, alpha=0.35, lw=0)
import statsmodels.api as sm
lw = sm.nonparametric.lowess(yv, tvals, frac=0.5)
ax.plot(lw[:, 0], lw[:, 1], color=C_HIGH, lw=2)
ax.set_ylim(-4, 4); ax.set_xlabel("Months"); ax.set_ylabel("Scaled Schoenfeld residual\n(TMB-high term)")
ax.text(0.98, 0.95, f"rank-time test p = {fmt_p(surv['ph_tests']['tmb_high20']['rank_p'])}; flat LOWESS = PH holds", transform=ax.transAxes,
        ha="right", va="top", fontsize=7, color=INK)
ax.grid(axis="y", color=GRID, lw=0.5)
fig.tight_layout(); fig.savefig(FIG / "figS2_schoenfeld_tmb.png"); plt.close(fig)

# Time-split model: HR in 0-12 months vs >12 months (episode split)
def split_at(df, cut):
    a = df.copy(); a["start"] = 0.0; a["stop"] = np.minimum(a.os_months, cut)
    a["ev"] = np.where(a.os_months <= cut, a.os_event, 0); a["late"] = 0
    b = df[df.os_months > cut].copy(); b["start"] = cut; b["stop"] = b.os_months; b["ev"] = b.os_event; b["late"] = 1
    return pd.concat([a, b], ignore_index=True)
from lifelines import CoxTimeVaryingFitter
Xs = X.copy(); Xs["id"] = np.arange(len(Xs))
sp = split_at(Xs, 12)
sp = sp[sp.stop > sp.start]
sp["tmb_early"] = sp.tmb_high20 * (1 - sp.late); sp["tmb_late"] = sp.tmb_high20 * sp.late
covs = [c for c in X.columns if c not in ("os_months", "os_event", "cancer_type", "tmb_high20")]
ctv = CoxTimeVaryingFitter().fit(sp[["id", "start", "stop", "ev", "cancer_type", "tmb_early", "tmb_late"] + covs],
                                 id_col="id", start_col="start", stop_col="stop", event_col="ev",
                                 strata=["cancer_type"])
surv["time_split"] = {k: {"HR": float(np.exp(ctv.params_[k])),
                          "lo": float(np.exp(ctv.confidence_intervals_.loc[k].iloc[0])),
                          "hi": float(np.exp(ctv.confidence_intervals_.loc[k].iloc[1])),
                          "p": float(ctv.summary.loc[k, "p"])} for k in ["tmb_early", "tmb_late"]}

# ------------------------------------------------------------------ Sensitivity analyses
sens = {}
def strat_hr(df, var, extra_cols=None):
    Xv = cox_df(df, [var] + (extra_cols or []))
    return hr_row(fit_strat(Xv), var)
sens["Top 20% within histology (primary)"] = surv["cox_primary"]
sens["Top 10% within histology"] = strat_hr(d, "tmb_high10pct")
sens["TMB ≥10 mut/Mb (fixed)"] = strat_hr(d, "tmb_ge10")
d["log2_tmb1_c"] = d.log2_tmb1
sens["Continuous, per doubling of (TMB+1)"] = strat_hr(d, "log2_tmb1_c")
# within-histology percentile rank (per 10 percentile points)
d["tmb_pctl10"] = d.groupby("cancer_type").tmb.rank(pct=True) * 10
sens["Within-histology percentile, per 10 points"] = strat_hr(d, "tmb_pctl10")
sens["Top 20%, excluding glioma"] = surv["cox_primary_excl_glioma"]
sens["Top 20%, excluding unmatched-normal (n=17)"] = strat_hr(d[d.matched_normal == 1], "tmb_high20")
# add sex + sample type + continuous age instead of age group
dd_ = d.copy(); dd_["age_c"] = dd_.age_years.fillna(dd_.groupby("age_group").age_years.transform("median")) / 10
Xx = pd.concat([dd_[["os_months", "os_event", "cancer_type", "tmb_high20", "age_c"]].reset_index(drop=True),
                (dd_.sex == "Female").astype(float).rename("female").reset_index(drop=True),
                (dd_.sample_type == "Metastasis").astype(float).rename("met_sample").reset_index(drop=True),
                pd.get_dummies(dd_.drug_class, prefix="drug", drop_first=True, dtype=float).reset_index(drop=True),
                pd.get_dummies(dd_.panel, prefix="panel", drop_first=True, dtype=float).reset_index(drop=True)], axis=1)
sens["Top 20%, + sex, sample type, continuous age"] = hr_row(fit_strat(Xx), "tmb_high20")
# TMB >= 10 within histologies where cut differs most
for h in ["Melanoma", "Non-Small Cell Lung Cancer", "Bladder Cancer"]:
    g = d[d.cancer_type == h].copy()
    g["age_c"] = g.age_years.fillna(g.age_years.median()) / 10
    m = CoxPHFitter().fit(g[["os_months", "os_event", "tmb_ge10", "age_c"]], "os_months", "os_event")
    sens[f"TMB ≥10, {SHORT[h]} only"] = hr_row(m, "tmb_ge10")

pd.DataFrame(sens).T.round(3).assign(p=lambda t: [f"{v:.2e}" for v in pd.DataFrame(sens).T.p]).to_csv(OUT / "sensitivity_analyses.csv")
R["sensitivity"] = sens

# Restricted cubic (natural) spline in log2(TMB+1), 4 df, stratified by histology
x = d.log2_tmb1.values
basis = dmatrix("cr(x, df=4) - 1", {"x": x}, return_type="dataframe")
basis.columns = [f"s{i}" for i in range(basis.shape[1])]
info = basis.design_info
Xsp = pd.concat([cox_df(d, []).reset_index(drop=True), basis.reset_index(drop=True)], axis=1)
# cr basis spans the intercept; drop one column for identifiability under Cox (no intercept)
Xsp = Xsp.drop(columns="s0")
cox_sp = fit_strat(Xsp, penal=0.0)
Xlin = cox_df(d, ["log2_tmb1_c"]); cox_lin = fit_strat(Xlin)
Xnull = cox_df(d, []); cox_null = fit_strat(Xnull)
sens_sp = {"lrt_overall": stats.chi2.sf(2 * (cox_sp.log_likelihood_ - cox_null.log_likelihood_), 3),
           "lrt_nonlinear": stats.chi2.sf(2 * (cox_sp.log_likelihood_ - cox_lin.log_likelihood_), 2),
           "aic_spline": cox_sp.AIC_partial_, "aic_linear": cox_lin.AIC_partial_}
grid = np.linspace(np.log2(0 + 1), np.log2(80 + 1), 200)
ref = np.log2(np.median(d.tmb) + 1)
Bg = np.asarray(build_design_matrices([info], {"x": grid})[0])[:, 1:]
Br = np.asarray(build_design_matrices([info], {"x": [ref]})[0])[:, 1:]
names = [f"s{i}" for i in range(1, 4)]
b = cox_sp.params_[names].values; V = cox_sp.variance_matrix_.loc[names, names].values
D = Bg - Br; lp = D @ b; se = np.sqrt(np.einsum("ij,jk,ik->i", D, V, D))
fig, ax = plt.subplots(figsize=(4.4, 2.9))
tm = 2 ** grid - 1
ax.fill_between(tm, np.exp(lp - 1.96 * se), np.exp(lp + 1.96 * se), color=C_HIGH, alpha=0.15, lw=0)
ax.plot(tm, np.exp(lp), color=C_HIGH, lw=2)
ax.axhline(1, color=INK2, lw=0.8, ls="--"); ax.axvline(10, color=C_LOW, lw=1, ls=":")
ax.text(10.5, 1.55, "10 mut/Mb", color=INK2, fontsize=7)
ax.set_xscale("symlog", linthresh=1); ax.set_xticks([0, 1, 2, 5, 10, 20, 50]); ax.set_xticklabels(["0", "1", "2", "5", "10", "20", "50"])
ax.set_yscale("log"); ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter()); ax.set_yticks([0.25, 0.5, 1, 2]); ax.set_yticklabels(["0.25", "0.5", "1", "2"])
ax.set_ylim(0.2, 2.2); ax.set_xlim(0, 80)
ax.set_xlabel("TMB (mut/Mb)"); ax.set_ylabel(f"Hazard ratio vs median TMB\n({np.median(d.tmb):.1f} mut/Mb)")
rug = d.tmb.clip(upper=80).sample(600, random_state=1)
ax.plot(rug, np.full(len(rug), 0.21), "|", color=INK2, alpha=0.3, ms=4)
ax.grid(axis="y", color=GRID, lw=0.5)
fig.tight_layout(); fig.savefig(FIG / "fig3_spline_tmb.png"); plt.close(fig)
for v in [2, 5, 10, 20, 40]:
    Bv = np.asarray(build_design_matrices([info], {"x": [np.log2(v + 1)]})[0])[:, 1:] - Br
    l = float((Bv @ b).item()); s = float(np.sqrt((Bv @ V @ Bv.T).item()))
    sens_sp[f"HR_at_{v}"] = [np.exp(l), np.exp(l - 1.96 * s), np.exp(l + 1.96 * s)]
R["spline"] = sens_sp
R["survival"] = surv

json.dump(R, open(OUT / "results.json", "w"), indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))
print(t1.to_string())
print(json.dumps({k: R[k] for k in ["categorical", "survival", "sensitivity", "spline"]}, indent=1,
                 default=lambda o: o.item() if hasattr(o, "item") else str(o))[:9000])
print(fr.round(3).to_string())
