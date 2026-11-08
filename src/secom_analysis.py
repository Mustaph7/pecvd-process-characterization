"""
Step 6: Yield driver analysis on real fab data (UCI SECOM).

1,567 production units, 590 anonymized sensor signals, pass/fail from
in-line test, with timestamps (Jul-Oct 2008). Reference: McCann & Johnston
(2008), UCI ML Repository, https://doi.org/10.24432/C54305 (CC BY 4.0).

Pipeline
  1. Profile: missing values, constant sensors, fail rate over time.
  2. Clean: drop constant sensors and sensors > 50% missing; median-impute the
     rest using TRAINING-period medians only.
  3. Time-ordered split: first 70% of units = train, last 30% = test.
     The process drifts (fail rate falls from 22% to 3-6%), so a model must be
     judged on later units than it learned from.
  4. Screening (train only): Mann-Whitney test per sensor with Benjamini-Hochberg
     FDR control, plus random forest importance (like JMP Predictor Screening).
  5. PCA: do fails separate from passes in sensor space?
  6. Classifiers: logistic regression on the top screened sensors and a random
     forest on all cleaned sensors, scored on the later test period with ROC AUC,
     PR AUC and "capture rate" (fails caught when inspecting the riskiest 10%).
     Compared with a random-split cross-validation to show how much an
     ordinary CV overstates performance on drifting data.

Usage (from the src folder):
    python secom_analysis.py
Needs data/secom.data and data/secom_labels.data. Writes data/secom_clean.csv
(for JMP), data/secom_screening.csv, and plots/secom_*.png.
"""
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve, precision_recall_curve

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
PLOTS = ROOT / "plots"
SEED = 42
TRAIN_FRAC = 0.70
MAX_MISSING = 0.50
TOP_K = 20
INSPECT_FRAC = 0.10


def load():
    X = pd.read_csv(DATA / "secom.data", sep=" ", header=None)
    X.columns = [f"S{i:03d}" for i in range(X.shape[1])]
    lab = pd.read_csv(DATA / "secom_labels.data", sep=" ", header=None, names=["label", "time"])
    t = pd.to_datetime(lab["time"], format="%d/%m/%Y %H:%M:%S")
    y = (lab["label"] == 1).astype(int).to_numpy()        # 1 = fail
    return X, y, t


def bh_fdr(p):
    p = np.asarray(p); n = len(p); order = np.argsort(p)
    q = p[order] * n / np.arange(1, n + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    out = np.empty(n); out[order] = np.clip(q, 0, 1)
    return out


def capture_rate(y, score, frac=INSPECT_FRAC):
    """Share of all fails caught if the riskiest `frac` of units are inspected."""
    k = max(1, int(round(frac * len(y))))
    top = np.argsort(-score)[:k]
    return y[top].sum() / max(1, y.sum())


def main():
    X, y, t = load()
    n, p = X.shape
    print(f"Units {n}, sensors {p}, fails {y.sum()} ({100 * y.mean():.1f}%)")
    print(f"Missing values: {X.isna().values.sum():,} ({100 * X.isna().values.mean():.2f}%)")
    fail_month = pd.Series(y).groupby(t.dt.to_period("M")).agg(["mean", "size"])
    print("Fail rate by month:", {str(k): f"{v['mean']:.1%} (n={v['size']})" for k, v in fail_month.iterrows()})

    # ---------------- clean ----------------
    constant = X.columns[X.nunique(dropna=True) <= 1]
    sparse = X.columns[X.isna().mean() > MAX_MISSING]
    drop = sorted(set(constant) | set(sparse))
    Xc = X.drop(columns=drop)
    n_train = int(round(TRAIN_FRAC * n))
    train = np.arange(n) < n_train
    med = Xc[train].median()
    Xc = Xc.fillna(med)
    # sensors constant within the training period carry no information for the model
    flat_train = Xc.columns[Xc[train].std() == 0]
    Xc = Xc.drop(columns=flat_train)
    print(f"\nDropped {len(constant)} constant + {len(set(sparse) - set(constant))} >50%-missing"
          f" + {len(flat_train)} flat-in-training sensors -> {Xc.shape[1]} kept")
    print(f"Train: first {n_train} units ({t[train].min():%d %b} - {t[train].max():%d %b}), "
          f"{y[train].sum()} fails ({100 * y[train].mean():.1f}%)")
    print(f"Test : last {n - n_train} units ({t[~train].min():%d %b} - {t[~train].max():%d %b}), "
          f"{y[~train].sum()} fails ({100 * y[~train].mean():.1f}%)")

    out = Xc.copy()
    out.insert(0, "Set", np.where(train, "Train", "Test"))
    out.insert(0, "Result", np.where(y == 1, "Fail", "Pass"))
    out.insert(0, "Timestamp", t.dt.strftime("%Y-%m-%d %H:%M"))
    out.to_csv(DATA / "secom_clean.csv", index=False)

    Xtr, ytr, Xte, yte = Xc[train].to_numpy(), y[train], Xc[~train].to_numpy(), y[~train]

    # ---------------- screening (train only) ----------------
    pvals, shift = [], []
    for j, c in enumerate(Xc.columns):
        a, b = Xtr[ytr == 1, j], Xtr[ytr == 0, j]
        pvals.append(stats.mannwhitneyu(a, b, alternative="two-sided").pvalue)
        sd = Xtr[:, j].std()
        shift.append((np.median(a) - np.median(b)) / sd if sd > 0 else 0.0)
    q = bh_fdr(pvals)
    rf_screen = RandomForestClassifier(n_estimators=500, min_samples_leaf=3, class_weight="balanced_subsample",
                                       random_state=SEED, n_jobs=-1).fit(Xtr, ytr)
    scr = pd.DataFrame({"Sensor": Xc.columns, "MW p": pvals, "FDR q": q,
                        "Median shift (SD units)": shift,
                        "RF importance": rf_screen.feature_importances_})
    scr["RF rank"] = scr["RF importance"].rank(ascending=False).astype(int)
    scr = scr.sort_values("MW p").reset_index(drop=True)
    scr.to_csv(DATA / "secom_screening.csv", index=False)
    n_sig = int((scr["FDR q"] < 0.05).sum())
    n_raw = int((scr["MW p"] < 0.05).sum())
    print(f"\nScreening on training data: {n_raw} sensors with p < 0.05, "
          f"{n_sig} survive FDR control (q < 0.05); ~{0.05 * len(scr):.0f} would pass p<0.05 by chance alone")
    print(scr.head(12).round(4).to_string(index=False))
    top_rf = scr.sort_values("RF importance", ascending=False)["Sensor"].head(TOP_K).tolist()
    top_mw = scr["Sensor"].head(TOP_K).tolist()
    print(f"Overlap of top-{TOP_K} lists (Mann-Whitney vs random forest): {len(set(top_rf) & set(top_mw))}")

    # do the screened sensors still separate pass/fail in the later test period?
    hold = []
    for s in scr["Sensor"].head(10):
        j = list(Xc.columns).index(s)
        hold.append(stats.mannwhitneyu(Xte[yte == 1, j], Xte[yte == 0, j]).pvalue)
    print("Top-10 screened sensors, Mann-Whitney p in TEST period:",
          ", ".join(f"{s} {pv:.3f}" for s, pv in zip(scr['Sensor'].head(10), hold)))

    # ---------------- PCA ----------------
    sc = StandardScaler().fit(Xtr)
    pca = PCA(random_state=SEED).fit(sc.transform(Xtr))
    cum = np.cumsum(pca.explained_variance_ratio_)
    n80 = int(np.searchsorted(cum, 0.80) + 1)
    Z = pca.transform(sc.transform(Xc.to_numpy()))
    pc_p = [stats.mannwhitneyu(Z[train][ytr == 1, k], Z[train][ytr == 0, k]).pvalue for k in range(5)]
    print(f"\nPCA: PC1 {100 * pca.explained_variance_ratio_[0]:.1f}%, PC2 {100 * pca.explained_variance_ratio_[1]:.1f}% "
          f"of variance; {n80} components needed for 80%")
    print("Pass/fail difference on PC1-PC5 (train, Mann-Whitney p):", [f"{v:.3f}" for v in pc_p])

    # ---------------- models ----------------
    cols_k = [list(Xc.columns).index(s) for s in top_rf]
    logit = make_pipeline(StandardScaler(), LogisticRegression(C=0.1, class_weight="balanced", max_iter=5000))
    rf = RandomForestClassifier(n_estimators=500, min_samples_leaf=3, class_weight="balanced_subsample",
                                random_state=SEED, n_jobs=-1)
    models = {f"Logistic (top {TOP_K} RF-screened)": (logit, cols_k),
              "Random forest (all sensors)": (rf, list(range(Xc.shape[1])))}
    res, curves = [], {}
    base = yte.mean()
    for name, (m, cols) in models.items():
        m.fit(Xtr[:, cols], ytr)
        s = m.predict_proba(Xte[:, cols])[:, 1]
        curves[name] = s
        # the common shortcut: random 5-fold CV over all units, screening done on ALL data
        cv = StratifiedKFold(5, shuffle=True, random_state=SEED)
        if "Logistic" in name:
            rf_all = RandomForestClassifier(n_estimators=500, min_samples_leaf=3, class_weight="balanced_subsample",
                                            random_state=SEED, n_jobs=-1).fit(Xc.to_numpy(), y)
            cols_all = np.argsort(-rf_all.feature_importances_)[:TOP_K]
            s_cv = cross_val_predict(m, Xc.to_numpy()[:, cols_all], y, cv=cv, method="predict_proba")[:, 1]
        else:
            s_cv = cross_val_predict(m, Xc.to_numpy(), y, cv=cv, method="predict_proba")[:, 1]
        res.append({"Model": name,
                    "Random-CV ROC AUC (optimistic)": roc_auc_score(y, s_cv),
                    "Time-split ROC AUC": roc_auc_score(yte, s),
                    "Time-split PR AUC": average_precision_score(yte, s),
                    f"Fails caught inspecting riskiest {INSPECT_FRAC:.0%}": capture_rate(yte, s)})
    # Variant: keep only sensors that separate pass/fail consistently in BOTH halves
    # of the training period (same direction, p < 0.05 in each). Uses train data only.
    h = np.arange(len(ytr)) < len(ytr) // 2
    stable = []
    for j in range(Xtr.shape[1]):
        ok = True; signs = []
        for part in (h, ~h):
            a, b = Xtr[part][ytr[part] == 1, j], Xtr[part][ytr[part] == 0, j]
            ok &= stats.mannwhitneyu(a, b).pvalue < 0.05
            signs.append(np.sign(np.median(a) - np.median(b)))
        if ok and signs[0] == signs[1]:
            stable.append(j)
    m_st = make_pipeline(StandardScaler(), LogisticRegression(C=0.1, class_weight="balanced", max_iter=5000))
    m_st.fit(Xtr[:, stable], ytr)
    s_st = m_st.predict_proba(Xte[:, stable])[:, 1]
    print(f"\nStability screen: {len(stable)} sensors consistent in both training halves "
          f"({', '.join(Xc.columns[stable])})")
    res.append({"Model": f"Logistic ({len(stable)} stable sensors)",
                "Random-CV ROC AUC (optimistic)": np.nan,
                "Time-split ROC AUC": roc_auc_score(yte, s_st),
                "Time-split PR AUC": average_precision_score(yte, s_st),
                f"Fails caught inspecting riskiest {INSPECT_FRAC:.0%}": capture_rate(yte, s_st)})
    # Post-hoc observation (chosen after looking at the test period, so NOT a validated model)
    j103 = list(Xc.columns).index("S103")
    print(f"Post-hoc: S103 alone has ROC AUC {roc_auc_score(ytr, Xtr[:, j103]):.2f} in train and "
          f"{roc_auc_score(yte, Xte[:, j103]):.2f} in test; the only top-10 screened sensor still significant in the test period")

    res = pd.DataFrame(res)
    pd.set_option("display.width", 200)
    print(f"\nModel performance (test = later {100 * (1 - TRAIN_FRAC):.0f}% of units, "
          f"{yte.sum()} fails, base fail rate {base:.1%}):")
    print(res.round(3).to_string(index=False))
    print(f"Random inspection of {INSPECT_FRAC:.0%} would catch ~{INSPECT_FRAC:.0%} of fails; "
          f"a no-skill PR AUC equals the fail rate ({base:.3f}).")
    res.to_csv(DATA / "secom_model_results.csv", index=False)

    # ---------------- plots ----------------
    fig, ax = plt.subplots(1, 3, figsize=(19, 5))
    wk = pd.Series(y).groupby(t.dt.to_period("W")).agg(["mean", "size"])
    wk = wk[wk["size"] >= 10]
    ax[0].plot(wk.index.to_timestamp(), 100 * wk["mean"], marker="o", color="#2a6f97")
    ax[0].axvline(t[n_train], color="red", ls="--"); ax[0].text(t[n_train], ax[0].get_ylim()[1] * 0.9, " test period", color="red")
    ax[0].set_title("Weekly fail rate (weeks with >= 10 units)"); ax[0].set_ylabel("% fail")
    ax[0].tick_params(axis="x", rotation=30)
    ax[1].hist(100 * X.isna().mean(), bins=40, color="#2a6f97")
    ax[1].axvline(100 * MAX_MISSING, color="red", ls="--"); ax[1].set_yscale("log")
    ax[1].set_title("Missing values per sensor"); ax[1].set_xlabel("% missing"); ax[1].set_ylabel("sensors (log)")
    top = scr.sort_values("RF importance", ascending=False).head(15)[::-1]
    ax[2].barh(top["Sensor"], top["RF importance"],
               color=["#2a6f97" if qq < 0.05 else "#b8c4ce" for qq in top["FDR q"]])
    ax[2].set_title("Top 15 sensors by random-forest importance\n(blue = also significant after FDR)")
    fig.tight_layout(); fig.savefig(PLOTS / "secom_overview.png", dpi=130)

    fig, ax = plt.subplots(1, 2, figsize=(13, 5.5))
    for lab, col, mk in [(0, "#b8c4ce", "o"), (1, "#d1495b", "x")]:
        mask = y == lab
        ax[0].scatter(Z[mask, 0], Z[mask, 1], s=10 if lab == 0 else 30, c=col, marker=mk,
                      label="Fail" if lab else "Pass", alpha=0.8)
    ax[0].set_xlabel(f"PC1 ({100 * pca.explained_variance_ratio_[0]:.1f}%)")
    ax[0].set_ylabel(f"PC2 ({100 * pca.explained_variance_ratio_[1]:.1f}%)")
    ax[0].set_title("PCA scores: fails do not form their own cluster"); ax[0].legend()
    ax[1].plot(np.arange(1, len(cum) + 1), 100 * cum, color="#2a6f97")
    ax[1].axhline(80, color="red", ls="--"); ax[1].axvline(n80, color="red", ls="--")
    ax[1].set_xlim(0, 200); ax[1].set_xlabel("Number of components"); ax[1].set_ylabel("Cumulative % variance")
    ax[1].set_title(f"{n80} components needed for 80% of the variance")
    fig.tight_layout(); fig.savefig(PLOTS / "secom_pca.png", dpi=130)

    fig, ax = plt.subplots(1, 3, figsize=(19, 5.5))
    for (name, s), col in zip(curves.items(), ["#2a6f97", "#e07a1f"]):
        fpr, tpr, _ = roc_curve(yte, s); ax[0].plot(fpr, tpr, label=f"{name} (AUC {roc_auc_score(yte, s):.2f})", color=col)
        pr, rc, _ = precision_recall_curve(yte, s); ax[1].plot(rc, pr, label=name, color=col)
        order = np.argsort(-s); capt = np.cumsum(yte[order]) / yte.sum()
        ax[2].plot(100 * np.arange(1, len(s) + 1) / len(s), 100 * capt, label=name, color=col)
    ax[0].plot([0, 1], [0, 1], "k:", lw=1); ax[0].set_xlabel("False positive rate"); ax[0].set_ylabel("True positive rate")
    ax[0].set_title("ROC, later test period"); ax[0].legend(fontsize=8)
    ax[1].axhline(base, color="k", ls=":", lw=1, label=f"no skill ({base:.3f})")
    ax[1].set_xlabel("Recall (fails caught)"); ax[1].set_ylabel("Precision"); ax[1].set_title("Precision-recall")
    ax[1].legend(fontsize=8)
    ax[2].plot([0, 100], [0, 100], "k:", lw=1, label="random inspection")
    ax[2].axvline(100 * INSPECT_FRAC, color="red", ls="--", lw=1)
    ax[2].set_xlabel("% of units inspected (riskiest first)"); ax[2].set_ylabel("% of fails caught")
    ax[2].set_title("Capture curve"); ax[2].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(PLOTS / "secom_models.png", dpi=130)
    print("\nSaved: data/secom_clean.csv, data/secom_screening.csv, data/secom_model_results.csv, "
          "plots/secom_overview.png, plots/secom_pca.png, plots/secom_models.png")


if __name__ == "__main__":
    main()
