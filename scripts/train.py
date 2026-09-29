"""Train and evaluate protein-expression predictors from slide features.

Two models, same patient-level 5-fold cross-validation:
    baseline  mean of patch features -> PCA -> logistic regression, per protein
    abmil     gated attention MIL, one attention branch + output per protein
              (multi-task); attention gives the heatmaps

Labels: each protein is split into High / Low per patient.
    --split median   High = above the median (all 145 patients used)
    --split tertile  top third vs bottom third; middle third left out
                     (cleaner classes, fewer patients)

    python scripts/train.py --features features/phikon --out results/phikon

Outputs (in --out):
    metrics.csv            AUC + 95% CI per model and protein, permutation p-value
    oof_predictions.csv    held-out prediction for every patient (for the demo)
    labels.csv             the High/Low labels used
    attention/<id>.npz     held-out attention per patch and protein (heatmaps)
    abmil_final.pt         ABMIL trained on all patients (for the API)
    config.json            settings, targets and thresholds
"""

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# Display name -> RPPA column
TARGETS = {
    "PD-L1": "PDL1",
    "p16": "P16INK4A",
    "Cyclin B1": "CYCLINB1",
    "phospho-Rb": "RB_pS807S811",
    "E-cadherin": "ECADHERIN",
}


# ---------------------------------------------------------------- data
def load_features(folder, patients):
    bags, coords = {}, {}
    for patient in patients:
        path = Path(folder) / f"{patient}.h5"
        if path.exists():
            with h5py.File(path, "r") as h5:
                bags[patient] = h5["features"][:]
                coords[patient] = h5["coords"][:]
    return bags, coords


def make_labels(rppa, split):
    labels = pd.DataFrame(index=rppa.index)
    for name, column in TARGETS.items():
        values = rppa[column]
        if split == "median":
            labels[name] = (values > values.median()).astype(float)
        else:
            low, high = values.quantile([1 / 3, 2 / 3])
            labels[name] = np.where(values >= high, 1.0, np.where(values <= low, 0.0, np.nan))
        labels.loc[values.isna(), name] = np.nan
    return labels


# ---------------------------------------------------------------- ABMIL
class MultiTaskABMIL(nn.Module):
    """Gated attention MIL (Ilse et al. 2018) with one attention branch and
    one classifier per protein, sharing the patch embedding."""

    def __init__(self, in_dim, n_tasks, hidden=256, attn=128, dropout=0.25):
        super().__init__()
        self.embed = nn.Sequential(nn.Linear(in_dim, hidden), nn.ReLU(), nn.Dropout(dropout))
        self.attn_v = nn.Sequential(nn.Linear(hidden, attn), nn.Tanh())
        self.attn_u = nn.Sequential(nn.Linear(hidden, attn), nn.Sigmoid())
        self.attn_w = nn.Linear(attn, n_tasks)
        self.classifiers = nn.Parameter(torch.randn(n_tasks, hidden) * 0.01)
        self.bias = nn.Parameter(torch.zeros(n_tasks))

    def forward(self, bag):                      # bag: [N, in_dim]
        h = self.embed(bag)                      # [N, hidden]
        scores = self.attn_w(self.attn_v(h) * self.attn_u(h))  # [N, T]
        attention = torch.softmax(scores, dim=0)                # over patches
        slide = attention.T @ h                                 # [T, hidden]
        logits = (slide * self.classifiers).sum(dim=1) + self.bias  # [T]
        return logits, attention


def train_abmil(bags, y, in_dim, args, device, seed):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = MultiTaskABMIL(in_dim, y.shape[1], dropout=args.dropout).to(device)
    optimiser = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loss_fn = nn.BCEWithLogitsLoss(reduction="none")
    targets = torch.tensor(y, dtype=torch.float32, device=device)
    mask = ~torch.isnan(targets)
    targets = torch.nan_to_num(targets)

    model.train()
    for _ in range(args.epochs):
        for i in rng.permutation(len(bags)):
            bag = bags[i]
            if len(bag) > args.max_train_patches:  # random patch subset each epoch
                bag = bag[rng.choice(len(bag), args.max_train_patches, replace=False)]
            logits, _ = model(torch.from_numpy(bag).to(device))
            if not mask[i].any():
                continue
            loss = loss_fn(logits, targets[i])[mask[i]].mean()
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
    return model.eval()


@torch.no_grad()
def predict_abmil(model, bags, device):
    probabilities, attentions = [], []
    for bag in bags:
        logits, attention = model(torch.from_numpy(bag).to(device))
        probabilities.append(torch.sigmoid(logits).cpu().numpy())
        attentions.append(attention.cpu().numpy())
    return np.array(probabilities), attentions


# ---------------------------------------------------------------- metrics
def auc_with_ci(y, p, n_boot=1000, seed=0):
    keep = ~np.isnan(y)
    y, p = y[keep], p[keep]
    if len(np.unique(y)) < 2:
        return np.nan, np.nan, np.nan, int(keep.sum())
    auc = roc_auc_score(y, p)
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(y), len(y))
        if len(np.unique(y[idx])) == 2:
            boots.append(roc_auc_score(y[idx], p[idx]))
    low, high = np.percentile(boots, [2.5, 97.5])
    return auc, low, high, int(keep.sum())


def baseline_oof(X, y, folds):
    """Out-of-fold probabilities for one protein with the baseline model."""
    oof = np.full(len(y), np.nan)
    for train, test in folds:
        train = train[~np.isnan(y[train])]
        if len(np.unique(y[train])) < 2:
            continue
        model = make_pipeline(StandardScaler(),
                              PCA(n_components=min(32, len(train) - 1), random_state=0),
                              LogisticRegression(C=0.1, max_iter=5000))
        model.fit(X[train], y[train])
        oof[test] = model.predict_proba(X[test])[:, 1]
    return oof


def permutation_p(X, y, folds, observed, n_perm, seed=0):
    """How often shuffled labels give an AUC at least as high as the real one."""
    rng = np.random.default_rng(seed)
    keep = ~np.isnan(y)
    hits = 0
    for _ in range(n_perm):
        shuffled = y.copy()
        shuffled[keep] = rng.permutation(y[keep])
        oof = baseline_oof(X, shuffled, folds)
        ok = keep & ~np.isnan(oof)
        if roc_auc_score(shuffled[ok], oof[ok]) >= observed:
            hits += 1
    return (hits + 1) / (n_perm + 1)


# ---------------------------------------------------------------- main
def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--features", default="features/phikon")
    parser.add_argument("--rppa", default="data/rppa_matrix.csv")
    parser.add_argument("--out", default="results/phikon")
    parser.add_argument("--split", default="median", choices=["median", "tertile"])
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.25)
    parser.add_argument("--max-train-patches", type=int, default=4096)
    parser.add_argument("--permutations", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    out = Path(args.out)
    (out / "attention").mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    rppa = pd.read_csv(args.rppa, index_col=0)
    bags, coords = load_features(args.features, rppa.index)
    patients = sorted(bags)
    if len(patients) < 20:
        raise SystemExit(f"Only {len(patients)} patients have features; extract more first.")
    labels = make_labels(rppa.loc[patients], args.split)
    labels.to_csv(out / "labels.csv")
    names = list(TARGETS)
    Y = labels[names].to_numpy()
    bag_list = [bags[p] for p in patients]
    in_dim = bag_list[0].shape[1]
    X_mean = np.stack([b.mean(axis=0) for b in bag_list])
    print(f"Patients: {len(patients)}  Feature dim: {in_dim}  Device: {device}  Split: {args.split}")

    folds = list(KFold(args.folds, shuffle=True, random_state=args.seed).split(patients))

    # Baseline
    oof_base = np.column_stack([baseline_oof(X_mean, Y[:, t], folds) for t in range(len(names))])

    # ABMIL, one model per fold (all proteins at once)
    oof_abmil = np.full(Y.shape, np.nan)
    for k, (train, test) in enumerate(folds):
        model = train_abmil([bag_list[i] for i in train], Y[train], in_dim, args, device, args.seed + k)
        probabilities, attentions = predict_abmil(model, [bag_list[i] for i in test], device)
        oof_abmil[test] = probabilities
        for i, attention in zip(test, attentions):
            np.savez_compressed(out / "attention" / f"{patients[i]}.npz",
                                attention=attention, coords=coords[patients[i]],
                                targets=np.array(names), fold=k)
        print(f"  fold {k + 1}/{args.folds} done")

    # Metrics
    rows = []
    for t, name in enumerate(names):
        for model_name, oof in [("baseline", oof_base), ("abmil", oof_abmil)]:
            auc, low, high, n = auc_with_ci(Y[:, t], oof[:, t])
            row = {"model": model_name, "protein": name, "auc": auc,
                   "ci_low": low, "ci_high": high, "n": n}
            if model_name == "baseline" and args.permutations and not np.isnan(auc):
                row["perm_p"] = permutation_p(X_mean, Y[:, t], folds, auc, args.permutations)
            rows.append(row)
    metrics = pd.DataFrame(rows).round(3)
    metrics.to_csv(out / "metrics.csv", index=False)

    predictions = pd.DataFrame(index=pd.Index(patients, name="patient"))
    for t, name in enumerate(names):
        predictions[f"{name}_true"] = Y[:, t]
        predictions[f"{name}_baseline"] = oof_base[:, t].round(3)
        predictions[f"{name}_abmil"] = oof_abmil[:, t].round(3)
    predictions.to_csv(out / "oof_predictions.csv")

    # Final model on all patients, for the API and new slides
    final = train_abmil(bag_list, Y, in_dim, args, device, args.seed)
    torch.save(final.state_dict(), out / "abmil_final.pt")
    config = {**vars(args), "targets": names, "rppa_columns": TARGETS,
              "in_dim": in_dim, "n_patients": len(patients),
              "thresholds": {n: float(rppa.loc[patients, c].median()) for n, c in TARGETS.items()}}
    (out / "config.json").write_text(json.dumps(config, indent=2))

    print("\n" + metrics.to_string(index=False))
    print(f"\nSaved results to {out}/")


if __name__ == "__main__":
    main()
