#!/usr/bin/env python
# coding: utf-8

# In[1]:


"""
RF training script + interpretability analysis (SHAP + ECFP4 fragments)
- Load data directly from CSV (requires columns: SMILES, Activity)
- Use Random split for 90/5/5 split
-Hui Peng @ 2024-06-01, script adapted from Xiaoyun Wang
-Modified by Jolie Miller @ 2026-10-14
"""


# In[2]:


# The exclamation mark runs this as a shell command instead of Python code.
# -q makes the install "quiet" (less output spam).
get_ipython().system('pip install rdkit -q')


# In[3]:


get_ipython().system('pip install shap')


# In[4]:


import os
import time
import pickle
from argparse import ArgumentParser
from typing import Dict, Tuple, List


# In[5]:


import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit.Chem import Draw


# In[6]:


import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader


# In[7]:


from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    roc_auc_score,
    average_precision_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)
from sklearn.ensemble import RandomForestClassifier


# In[8]:


import matplotlib.pyplot as plt
import shap


# In[9]:


# ===== Global configuration =====
FP_NBITS = 512
RADIUS = 2
BATCH_SIZE = 512
EPOCHS = 200
PATIENCE = 20
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-5
DROPOUT = 0.2


# ==================== Fingerprint and scaffold tools ====================

# In[10]:


def smiles_to_ecfp(smiles: str, n_bits: int = FP_NBITS, radius: int = RADIUS):

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
    from rdkit import DataStructs
    arr = np.zeros((n_bits,), dtype=np.int8)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr.astype(np.float32)


# In[11]:


def build_fingerprints(smiles_list: List[str]) -> np.ndarray:

    fps = []
    for smi in smiles_list:
        fp = smiles_to_ecfp(smi)
        if fp is None:
            raise ValueError(f"Invalid SMILES detected: {smi}")
        fps.append(fp)
    return np.stack(fps, axis=0)


# In[12]:


def convert_smiles_to_mols(smiles_list: List[str]):
    """SMILES -> RDKit Mol list"""
    mols = []
    for s in smiles_list:
        m = Chem.MolFromSmiles(s)
        if m is None:
            raise ValueError(f"Invalid SMILES detected: {s}")
        mols.append(m)
    return mols


# In[13]:


def generate_scaffolds(mols):

    scaffolds = {}
    for idx, m in enumerate(mols):
        core = MurckoScaffold.GetScaffoldForMol(m)
        if core is None:
            key = "NO_SCAFFOLD"
        else:
            key = Chem.MolToSmiles(core, isomericSmiles=True)
        scaffolds.setdefault(key, []).append(idx)
    return scaffolds


# In[14]:


def scaffold_split_indices(
    n_samples: int,
    scaffolds_dict: Dict[str, list],
    sizes=(0.9, 0.05, 0.05),
    seed: int = 0,
):
    """
    Can perform 90/5/5 split based on scaffold, return train/val/test index arrays.
    """
    assert abs(sum(sizes) - 1.0) < 1e-6, "sizes must sum to 1"
    rng = np.random.RandomState(seed)
    scaffold_keys = list(scaffolds_dict.keys())
    rng.shuffle(scaffold_keys)

    n_total = n_samples
    n_train = int(sizes[0] * n_total)
    n_val = int(sizes[1] * n_total)

    train_idx, val_idx, test_idx = [], [], []
    for key in scaffold_keys:
        idxs = scaffolds_dict[key]
        if len(train_idx) + len(idxs) <= n_train:
            train_idx.extend(idxs)
        elif len(val_idx) + len(idxs) <= n_val:
            val_idx.extend(idxs)
        else:
            test_idx.extend(idxs)

    train_idx = np.array(sorted(train_idx), dtype=int)
    val_idx = np.array(sorted(val_idx), dtype=int)
    test_idx = np.array(sorted(test_idx), dtype=int)

    return train_idx, val_idx, test_idx


# ==================== MAIN ====================

# In[15]:


from types import SimpleNamespace

# Plain object with the same attribute names the steps already use (args.seed, etc.)
args = SimpleNamespace(
    dataset_path="CAR.csv",
    output_root="CAR_model_output_root_scaffold_split_clean",
    seed=135,
    predict_csv=None,
    model_path=None,
    predict_output=None,
)


# In[16]:


torch.manual_seed(args.seed)
np.random.seed(args.seed)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
model_dir = os.path.join(args.output_root, "models")
metric_dir = os.path.join(args.output_root, "metrics")
split_dir = os.path.join(args.output_root, "splits")
pred_dir = os.path.join(args.output_root, "predictions")
shap_dir = os.path.join(args.output_root, "SHAP_Analysis")
os.makedirs(model_dir, exist_ok=True)
os.makedirs(metric_dir, exist_ok=True)
os.makedirs(split_dir, exist_ok=True)
os.makedirs(pred_dir, exist_ok=True)
os.makedirs(shap_dir, exist_ok=True)


# In[17]:


# ===== Step 1: Load CSV data =====
print("\n" + "="*60)
print("Step 1: Load data from CSV")
print("="*60)

df = pd.read_csv(args.dataset_path)
required_cols = ["SMILES", "Activity", "logP"]
for c in required_cols:
    if c not in df.columns:
        raise ValueError(f"CSV is missing required column: {c}")
smiles = df["SMILES"].astype(str).values
logP = df["logP"].astype(float).values.reshape(-1, 1)
n_samples = len(smiles)
print(f"Number of loaded samples: {n_samples}")


# In[18]:


# ===== Step 2: Scaffold Split =====
print("\n" + "="*60)
print("Step 2: Scaffold split (75/25)")
print("="*60)

# SMILES strings -> RDKit molecule objects (raises an error on invalid SMILES)
mols = convert_smiles_to_mols(list(smiles))

# Group molecule indices by Murcko scaffold, e.g. {"c1ccccc1": [0, 5, 9], ...}
scaffolds = generate_scaffolds(mols)
print(f"Unique scaffolds: {len(scaffolds)}")

# Assign whole scaffold groups to train or test, so no scaffold appears in both.
# sizes = (train, val, test); val is 0 because the later CV steps replace it.
train_idx, val_idx, test_idx = scaffold_split_indices(
    n_samples, scaffolds, sizes=(0.75, 0.0, 0.25), seed=args.seed
)

print(f"Training set: {len(train_idx)} samples")
print(f"Validation set: {len(val_idx)} samples")  # will be 0
print(f"Test set: {len(test_idx)} samples")

# Check class balance in each set (the split ignores Activity labels)
act = df["Activity"].values
print(f"Activity=1 fraction | train: {act[train_idx].mean():.2f}, test: {act[test_idx].mean():.2f}")

split_dict = {
    "train_idx": train_idx,
    "val_idx": val_idx,
    "test_idx": test_idx,
    "seed": args.seed,
    "total_samples": n_samples,
}
split_path = os.path.join(split_dir, f"splits_75_25_seed{args.seed}.pkl")  # filename updated
with open(split_path, "wb") as f:
    pickle.dump(split_dict, f)
print(f"Split indices saved to: {split_path}")

# ----- Save scaffold -> train/test assignment -----
train_set, test_set = set(train_idx), set(test_idx)   # sets make the lookups below fast
rows = []
for scaf_smiles, idxs in scaffolds.items():           # scaffolds = {scaffold SMILES: [molecule indices]}
    n_tr = sum(i in train_set for i in idxs)          # compounds with this scaffold in train
    n_te = sum(i in test_set for i in idxs)           # compounds with this scaffold in test
    rows.append({
        "scaffold": scaf_smiles if scaf_smiles else "(no ring)",   # empty scaffold = acyclic molecules
        "n_compounds": len(idxs),
        "set": "train" if n_te == 0 else ("test" if n_tr == 0 else "both"),
        "n_train": n_tr,
        "n_test": n_te,
        "activity_1_fraction": round(act[idxs].mean(), 2),         # extra: fraction active in this group
    })
scaf_df = pd.DataFrame(rows).sort_values("n_compounds", ascending=False)
scaf_df.to_csv(os.path.join(split_dir, "scaffold_assignments.csv"), index=False)
print(scaf_df.to_string(index=False))


# In[19]:


# ===== Step 3: Labels =====
print("\n" + "="*60)
print("Step 3: Prepare labels")
print("="*60)

y_dict = {
    "Activity": df["Activity"].astype(float).values
}

print(f"Activity label shape: {y_dict['Activity'].shape}")


# In[20]:


# ===== Step 4: Fingerprints + logP =====
print("\n" + "="*60)
print("Step 4: Calculate ECFP4 fingerprints and concatenate logP")
print("="*60)
X_ecfp = build_fingerprints(list(smiles))
print(f"Fingerprint matrix shape: {X_ecfp.shape}")

X = np.concatenate([X_ecfp, logP], axis=1)##put logP as the input feature
print(f"Final feature matrix shape (ECFP4 + logP): {X.shape}")


# In[21]:


# ===== Step 5: Model variability checks and parameter optimization =====
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import roc_auc_score, f1_score

# Slice the full arrays into train/test using the scaffold-split indices
X_tr, y_tr = X[train_idx], y_dict["Activity"][train_idx].astype(int)
X_te, y_te = X[test_idx],  y_dict["Activity"][test_idx].astype(int)

def make_rf(seed):
    """Same RF settings as the rest of the notebook; only the seed varies."""
    return RandomForestClassifier(n_estimators=100, max_depth=5, max_features=0.5,
                                  class_weight="balanced",
                                  random_state=seed, n_jobs=-1)

# --- Repeat 5-fold CV on training set with different shuffles ---
# n_repeats=100 -> 100 different random fold assignments x 5 folds = 500 fits
rskf = RepeatedStratifiedKFold(n_splits=5, n_repeats=100, random_state=args.seed)
cv_rows = []
for k, (tr, va) in enumerate(rskf.split(X_tr, y_tr)):
    m = make_rf(args.seed).fit(X_tr[tr], y_tr[tr])      # fit on 4 folds
    prob = m.predict_proba(X_tr[va])[:, 1]               # P(Activity=1) on held-out fold
    cv_rows.append({
        "repeat": k // 5,                                # which shuffle (0-9)
        "fold": k % 5,
        "roc_auc": roc_auc_score(y_tr[va], prob),
        "f1": f1_score(y_tr[va], m.predict(X_tr[va])),
    })
cv_df = pd.DataFrame(cv_rows)

# Mean of each repeat's 5 folds; the spread of these = variability from fold assignment
rep_means = cv_df.groupby("repeat")[["roc_auc", "f1"]].mean()
print("Repeated CV on training set")
print(f"  ROC-AUC: {rep_means.roc_auc.mean():.3f} +/- {rep_means.roc_auc.std():.3f} (SD across repeats)")
print(f"  F1:      {rep_means.f1.mean():.3f} +/- {rep_means.f1.std():.3f}")


# In[22]:


# ===== Step 6: Train final model on training data for prediction =====
print("\n" + "="*60)
print("Step 6: Train final model on training data for prediction use")
print("="*60)
from sklearn.ensemble import RandomForestClassifier
import joblib
final_model = RandomForestClassifier(n_estimators=100, max_depth=5, max_features=0.5, class_weight='balanced', random_state=args.seed, n_jobs=-1)
final_model.fit(X[train_idx], y_dict["Activity"][train_idx])
final_model_path = os.path.join(model_dir, "final_rf_model.pkl")
joblib.dump(final_model, final_model_path)
print(f"Final model trained on training data and saved to: {final_model_path}")


# In[23]:


# ===== Step 7: Evaluate final model on training set =====

from sklearn.metrics import (roc_auc_score, average_precision_score, f1_score,
                             precision_score, recall_score, confusion_matrix,
                             ConfusionMatrixDisplay)
from sklearn.model_selection import cross_val_predict, StratifiedKFold

# (a) Resubstitution: the final model predicting the same molecules it trained on
pred_train = final_model.predict(X_tr)
cm_train = confusion_matrix(y_tr, pred_train)

# (b) Out-of-fold: each training molecule is predicted by a model that never saw it
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=args.seed)
pred_oof = cross_val_predict(final_model, X_tr, y_tr, cv=cv)   # clones and refits the model per fold
cm_oof = confusion_matrix(y_tr, pred_oof)

# Plot the two side by side
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
for ax, cm_, title in [(axes[0], cm_train, "Training set (resubstitution)"),
                       (axes[1], cm_oof,   "Training set (5-fold out-of-fold)")]:
    ConfusionMatrixDisplay(cm_, display_labels=["Inactive (0)", "Active (1)"]).plot(
        ax=ax, cmap="Blues", values_format="d", colorbar=False)
    ax.set_title(title)
plt.tight_layout()
plt.savefig(os.path.join(metric_dir, "train_confusion_matrices.png"), dpi=300, bbox_inches="tight")
plt.show()

print("Resubstitution  [TN FP / FN TP]:", cm_train.ravel())
print("Out-of-fold     [TN FP / FN TP]:", cm_oof.ravel())


# In[24]:


# ===== Step 8: Final evaluation on the held-out scaffold test set =====
from sklearn.metrics import (roc_auc_score, average_precision_score, f1_score,
                             precision_score, recall_score, confusion_matrix,
                             ConfusionMatrixDisplay)
import matplotlib.pyplot as plt

# Slice the full arrays with the scaffold-split indices
X_tr, y_tr = X[train_idx], y_dict["Activity"][train_idx].astype(int)
X_te, y_te = X[test_idx],  y_dict["Activity"][test_idx].astype(int)

# Train on the training set only
final_model = RandomForestClassifier(
    n_estimators=100, max_depth=5, max_features=0.5,
    class_weight="balanced", random_state=args.seed, n_jobs=-1,
).fit(X_tr, y_tr)

prob = final_model.predict_proba(X_te)[:, 1]   # P(Activity=1) for each test molecule
pred = final_model.predict(X_te)               # class labels at the default 0.5 threshold

# Threshold-free metrics use probabilities; F1/precision/recall use the labels
metrics = {
    "roc_auc":   roc_auc_score(y_te, prob),
    "pr_auc":    average_precision_score(y_te, prob),
    "f1":        f1_score(y_te, pred),
    "precision": precision_score(y_te, pred, zero_division=0),
    "recall":    recall_score(y_te, pred),
}
print(f"Test set: n={len(y_te)}, Activity=1 fraction={y_te.mean():.2f}")
for k, v in metrics.items():
    print(f"  {k}: {v:.3f}")

# Confusion matrix: rows = true class, columns = predicted class
cm = confusion_matrix(y_te, pred)
tn, fp, fn, tp = cm.ravel()
print(f"Confusion matrix: TN={tn}, FP={fp}, FN={fn}, TP={tp}")

# ----- Font sizes (edit these independently) -----
TITLE_FS  = 18    # plot title
LABEL_FS  = 17    # axis titles ("True label", "Predicted label")
TICK_FS   = 14    # class names on the axes ("Inactive (0)", "Active (1)")
NUMBER_FS = 20    # counts inside the grid cells

disp = ConfusionMatrixDisplay(cm, display_labels=["Inactive (0)", "Active (1)"])
disp.plot(cmap="Blues", values_format="d",
          text_kw={"fontsize": NUMBER_FS})            # grid number size

disp.ax_.set_title("Test set confusion matrix", fontsize=TITLE_FS)
disp.ax_.set_xlabel("Predicted label", fontsize=LABEL_FS)
disp.ax_.set_ylabel("True label", fontsize=LABEL_FS)
disp.ax_.tick_params(axis="both", labelsize=TICK_FS)  # class-name labels
disp.im_.colorbar.ax.tick_params(labelsize=TICK_FS)   # colorbar numbers

plt.savefig(os.path.join(metric_dir, "test_confusion_matrix.png"), dpi=300, bbox_inches="tight")
plt.show()

# Save metrics and the model
pd.DataFrame([{**metrics, "tn": tn, "fp": fp, "fn": fn, "tp": tp,
               "max_features": "0.5"}]).to_csv(
    os.path.join(metric_dir, "test_metrics.csv"), index=False)

# ===== Test-set predictions: correct vs incorrect =====
# Scaffold of every molecule (works for both scaffold and random splits)
scaf_map = generate_scaffolds(convert_smiles_to_mols(list(smiles)))
scaf_of = {i: (k if k else "(no ring)") for k, idxs in scaf_map.items() for i in idxs}

# Label each test molecule: TP/TN = correct, FP/FN = incorrect
outcome = np.where((pred == 1) & (y_te == 1), "TP",
          np.where((pred == 0) & (y_te == 0), "TN",
          np.where((pred == 1) & (y_te == 0), "FP", "FN")))

test_df = pd.DataFrame({
    "row_index": test_idx,                      # 0-based row position in CAR.csv (excluding header)
    "SMILES": smiles[test_idx],
    "scaffold": [scaf_of[i] for i in test_idx],
    "logP": logP[test_idx, 0],
    "True": y_te,
    "Pred": pred,
    "Prob": prob.round(3),                      # predicted P(Activity=1)
    "outcome": outcome,
    "correct": pred == y_te,
}).sort_values(["correct", "Prob"], ascending=[True, False])   # incorrect first, most confident first

test_df.to_csv(os.path.join(metric_dir, "test_predictions.csv"), index=False)

print(test_df["outcome"].value_counts().to_string())
print("\nIncorrect predictions:")
print(test_df[~test_df["correct"]].to_string(index=False))


# In[25]:


# ===== Step 9: Interpretability =====

# ============================================================
# SHAP analysis: which features drive "active" predictions (class 1)
# Explains final_model (Step 6, trained on the training set) using
# shap.TreeExplainer, computed on the TRAINING molecules only.
# Features: 512 ECFP4 bits (Bit_0 ... Bit_511) + logP (the last column).
# ============================================================

import shap
from tqdm import tqdm
from collections import defaultdict

N_TOP_BITS = 10        # how many top features to plot and summarize
TARGET_CLASS = 1       # 1 = "active"; set to 0 to see what drives "inactive"

# Feature names: column 512 is logP, not a fingerprint bit
feature_names = [f"Bit_{i}" for i in range(FP_NBITS)] + ["logP"]
assert len(feature_names) == X.shape[1]

# -----------------------------
# SHAP values on the training molecules, final model
# -----------------------------
X_shap = X[train_idx]                 # rows the final model was trained on
smiles_shap = smiles[train_idx]

print("Computing SHAP values with TreeExplainer...")
explainer = shap.TreeExplainer(final_model)
raw_shap = explainer.shap_values(X_shap, check_additivity=False)

# Handle both shap output formats (list of per-class arrays, or 3D array)
if isinstance(raw_shap, list):
    shap_values = raw_shap[TARGET_CLASS]
elif raw_shap.ndim == 3:
    shap_values = raw_shap[:, :, TARGET_CLASS]
else:
    shap_values = raw_shap

# Sanity check: base value + sum of SHAP values must equal the model's predicted P(class)
base = np.atleast_1d(explainer.expected_value)
base = base[TARGET_CLASS] if len(base) > 1 else base[0]
pred_p = final_model.predict_proba(X_shap)[:, TARGET_CLASS]
err = np.abs(shap_values.sum(axis=1) + base - pred_p).max()
print(f"SHAP additivity check, max error: {err:.2e} (should be ~0)")
assert err < 1e-6, "SHAP values do not add up to the model output; do not trust them"

# -----------------------------
# Summary plots
# -----------------------------
shap.summary_plot(shap_values, X_shap, feature_names=feature_names,
                  max_display=N_TOP_BITS, plot_size=(6, 6), show=False)
plt.tight_layout()
plt.savefig(os.path.join(shap_dir, "shap_summary_beeswarm_top10.png"), dpi=300, bbox_inches="tight")
plt.close()

shap.summary_plot(shap_values, X_shap, feature_names=feature_names,
                  plot_type="bar", max_display=N_TOP_BITS, show=False)
plt.tight_layout()
plt.savefig(os.path.join(shap_dir, "shap_importance_bar_top10.png"), dpi=300, bbox_inches="tight")
plt.close()
print(f"Summary plots saved to: {shap_dir}")

# -----------------------------
# Rank every feature by mean |SHAP|
# -----------------------------
importance_df = pd.DataFrame({
    "Feature": feature_names,
    "Column": range(X.shape[1]),
    "Mean_Abs_SHAP": np.abs(shap_values).mean(axis=0),
}).sort_values("Mean_Abs_SHAP", ascending=False).reset_index(drop=True)
importance_df.to_csv(os.path.join(shap_dir, "shap_bit_importance.csv"), index=False)
print(f"Full ranking saved to: {os.path.join(shap_dir, 'shap_bit_importance.csv')}")

# ============================================================
# Bit -> fragment dictionary (fingerprint bits only; logP is not a fragment)
# nBits MUST equal FP_NBITS (512), the size used to train the model; otherwise
# bit numbers refer to different fragments than the ones SHAP ranked.
# Scans training molecules only, to match the data SHAP explained.
# ============================================================
bit_to_fragments = defaultdict(lambda: defaultdict(lambda: [0, None]))

print("Scanning training molecules to build the bit -> fragment map...")
for smi in tqdm(smiles_shap):
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        continue
    bit_info = {}
    AllChem.GetMorganFingerprintAsBitVect(mol, RADIUS, nBits=FP_NBITS, bitInfo=bit_info)
    for bit_idx, envs in bit_info.items():
        for atom_idx, rad in envs:
            env = Chem.FindAtomEnvironmentOfRadiusN(mol, rad, atom_idx)
            submol = Chem.PathToSubmol(mol, env, atomMap={})
            if submol is None:
                continue
            entry = bit_to_fragments[bit_idx][Chem.MolToSmiles(submol)]
            entry[0] += 1
            if entry[1] is None:
                entry[1] = smi       # first molecule seen, kept as an example

all_bit_rows = []
for bit_idx in range(FP_NBITS):      # 0..511 only
    if bit_idx not in bit_to_fragments:
        all_bit_rows.append({"Bit": bit_idx, "Fragment_SMILES": "Not present in training set",
                             "Times_Seen": 0, "Example_SMILES": "N/A"})
        continue
    for frag_smi, (count, example_smi) in bit_to_fragments[bit_idx].items():
        all_bit_rows.append({"Bit": bit_idx, "Fragment_SMILES": frag_smi,
                             "Times_Seen": count, "Example_SMILES": example_smi})
all_bits_df = pd.DataFrame(all_bit_rows).sort_values(["Bit", "Times_Seen"], ascending=[True, False])
all_bits_df.to_csv(os.path.join(shap_dir, "all_bits_to_fragments.csv"), index=False)
print(f"Bit -> fragment map saved ({len(all_bits_df)} rows)")

# ============================================================
# Top features with direction and fragments in one table
# ============================================================
rows = []
for _, r in importance_df.head(N_TOP_BITS).iterrows():
    col = int(r["Column"])
    present = X_shap[:, col] > 0                     # molecules with this bit on
    frags = ""
    if col < FP_NBITS:                               # fingerprint bit -> look up fragments
        sub = all_bits_df[all_bits_df["Bit"] == col]
        frags = "; ".join(f"{f} (n={n})" for f, n in zip(sub["Fragment_SMILES"], sub["Times_Seen"]))
    rows.append({
        "Feature": r["Feature"],
        "Mean_Abs_SHAP": round(r["Mean_Abs_SHAP"], 4),
        # signed mean SHAP where the bit is present: >0 pushes toward class 1, <0 away from it
        "Mean_SHAP_when_present": round(shap_values[present, col].mean(), 4) if present.any() else np.nan,
        "N_train_with_bit": int(present.sum()),
        "Fragments": frags,
    })
top_df = pd.DataFrame(rows)
top_df.to_csv(os.path.join(shap_dir, "top_features_summary.csv"), index=False)
print(top_df.to_string(index=False))


# In[ ]:




