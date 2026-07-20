import json, numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, confusion_matrix

plt.rcParams.update({"font.size": 11, "figure.dpi": 150})

r = json.load(open("results.json"))
art = np.load("artifacts.npz", allow_pickle=True)

# ---------- 1. Privacy-utility tradeoff ----------
sweep = r["dp_sgd_sweep"]
eps = [s["epsilon"] for s in sweep]
acc = [s["mean"]["accuracy"] for s in sweep]
acc_sd = [s["std"]["accuracy"] for s in sweep]
auc = [s["mean"]["roc_auc"] for s in sweep]
auc_sd = [s["std"]["roc_auc"] for s in sweep]
base_acc = r["baseline_nonprivate"]["mean"]["accuracy"]
base_auc = r["baseline_nonprivate"]["mean"]["roc_auc"]

fig, ax = plt.subplots(figsize=(6.5, 4.5))
ax.errorbar(eps, acc, yerr=acc_sd, marker="o", capsize=3, label="Accuracy (DP-SGD MLP)", color="#2b6cb0")
ax.errorbar(eps, auc, yerr=auc_sd, marker="s", capsize=3, label="ROC-AUC (DP-SGD MLP)", color="#c05621")
ax.axhline(base_acc, ls="--", color="#2b6cb0", alpha=0.6, label="Non-private accuracy")
ax.axhline(base_auc, ls="--", color="#c05621", alpha=0.6, label="Non-private ROC-AUC")
ax.set_xscale("log")
ax.set_xlabel(r"Privacy budget $\varepsilon$ (log scale, smaller = stronger privacy)")
ax.set_ylabel("Score")
ax.set_title("Privacy-Utility Trade-off: DP-SGD Optimized MLP")
ax.legend(fontsize=8, loc="lower right")
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig("fig_privacy_utility.png")
plt.close(fig)

# ---------- 2. ROC curve (non-private baseline) ----------
proba = art["proba_base"]; y_test = art["y_test"]
fpr, tpr, _ = roc_curve(y_test, proba)
fig, ax = plt.subplots(figsize=(5, 4.5))
ax.plot(fpr, tpr, color="#2b6cb0", lw=2, label=f"MLP (AUC = {base_auc:.3f})")
ax.plot([0, 1], [0, 1], ls="--", color="gray")
ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")
ax.set_title("ROC Curve - Non-Private Baseline MLP")
ax.legend(fontsize=9)
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig("fig_roc.png")
plt.close(fig)

# ---------- 3. Confusion matrix ----------
pred = (proba >= 0.5).astype(int)
cm = confusion_matrix(y_test, pred)
fig, ax = plt.subplots(figsize=(4.5, 4))
im = ax.imshow(cm, cmap="Blues")
for i in range(2):
    for j in range(2):
        ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                 color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=14)
ax.set_xticks([0, 1]); ax.set_xticklabels(["Rejected", "Approved"])
ax.set_yticks([0, 1]); ax.set_yticklabels(["Rejected", "Approved"])
ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
ax.set_title("Confusion Matrix - Non-Private Baseline")
fig.tight_layout()
fig.savefig("fig_confusion.png")
plt.close(fig)

# ---------- 4. Training loss curve ----------
loss_hist = art["loss_hist_base"]
fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(range(1, len(loss_hist) + 1), loss_hist, color="#2b6cb0", marker="o", ms=3)
ax.set_xlabel("Epoch (full-batch step)")
ax.set_ylabel("Binary cross-entropy loss")
ax.set_title("Training Loss - Non-Private Baseline MLP")
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig("fig_loss.png")
plt.close(fig)

# ---------- 5. Feature importance (permutation + integrated gradients) ----------
feat = list(art["feature_cols"])
perm = art["perm_imp"]
ig = art["ig_mean_abs"]

def sorted_bar(vals, title, fname, xlabel):
    order = np.argsort(vals)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.barh([feat[i] for i in order], [vals[i] for i in order], color="#2b6cb0")
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(fname)
    plt.close(fig)

sorted_bar(perm, "Permutation Importance (\u0394 ROC-AUC) - Non-Private MLP",
           "fig_perm_importance.png", "Mean AUC drop when feature is shuffled")
sorted_bar(ig, "Integrated Gradients - Mean |Attribution| - Non-Private MLP",
           "fig_integrated_gradients.png", "Mean |attribution| on 40 test samples")

# ---------- 6. Calibration curve ----------
bins = np.linspace(0, 1, 11)
bin_idx = np.digitize(proba, bins) - 1
bin_idx = np.clip(bin_idx, 0, 8)
obs, pred_mean = [], []
for b in range(9):
    mask = bin_idx == b
    if mask.sum() > 0:
        obs.append(y_test[mask].mean())
        pred_mean.append(proba[mask].mean())
fig, ax = plt.subplots(figsize=(5, 4.5))
ax.plot(pred_mean, obs, marker="o", color="#2b6cb0", label="MLP")
ax.plot([0, 1], [0, 1], ls="--", color="gray", label="Perfect calibration")
ax.set_xlabel("Mean predicted probability")
ax.set_ylabel("Observed approval rate")
ax.set_title("Calibration Curve - Non-Private Baseline MLP")
ax.legend(fontsize=9)
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig("fig_calibration.png")
plt.close(fig)

# ---------- 7. Class-balancing comparison bar chart ----------
cbc = r["class_balance_comparison"]
methods = list(cbc.keys())
metrics_to_plot = ["accuracy", "f1", "roc_auc"]
x = np.arange(len(methods)); width = 0.25
fig, ax = plt.subplots(figsize=(6.5, 4.5))
for i, m in enumerate(metrics_to_plot):
    vals = [cbc[meth][m] for meth in methods]
    ax.bar(x + i * width, vals, width, label=m)
ax.set_xticks(x + width)
ax.set_xticklabels(["No balancing", "SMOTE", "Class weighting"])
ax.set_ylabel("Score")
ax.set_title("Class-Imbalance Handling Strategies (Non-Private MLP)")
ax.legend(fontsize=9)
ax.grid(alpha=0.3, axis="y")
fig.tight_layout()
fig.savefig("fig_class_balance.png")
plt.close(fig)

print("All figures saved.")
