import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

LGREY = "#EBEBEB"
GREY  = "#444444"
COLORS = ["#4C72B0", "#DD8452", "#C44E52", "#8172B2", "#55A868", "#937860"]
METHODS = ["Baseline", "AnyBCQ\n4-bit", "AnyBCQ\n3-bit", "AnyBCQ\n2-bit",
           "AWQ\n4-bit", "Wanda\n50%"]

plt.rcParams.update({
    "font.family":     "serif",
    "font.size":       10,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "figure.dpi":        150,
})


#data from paper
MMLU      = [74.95, 70.48, 66.56, 33.75, 73.94, 63.43]
GSM8K_STR = [75.59, 40.49,  1.14,  0.00, 80.59, 34.50]
VRAM      = [15.29,  7.00,  7.00,  7.00, 15.44, 15.29]
CKPT      = [15.27,  6.87,  6.87,  6.87,  5.68, 15.27]
DECODE    = [48.07, 55.79, 55.11, 55.07, 47.91, 48.23]


#quality bar chart
fig1, ax = plt.subplots(figsize=(7.5, 4.2))

x = np.arange(len(METHODS))
w = 0.35

bars1 = ax.bar(x - w/2, MMLU,      width=w, color=COLORS, edgecolor="white")
bars2 = ax.bar(x + w/2, GSM8K_STR, width=w, color=COLORS,
               edgecolor="white", hatch="//", alpha=0.75)

for b, v in zip(bars1, MMLU):
    ax.text(b.get_x() + b.get_width()/2, v + 1,
            f"{v:.1f}", ha="center", va="bottom", fontsize=7.5, color=GREY)
for b, v in zip(bars2, GSM8K_STR):
    ax.text(b.get_x() + b.get_width()/2, v + 1,
            f"{v:.1f}", ha="center", va="bottom", fontsize=7.5, color=GREY)

ax.set_xticks(x)
ax.set_xticklabels(METHODS, ha="center")
ax.set_ylabel("Accuracy (%)")
ax.set_ylim(0, 105)
ax.yaxis.grid(True, color=LGREY, zorder=0)
ax.set_axisbelow(True)

p1 = mpatches.Patch(facecolor="grey",             label="MMLU")
p2 = mpatches.Patch(facecolor="grey", hatch="//", label="GSM8K Strict", alpha=0.75)
ax.legend(handles=[p1, p2], frameon=False, loc="upper right")
ax.set_title("Downstream Quality by Compression Method  (higher is better)",
             pad=8, fontsize=11)

fig1.tight_layout()
fig1.savefig("fig61_quality.pdf", bbox_inches="tight")
print("fig61_quality.pdf saved")

#tradeoff scatter
fig2, ax = plt.subplots(figsize=(6.5, 4.2))

labels_sc = ["Baseline", "AnyBCQ 4-bit", "AnyBCQ 3-bit",
             "AnyBCQ 2-bit", "AWQ 4-bit", "Wanda 50%"]

for i, (lbl, col) in enumerate(zip(labels_sc, COLORS)):
    ax.scatter(VRAM[i], MMLU[i], color=col, s=180, zorder=5,
               edgecolors="white", linewidths=0.8)

label_xy = {
    "Baseline":      (15.29, 74.95, -1.6,  1.5),
    "AnyBCQ 4-bit":  ( 7.00, 70.48,  0.2,  1.5),
    "AnyBCQ 3-bit":  ( 7.00, 66.56,  0.2, -3.0),
    "AnyBCQ 2-bit":  ( 7.00, 33.75,  0.2,  1.5),
    "AWQ 4-bit":     (15.44, 73.94, -5.5, -3.5),
    "Wanda 50%":     (15.29, 63.43, -5.3,  1.5),
}
for lbl, col in zip(labels_sc, COLORS):
    x0, y0, dx, dy = label_xy[lbl]
    ax.text(x0 + dx, y0 + dy, lbl, fontsize=8.5, color=col,
            fontweight="bold", ha="left" if dx >= 0 else "right")

ax.axhline(MMLU[0], color=COLORS[0], lw=0.8, ls="--", alpha=0.5, zorder=1)
ax.axvline(VRAM[0], color=COLORS[0], lw=0.8, ls="--", alpha=0.5, zorder=1)

ax.fill_betweenx([MMLU[0], 85], 0, VRAM[0],
                 color="#55A868", alpha=0.07, zorder=0)
ax.text(3.2, 79, "low VRAM\nhigh accuracy", fontsize=8,
        color="#2A7D4F", style="italic", va="top")

ax.set_xlabel("Peak VRAM (GiB)  \u2190 lower is better")
ax.set_ylabel("MMLU Accuracy (%)  higher is better \u2192")
ax.set_xlim(2, 19)
ax.set_ylim(25, 85)
ax.yaxis.grid(True, color=LGREY, zorder=0)
ax.xaxis.grid(True, color=LGREY, zorder=0)
ax.set_axisbelow(True)
ax.set_title("Quality-Resource Trade-off: MMLU vs. Peak VRAM\n"
             "(TurboQuant excluded)", pad=8)

fig2.tight_layout()
fig2.savefig("fig62_tradeoff.pdf", bbox_inches="tight")
print("fig62_tradeoff.pdf saved")

#checkpoint and size
fig3, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(8.5, 4.0))
fig3.subplots_adjust(wspace=0.30)

x = np.arange(len(METHODS))
w = 0.55

#size
bars_l = ax_l.bar(x, CKPT, width=w, color=COLORS, edgecolor="white")
ax_l.set_xticks(x)
ax_l.set_xticklabels(METHODS, ha="center")
ax_l.set_ylabel("Checkpoint Size (GiB)")
ax_l.set_ylim(0, 18)
ax_l.yaxis.grid(True, color=LGREY, zorder=0)
ax_l.set_axisbelow(True)
ax_l.set_title("Checkpoint Size\n(lower is better)", pad=6)
for bar, val in zip(bars_l, CKPT):
    ax_l.text(bar.get_x() + bar.get_width()/2, val + 0.3,
              f"{val:.2f}", ha="center", va="bottom", fontsize=8, color=GREY)
ax_l.axhline(CKPT[0], color=COLORS[0], lw=1.0, ls="--", alpha=0.6,
             label=f"Baseline ({CKPT[0]:.2f} GiB)")
ax_l.legend(fontsize=8, frameon=False)


bars_r = ax_r.bar(x, DECODE, width=w, color=COLORS, edgecolor="white")
ax_r.set_xticks(x)
ax_r.set_xticklabels(METHODS, ha="center")
ax_r.set_ylabel("Decode Throughput (tok/s)")
ax_r.set_ylim(0, 70)
ax_r.yaxis.grid(True, color=LGREY, zorder=0)
ax_r.set_axisbelow(True)
ax_r.set_title("Decode Throughput\n(higher is better)", pad=6)
for bar, val in zip(bars_r, DECODE):
    ax_r.text(bar.get_x() + bar.get_width()/2, val + 0.5,
              f"{val:.1f}", ha="center", va="bottom", fontsize=8, color=GREY)
ax_r.axhline(DECODE[0], color=COLORS[0], lw=1.0, ls="--", alpha=0.6,
             label=f"Baseline ({DECODE[0]:.1f} tok/s)")
ax_r.legend(fontsize=8, frameon=False)

fig3.suptitle("Storage Footprint and Inference Throughput by Method\n"
              "(TurboQuant excluded)", y=1.02, fontsize=11)
fig3.tight_layout()
fig3.savefig("fig63_resources.pdf", bbox_inches="tight")
print("fig63_resources.pdf saved")
