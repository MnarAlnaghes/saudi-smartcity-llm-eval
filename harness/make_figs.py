import sys, pandas as pd, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
res, out = sys.argv[1], sys.argv[2]
U = pd.read_csv(f"{res}/summary.csv")
get = lambda u, m: U[(U.unit == u) & (U.model == m)]["mean"].iloc[0]
NAMES = ["ALLaM-7B", "GPT-OSS-20B", "Gemma 4 12B", "Qwen3-8B", "Llama 3.1 8B"]
PARAMS = {"ALLaM-7B": 7.0, "GPT-OSS-20B": 21.0, "Gemma 4 12B": 12.0, "Qwen3-8B": 8.2, "Llama 3.1 8B": 8.0}
COL = {"ALLaM-7B": "#2a7f62", "GPT-OSS-20B": "#3b5bdb", "Gemma 4 12B": "#e8590c", "Qwen3-8B": "#9c36b5", "Llama 3.1 8B": "#868e96"}
matplotlib.rcParams.update({"font.size": 7.5, "pdf.fonttype": 42, "font.family": "DejaVu Sans"})
AX = [("KC", "Knowledge"), ("STR", "ST reasoning"), ("AC", "Computation"), ("SR", "Traffic\nsemantics"),
      ("DA", "Applications"), ("KSA-Knowledge", "Saudi\ngeography"), ("arabic_mmlu", "Arabic"),
      ("simpleqa", "Factuality"), ("ifeval", "Instruction\nfollowing"), ("humaneval", "Code"), ("xstest", "No over-\nrefusal")]
ang = np.linspace(0, 2*np.pi, len(AX), endpoint=False).tolist(); ang += ang[:1]
fig = plt.figure(figsize=(3.5, 3.7)); ax = plt.subplot(111, polar=True)
ax.set_theta_offset(np.pi/2); ax.set_theta_direction(-1)
for n in NAMES:
    v = [(1-get(u, n)) if u == "xstest" else get(u, n) for u, _ in AX]; v += v[:1]
    ax.plot(ang, v, lw=1.3, color=COL[n], label=n); ax.fill(ang, v, color=COL[n], alpha=0.04)
ax.set_xticks(ang[:-1]); ax.set_xticklabels([l for _, l in AX], fontsize=6.3)
ax.tick_params(axis="x", pad=6)
ax.set_ylim(0, 1); ax.set_yticks([.25, .5, .75, 1]); ax.set_yticklabels(["25", "50", "75", "100"], fontsize=5.5, color="#555")
ax.set_rlabel_position(16); ax.grid(color="#ccc", lw=0.5); ax.spines["polar"].set_color("#bbb")
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.08), ncol=3, fontsize=6.2, frameon=False, handlelength=1.5, columnspacing=1)
plt.tight_layout(); plt.savefig(f"{out}/radar.pdf", bbox_inches="tight"); plt.savefig(f"{out}/radar.png", dpi=150, bbox_inches="tight"); plt.close()

score = {n: np.mean([get("ST-overall", n), get("KSA-overall", n)]) for n in NAMES}
fig, ax = plt.subplots(figsize=(3.4, 2.3))
off = {"ALLaM-7B": (6, -9), "Llama 3.1 8B": (6, 4), "Qwen3-8B": (6, 3), "Gemma 4 12B": (6, -3), "GPT-OSS-20B": (-58, -10)}
for n in NAMES:
    x, y = PARAMS[n], 100*score[n]
    ax.scatter(x, y, s=22, color=COL[n], zorder=3)
    ax.annotate(n, (x, y), fontsize=6.3, xytext=off[n], textcoords="offset points", color="#222")
ax.set_xlabel("Parameters (billions, total)"); ax.set_ylabel("Mean urban accuracy (%)")
ax.set_xlim(5, 23); ax.set_ylim(33, 63)
ax.grid(alpha=0.3, lw=0.5); [ax.spines[k].set_visible(False) for k in ("top", "right")]
plt.tight_layout(); plt.savefig(f"{out}/pareto.pdf", bbox_inches="tight"); plt.savefig(f"{out}/pareto.png", dpi=150, bbox_inches="tight"); plt.close()
print({n: round(100*score[n], 1) for n in NAMES})
