# Q1: JL for k-means clustering
import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from prep import load_data

k = 15
m_list = [5, 10, 15, 20, 25]
os.makedirs("results", exist_ok=True)

blue = "#2a78d6"
orange = "#eb6834"
grey = "#52514e"


# cost(D, B): each row goes to its closest centre, then add the squared distances
def get_cost(D, B):
    dist = (D ** 2).sum(axis=1).reshape(-1, 1) - 2 * D @ B.T + (B ** 2).sum(axis=1)
    dist = np.maximum(dist, 0)
    return float(dist.min(axis=1).sum())


# centres = average of the rows of X in each cluster
def get_centres(X, labels, k):
    centres = []
    for j in range(k):
        centres.append(X[labels == j].mean(axis=0))
    return np.array(centres)


D, y, cols, dropped = load_data()
n, d = D.shape
print("n =", n, " d =", d)

# small warm-up run (result not used) so that the first timing does not include library start-up
KMeans(n_clusters=k, n_init=5, random_state=0).fit(D[:20000])


# ---------------- (a) k-means on D ----------------
print("\n----- (a) -----")
start = time.perf_counter()
km = KMeans(n_clusters=k, n_init=5, random_state=0)
km.fit(D)
time_D = time.perf_counter() - start

B = km.cluster_centers_
labels_D = km.labels_
cost_D = get_cost(D, B)

print("cost(D, B) =", round(cost_D, 2))
print("inertia from sklearn =", round(km.inertia_, 2))
print("time taken =", round(time_D, 2), "sec")
print("cluster sizes =", np.bincount(labels_D))


# ---------------- (b), (c), (d) ----------------
print("\n----- (b), (c), (d) -----")
rng = np.random.default_rng(1)
rows = []
for m in m_list:
    for r in range(5):
        # random matrix A with N(0, 1/m) entries
        A = rng.normal(0, 1 / np.sqrt(m), size=(m, d))

        # (d) time = computing E + k-means on E
        start = time.perf_counter()
        E = D @ A.T
        km = KMeans(n_clusters=k, n_init=5, random_state=0)
        km.fit(E)
        t = time.perf_counter() - start

        # (b) B_hat: average the original rows of D in each cluster
        B_hat = get_centres(D, km.labels_, k)
        cost_hat = get_cost(D, B_hat)

        # (c) B_plus: map the centres back with the pseudo inverse of A
        A_pinv = np.linalg.pinv(A)
        B_plus = km.cluster_centers_ @ A_pinv.T
        cost_plus = get_cost(D, B_plus)

        rows.append([m, r + 1, cost_hat, cost_hat / cost_D, cost_plus, cost_plus / cost_D, t])
        print("m =", m, "run", r + 1, "| cost B_hat =", round(cost_hat, 2),
              "| ratio B_hat =", round(cost_hat / cost_D, 4),
              "| ratio B_plus =", round(cost_plus / cost_D, 2), "| time =", round(t, 2))

runs = pd.DataFrame(rows, columns=["m", "run", "cost_hat", "ratio_hat", "cost_plus", "ratio_plus", "time"])
runs.to_csv("results/q1_bcd_all_runs.csv", index=False)

summary = runs.groupby("m").agg(
    mean_ratio_hat=("ratio_hat", "mean"),
    min_ratio_hat=("ratio_hat", "min"),
    max_ratio_hat=("ratio_hat", "max"),
    mean_ratio_plus=("ratio_plus", "mean"),
    min_ratio_plus=("ratio_plus", "min"),
    max_ratio_plus=("ratio_plus", "max"),
    mean_time=("time", "mean"),
).reset_index()
summary.to_csv("results/q1_bcd_summary.csv", index=False)
print()
print(summary.round(4).to_string(index=False))

# plot for (b): mean ratio with min-max range
m_arr = summary["m"]
mean = summary["mean_ratio_hat"]
low = mean - summary["min_ratio_hat"]
high = summary["max_ratio_hat"] - mean

plt.figure(figsize=(6, 4))
plt.errorbar(m_arr, mean, yerr=[low, high], fmt="o-", color=blue, lw=2, ms=8, capsize=5,
             label="mean (bars = min to max of 5 runs)")
plt.axhline(1, color=grey, ls="--", lw=1, label="ratio = 1 (same cost as k-means on D)")
plt.xticks(m_arr)
plt.xlabel("m (projected dimension)")
plt.ylabel("cost(D, B_hat) / cost(D, B)")
plt.title("Q1(b): cost ratio of B_hat vs m")
plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig("results/q1b_ratio.png", dpi=150)

# plot for (d): time vs m
plt.figure(figsize=(6, 4))
plt.plot(m_arr, summary["mean_time"], "o-", color=blue, lw=2, ms=8, label="compute E + k-means on E (mean of 5)")
plt.axhline(time_D, color=orange, ls="--", lw=2, label="k-means on D, part (a) = " + str(round(time_D, 2)) + " s")
plt.xticks(m_arr)
plt.ylim(0, None)
plt.xlabel("m (projected dimension)")
plt.ylabel("wall-clock time (s)")
plt.title("Q1(d): running time vs m")
plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig("results/q1d_time.png", dpi=150)


# ---------------- (e) ----------------
print("\n----- (e) -----")
rng = np.random.default_rng(2)

# pick 10^4 random pairs (i, l) with D_i != D_l, same pairs for every m
pair_i = []
pair_l = []
while len(pair_i) < 10000:
    i = rng.integers(n)
    l = rng.integers(n)
    if np.any(D[i] != D[l]):
        pair_i.append(i)
        pair_l.append(l)
pair_i = np.array(pair_i)
pair_l = np.array(pair_l)
diff = D[pair_i] - D[pair_l]
dist_D = (diff ** 2).sum(axis=1)

# cost_D(C_D) for the clusters from (a)
mu_D = get_centres(D, labels_D, k)
cost_D_CD = ((D - mu_D[labels_D]) ** 2).sum()
print("cost_D(C_D) =", round(cost_D_CD, 2))

pair_ratios = {}
cost_ratios = {}
rows = []
for m in m_list:
    # (i) one draw of A, ratio for every pair
    A = rng.normal(0, 1 / np.sqrt(m), size=(m, d))
    dist_E = ((diff @ A.T) ** 2).sum(axis=1)
    pair_ratios[m] = dist_E / dist_D

    # (ii) 20 draws of A, ratio of the cost of the same clusters C_D
    ratios = []
    for draw in range(20):
        A = rng.normal(0, 1 / np.sqrt(m), size=(m, d))
        E = D @ A.T
        mu_E = get_centres(E, labels_D, k)
        cost_E = ((E - mu_E[labels_D]) ** 2).sum()
        ratios.append(cost_E / cost_D_CD)
    cost_ratios[m] = np.array(ratios)

    for name, r in [("pairs (i)", pair_ratios[m]), ("cost (ii)", cost_ratios[m])]:
        q1, q3 = np.percentile(r, [25, 75])
        rows.append([m, name, r.mean(), np.median(r), r.std(), q3 - q1, r.min(), r.max()])

stats = pd.DataFrame(rows, columns=["m", "type", "mean", "median", "std", "IQR", "min", "max"])
stats.to_csv("results/q1e_stats.csv", index=False)
print(stats.round(4).to_string(index=False))

# box plots of (i) and (ii) against m in one figure
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
pos = np.arange(len(m_list))

b1 = ax1.boxplot([pair_ratios[m] for m in m_list], positions=pos - 0.18, widths=0.3,
                 patch_artist=True, showfliers=False)
b2 = ax1.boxplot([cost_ratios[m] for m in m_list], positions=pos + 0.18, widths=0.3,
                 patch_artist=True, showfliers=False)
for box in b1["boxes"]:
    box.set_facecolor(blue)
for box in b2["boxes"]:
    box.set_facecolor(orange)
for line in b1["medians"] + b2["medians"]:
    line.set_color("black")
ax1.axhline(1, color=grey, ls="--", lw=1)
ax1.set_xticks(pos)
ax1.set_xticklabels(m_list)
ax1.set_xlabel("m")
ax1.set_ylabel("ratio")
ax1.set_title("(i) pair distance ratio vs (ii) cost ratio")
ax1.legend([b1["boxes"][0], b2["boxes"][0]],
           ["(i) ||E_i - E_l||^2 / ||D_i - D_l||^2  (10^4 pairs)", "(ii) cost_E(C_D) / cost_D(C_D)  (20 draws)"],
           loc="upper left", fontsize=8)
ax1.set_ylim(0, 3.1)
ax1.grid(alpha=0.3)

# zoom on (ii), its boxes are too small to see on the left
b3 = ax2.boxplot([cost_ratios[m] for m in m_list], positions=pos, widths=0.4, patch_artist=True)
for box in b3["boxes"]:
    box.set_facecolor(orange)
for line in b3["medians"]:
    line.set_color("black")
ax2.axhline(1, color=grey, ls="--", lw=1)
ax2.set_xticks(pos)
ax2.set_xticklabels(m_list)
ax2.set_xlabel("m")
ax2.set_ylabel("cost_E(C_D) / cost_D(C_D)")
ax2.set_title("zoom on (ii)")
ax2.grid(alpha=0.3)

plt.tight_layout()
plt.savefig("results/q1e_boxplots.png", dpi=150)
print("\nplots saved in results/")
