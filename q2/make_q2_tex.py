"""Write q2.tex, the LaTeX version of the Question 2 answers, from results/.

Run the notebook Q2_JL_linear_regression.ipynb first, then:
    python make_q2_tex.py <output folder>
The output folder gets q2.tex and the three figures. q2.tex is meant to be pulled into the
Question 1 Overleaf project with \\input{q2}, so it only uses packages main.tex already loads.
"""
import json
import shutil
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
RES, FIG = HERE / "results", HERE / "figures"
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / "overleaf_q2"
OUT.mkdir(parents=True, exist_ok=True)

R = json.loads((RES / "q2_results.json").read_text())
runs_b = pd.read_csv(RES / "q2b_all_runs.csv")
sum_b = pd.read_csv(RES / "q2b_summary.csv", index_col="m")
sum_c = pd.read_csv(RES / "q2c_times.csv", index_col="m")
sum_d = pd.read_csv(RES / "q2d_summary.csv", index_col="m")
runs_e = pd.read_csv(RES / "q2e_all_runs.csv")
sum_e = pd.read_csv(RES / "q2e_summary.csv", index_col="sketch")
full_e = pd.read_csv(RES / "q2e_full_D_check.csv", index_col=0)

n, d, r = R["n"], R["d"], R["rank_D"]
loss_b, time_a = R["loss_b"], R["time_a"]


def num(x, digits=2):
    # 2949.788 -> 2{,}949.79, the same thousands separator as the Q1 tables
    return f"{x:,.{digits}f}".replace(",", "{,}")


def sci(x, digits=1):
    # 9.31e-25 -> $9.3\times10^{-25}$
    mant, exp = f"{x:.{digits}e}".split("e")
    return f"${mant}\\times10^{{{int(exp)}}}$"


def small_or_num(x, digits=2):
    return sci(x) if x < 1e-6 else num(x, digits)


# ---------- tables ----------
rows = []
for m, grp in runs_b.groupby("m"):
    grp = grp.sort_values("rep")
    rows.append(f"\\multirow{{3}}{{*}}{{{m}}} & Loss$(D,y,\\tilde a)$ & " + " & ".join(num(x) for x in grp.loss) + " \\\\")
    rows.append("     & ratio & " + " & ".join(f"{x:.4f}" for x in grp.ratio) + " \\\\")
    rows.append("     & sketched loss & " + " & ".join(small_or_num(x) for x in grp.sketched_loss) + " \\\\")
    rows.append("\\midrule")
t_b_runs = "\n".join(rows[:-1])

rows = []
for m, q in sum_b.iterrows():
    ref = "--" if pd.isna(q.ref_ratio_gauss) else f"{q.ref_ratio_gauss:.3f}"
    sk = sci(q.sketched_over_loss_b) if q.sketched_over_loss_b < 1e-6 else f"{q.sketched_over_loss_b:.3f}"
    sk_ref = "0" if q.ref_sketched_over_loss_b_gauss == 0 else f"{q.ref_sketched_over_loss_b_gauss:.3f}"
    rows.append(f"{m} & {q.ratio_mean:.3f} & {q.ratio_min:.3f} & {q.ratio_max:.3f} & {ref} & {sk} & {sk_ref}"
                f" & {int(q.rank_SD)} & {q.smin_SD_mean:.3f} & {q.norm_a_mean:.1f} \\\\")
t_b_sum = "\n".join(rows)

rows = [f"{m} & {1e3 * q.t_draw:.1f} & {1e3 * q.t_form:.0f} & {1e3 * q.t_solve:.1f} & {q.t_total:.3f} & {q.speedup:.1f}$\\times$ \\\\"
        for m, q in sum_c.iterrows()]
t_c = "\n".join(rows)

rows = [f"{m} & {q['mean']:.4f} & {q['std']:.4f} & {q.theory_std:.4f} & {q['min']:.4f} & {q['max']:.4f} \\\\"
        for m, q in sum_d.iterrows()]
t_d = "\n".join(rows)

piv = runs_e.pivot(index="rep", columns="sketch", values=["t_sketch", "ratio"])
rows = [f"{rep} & {1e3 * piv.loc[rep, ('t_sketch', 'CountSketch')]:.1f} & {piv.loc[rep, ('ratio', 'CountSketch')]:.4f}"
        f" & {1e3 * piv.loc[rep, ('t_sketch', 'Gaussian')]:.1f} & {piv.loc[rep, ('ratio', 'Gaussian')]:.4f} \\\\" for rep in piv.index]
cs, ga = sum_e.loc["CountSketch"], sum_e.loc["Gaussian"]
rows.append("\\midrule")
rows.append(f"Mean & {1e3 * cs.time_mean:.1f} & {cs.ratio_mean:.4f} & {1e3 * ga.time_mean:.1f} & {ga.ratio_mean:.4f} \\\\")
t_e = "\n".join(rows)

fc, fg = full_e.loc["CountSketch"], full_e.loc["Gaussian"]
t_e_full = "\n".join([
    f"CountSketch & {fc.t_sketch:.2f} s & {fc.S_storage_MB:.1f} MB ($h$ and $\\sigma$) & {fc.ratio:.3f} \\\\",
    f"Dense Gaussian & {fg.t_sketch:.2f} s & {num(fg.S_storage_MB, 0)} MB & {fg.ratio:.3f} \\\\",
])

# ---------- numbers used in the text ----------
b50, b100 = sum_b.loc[50], sum_b.loc[100]
above = sum_b[sum_b.index > r]
below = runs_b[runs_b.m < r]
v = dict(
    n=num(n, 0), d=d, r=r, n_attack=num(R["n_attack"], 0), n_normal=num(n - R["n_attack"], 0),
    frac=f"{100 * R['frac_attack']:.1f}", seed=R["seed"],
    loss_b=num(loss_b, 2), time_a=f"{time_a:.2f}", times_a=", ".join(f"{t:.2f}" for t in R["times_a"][:-1]) + f" and {R['times_a'][-1]:.2f}",
    tss=num(R["tss"], 1), r2=f"{1 - loss_b / R['tss']:.3f}", norm_b=f"{R['norm_b']:.1f}", smin_D=f"{R['smin_D']:.3f}",
    sk_lo=sci(below.sketched_loss.min()), sk_hi=sci(below.sketched_loss.max()),
    below_lo=f"{below.ratio.min():.1f}", below_hi=f"{below.ratio.max():.1f}",
    r50=f"{b50.ratio_mean:.2f}", r50lo=f"{b50.ratio_min:.2f}", r50hi=f"{b50.ratio_max:.2f}",
    r100=f"{b100.ratio_mean:.1f}", r100lo=f"{b100.ratio_min:.1f}", r100hi=f"{b100.ratio_max:.1f}",
    smin50=f"{b50.smin_SD_mean:.1f}", smin100=f"{b100.smin_SD_mean:.2f}",
    na50=f"{b50.norm_a_mean:.1f}", na100=f"{b100.norm_a_mean:.1f}",
    m_above=f"$m = {above.index[0]}$, " + ", ".join(f"${m}$" for m in above.index[1:-1]) + f" and ${above.index[-1]}$",
    n_above=5 * len(above),
    ratios_above=" $\\to$ ".join(f"{q:.3f}" for q in above.ratio_mean),
    sk_above=" $\\to$ ".join(f"{q:.2f}" for q in above.sketched_over_loss_b),
    smin_first=f"{above.smin_SD_mean.iloc[0]:.3f}", smin_last=f"{above.smin_SD_mean.iloc[-1]:.3f}",
    m_first=above.index[0], m_last=above.index[-1],
    t_lo=f"{sum_c.t_total.min():.2f}", t_hi=f"{sum_c.t_total.max():.2f}",
    sp_lo=f"{sum_c.speedup.min():.1f}", sp_hi=f"{sum_c.speedup.max():.1f}",
    form_share=f"{100 * (sum_c.t_form / sum_c.t_total).min():.0f}", t_form=f"{sum_c.t_form.mean():.2f}",
    t_draw=f"{1e3 * sum_c.t_draw.mean():.0f}", ns_per=f"{1e9 * sum_c.t_form.mean() / (n * d):.0f}",
    solve_lo=f"{1e3 * sum_c.t_solve.min():.1f}", solve_hi=f"{1e3 * sum_c.t_solve.max():.1f}",
    nd=f"{n * d / 1e6:.1f}", r1000=f"{sum_b.loc[1000].ratio_mean:.3f}", r5000=f"{sum_b.loc[5000].ratio_mean:.3f}",
    sd50=f"{sum_d.loc[50, 'std']:.3f}", sd500=f"{sum_d.loc[500, 'std']:.3f}", sd5000=f"{sum_d.loc[5000, 'std']:.3f}",
    mn50=f"{sum_d.loc[50, 'min']:.2f}", mx50=f"{sum_d.loc[50, 'max']:.2f}",
    mn5000=f"{sum_d.loc[5000, 'min']:.3f}", mx5000=f"{sum_d.loc[5000, 'max']:.3f}",
    flat=sci(1 - R["v_flatness"]).strip("$"),
    r1=R["rank_D1"], loss_b1=num(R["loss_b1"], 2), ga_ref=f"{1 + R['rank_D1'] / (500 - R['rank_D1'] - 1):.3f}",
    cs_t=f"{1e3 * cs.time_mean:.0f}", ga_t=f"{1e3 * ga.time_mean:.0f}", e_speed=f"{ga.time_mean / cs.time_mean:.1f}",
    cs_r=f"{cs.ratio_mean:.3f}", ga_r=f"{ga.ratio_mean:.3f}",
    fc_t=f"{fc.t_sketch:.2f}", fg_t=f"{fg.t_sketch:.2f}", full_speed=f"{fg.t_sketch / fc.t_sketch:.1f}",
    fg_gb=f"{fg.S_storage_MB / 1e3:.2f}", fc_mb=f"{fc.S_storage_MB:.1f}", m_over_r=f"{500 / r:.1f}",
    cpu=R["processor"],
)

TEX = r"""% ================= QUESTION 2 =================
% This file is generated by make_q2_tex.py from the results of Q2_JL_linear_regression.ipynb.
\clearpage
\section*{Question 2: JL for linear regression}

We use the same $D$ as in Question 1 (see Data), with $n = @@n@@$ rows and $d = @@d@@$ columns. The label is
$y_i = 1$ if connection $i$ is an attack (its label is not \texttt{normal.}) and $y_i = 0$ otherwise,
which gives @@n_attack@@ attacks and @@n_normal@@ normal connections (@@frac@@\% attacks).
The notebook for this question builds $D$ with the same three steps, and we checked that the result is
identical to the output of \texttt{prep.py}.

Every least-squares problem is solved with \texttt{numpy.linalg.lstsq(M, z, rcond=None)}. The random
numbers come from \texttt{np.random.SeedSequence(@@seed@@).spawn(6)}: one generator each for (b), (d),
the subset in (e), the CountSketches in (e), the Gaussian sketches in (e), and the extra check on the full
$D$ in (e). Running the notebook again gives the same losses and ratios (except the times). The code is in
\texttt{q2/Q2\_JL\_linear\_regression.ipynb}. All times are on an @@cpu@@.

\textbf{Rank of $D$.} $\operatorname{rank}(D) = @@r@@$, not @@d@@, because of three exact linear relations
between the one-hot columns. The \texttt{protocol\_type}, \texttt{service} and \texttt{flag} columns each
add up to the all-ones vector (two independent relations), and \texttt{protocol\_type\_icmp} equals the sum
of the six ICMP-only service columns (\texttt{eco\_i}, \texttt{ecr\_i}, \texttt{red\_i}, \texttt{tim\_i},
\texttt{urh\_i}, \texttt{urp\_i}). The notebook checks all three. We write $r = \operatorname{rank}(D) = @@r@@$
and $L^* = \text{Loss}(D,y,b)$. Since the all-ones vector is in the column space of $D$, the model already
has an intercept.

\textbf{Why sketch-and-solve works.} Instead of solving $\min_a \|Da-y\|_2^2$, which has $n$ equations,
we solve $\min_a \|SDa-Sy\|_2^2$, which has only $m$. Every vector $Da-y$ lies in
$W = \operatorname{span}(\text{columns of } D,\ y)$, a fixed subspace of dimension at most $r+1$.
Suppose that for every $w \in W$,
$(1-\varepsilon)\|w\|^2 \le \|Sw\|^2 \le (1+\varepsilon)\|w\|^2$. Then
\[
\text{Loss}(D,y,\tilde a)
\le \frac{\|S(D\tilde a-y)\|^2}{1-\varepsilon}
\le \frac{\|S(Db-y)\|^2}{1-\varepsilon}
\le \frac{1+\varepsilon}{1-\varepsilon}\,\text{Loss}(D,y,b),
\]
where the middle step holds because $\tilde a$ minimises the sketched loss. So we need the JL property for
a whole subspace, not for $n$ fixed points. This is impossible when $m < r$, since some nonzero $Dz$ is then
mapped to $0$. A Gaussian $S$ works with $m = O((r+\log(1/\delta))/\varepsilon^2)$ and a CountSketch with
$m = O(r^2/(\delta\varepsilon^2))$ (Clarkson and Woodruff, 2013), but $SD$ for a CountSketch costs only one
pass over $D$.

% ---------------- (a) ----------------
\subsection*{(a)}

\begin{enumerate}
    \item Solve $b = \arg\min_a \|Da-y\|_2^2$ with \texttt{lstsq} on the full $D$. Because
    $\operatorname{rank}(D) = @@r@@ < @@d@@$, the minimiser is not unique. \texttt{lstsq} returns the one with
    the smallest norm, and the minimum loss is the same for all of them.
    \item Time the call with \texttt{time.perf\_counter}, 5 times, and take the mean.
\end{enumerate}

\begin{table}[H]
\centering
\begin{tabular}{lr}
\toprule
Loss$(D,y,b) = \|Db-y\|_2^2$ & @@loss_b@@ \\
Loss of the best constant, $\|y-\bar y\|_2^2$ & @@tss@@ \\
$\|b\|_2$ (minimum-norm solution) & @@norm_b@@ \\
Wall-clock time (mean of 5 runs) & @@time_a@@~s \\
\bottomrule
\end{tabular}
\caption{Least squares on the full $D$. The 5 times were @@times_a@@~s.}
\end{table}

Loss$(D,y,b) = @@loss_b@@$ and computing $b$ takes @@time_a@@~s. For scale, predicting the constant $\bar y$
gives a loss of @@tss@@, so the linear fit explains $1 - L^*/\|y-\bar y\|^2 = @@r2@@$ of the variance of $y$.

% ---------------- (b) ----------------
\subsection*{(b)}

\begin{enumerate}
    \item \textbf{Draw $S$.} For each row $i$, draw a bucket $h(i) \in \{1,\dots,m\}$ and a sign
    $\sigma(i) \in \{-1,+1\}$, uniformly and independently. $S$ is stored as these two arrays of length $n$,
    never as an $m \times n$ matrix.
    \item \textbf{Form $SD$ and $Sy$ in one pass.} Row $i$ of $D$, times $\sigma(i)$, is added to row $h(i)$ of
    $SD$ with \texttt{np.add.at(SD, h, sigma[:, None] * D)}, and the same for $y$. A test in the notebook checks
    this against an explicit dense $S$.
    \item \textbf{Solve} $\tilde a = \arg\min_a \|SDa-Sy\|_2^2$ with \texttt{lstsq}.
    \item \textbf{Record} Loss$(D,y,\tilde a)$, its ratio to $L^*$ and the sketched loss $\|SD\tilde a-Sy\|_2^2$.
    To explain the results we also record $\operatorname{rank}(SD)$, the smallest nonzero singular value
    $\sigma_{\min}(SD)$ and $\|\tilde a\|_2$.
    \item Repeat 5 times for each $m$, with a new $S$ each time (35 runs).
\end{enumerate}

\begin{table}[H]
\centering
\small
\begin{tabular}{clrrrrr}
\toprule
$m$ & & Run 1 & Run 2 & Run 3 & Run 4 & Run 5 \\
\midrule
@@t_b_runs@@
\bottomrule
\end{tabular}
\caption{Loss$(D,y,\tilde a)$, ratio $=$ Loss$(D,y,\tilde a)/L^*$ and sketched loss $\|SD\tilde a-Sy\|_2^2$
for each run ($L^* = @@loss_b@@$).}
\end{table}

\begin{table}[H]
\centering
\small
\setlength{\tabcolsep}{4.5pt}
\begin{tabular}{r rrrr rr rrr}
\toprule
 & \multicolumn{4}{c}{Loss$(D,y,\tilde a)/L^*$} & \multicolumn{2}{c}{sketched loss$/L^*$} & & & \\
\cmidrule(lr){2-5}\cmidrule(lr){6-7}
$m$ & Mean & Min & Max & Ref. & Mean & Ref. & $\operatorname{rank}(SD)$ & $\sigma_{\min}(SD)$ & $\|\tilde a\|_2$ \\
\midrule
@@t_b_sum@@
\bottomrule
\end{tabular}
\caption{Mean, smallest and largest ratio over the 5 runs, the mean sketched loss, and the mean of the three
extra quantities. The two ``Ref.'' columns are the exact means for a dense Gaussian $S$, derived at the end
of this part. For comparison, $\sigma_{\min}(D) = @@smin_D@@$ and $\|b\|_2 = @@norm_b@@$.}
\label{tab:q2b}
\end{table}

\begin{figure}[H]
\centering
\includegraphics[width=\textwidth]{q2b_ratio_loglog.png}
\caption{Left: mean ratio Loss$(D,y,\tilde a)/L^*$ against $m$ on log--log axes; the bars go from the smallest
to the largest of the 5 runs. Right: the sketched loss divided by $L^*$. The dashed line is at
$m = \operatorname{rank}(D) = @@r@@$.}
\end{figure}

\textbf{Below $m = \operatorname{rank}(D) = @@r@@$ ($m = 50, 100$).}
\begin{itemize}
    \item \textbf{The sketched loss is zero.} $SD$ is $m \times @@d@@$ with rank $m$ in every run, so the $m$
    equations $SDa = Sy$ can be solved exactly. The sketched loss is between @@sk_lo@@ and @@sk_hi@@, which is
    zero up to rounding, while the true loss is in the thousands.
    \item \textbf{The ratio is large and varies a lot}: @@r50@@ (@@r50lo@@ to @@r50hi@@) at $m = 50$ and
    @@r100@@ (@@r100lo@@ to @@r100hi@@) at $m = 100$. $S$ maps a subspace of $\operatorname{col}(D)$ of dimension
    at least $r-m$ to $0$, so the sketched problem cannot see those directions, and \texttt{lstsq} just returns
    the minimum-norm point among a whole family of exact solutions.
    \item \textbf{It gets worse as $m$ gets close to $r$.} At $m = 100$ the sketch has to fit 100 equations
    exactly with barely more useful unknowns (@@r@@), so the equations are close to dependent:
    $\sigma_{\min}(SD)$ falls from @@smin50@@ at $m = 50$ to @@smin100@@ at $m = 100$. Fitting the sketched noise
    exactly then needs large coefficients ($\|\tilde a\|_2$ grows from @@na50@@ to @@na100@@), and the true
    loss blows up. This peak just below $m = \operatorname{rank}(D)$ is known as the interpolation threshold
    (or ``double descent'').
\end{itemize}

\textbf{Above $m = \operatorname{rank}(D)$ (@@m_above@@).}
\begin{itemize}
    \item $\operatorname{rank}(SD) = @@r@@ = \operatorname{rank}(D)$ in all @@n_above@@ runs, so $S$ is one-to-one on
    $\operatorname{col}(D)$ and the sketched problem is overdetermined. As $m$ grows, $\sigma_{\min}(SD)$ gets
    close to $\sigma_{\min}(D) = @@smin_D@@$ (@@smin_first@@ at $m = @@m_first@@$, @@smin_last@@ at
    $m = @@m_last@@$), i.e.\ $S$ becomes a better subspace embedding.
    \item \textbf{The ratio falls steadily towards 1}: @@ratios_above@@, and the gap between the best and worst
    run shrinks too. It follows $1 + r/(m-r-1) \approx 1 + r/m$, so the extra loss roughly halves when $m$
    doubles. On the log--log plot the curve flattens out towards 1 from above.
    \item \textbf{No ratio is below 1}, in any of the 35 runs, and none can be: $b$ minimises
    Loss$(D,y,\cdot)$ over all $a$, so Loss$(D,y,\tilde a) \ge$ Loss$(D,y,b)$ for every $\tilde a$.
    \item \textbf{The sketched loss is no longer zero and rises towards $L^*$ from below.} Divided by $L^*$
    it goes @@sk_above@@, close to $(m-r)/m$. It is always an underestimate:
    $\|SD\tilde a-Sy\|^2 \le \|S(Db-y)\|^2 \approx \|Db-y\|^2$, and the $r$ fitted coefficients absorb about a
    fraction $r/m$ of the sketched residual.
\end{itemize}

\textbf{Where the reference values come from.} Take a dense Gaussian $S$ with $N(0,1/m)$ entries and
$m > r+1$. Let $U \in \mathbb{R}^{n\times r}$ be an orthonormal basis of $\operatorname{col}(D)$, write
$Db = Uc_b$, and let $v = Db-y$, which is orthogonal to $\operatorname{col}(D)$.
\begin{enumerate}
    \item Every fitted vector is $Da = Uc$, and by Pythagoras Loss$(D,y,a) = \|v\|^2 + \|c-c_b\|^2$.
    \item $Sy = SUc_b - Sv$, so the sketched solution is $\tilde c = c_b - (SU)^+Sv$ and
    Loss$(D,y,\tilde a) - L^* = \|(SU)^+Sv\|^2$.
    \item Because $U \perp v$, the Gaussian matrices $SU$ and $Sv \sim N(0, \tfrac{\|v\|^2}{m} I_m)$ are
    independent. Using $\mathbb{E}\big[((SU)^\top SU)^{-1}\big] = \tfrac{m}{m-r-1} I_r$ (inverse Wishart),
    $\mathbb{E}\|(SU)^+Sv\|^2 = \tfrac{\|v\|^2}{m}\operatorname{tr}\mathbb{E}\big[((SU)^\top SU)^{-1}\big]
    = \|v\|^2\,\tfrac{r}{m-r-1}$. So the mean ratio is $1 + \tfrac{r}{m-r-1}$.
    \item The sketched loss is $\|(I-P)Sv\|^2$, where $P$ is the projection onto the $r$-dimensional
    $\operatorname{col}(SU)$, which is independent of $Sv$. So its mean is $\|v\|^2\,\tfrac{m-r}{m}$.
\end{enumerate}
The CountSketch results in Table~\ref{tab:q2b} are close to both reference values.

% ---------------- (c) ----------------
\subsection*{(c)}

In each run of (b) we timed three steps separately: drawing $S$ ($h$ and $\sigma$), forming $SD$ and $Sy$,
and solving the $m \times @@d@@$ problem.

\begin{table}[H]
\centering
\begin{tabular}{rrrrrr}
\toprule
$m$ & Draw $S$ (ms) & Form $SD$, $Sy$ (ms) & Solve (ms) & Total (s) & Speed-up vs (a) \\
\midrule
@@t_c@@
\bottomrule
\end{tabular}
\caption{Time of sketch-and-solve, averaged over the 5 runs of (b). Computing $b$ in (a) took @@time_a@@~s.}
\end{table}

\begin{figure}[H]
\centering
\includegraphics[width=0.7\textwidth]{q2c_times.png}
\caption{Total sketch-and-solve time (mean of 5 runs) and the solve step alone, against the time of the full
solve from (a).}
\end{figure}

Sketch-and-solve takes @@t_lo@@--@@t_hi@@~s for every $m$, against @@time_a@@~s for the full solve, so it is
about @@sp_lo@@--@@sp_hi@@ times faster.
\begin{itemize}
    \item \textbf{Forming $SD$ and $Sy$ takes most of the time} (at least @@form_share@@\%, about @@t_form@@~s),
    and it does not depend on $m$. It is one pass over the $nd \approx @@nd@@$ million entries of $D$: each row is
    added once, wherever it lands.
    \item Drawing $S$ needs only $2n$ random numbers (about @@t_draw@@~ms). Solving the small problem costs
    $O(md^2)$. It grows with $m$ (@@solve_lo@@ to @@solve_hi@@~ms) but stays a few percent of the total.
    \item The full solve costs $O(nd^2) \approx 6.6\times10^9$ operations, against $O(nd + md^2)$ for the
    sketch, so we expected a larger speed-up. It is only @@sp_lo@@--@@sp_hi@@ times because $d = @@d@@$ is small,
    the SVD inside \texttt{lstsq} is very fast, and \texttt{np.add.at} works one element at a time (about
    @@ns_per@@ ns per entry). The gap would be larger for a bigger $d$, or for a sparse $D$, where the sketch
    costs $O(\operatorname{nnz}(D))$.
    \item Together with (b): in about a third of the time, $m = 1000$ gives a loss within @@r1000@@ times the
    optimum and $m = 5000$ within @@r5000@@ times.
\end{itemize}

% ---------------- (d) ----------------
\subsection*{(d)}

\begin{enumerate}
    \item Compute $v = Db-y$ once, so $\|v\|^2 = L^*$. It is fixed before any $S$ is drawn, which is exactly
    the setting of the JL lemma.
    \item For $m \in \{50, 500, 5000\}$, draw 100 independent CountSketches, form $Sv$ in one pass, and record
    $\|Sv\|_2^2/\|v\|_2^2$.
    \item Plot a histogram for each $m$ on the same $x$-axis, so the spreads can be compared directly.
\end{enumerate}

\begin{figure}[H]
\centering
\includegraphics[width=\textwidth]{q2d_histograms.png}
\caption{Histograms of the 100 values of $\|Sv\|^2/\|v\|^2$ for each $m$. The dashed line is at 1.}
\end{figure}

\begin{table}[H]
\centering
\begin{tabular}{rrrrrr}
\toprule
$m$ & Mean & Std & Theory std & Min & Max \\
\midrule
@@t_d@@
\bottomrule
\end{tabular}
\caption{Summary of $\|Sv\|^2/\|v\|^2$ over 100 sketches. The theory std is derived below.}
\end{table}

\textbf{Mean and variance for a fixed $v$.} Write $\delta_{ik} = \mathbf{1}[h(i)=h(k)]$. Then
\[
\|Sv\|^2 = \sum_{j=1}^m \Big(\sum_{i:\,h(i)=j} \sigma_i v_i\Big)^2
= \sum_i v_i^2 + \sum_{i\ne k} \delta_{ik}\,\sigma_i\sigma_k\,v_i v_k .
\]
Since $\mathbb{E}[\sigma_i\sigma_k] = 0$ for $i \ne k$, $\mathbb{E}\|Sv\|^2 = \|v\|^2$, so CountSketch is
unbiased. The cross terms of different pairs are uncorrelated (because of the random signs) and
$\Pr[h(i)=h(k)] = 1/m$, so
\[
\operatorname{Var}\|Sv\|^2 = 4\sum_{i<k}\frac{v_i^2v_k^2}{m} = \frac{2}{m}\big(\|v\|_2^4 - \|v\|_4^4\big),
\qquad
\operatorname{std}\frac{\|Sv\|^2}{\|v\|^2} = \sqrt{\frac{2}{m}\Big(1 - \frac{\|v\|_4^4}{\|v\|_2^4}\Big)}
\approx \sqrt{\frac{2}{m}} .
\]
Here $\|v\|_4^4/\|v\|_2^4 = @@flat@@$ ($v$ is spread over many rows), so the theory std is $\sqrt{2/m}$ in
practice.

\textbf{Results.} The mean is 1 up to sampling error for every $m$. The standard deviation is @@sd50@@,
@@sd500@@ and @@sd5000@@ for $m = 50$, $500$ and $5000$, close to $\sqrt{2/m} = 0.200$, $0.063$ and $0.020$.

\textbf{How the spread changes with $m$.} The spread shrinks like $1/\sqrt{m}$: every time $m$ grows 10 times,
the histogram gets about $\sqrt{10} \approx 3.2$ times narrower. The range goes from
@@mn50@@--@@mx50@@ at $m = 50$ to @@mn5000@@--@@mx5000@@ at $m = 5000$. By Chebyshev,
$\Pr\big[\,\big|\|Sv\|^2/\|v\|^2 - 1\big| > \varepsilon\big] \le 2/(m\varepsilon^2)$, so
$m = 2/(\delta\varepsilon^2)$ rows keep the length of one vector within $1 \pm \varepsilon$ with probability
$1-\delta$. This is the JL guarantee of CountSketch. Its dependence on $\delta$ is worse than the
$\log(1/\delta)$ of a Gaussian, which is the price of having only one nonzero per column.

Compare this with (b): there $S$ has to preserve a whole @@r@@-dimensional subspace at once, not one vector.
That is why (b) needs $m$ much larger than $r$, while here $m = 50$ is already fairly accurate.

% ---------------- (e) ----------------
\subsection*{(e)}

\begin{enumerate}
    \item \textbf{Choose the subset once.} Draw 50{,}000 row indices uniformly without replacement with a
    fixed-seed generator. $D'$ is $50{,}000 \times @@d@@$ with $\operatorname{rank}(D') = @@r1@@$, because some rare
    categories do not occur in the subset. The same subset is used for both sketches and all 5 runs.
    \item \textbf{Full solution on the subset.} $b' = \arg\min_a \|D'a-y'\|^2$ with \texttt{lstsq} gives
    Loss$(D',y',b') = @@loss_b1@@$.
    \item \textbf{Sketch with $m = 500$}, with a new $S$ in each of 5 runs: (i) a CountSketch as in (b); (ii) a
    dense $S \in \mathbb{R}^{500\times 50000}$ with i.i.d.\ $N(0,1/m)$ entries
    (\texttt{rng.normal(0, 1/np.sqrt(m), size=(m, 50000))}), with $SD'$ and $Sy'$ computed by matrix products.
    We time drawing $S$ and forming $SD'$ and $Sy'$, then solve for $\tilde a$ and compute
    Loss$(D',y',\tilde a)/\text{Loss}(D',y',b')$.
\end{enumerate}

\begin{table}[H]
\centering
\begin{tabular}{lrrrr}
\toprule
 & \multicolumn{2}{c}{CountSketch} & \multicolumn{2}{c}{Gaussian} \\
\cmidrule(lr){2-3}\cmidrule(lr){4-5}
Run & Time (ms) & Ratio & Time (ms) & Ratio \\
\midrule
@@t_e@@
\bottomrule
\end{tabular}
\caption{Time to draw $S$ and form $SD'$, $Sy'$, and the ratio Loss$(D',y',\tilde a)/\text{Loss}(D',y',b')$,
on the 50{,}000-row subset with $m = 500$. The Gaussian theory gives a mean ratio of
$1 + r'/(m-r'-1) = @@ga_ref@@$, where $r' = \operatorname{rank}(D') = @@r1@@$.}
\end{table}

As an extra check for the last question, we also ran each sketch once on the full $D$ with $m = 500$:

\begin{table}[H]
\centering
\small
\begin{tabular}{lccc}
\toprule
Sketch on the full $D$ & Time to draw $S$, form $SD$, $Sy$ & Storage for $S$ & Ratio \\
\midrule
@@t_e_full@@
\bottomrule
\end{tabular}
\caption{One run of each sketch on the full $D$, $m = 500$. Ratio $=$ Loss$(D,y,\tilde a)/L^*$.}
\end{table}

\textbf{Comparison.} On the 50{,}000-row subset the two sketches are about equally accurate (mean ratio
@@cs_r@@ for CountSketch and @@ga_r@@ for the Gaussian, against @@ga_ref@@ from the Gaussian theory), but
CountSketch is @@e_speed@@ times faster to draw and apply (@@cs_t@@~ms against @@ga_t@@~ms). The Gaussian's
time goes into drawing $500 \cdot 50{,}000 = 2.5\times10^7$ normal numbers and the dense product
($m \cdot 50{,}000 \cdot d \approx 2.9\times10^9$ operations). CountSketch needs $10^5$ random numbers and one
pass over the $5.8\times10^6$ entries of $D'$.

\textbf{Which one we would use on the full $D$: CountSketch.}
\begin{itemize}
    \item \textbf{Time.} A dense $S$ for the full $D$ is $500 \times @@n@@$, so applying it costs
    $m\,n\,d \approx 2.9\times10^{10}$ operations, against $nd \approx 5.7\times10^7$ for CountSketch. We measured
    @@fc_t@@~s against @@fg_t@@~s (@@full_speed@@ times faster). The gap is smaller than the operation count
    suggests only because dense matrix products are very fast.
    \item \textbf{Memory.} The dense $S$ alone takes @@fg_gb@@ GB in float64. CountSketch only stores $h$ and
    $\sigma$ (@@fc_mb@@ MB).
    \item \textbf{Accuracy.} At $m = 500 \approx @@m_over_r@@\,r$ both have about the same accuracy (see the two
    tables above).
    \item \textbf{One pass.} Each row only needs its own $h(i)$ and $\sigma(i)$, so $SD$ can be built while
    reading the rows one by one. For sparse data it costs $O(\operatorname{nnz}(D))$.
\end{itemize}
One caveat: the worst-case theory needs a larger $m$ for CountSketch ($O(r^2/\varepsilon^2)$) than for a
Gaussian ($O(r/\varepsilon^2)$). If a guarantee at a small $m$ were needed, one could apply a CountSketch first
and then a small Gaussian sketch to $SD$, which keeps the speed of both.

\textbf{Sources used for Question 2.}
\begin{itemize}
    \item CSE558 Lectures 18--20 (the JL lemma and JL for $k$-means).
    \item K. L. Clarkson and D. P. Woodruff, \emph{Low rank approximation and regression in input sparsity
    time}, STOC 2013. \url{https://arxiv.org/abs/1207.6365}
    \item D. P. Woodruff, \emph{Sketching as a tool for numerical linear algebra}, 2014.
    \url{https://arxiv.org/abs/1411.4357}
\end{itemize}

"""

fills = {"t_b_runs": t_b_runs, "t_b_sum": t_b_sum, "t_c": t_c, "t_d": t_d, "t_e": t_e, "t_e_full": t_e_full,
         **{k: str(x) for k, x in v.items()}}
out = TEX
for k, x in fills.items():
    out = out.replace(f"@@{k}@@", x)
assert "@@" not in out, [l for l in out.splitlines() if "@@" in l][:3]
(OUT / "q2.tex").write_text(out)
for png in ["q2b_ratio_loglog.png", "q2c_times.png", "q2d_histograms.png"]:
    shutil.copy(FIG / png, OUT / png)
print("wrote", OUT / "q2.tex", "and 3 figures")
