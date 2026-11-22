"""Build the Question 2 answer sheet (Q2_report.html and Q2_report.pdf) from results/ and figures/.

Run the notebook Q2_JL_linear_regression.ipynb first, then:
    python make_report.py
Edit GROUP / DISCUSSED_WITH below before you submit. The PDF is printed with headless Google Chrome;
if Chrome is not found, open Q2_report.html in a browser and use Print -> Save as PDF.
"""
import json
import shutil
import subprocess
import tempfile
from pathlib import Path

import pandas as pd

GROUP = "Roll No. 1 &ndash; Name 1 &nbsp;&middot;&nbsp; Roll No. 2 &ndash; Name 2"
DISCUSSED_WITH = "&mdash; (add the names of any groups you discussed with)"

HERE = Path(__file__).resolve().parent
RES, FIG = HERE / "results", HERE / "figures"
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


def sci(x, digits=2):
    # 1.23e-24 -> 1.23 &times; 10<sup>-24</sup>
    mant, exp = f"{x:.{digits}e}".split("e")
    return f"{mant}&thinsp;&times;&thinsp;10<sup>{int(exp)}</sup>"


def num(x, digits=2):
    return f"{x:,.{digits}f}"


def table(header, rows, cls=""):
    th = "".join(f"<th>{h}</th>" for h in header)
    trs = "".join("<tr>" + "".join(f"<td>{c}</td>" for c in row) + "</tr>" for row in rows)
    return f'<table class="{cls}"><thead><tr>{th}</tr></thead><tbody>{trs}</tbody></table>'


# ---------- tables ----------
t_data = table(["quantity", "value"], [
    ["rows n (connections)", f"{n:,}"],
    ["columns after one-hot encoding (38 numeric + 3 + 66 + 11)", R["n_onehot"]],
    ["constant columns dropped in step (ii)", ", ".join(f"<code>{c}</code>" for c in R["constant_cols"])],
    ["columns d of D", d],
    ["rank(D)", r],
    ["attacks (y = 1) / normal (y = 0)", f"{R['n_attack']:,} / {n - R['n_attack']:,} ({100 * R['frac_attack']:.1f}% attacks)"],
], "kv")

t_a = table(["quantity", "value"], [
    ["Loss(D, y, b) = &#8214;Db &minus; y&#8214;<sub>2</sub><sup>2</sup>", f"<b>{num(loss_b, 4)}</b>"],
    ["mean squared residual Loss / n", f"{loss_b / n:.6f} (RMS {(loss_b / n) ** 0.5:.4f})"],
    ["&#8214;y &minus; &#563;&#8214;<sup>2</sup> (loss of the best constant)", num(R["tss"], 1)],
    ["&#8214;b&#8214;<sub>2</sub> (minimum-norm solution)", num(R["norm_b"], 2)],
    ["time to compute b with <code>lstsq</code>, 5 runs", ", ".join(f"{t:.3f}" for t in R["times_a"]) + " s"],
    ["mean time", f"<b>{time_a:.3f} s</b>"],
], "kv")

rows = []
for _, q in runs_b.iterrows():
    sk = sci(q.sketched_loss) if q.sketched_loss < 1e-6 else num(q.sketched_loss, 1)
    rows.append([int(q.m), int(q.rep), num(q.loss, 1), f"{q.ratio:.4f}", sk, int(q.rank_SD)])
t_b_all = table(["m", "run", "Loss(D, y, &atilde;)", "ratio to Loss(D, y, b)", "sketched loss &#8214;SD&atilde; &minus; Sy&#8214;<sup>2</sup>", "rank(SD)"], rows, "small breakable")

rows = []
for m, q in sum_b.iterrows():
    sk = sci(q.sketched_loss_mean) if q.sketched_loss_mean < 1e-6 else num(q.sketched_loss_mean, 1)
    ref = "&mdash;" if pd.isna(q.ref_ratio_gauss) else f"{q.ref_ratio_gauss:.3f}"
    rows.append([m, f"<b>{q.ratio_mean:.3f}</b>", f"{q.ratio_min:.3f}", f"{q.ratio_max:.3f}", ref, sk,
                 f"{q.sketched_over_loss_b:.3f}", f"{q.ref_sketched_over_loss_b_gauss:.3f}",
                 int(q.rank_SD), f"{q.smin_SD_mean:.3f}", f"{q.norm_a_mean:.1f}"])
t_b_sum = table(["m", "mean ratio", "min", "max", "ref. 1+r/(m&minus;r&minus;1)", "mean sketched loss",
                 "sketched / Loss(b)", "ref. (m&minus;r)/m", "rank(SD)", "&sigma;<sub>min</sub>(SD)", "&#8214;&atilde;&#8214;"], rows, "small")

rows = [[m, f"{1e3 * q.t_draw:.1f}", f"{1e3 * q.t_form:.0f}", f"{1e3 * q.t_solve:.1f}", f"<b>{q.t_total:.3f}</b>",
         f"{q.speedup:.1f}&times;"] for m, q in sum_c.iterrows()]
t_c = table(["m", "draw S (ms)", "form SD, Sy (ms)", "solve (ms)", "total (s)", "speed-up vs (a)"], rows)

rows = [[m, f"{q['mean']:.4f}", f"<b>{q['std']:.4f}</b>", f"{q.theory_std:.4f}", f"{q['min']:.3f}", f"{q['max']:.3f}"]
        for m, q in sum_d.iterrows()]
t_d = table(["m", "mean", "std", "theory std &radic;(2/m &middot; (1 &minus; &#8214;v&#8214;<sub>4</sub><sup>4</sup>/&#8214;v&#8214;<sub>2</sub><sup>4</sup>))", "min", "max"], rows)

piv = runs_e.pivot(index="rep", columns="sketch", values=["t_sketch", "ratio"])
rows = [[rep, f"{1e3 * piv.loc[rep, ('t_sketch', 'CountSketch')]:.1f}", f"{piv.loc[rep, ('ratio', 'CountSketch')]:.4f}",
         f"{1e3 * piv.loc[rep, ('t_sketch', 'Gaussian')]:.1f}", f"{piv.loc[rep, ('ratio', 'Gaussian')]:.4f}"] for rep in piv.index]
cs, ga = sum_e.loc["CountSketch"], sum_e.loc["Gaussian"]
rows.append(["<b>mean</b>", f"<b>{1e3 * cs.time_mean:.1f}</b>", f"<b>{cs.ratio_mean:.4f}</b>",
             f"<b>{1e3 * ga.time_mean:.1f}</b>", f"<b>{ga.ratio_mean:.4f}</b>"])
t_e = table(["run", "CountSketch: time (ms)", "CountSketch: ratio", "Gaussian: time (ms)", "Gaussian: ratio"], rows)

fc, fg = full_e.loc["CountSketch"], full_e.loc["Gaussian"]
t_e_full = table(["sketch on the full D (m = 500, one run)", "time to draw S and form SD, Sy", "storage for S", "ratio Loss(D,y,&atilde;)/Loss(D,y,b)"], [
    ["CountSketch", f"{fc.t_sketch:.2f} s", f"{fc.S_storage_MB:.1f} MB (h and &sigma;)", f"{fc.ratio:.3f}"],
    ["dense Gaussian", f"{fg.t_sketch:.2f} s", f"{fg.S_storage_MB:,.0f} MB", f"{fg.ratio:.3f}"],
], "fl")

# ---------- numbers used in the text ----------
b50, b100 = sum_b.loc[50], sum_b.loc[100]
above = sum_b[sum_b.index > r]
v = dict(
    n=f"{n:,}", d=d, r=r, nd=f"{n * d / 1e6:.1f}", loss_b=num(loss_b, 2), time_a=f"{time_a:.2f}",
    r2=f"{1 - loss_b / R['tss']:.3f}", tss=num(R["tss"], 1), smin_D=f"{R['smin_D']:.3f}", norm_b=f"{R['norm_b']:.1f}",
    m_over_r=f"{500 / r:.1f}", t_draw=f"{1e3 * sum_c.t_draw.mean():.0f}",
    below_lo=f"{runs_b[runs_b.m < r].ratio.min():.1f}", below_hi=f"{runs_b[runs_b.m < r].ratio.max():.1f}",
    r50=f"{b50.ratio_mean:.2f}", r50lo=f"{b50.ratio_min:.2f}", r50hi=f"{b50.ratio_max:.2f}",
    r100=f"{b100.ratio_mean:.1f}", r100lo=f"{b100.ratio_min:.1f}", r100hi=f"{b100.ratio_max:.1f}",
    smin50=f"{b50.smin_SD_mean:.1f}", smin100=f"{b100.smin_SD_mean:.2f}",
    na50=f"{b50.norm_a_mean:.1f}", na100=f"{b100.norm_a_mean:.1f}",
    sk_lo=sci(runs_b[runs_b.m < r].sketched_loss.min(), 1), sk_hi=sci(runs_b[runs_b.m < r].sketched_loss.max(), 1),
    ratios_above=" &rarr; ".join(f"{q:.3f}" for q in above.ratio_mean),
    m_above=", ".join(str(m) for m in above.index),
    sk_above=" &rarr; ".join(f"{q:.2f}" for q in above.sketched_over_loss_b),
    smin_first=f"{above.smin_SD_mean.iloc[0]:.3f}", smin_last=f"{above.smin_SD_mean.iloc[-1]:.4f}",
    m_first=above.index[0], m_last=above.index[-1],
    t_lo=f"{sum_c.t_total.min():.2f}", t_hi=f"{sum_c.t_total.max():.2f}",
    sp_lo=f"{sum_c.speedup.min():.1f}", sp_hi=f"{sum_c.speedup.max():.1f}",
    form_share=f"{100 * (sum_c.t_form / sum_c.t_total).min():.0f}",
    t_form=f"{sum_c.t_form.mean():.2f}", ns_per=f"{1e9 * sum_c.t_form.mean() / (n * d):.0f}",
    solve_lo=f"{1e3 * sum_c.t_solve.min():.1f}", solve_hi=f"{1e3 * sum_c.t_solve.max():.1f}",
    r1000=f"{sum_b.loc[1000].ratio_mean:.3f}", r5000=f"{sum_b.loc[5000].ratio_mean:.3f}",
    sd50=f"{sum_d.loc[50, 'std']:.3f}", sd500=f"{sum_d.loc[500, 'std']:.3f}", sd5000=f"{sum_d.loc[5000, 'std']:.3f}",
    mn50=f"{sum_d.loc[50, 'min']:.2f}", mx50=f"{sum_d.loc[50, 'max']:.2f}",
    mn5000=f"{sum_d.loc[5000, 'min']:.3f}", mx5000=f"{sum_d.loc[5000, 'max']:.3f}",
    flat=(lambda x: f"{x.split('e')[0]}\\times10^{{{int(x.split('e')[1])}}}")(f"{1 - R['v_flatness']:.1e}"),
    r1=R["rank_D1"], loss_b1=num(R["loss_b1"], 2),
    cs_t=f"{1e3 * cs.time_mean:.0f}", ga_t=f"{1e3 * ga.time_mean:.0f}", e_speed=f"{ga.time_mean / cs.time_mean:.1f}",
    cs_r=f"{cs.ratio_mean:.3f}", ga_r=f"{ga.ratio_mean:.3f}", ga_ref=f"{1 + R['rank_D1'] / (500 - R['rank_D1'] - 1):.3f}",
    fc_t=f"{fc.t_sketch:.2f}", fg_t=f"{fg.t_sketch:.2f}", fg_mb=f"{fg.S_storage_MB / 1e3:.2f}", fc_mb=f"{fc.S_storage_MB:.1f}",
    cpu=R["processor"], numpy=R["numpy"], seed=R["seed"],
)
v["full_speed"] = f"{fg.t_sketch / fc.t_sketch:.1f}"

HTML = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>CSE558 Assignment Q2</title>
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/katex.min.css">
<script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/katex.min.js"></script>
<script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/contrib/auto-render.min.js"
  onload="renderMathInElement(document.body,{delimiters:[{left:'$$',right:'$$',display:true},{left:'$',right:'$',display:false}],throwOnError:false})"></script>
<style>
  @page { size: A4; margin: 15mm 15mm 16mm; }
  :root { --ink:#1b1b1a; --ink2:#52514e; --rule:#dcdbd6; --tint:#f4f4f1; --accent:#2a78d6; }
  html { background:#fff; }
  body { font-family: -apple-system, "Helvetica Neue", Arial, sans-serif; font-size: 10.2pt; line-height: 1.45;
         color: var(--ink); background:#fff; max-width: 185mm; margin: 0 auto; padding: 0 4mm; }
  h1 { font-size: 17pt; margin: 0 0 2mm; }
  h2 { font-size: 13pt; margin: 7mm 0 2mm; padding-bottom: 1mm; border-bottom: 1.5px solid var(--ink); break-after: avoid; }
  h3 { font-size: 10.8pt; margin: 4mm 0 1.5mm; break-after: avoid; }
  p { margin: 1.5mm 0; }
  .meta { color: var(--ink2); font-size: 9.4pt; border-bottom: 1px solid var(--rule); padding-bottom: 2mm; margin-bottom: 3mm; }
  .meta b { color: var(--ink); }
  ol.steps { padding-left: 6mm; margin: 1.5mm 0; }
  ol.steps > li { margin: 1mm 0; }
  .answer { background: var(--tint); border-left: 3px solid var(--accent); padding: 2mm 4mm; margin: 2.5mm 0; break-inside: avoid; }
  .answer > p:first-child { margin-top: 0; }
  .note { color: var(--ink2); font-size: 9.2pt; }
  .derive { border: 1px solid var(--rule); border-radius: 4px; padding: 2mm 4mm; margin: 2.5mm 0; break-inside: avoid; }
  table { border-collapse: collapse; margin: 2mm 0; font-size: 9pt; break-inside: avoid; font-variant-numeric: tabular-nums; }
  th { background: var(--tint); font-weight: 600; text-align: right; padding: 1.2mm 2.4mm; border-bottom: 1px solid var(--ink2); }
  td { text-align: right; padding: 0.9mm 2.4mm; border-bottom: 1px solid var(--rule); }
  table.kv th:first-child, table.kv td:first-child { text-align: left; }
  table.kv td:last-child { text-align: left; }
  table.small { font-size: 8.3pt; }
  table.breakable { break-inside: auto; }
  table.fl th:first-child, table.fl td:first-child { text-align: left; }
  table.small td, table.small th { padding: 0.6mm 2mm; }
  figure { margin: 3mm 0; break-inside: avoid; }
  figure img { width: 100%; }
  figcaption { font-size: 8.8pt; color: var(--ink2); margin-top: 1mm; }
  code { font-size: 8.8pt; background: var(--tint); padding: 0 1mm; border-radius: 2px; }
  .cols { display: flex; gap: 6mm; align-items: flex-start; }
  .cols > * { flex: 1; }
  .pb { break-before: page; }
  ul { margin: 1.5mm 0; padding-left: 6mm; }
  @media screen { body { padding: 8mm 4mm; } }
</style></head>
<body>

<h1>CSE558 Data Science &middot; Assignment &middot; Question 2</h1>
<div class="meta">
  <b>JL for linear regression</b> (sketch-and-solve with CountSketch) &nbsp;|&nbsp; Group: <b>@@GROUP@@</b><br>
  Discussed with: @@DISCUSSED@@<br>
  Sources: CSE558 Lectures 18&ndash;20 (JL lemma, JL for k-means);
  K. L. Clarkson and D. P. Woodruff, <i>Low rank approximation and regression in input sparsity time</i>, STOC 2013, https://arxiv.org/abs/1207.6365 (CountSketch as a subspace embedding);
  D. P. Woodruff, <i>Sketching as a tool for numerical linear algebra</i>, 2014, https://arxiv.org/abs/1411.4357.<br>
  Code: <code>Q2_JL_linear_regression.ipynb</code> (every number below is printed there). Master seed @@seed@@; machine: @@cpu@@, numpy @@numpy@@.
  Losses and ratios are exactly reproducible; wall-clock times vary slightly between runs.
</div>

<h2>Step 0 &middot; The data matrix D and the labels y</h2>
<ol class="steps">
  <li><b>Load</b> the 10% KDD Cup 1999 data with <code>fetch_kddcup99(subset=None, as_frame=True)</code>: 494,021 connections, 41 features.</li>
  <li><b>(i) One-hot encode</b> <code>protocol_type</code> (3 values), <code>service</code> (66) and <code>flag</code> (11) into 80 columns, and convert the other 38 features to float: 118 columns.</li>
  <li><b>(ii) Drop constant columns</b>: <code>num_outbound_cmds</code> and <code>is_host_login</code> are 0 on every row, leaving 116 columns.</li>
  <li><b>(iii) Rescale</b> every column to $[0,1]$ with $x\mapsto (x-\min)/(\max-\min)$.</li>
  <li><b>Labels</b>: <code>y = (kdd.target != b'normal.').to_numpy(dtype=float)</code>, so $y_i=1$ for an attack.</li>
</ol>
@@t_data@@
<p><b>Why rank(D) = @@r@@ and not @@d@@.</b> The one-hot columns stay 0/1 after rescaling, and three exact linear relations hold:
the <code>protocol_type</code>, <code>service</code> and <code>flag</code> columns each add up to the all-ones vector (2 independent relations),
and <code>protocol_type_icmp</code> equals the sum of the six ICMP-only service columns (<code>eco_i, ecr_i, red_i, tim_i, urh_i, urp_i</code>).
The notebook checks all three. So $r=\operatorname{rank}(D)=116-3=@@r@@$. The all-ones vector lies in the column space of $D$,
so the model already contains an intercept. That is why the question says no intercept column is needed.</p>

<h2>Background &middot; Why sketch-and-solve works</h2>
<p>We replace $\min_a\|Da-y\|_2^2$ (with $n$ = @@n@@ equations) by $\min_a\|SDa-Sy\|_2^2$ (with only $m$ equations). All the vectors $Da-y$ lie in
$W=\operatorname{span}(\text{columns of }D,\ y)$, a fixed subspace of dimension at most $r+1$.</p>
<div class="derive">
<p><b>Lemma (subspace embedding &rArr; near-optimal solution).</b> Suppose that for every $w\in W$, $(1-\varepsilon)\|w\|^2\le\|Sw\|^2\le(1+\varepsilon)\|w\|^2$. Then
$\text{Loss}(D,y,\tilde a)\le\frac{1+\varepsilon}{1-\varepsilon}\,\text{Loss}(D,y,b)$.</p>
<p><i>Proof, in three steps.</i> (1) $D\tilde a-y\in W$, so $(1-\varepsilon)\|D\tilde a-y\|^2\le\|S(D\tilde a-y)\|^2$.
(2) $\tilde a$ minimises the sketched loss, so $\|S(D\tilde a-y)\|^2\le\|S(Db-y)\|^2$.
(3) $Db-y\in W$, so $\|S(Db-y)\|^2\le(1+\varepsilon)\|Db-y\|^2$. Chain (1)&ndash;(3) and divide by $1-\varepsilon$. &#9633;</p>
</div>
<ul>
  <li>This is the JL lemma applied to a whole <i>subspace</i>, not to $n$ fixed points. A subspace of dimension $r$ cannot be embedded into $\mathbb{R}^m$ with $m&lt;r$:
  some nonzero $Dz$ is sent to $0$. So nothing can work below $m=\operatorname{rank}(D)$. This is the threshold the question asks about.</li>
  <li>A Gaussian $S$ is a subspace embedding with $m=O((r+\log(1/\delta))/\varepsilon^2)$. A CountSketch is one with
  $m=O(r^2/(\delta\varepsilon^2))$ (Clarkson&ndash;Woodruff 2013), and $SD$ costs only one pass over $D$.</li>
  <li>For <i>one fixed</i> vector, CountSketch is a JL map: $\mathbb{E}\|Sv\|^2=\|v\|^2$ and $\operatorname{Var}\approx 2\|v\|^4/m$. Part (d) checks this.</li>
</ul>

<h2>Part (a) &middot; The full solution b</h2>
<ol class="steps">
  <li>Solve $b=\arg\min_a\|Da-y\|_2^2$ with <code>numpy.linalg.lstsq(D, y, rcond=None)</code> on all @@n@@ &times; @@d@@ entries.
  Because $\operatorname{rank}(D)=@@r@@&lt;@@d@@$, the minimiser is not unique. <code>lstsq</code> returns the minimum-norm one, and the minimum loss is unique.</li>
  <li>Time the call with <code>time.perf_counter</code>; repeat 5 times and average.</li>
</ol>
@@t_a@@
<div class="answer"><p><b>Answer (a).</b> $\text{Loss}(D,y,b)=$ <b>@@loss_b@@</b>, and computing $b$ takes <b>@@time_a@@ s</b> (mean of 5 runs).
For scale: predicting the constant $\bar y$ has loss @@tss@@, so the linear fit explains $1-\text{Loss}/\|y-\bar y\|^2=@@r2@@$ of the variance of $y$.</p></div>

<h2>Part (b) &middot; CountSketch for m = 50 &hellip; 5000</h2>
<ol class="steps">
  <li><b>Draw S.</b> For each row $i$, draw a bucket $h(i)\in\{1,\dots,m\}$ and a sign $\sigma(i)\in\{\pm1\}$, uniformly and independently. $S$ is stored as these two length-$n$ arrays, never as an $m\times n$ matrix.</li>
  <li><b>Form SD and Sy in one pass</b>: row $i$ of $D$, times $\sigma(i)$, is added to row $h(i)$ of $SD$, with <code>np.add.at(SD, h, sigma[:, None] * D)</code>. The same is done for $y$. A unit test in the notebook checks this against an explicit dense $S$.</li>
  <li><b>Solve</b> $\tilde a=\arg\min_a\|SDa-Sy\|_2^2$ with <code>lstsq</code>.</li>
  <li><b>Record</b> $\text{Loss}(D,y,\tilde a)$, its ratio to $\text{Loss}(D,y,b)$, the sketched loss $\|SD\tilde a-Sy\|_2^2$, and (to explain the curve) $\operatorname{rank}(SD)$,
  the smallest nonzero singular value $\sigma_{\min}(SD)$ and $\|\tilde a\|$.</li>
  <li>Repeat 5 times per $m$, with a fresh $S$ each time (35 runs).</li>
</ol>
<h3>Summary per m over the 5 runs (r = rank(D) = @@r@@). All 35 individual runs are listed in Appendix A.</h3>
@@t_b_sum@@
<p class="note">For comparison: $\sigma_{\min}(D)=@@smin_D@@$ and $\|b\|=@@norm_b@@$. The two "ref." columns are the exact means for a dense Gaussian $S$ (derived in the box below). CountSketch tracks them closely.</p>
<figure><img src="figures/q2b_ratio_loglog.png" alt="Ratio and sketched loss against m">
<figcaption><b>Figure 1.</b> Left: mean ratio $\text{Loss}(D,y,\tilde a)/\text{Loss}(D,y,b)$ against $m$ on log&ndash;log axes. The bars run from the smallest to the largest of the 5 ratios. Right: the sketched loss divided by $\text{Loss}(D,y,b)$. The dashed line marks $m=\operatorname{rank}(D)=@@r@@$.</figcaption></figure>

<div class="answer">
<p><b>Answer (b): below m = rank(D) = @@r@@ (m = 50, 100).</b></p>
<ul>
  <li><b>Sketched loss is zero.</b> $SD$ is $m\times @@d@@$ with rank $m$ in every run, so the $m$ equations $SDa=Sy$ can be solved <i>exactly</i>:
  the sketched loss is @@sk_lo@@ to @@sk_hi@@, which is zero up to rounding. It says nothing about the true loss, which is in the thousands.</li>
  <li><b>The ratio is large and erratic</b>: @@r50@@ (from @@r50lo@@ to @@r50hi@@) at $m=50$ and @@r100@@ (from @@r100lo@@ to @@r100hi@@) at $m=100$.
  $S$ maps a subspace of $\operatorname{col}(D)$ of dimension at least $r-m$ to $0$. The sketched problem cannot see those directions,
  and <code>lstsq</code> just returns the minimum-norm point of a $(@@d@@-m)$-dimensional family of exact solutions.</li>
  <li><b>It gets worse as m approaches r.</b> At $m=100$ the sketch has to <i>interpolate</i> 100 equations with barely more effective unknowns (113).
  The equations become nearly dependent: $\sigma_{\min}(SD)$ falls from @@smin50@@ at $m=50$ to @@smin100@@ at $m=100$.
  Fitting the sketched noise $Sv$ exactly then forces large moves ($\|\tilde a\|$ grows from @@na50@@ to @@na100@@), and the loss blows up.
  This peak at $m\approx\operatorname{rank}$ is the interpolation threshold (also called "double descent").</li>
</ul>
<p><b>Answer (b): above m = rank(D) (m = @@m_above@@).</b></p>
<ul>
  <li>$\operatorname{rank}(SD)=@@r@@=\operatorname{rank}(D)$ in all 25 runs. $S$ is now one-to-one on $\operatorname{col}(D)$, and the sketched problem is overdetermined.
  As $m$ grows, $\sigma_{\min}(SD)$ approaches $\sigma_{\min}(D)=@@smin_D@@$ (@@smin_first@@ at $m=@@m_first@@$, @@smin_last@@ at $m=@@m_last@@$). $S$ becomes a better subspace embedding.</li>
  <li><b>The ratio falls steadily towards 1</b>: @@ratios_above@@. The min&ndash;max spread shrinks too.
  It follows $1+r/(m-r-1)\approx1+r/m$, so the excess loss halves when $m$ doubles. The log&ndash;log curve approaches 1 (the horizontal asymptote) from above.</li>
  <li><b>No ratio is below 1</b>, in any of the 35 runs, and none can be: $b$ minimises $\text{Loss}(D,y,\cdot)$ over all $a$, so $\text{Loss}(D,y,\tilde a)\ge\text{Loss}(D,y,b)$ for every $\tilde a$.</li>
  <li><b>The sketched loss is nonzero and climbs towards Loss(D, y, b) from below.</b> Relative to Loss(b) it goes @@sk_above@@, matching $(m-r)/m$.
  It is always an <i>under</i>-estimate: $\|SD\tilde a-Sy\|^2\le\|S(Db-y)\|^2=\|Sv\|^2\approx\|v\|^2$, and the $r$ fitted parameters absorb about a fraction $r/m$ of the sketched residual.</li>
</ul>
</div>

<div class="derive">
<p><b>Where the reference curves come from (dense Gaussian S with N(0,1/m) entries, m &gt; r + 1).</b>
Let $U\in\mathbb{R}^{n\times r}$ be an orthonormal basis of $\operatorname{col}(D)$, $Db=Uc_b$, and let $v=Db-y$, which is orthogonal to $\operatorname{col}(D)$.</p>
<ol class="steps">
  <li>Every fitted vector is $Da=Uc$. By Pythagoras, $\text{Loss}(D,y,a)=\|v\|^2+\|c-c_b\|^2$.</li>
  <li>$Sy=SUc_b-Sv$, so the sketched solution is $\tilde c=c_b-(SU)^{+}Sv$, and $\text{Loss}(D,y,\tilde a)-\text{Loss}(D,y,b)=\|(SU)^{+}Sv\|^2$.</li>
  <li>Because $U\perp v$, the Gaussian matrices $SU$ (i.i.d. $N(0,1/m)$) and $Sv\sim N(0,\tfrac{\|v\|^2}{m}I_m)$ are independent. With $\mathbb{E}[((SU)^\top SU)^{-1}]=\tfrac{m}{m-r-1}I_r$ (inverse Wishart),
  $\mathbb{E}\|(SU)^{+}Sv\|^2=\tfrac{\|v\|^2}{m}\operatorname{tr}\mathbb{E}[((SU)^\top SU)^{-1}]=\|v\|^2\tfrac{r}{m-r-1}$. So the mean ratio is $1+\tfrac{r}{m-r-1}$.</li>
  <li>The sketched loss is $\|(I-P)Sv\|^2$, where $P$ projects onto the $r$-dimensional $\operatorname{col}(SU)$, which is independent of $Sv$. So its mean is $\|v\|^2\,\tfrac{m-r}{m}$.</li>
</ol>
</div>

<h2>Part (c) &middot; Time of sketch-and-solve versus the full solve</h2>
<ol class="steps">
  <li>In each run of (b), time three steps: draw $S$ ($h,\sigma$), form $SD$ and $Sy$, and solve the $m\times @@d@@$ problem. Average over the 5 runs.</li>
  <li>Compare the total with the @@time_a@@ s needed for $b$ in (a).</li>
</ol>
<div class="cols">
<div>@@t_c@@</div>
<figure><img src="figures/q2c_times.png" alt="Sketch-and-solve time against m"><figcaption><b>Figure 2.</b> Total sketch-and-solve time (mean of 5) and its solve step, against the full solve.</figcaption></figure>
</div>
<div class="answer">
<p><b>Answer (c).</b> Sketch-and-solve takes <b>@@t_lo@@&ndash;@@t_hi@@ s for every m</b>, against <b>@@time_a@@ s</b> for the full solve: about <b>@@sp_lo@@&ndash;@@sp_hi@@&times; faster</b>.</p>
<ul>
  <li><b>Forming SD and Sy dominates</b> (at least @@form_share@@% of the time, about @@t_form@@ s), and <b>it does not depend on m</b>. It is one pass over the $nd\approx@@nd@@$ million entries of $D$; each row is added once, wherever it lands.</li>
  <li>Drawing $S$ needs only $2n$ random numbers (about @@t_draw@@ ms). Solving the small problem costs $O(md^2)$: it grows with $m$ (@@solve_lo@@ ms to @@solve_hi@@ ms) but stays a few percent of the total.</li>
  <li>The full solve costs $O(nd^2)\approx 6.6\times10^9$ operations, against $O(nd + md^2)$ for the sketch, so the cost model predicts a much larger gain.
  The measured gain is only @@sp_lo@@&ndash;@@sp_hi@@&times; because $d=@@d@@$ is small, LAPACK's SVD runs at many GFLOP/s on several cores, and <code>np.add.at</code> works one element at a time (about @@ns_per@@ ns per entry).
  The gap would widen with larger $d$, or with sparse $D$, where the sketch costs $O(\operatorname{nnz}(D))$.</li>
  <li>Taken with (b): for about a third of the time, $m=1000$ gives a loss within @@r1000@@&times; of the optimum and $m=5000$ within @@r5000@@&times;.</li>
</ul>
</div>

<h2>Part (d) &middot; JL for one fixed vector: &#8214;Sv&#8214;&sup2; / &#8214;v&#8214;&sup2;</h2>
<ol class="steps">
  <li>Compute the residual of the full solution, $v=Db-y$, once ($\|v\|^2=\text{Loss}(D,y,b)$). It is fixed before any $S$ is drawn: this is exactly the JL setting.</li>
  <li>For $m\in\{50,500,5000\}$, draw 100 independent CountSketches, form $Sv$ in one pass, and record $\|Sv\|_2^2/\|v\|_2^2$.</li>
  <li>Plot a histogram for each $m$ (shared $x$-axis, so the spreads can be compared directly) and report the mean and standard deviation.</li>
</ol>
<figure><img src="figures/q2d_histograms.png" alt="Histograms of the norm ratio"><figcaption><b>Figure 3.</b> 100 values of $\|Sv\|^2/\|v\|^2$ per $m$. The dashed line is at 1.</figcaption></figure>
@@t_d@@
<div class="derive">
<p><b>Mean and variance of a CountSketch on a fixed v.</b> Write $\delta_{ik}=\mathbf{1}[h(i)=h(k)]$. Then
$$\|Sv\|^2=\sum_{j=1}^m\Big(\sum_{i:h(i)=j}\sigma_iv_i\Big)^2=\sum_i v_i^2+\sum_{i\ne k}\delta_{ik}\,\sigma_i\sigma_k\,v_iv_k .$$
Since $\mathbb{E}[\sigma_i\sigma_k]=0$ for $i\ne k$, $\mathbb{E}\|Sv\|^2=\|v\|^2$: CountSketch is unbiased. The cross terms of different pairs are uncorrelated (random signs), and $\Pr[h(i)=h(k)]=1/m$, so
$$\operatorname{Var}\|Sv\|^2=4\sum_{i&lt;k}\frac{v_i^2v_k^2}{m}=\frac{2}{m}\big(\|v\|_2^4-\|v\|_4^4\big),\qquad
\operatorname{std}\frac{\|Sv\|^2}{\|v\|^2}=\sqrt{\frac{2}{m}\Big(1-\frac{\|v\|_4^4}{\|v\|_2^4}\Big)}\approx\sqrt{\frac{2}{m}}.$$
Here $\|v\|_4^4/\|v\|_2^4=@@flat@@$ ($v$ is spread over many rows), so the theory std is $\sqrt{2/m}$ for practical purposes.</p>
</div>
<div class="answer">
<p><b>Answer (d).</b> The mean is 1 within sampling error for every $m$ (unbiased). The standard deviation is <b>@@sd50@@, @@sd500@@ and @@sd5000@@</b> for $m=50,500,5000$, matching $\sqrt{2/m}=0.200,\ 0.063,\ 0.020$.
<b>The spread shrinks like $1/\sqrt m$</b>: every 10&times; increase in $m$ makes the histogram about $\sqrt{10}\approx3.2$&times; narrower.
The range goes from @@mn50@@&ndash;@@mx50@@ at $m=50$ to @@mn5000@@&ndash;@@mx5000@@ at $m=5000$.
By Chebyshev, $\Pr\big[\,|\|Sv\|^2/\|v\|^2-1|&gt;\varepsilon\big]\le 2/(m\varepsilon^2)$, so $m=2/(\delta\varepsilon^2)$ rows keep one vector's length within $1\pm\varepsilon$.
That is the JL guarantee of CountSketch. It has a worse dependence on $\delta$ than a Gaussian's $\log(1/\delta)$, which is the price of having one nonzero per column.</p>
<p>Contrast with (b): there $S$ must preserve a whole $@@r@@$-dimensional subspace at once, not one vector. So (b) needs $m\gg r$, while (d) is already accurate at $m=50$.</p>
</div>

<h2>Part (e) &middot; CountSketch versus a dense Gaussian sketch on 50,000 rows</h2>
<ol class="steps">
  <li><b>Choose the subset once.</b> Draw 50,000 row indices uniformly without replacement with a fixed-seed generator. $D'$ is 50,000 &times; @@d@@ with $\operatorname{rank}(D')=@@r1@@$, because some rare categories do not occur in the subset. The same subset is used for both sketches and all 5 repetitions.</li>
  <li><b>Full solution on the subset.</b> $b'=\arg\min_a\|D'a-y'\|^2$ via <code>lstsq</code>: $\text{Loss}(D',y',b')=@@loss_b1@@$.</li>
  <li><b>Sketch with m = 500</b>, drawing a fresh $S$ in each of 5 repetitions: (i) a CountSketch, as in (b); (ii) a dense $S\in\mathbb{R}^{500\times50000}$ with i.i.d. $N(0,1/m)$ entries
  (<code>rng.normal(0, 1/np.sqrt(m), size=(m, 50000))</code>), then $SD'$ and $Sy'$ by matrix products.
  Time drawing $S$ and forming $SD'$ and $Sy'$. Then solve for $\tilde a$ and compute $\text{Loss}(D',y',\tilde a)/\text{Loss}(D',y',b')$.</li>
</ol>
@@t_e@@
<p class="note">Reference for the Gaussian sketch: $1+r'/(m-r'-1)=1+@@r1@@/(500-@@r1@@-1)=@@ga_ref@@$. As an extra check of the last question, we also ran each sketch once on the <i>full</i> $D$ with $m=500$:</p>
@@t_e_full@@
<div class="answer">
<p><b>Answer (e).</b> On the 50,000-row subset the two sketches are <b>about equally accurate</b> (mean ratio @@cs_r@@ for CountSketch and @@ga_r@@ for the Gaussian; the Gaussian theory gives @@ga_ref@@).
But CountSketch is <b>@@e_speed@@&times; faster</b> to draw and apply (@@cs_t@@ ms against @@ga_t@@ ms).
The Gaussian's time goes into drawing $m\cdot50{,}000=2.5\times10^7$ normal numbers and the dense product ($m\cdot50{,}000\cdot d\approx2.9\times10^9$ operations).
CountSketch needs $10^5$ random numbers and one pass over the $5.8\times10^6$ entries of $D'$.</p>
<p><b>On the full D, I would use the CountSketch.</b></p>
<ul>
  <li><b>Time.</b> A dense $S$ for the full $D$ is $500\times@@n@@$, so applying it costs $m\,n\,d\approx2.9\times10^{10}$ operations, against $nd\approx5.7\times10^7$ for CountSketch.
  Measured: @@fc_t@@ s against @@fg_t@@ s (@@full_speed@@&times;). The gap is smaller than the operation count suggests only because BLAS matrix products are very fast.</li>
  <li><b>Memory.</b> The dense $S$ alone takes @@fg_mb@@ GB in float64. CountSketch stores $h$ and $\sigma$: @@fc_mb@@ MB.</li>
  <li><b>Accuracy.</b> At $m=500\approx@@m_over_r@@\,r$ the two have the same accuracy (ratios @@cs_r@@ vs @@ga_r@@ on $D'$; one run each on $D$ above).</li>
  <li><b>One pass.</b> Each row only needs its own $h(i),\sigma(i)$, so $SD$ can be built while streaming the rows. For sparse data it costs $O(\operatorname{nnz}(D))$.</li>
</ul>
<p class="note">Caveat: the worst-case theory asks more of CountSketch ($m=O(r^2/\varepsilon^2)$) than of a Gaussian ($O(r/\varepsilon^2)$). If a guarantee at small $m$ were needed, one could apply CountSketch first and then a small Gaussian sketch to $SD$, and keep the speed of both.</p>
</div>

<h2>Summary</h2>
<ul>
  <li>(a) $\text{Loss}(D,y,b)=@@loss_b@@$, found in @@time_a@@ s on all $n$ = @@n@@ rows.</li>
  <li>(b) Below $m=\operatorname{rank}(D)=@@r@@$, the sketched problem is solved exactly (sketched loss 0), and the true loss is @@below_lo@@&ndash;@@below_hi@@&times; the optimum, worst just below the rank.
  Above the rank, the ratio decreases like $1+r/m$ towards 1 (@@r5000@@ at $m=5000$), never below 1. The sketched loss rises towards Loss(b) from below, like $(m-r)/m$.</li>
  <li>(c) Sketch-and-solve takes @@t_lo@@&ndash;@@t_hi@@ s for every $m$ (forming $SD$ dominates and does not depend on $m$), @@sp_lo@@&ndash;@@sp_hi@@&times; faster than the full solve here.</li>
  <li>(d) $\|Sv\|^2/\|v\|^2$ has mean 1 and std $\approx\sqrt{2/m}$: the spread shrinks like $1/\sqrt m$.</li>
  <li>(e) Same accuracy as a dense Gaussian sketch, @@e_speed@@&times; less time on $D'$ and far less memory, and one pass over the data: use CountSketch on the full $D$.</li>
</ul>

<h2>Appendix A &middot; All 35 runs of part (b)</h2>
<p>Each row is one fresh CountSketch $S$. The sketched loss is $\|SD\tilde a-Sy\|_2^2$; $\text{Loss}(D,y,b)=@@loss_b@@$.</p>
@@t_b_all@@

<h2>Appendix B &middot; Files and how to reproduce</h2>
<ul>
  <li><code>Q2_JL_linear_regression.ipynb</code>: all the code for Question 2, with its outputs. Run it top to bottom (about 1 minute on an @@cpu@@; it downloads the data on the first run).</li>
  <li><code>figures/</code>: the three figures. <code>results/</code>: every table as CSV, plus <code>q2_results.json</code>.</li>
  <li><code>make_report.py</code>: rebuilds this document (<code>Q2_report.html</code> and <code>Q2_report.pdf</code>) from <code>results/</code> and <code>figures/</code>.</li>
  <li>Random numbers: <code>numpy.random.SeedSequence(@@seed@@).spawn(6)</code> gives one generator each for (b), (d), the subset in (e), the CountSketches in (e), the Gaussian sketches in (e), and the full-D check.
  Rerunning gives the same losses and ratios to the last digit. Times change by a few percent.</li>
</ul>
</body></html>
"""

fills = {"GROUP": GROUP, "DISCUSSED": DISCUSSED_WITH, "t_data": t_data, "t_a": t_a, "t_b_all": t_b_all,
         "t_b_sum": t_b_sum, "t_c": t_c, "t_d": t_d, "t_e": t_e, "t_e_full": t_e_full, **{k: str(x) for k, x in v.items()}}
out = HTML
for k, x in fills.items():
    out = out.replace(f"@@{k}@@", x)
assert "@@" not in out, [l for l in out.splitlines() if "@@" in l][:3]
html_path = HERE / "Q2_report.html"
html_path.write_text(out)
print("wrote", html_path)

chrome = next((p for p in ["/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
                           shutil.which("google-chrome") or "", shutil.which("chromium") or ""] if p and Path(p).exists()), None)
if chrome is None:
    print("Chrome not found: open Q2_report.html in a browser and print it to PDF.")
else:
    pdf_path = HERE / "Q2_report.pdf"
    with tempfile.TemporaryDirectory() as profile:
        subprocess.run([chrome, "--headless=new", "--disable-gpu", "--no-pdf-header-footer", f"--user-data-dir={profile}",
                        "--virtual-time-budget=20000", "--run-all-compositor-stages-before-draw",
                        f"--print-to-pdf={pdf_path}", html_path.as_uri()],
                       check=True, capture_output=True)
    print("wrote", pdf_path)
