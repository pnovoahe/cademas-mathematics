# Case-study figures briefing (for LLM drafting)

Purpose: give an LLM enough context to draft / revise the **Results and Interpretation**
subsection of the companion paper (Employee Attrition Case Study), using only
what the figures show and the quantitative results behind them.

Language of the manuscript: **English**, third person preferred elsewhere in
the paper; this briefing is in English for drafting convenience.

---

## 1. Experimental setting (shared by all figures)

| Item | Value |
|------|--------|
| Cohort size | \(n=100\) employees |
| Intervention capacity | \(K=10\) (primary analysis) |
| Predictive score | \(R_i\in[0,1]\) (AUC-weighted average of 4 H2O AutoML units) |
| Contextual score | \(Q_i\in[0,1]\) (Digital/Technological Transformation fuzzy context; Gödel AND → \(Q_i=0\) is a contextual veto) |
| Observed attrition label | **Not** used in \(R\), \(Q\), or \(P\) (audit only) |
| Operators in the figures | **Seven** configs: \(A_L(0.1)\), \(A_L(0.5)\), \(A_L(0.9)\), \(A_G(0.1)\), \(A_G(0.5)\), \(A_G(0.9)\), \(A_M=\min(R,Q)\) |
| Tie-break for Top-\(K\) / ranks | Higher \(P\), then higher \(R\), then row index (deterministic) |

**Important consistency note for drafting:** some earlier paragraphs in the
manuscript still say “five configurations” and call \(A_M\) an
“arithmetic-mean” operator. The **figures and simulation use seven
configs** and \(A_M=\min(R,Q)\). Prefer the seven-operator / minimum
wording when writing Results.

**Population snapshot (from `results/cases_attrition_100_operator_scores.csv`):**

- \(35/100\) employees have \(Q_i=0\) (contextual vetoes).
- \(R\) spans roughly \([0.00,0.99]\) (mean \(\approx 0.47\)).
- Non-veto \(Q\) spans roughly \([0.06,0.73]\).

Source experiment: `src/03_attrition_case_study/`.
Figures are written under `src/03_attrition_case_study/figures/`.

---

## 2. Figures available

### 2.1 In the manuscript (must be discussed)

| LaTeX label | File | Role |
|-------------|------|------|
| `fig:attrition-case-overview` | `figures/attrition_case_overview.pdf` | Distributional overview of the cohort and of \(P\) under the seven configs |
| `fig:attrition-rank-bump-delta` | `figures/attrition_rank_bump_delta.pdf` | Rank trajectories of the Top-10 under \(A_L(0.5)\), with signed rank changes on non-baseline columns |

### 2.2 Generated but **not** currently in the manuscript

| File | Role |
|------|------|
| `attrition_rank_bump.pdf` | Same trajectories with **absolute** ranks in every column (no \(\Delta\)) |

Do **not** invent a third manuscript figure unless asked; the absolute bump is
auxiliary.

---

## 3. Figure A — `fig:attrition-case-overview`

**Layout:** one row, two panels.

### Panel (a) — \((R,Q)\) scatter

- Each point = one employee.
- Axes: \(R\) (horizontal), \(Q\) (vertical), both in \([0,1]\).
- Dashed guides at \(R=0.5\) and \(Q=0.5\).
- Stratum colours (same palette as other simulation figures):
  - **Weak** (gray): low \(R\), low \(Q\)
  - **Strong** (green): high \(R\), high \(Q\)
  - **Rweak** (blue): low \(R\), high \(Q\)
  - **Qweak** (yellow): high \(R\), low \(Q\)
  - **Q vetoed** (orange): \(Q_i=0\) (points on the horizontal axis)

**What to take away for text:** the cohort mixes vetoed and non-vetoed
employees; predictive risk and contextual alignment are only partially
aligned (many high-\(R\) cases have modest \(Q\), and many vetoes exist).

### Panel (b) — seven violins of \(P\)

- One violin per config, left→right in the order  
  \(A_L(0.1),\ A_L(0.5),\ A_L(0.9),\ A_G(0.1),\ A_G(0.5),\ A_G(0.9),\ A_M\).
- Points inside violins use the same stratum colours as (a).
- Horizontal dashed line = Top-\(K=10\) cutoff (score of the 10th selected
  employee under that operator).
- “Top-\(K\)” arrows annotate the first two violins.

**Approximate cutoffs (10th score) and vetoes inside Top-10:**

| Config | Top-10 cutoff \(P\) (approx.) | Vetoed (\(Q=0\)) in Top-10 |
|--------|-------------------------------|----------------------------|
| \(A_L(0.1)\) | 0.48 | 0 |
| \(A_L(0.5)\) | 0.64 | 0 |
| \(A_L(0.9)\) | 0.89 | **1** |
| \(A_G(0.1)\) | 0.40 | 0 |
| \(A_G(0.5)\) | 0.57 | 0 |
| \(A_G(0.9)\) | 0.86 | 0 |
| \(A_M\) | 0.33 | 0 |

**What to take away for text:**

- Raising \(\lambda\) in \(A_L\) shifts mass and the Top-\(K\) cutoff upward
  (more predictive weight).
- Zero-absorbing operators (\(A_G\), \(A_M\)) keep vetoed cases at \(P=0\),
  so they do not enter Top-10 here (except the compensatory high-\(\lambda\)
  linear case, which admits **one** vetoed employee at \(K=10\)).
- The violins show that aggregation semantics change both the **shape of
  \(P\)** and **who sits above the Top-\(K\) line**, not only a uniform
  rescaling.

---

## 4. Figure B — `fig:attrition-rank-bump-delta` (manuscript bump)

**Question answered:** holding the cohort fixed, how do the ranks of the
employees selected under the **reference** operator \(A_L(0.5)\) change when
we switch to the other six configs?

### Visual encoding

- **X-axis (7 tracks):** same order as the violins.
- **Y-axis:** rank (1 = highest priority at the top); **no y tick labels**.
- **Only 10 curves:** employees in the Top-10 under \(A_L(0.5)\). No “ghost”
  employees that enter Top-10 only under other operators.
- **Left labels:** employee names (`Case_ID`), colour-matched to their series.
- **Reference column \(A_L(0.5)\):** hollow white balls with coloured rim;
  interior number = **absolute rank** \(1,\ldots,10\).
- **Other columns:**
  - if rank unchanged vs \(A_L(0.5)\): **solid** series-coloured ball;
  - if changed: hollow white ball with black **signed \(\Delta\)**,
    \(\Delta=\mathrm{rank}-\mathrm{rank}_{A_L(0.5)}\).
    - Positive \(\Delta\) = worse rank (e.g. \(5\to 12\) → `+7`).
    - Negative \(\Delta\) = better rank (e.g. \(5\to 1\) → `-4`).
- **Shaded gray band + vertical “Top-10” label on the right:** ranks
  \(1\)–\(10\). Markers **below** the band have left the intervention set
  under that operator.

### Reference Top-10 under \(A_L(0.5)\) (absolute order)

| Rank @ \(A_L(0.5)\) | Employee | \(R\) | \(Q\) |
|--------------------:|----------|------:|------:|
| 1 | Laura Baker | 0.936 | 0.600 |
| 2 | Andrew Wood | 0.929 | 0.567 |
| 3 | Alice Brown | 0.929 | 0.560 |
| 4 | Andrew Ross | 0.930 | 0.467 |
| 5 | Caroline Bennett | 0.937 | 0.457 |
| 6 | Jessica Wilson | 0.944 | 0.400 |
| 7 | Noah Murphy | 0.974 | 0.360 |
| 8 | Eleanor Foster | 0.957 | 0.333 |
| 9 | Christian Bennett | 0.969 | 0.320 |
| 10 | Madison Foster | 0.919 | 0.367 |

All ten have relatively high \(R\) and **non-zero** \(Q\) (none are vetoed).

### Rank displacements vs \(A_L(0.5)\) (same order of employees)

Columns: \(\Delta\) at \(A_L(0.1)\), \(A_L(0.9)\), \(A_G(0.1)\), \(A_G(0.5)\), \(A_G(0.9)\), \(A_M\).
(Absolute ranks at the reference are \(1\)–\(10\) by construction.)

| Employee | \(\Delta\) \(A_L(0.1)\) | \(\Delta\) \(A_L(0.9)\) | \(\Delta\) \(A_G(0.1)\) | \(\Delta\) \(A_G(0.5)\) | \(\Delta\) \(A_G(0.9)\) | \(\Delta\) \(A_M\) | Leaves Top-10 under |
|----------|------------------------:|------------------------:|------------------------:|------------------------:|------------------------:|------------------:|---------------------|
| Laura Baker | +1 | +3 | 0 | 0 | 0 | 0 | — (always Top-10) |
| Andrew Wood | +1 | +7 | 0 | 0 | 0 | 0 | — |
| Alice Brown | +1 | +5 | 0 | 0 | 0 | 0 | — |
| Andrew Ross | +4 | **+13** | 0 | 0 | +2 | 0 | \(A_L(0.9)\) (rank 17) |
| Caroline Bennett | +4 | **+9** | 0 | 0 | 0 | 0 | \(A_L(0.9)\) (rank 14) |
| Jessica Wilson | +6 | +6 | +2 | 0 | +2 | 0 | \(A_L(0.1)\), \(A_L(0.9)\) |
| Noah Murphy | +8 | **−5** | +5 | 0 | −3 | +2 | \(A_L(0.1)\), \(A_G(0.1)\) |
| Eleanor Foster | +13 | −2 | +7 | +2 | +2 | +2 | \(A_L(0.1)\), \(A_G(0.1)\) |
| Christian Bennett | +15 | **−6** | +9 | +3 | −2 | +3 | \(A_L(0.1)\), \(A_G(0.1)\), \(A_G(0.5)\), \(A_M\) |
| Madison Foster | +4 | **+27** | 0 | −2 | +5 | −3 | \(A_L(0.1)\), \(A_L(0.9)\), \(A_G(0.9)\) |

### Overlap of Top-10 sets vs the \(A_L(0.5)\) Top-10

| Config | Intersection size | Jaccard | Outsiders entering Top-10 (names) |
|--------|------------------:|--------:|-----------------------------------|
| \(A_L(0.1)\) | 5 | 0.33 | Caroline Adams, Caroline Roberts, Claire Bailey, Hunter Adams, Julia Smith |
| \(A_L(0.5)\) | 10 | 1.00 | — |
| \(A_L(0.9)\) | 6 | 0.43 | Amelia Davis, Camila Phillips, Dominic Collins, Leah Cooper |
| \(A_G(0.1)\) | 7 | 0.54 | Caroline Adams, Caroline Roberts, Julia Smith |
| \(A_G(0.5)\) | 9 | 0.82 | Lucy Scott |
| \(A_G(0.9)\) | 9 | 0.82 | Amelia Davis |
| \(A_M\) | 9 | 0.82 | Lucy Scott |

*(The bump figure does **not** draw those outsiders; only the reference
Top-10 trajectories. Mention outsiders only if discussing set replacement.)*

### Narrative takeaways for the bump (safe for drafting)

1. **Baseline \(A_L(0.5)\) as reference.** The Top-10 is a balanced
   compromise between high \(R\) and moderate-to-high \(Q\).
2. **High predictive weight \(A_L(0.9)\) destabilizes that set.** Several
   mid/lower Top-10 employees fall out of Top-10 (large positive \(\Delta\));
   Madison Foster collapses to rank 37 (`+27`). Conversely, high-\(R\) /
   lower-\(Q\) members of the reference set (Noah Murphy, Christian Bennett)
   **improve** under \(A_L(0.9)\).
3. **Low predictive weight \(A_L(0.1)\) also reshuffles.** Context-heavy
   linear scoring demotes several high-\(R\) / mid-\(Q\) reference members
   (large positive \(\Delta\) for Eleanor / Christian / Noah); Jaccard with
   the reference Top-10 drops to \(0.33\).
4. **Zero-absorbing family is more stable near \(\lambda=0.5\).** Under
   \(A_G(0.5)\) and \(A_M\), most reference Top-10 members stay inside or at
   the boundary of Top-10 (Jaccard \(0.82\)); displacements are small except
   for a few borderline cases (e.g. Christian Bennett leaves under \(A_G(0.5)\)
   and \(A_M\)).
5. **Policy implication.** Even with identical \((R_i,Q_i)\), changing only
   the aggregation semantics can **replace several of the ten intervention
   slots**—especially when moving \(\lambda\) in \(A_L\) away from \(0.5\).

---

## 5. Suggested structure for the LaTeX subsection

Current subsection: `\subsection{Results and Interpretation}`  
(after Experimental Setup; before Discussion).

Recommended paragraph flow:

1. **Point to Figure A.** Describe panel (a) cohort geometry (vetoes + strata)
   and panel (b) seven \(P\) distributions / Top-\(K\) cutoffs; note the
   single veto admitted by \(A_L(0.9)\) at \(K=10\).
2. **Bridge to Figure B.** Same \(K=10\), but now follow the identities of the
   reference Top-10 under operator changes.
3. **Interpret the bump.** Contrast \(A_L(0.9)\) reshuffling vs relative
   stability of \(A_G(0.5)\) / \(A_M\); optionally cite 1–2 named employees as
   concrete examples (e.g. Madison Foster `+27` under \(A_L(0.9)\); Laura
   Baker remaining near rank 1 under zero-absorbing operators).
4. **Close.** Aggregation choice is not cosmetic: it changes who receives
   the scarce intervention under a fixed budget \(K\).

Existing short bridge paragraph and captions in the `.tex` can be kept or
expanded; do not contradict the encodings in §3–§4.

---

## 6. Paths and regeneration

```text
src/03_attrition_case_study/figures/attrition_case_overview.{pdf,png}
src/03_attrition_case_study/figures/attrition_rank_bump_delta.{pdf,png}
src/03_attrition_case_study/figures/attrition_rank_bump.{pdf,png}

Regenerate:
  cd simulations && .venv/bin/python 03_attrition_case_study/run.py --figures-only
```

Data: `results/cases_attrition_100_operator_scores.csv` (`;` separator).
