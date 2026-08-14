# First-round reviews

Manuscript: *Aggregation Semantics for Prioritization in Cooperative Automated Decision Making*

---

## Reviewer 1

### Overall assessment

The manuscript investigates the influence of different aggregation operators on the integration of predictive and contextual information within the CADEMAS framework. The authors compare linear aggregation, the minimum operator, and the weighted geometric mean through theoretical analysis, Monte Carlo simulations, and an employee-attrition case study.

The topic is relevant, and the manuscript is generally well organized and clearly written. In particular, interpreting the choice of an aggregation operator as a governance decision rather than merely a numerical implementation detail is potentially valuable. However, the mathematical contribution is currently limited, and several theoretical and experimental issues should be addressed before the manuscript can be considered for publication.

**Recommendation: major revision.**

---

### Major comments

1. **Novelty.** The novelty of the manuscript should be explained more precisely. The considered operators and their principal properties are well known in aggregation theory, fuzzy logic, and multi-criteria decision making. At present, the contribution appears mainly to consist of applying these operators within the CADEMAS framework. The authors should clearly distinguish their original theoretical or methodological contribution from a comparative application of established aggregation mechanisms. A dedicated comparison with the closest existing studies would be helpful.

2. **Elementary theory.** The theoretical analysis is rather elementary. Proposition 1 is obtained by directly rearranging the corresponding score inequalities, while Proposition 2 follows immediately from strict monotonicity. Although these statements are correct, their mathematical novelty is limited. The authors should consider formulating results for broader classes of aggregation operators characterized by properties such as zero absorption, strict monotonicity, compensability, veto preservation, and sensitivity to imbalance.

3. **Terminology in Proposition 1.** The terminology used in Proposition 1 requires correction. The manuscript refers to a “strict conjunctive aggregator” and states that this class includes both the minimum and the algebraic product. In the standard classification of triangular norms, the minimum operator is not a strict t-norm. The authors should instead state the exact assumptions required for the result, for example
   \[
   A(x,0)=0
   \]
   and
   \[
   A(x,y)>0 \qquad \text{whenever } x>0 \text{ and } y>0.
   \]
   The proposition should then be formulated in terms of these properties.

4. **Weighted geometric boundary cases.** The definition of the weighted geometric aggregation operator
   \[
   A_G(R,Q)=R^{\lambda}Q^{1-\lambda},\qquad \lambda\in[0,1],
   \]
   requires additional care at the boundary values \(\lambda=0\) and \(\lambda=1\). If one of the inputs is zero, expressions involving \(0^0\) may occur. Moreover, the claimed veto property for \(Q=0\) does not hold in the same form when \(\lambda=1\). The authors should either restrict the parameter to \(\lambda\in(0,1)\) or provide precise conventions and treat the boundary cases separately.

5. **Redundant “normalization” property.** The list of properties imposed on aggregation operators contains a formal redundancy. The statement
   \[
   A\colon [0,1]^2\longrightarrow[0,1]
   \]
   is presented as “normalization”, although it merely specifies the domain and codomain of the operator. The authors should provide an appropriate definition of normalization or remove this item. It would also be useful to state whether continuity, commutativity, idempotency, or strict monotonicity are assumed and which considered operators satisfy these properties.

6. **Experiments vs. algebraic properties.** A substantial part of the experimental findings follows directly from the algebraic properties of the selected operators. In particular, the conclusion that the minimum and weighted geometric operators exclude alternatives with \(Q_i=0\) follows immediately from their zero-absorption property. The authors should explain what additional information is provided by the Monte Carlo experiments and include experiments involving nonzero intermediate context scores, where the behavior of the operators is less immediate.

7. **Experimental protocol.** The experimental protocol should be strengthened. Only \(N=30\) Monte Carlo trials are used to construct empirical 95% confidence intervals. This number is relatively small for percentile-based interval estimation. The authors should substantially increase the number of trials, explain the precise construction of the confidence intervals, and report the random seeds or all other information required to reproduce the results.

8. **Robustness claim (Section 4.3).** The claim that non-linear operators are more robust to contextual uncertainty is based on a narrow experimental setting. The comparison in Section 4.3 uses only one value, \(\lambda=0.35\), one noise distribution, and one synthetic population model. Moreover, the reported difference in Kendall’s rank correlation is moderate. The authors should investigate several values of \(\lambda\), different noise models and population distributions, and possibly additional ranking-stability measures. Otherwise, the conclusions should be restricted to the experimental conditions considered.

9. **Link between Proposition 2 and uncertainty experiment.** Proposition 2 does not directly explain the contextual-uncertainty experiment. The proposition concerns alternatives with identical context scores, whereas Section 4.3 considers heterogeneous context scores subjected to random perturbations. The claimed relationship between the proposition and the simulation should therefore be clarified. A sensitivity result involving, for example,
   \[
   \bigl|A(R,Q+\varepsilon)-A(R,Q)\bigr|,
   \]
   would provide a more appropriate theoretical basis for this experiment.

10. **Employee-attrition example.** The employee-attrition example is useful as an illustration, but it should not be presented as a strong independent empirical validation. The study uses precomputed predictive and contextual scores from an earlier CADEMAS-ML application rather than independently evaluating the underlying models. The authors should clarify this limitation and explain how the employees were selected, how the context scores were obtained, how ties were handled, and whether the conclusions remain stable for other values of \(K\).

11. **Policy violation metric.** The policy violation metric is defined by
    \[
    V_K=\frac{\lvert\Pi_K\cap X_{\mathrm{veto}}\rvert}{K}.
    \]
    However, the definition of the average predictive score for compliant alternatives should explicitly address the case
    \[
    \Pi_K\setminus X_{\mathrm{veto}}=\varnothing,
    \]
    in which the denominator becomes zero. Even if this situation does not occur in the reported experiments, the metric should be mathematically well defined.

12. **Choice of operators.** The choice of the three operators requires further justification. The Introduction mentions several broad families of aggregation operators, but the empirical analysis considers only the linear operator, minimum, and weighted geometric mean. The algebraic product is introduced theoretically but is not systematically evaluated, while the maximum operator is immediately excluded. The authors should either provide a stronger rationale for this selection or include additional representative operators.

---

### Minor comments

1. Consistently distinguish between an aggregation operator, a t-norm, and a conjunctive aggregation function.

2. Make the notation for the minimum and product operators consistent throughout the manuscript.

3. Define “predictive overconfidence” more carefully. In the simulations it denotes an artificial upward shift in predictive scores, but no probabilistic calibration or conventional overconfidence measure is considered.

4. Explain how ties generated by the minimum operator are resolved before computing rankings and Kendall’s correlation coefficient.

5. The notation \(\operatorname{argsort}\) should specify whether the ordering is ascending or descending and how equal scores are treated.

6. Check section titles and terminology for consistent capitalization, particularly the title of Section 4.3.

7. Careful English-language editing. For example, replace “responsible of such combination” by “responsible for such a combination”, and simplify several long sentences in the Introduction.

---

### Closing remark (Reviewer 1)

The manuscript addresses a relevant problem and provides a clear illustration of how different aggregation semantics can affect prioritization within cooperative automated decision-making systems. Nevertheless, the present theoretical contribution is limited, and some experimental conclusions are broader than the available evidence supports. The manuscript could become suitable for publication after a substantial revision that clarifies its novelty, strengthens and generalizes the theoretical results, improves the experimental protocol, and moderates or further supports the claims concerning robustness and policy compliance.

---

## Reviewer 2

### Overall assessment

The manuscript addresses an interesting and timely problem concerning the role of aggregation operators in Cooperative Automated Decision-Making Systems (CADEMAS). The central idea is potentially valuable, and the overall presentation is technically sound. However, several aspects require substantial clarification and strengthening before the manuscript can be considered for publication.

---

### Comments

1. **Conditions for veto and rank reversal.** Discuss the precise conditions under which the minimum and weighted geometric mean preserve contextual exclusions, and explicitly establish the conditions leading to rank reversal under linear aggregation.

2. **Novelty and related work.** More explicitly distinguish the contribution from existing work on cooperative automated decision making, context-aware decision support, aggregation operators, and multi-criteria decision-making. A dedicated discussion of the novelty and theoretical/practical contributions would improve the manuscript.

3. **Literature review, research questions, and gaps.** In the literature review section, add some recent and relevant articles. What are the research questions that this study wants to address? Research gaps and contributions should be better linked to the literature review.

4. **Chronological related-work table.** To demonstrate the true innovation of the work, consider including a chronological table of related publications at the end of the Introduction. This would highlight the hierarchy of the literature study and succinctly showcase the novelty of the work.

5. **Universality of properties.** Discuss whether the reported properties hold universally or only under specific normalization, weighting, or score-domain assumptions.

6. **Choice of operators.** The comparison is restricted to the standard linear operator, conjunctive minimum, and weighted geometric mean. Explain more clearly why these three operators are sufficient to support the main conclusions. There are other operators such as OWA operators, parameterized t-norm/t-conorm families, Choquet-type integrals, or other generalized means.

7. **Sensitivity analyses.** Report sensitivity analyses to demonstrate whether the conclusions remain stable under substantially different parameter settings.

8. **Writing style.** Scientific articles are typically written in the third person to maintain objectivity. Avoid first-person formulations such as “We provide”, “We demonstrate”, “We compare”, etc.
