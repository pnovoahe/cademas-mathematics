# Response to reviewers' comments

## Reviewer 1
---
### Comments 1: Novelty. 
The novelty of the manuscript should be explained more precisely. The considered operators and their principal properties are well known in aggregation theory, fuzzy logic, and multi-criteria decision making. At present, the contribution appears mainly to consist of applying these operators within the CADEMAS framework. The authors should clearly distinguish their original theoretical or methodological contribution from a comparative application of established aggregation mechanisms. A dedicated comparison with the closest existing studies would be helpful.  

#### Response 1
Thank you for this important observation. We agree that the original version of the manuscript did not sufficiently distinguish the methodological contribution of the work from the use of well-established aggregation operators.

The contribution of this paper is not the introduction of new aggregation operators nor the extension of their mathematical properties. Instead, our contribution is methodological: we reinterpret aggregation within the CADEMAS framework as a governance and policy-design mechanism that regulates the interaction between predictive evidence and contextual knowledge. From this perspective, different aggregation operators represent alternative policy semantics with different levels of compensation between automated recommendations and contextual constraints.

To clarify this point, we substantially revised the Introduction. In particular, we added a dedicated discussion positioning aggregation as a policy-enforcement mechanism in hybrid Automated Decision-Making architectures and explaining how this perspective differs from the traditional use of aggregation operators in fuzzy MCDM studies. We also added a comparison with related work to emphasize that our objective is not to compare operators in isolation, but to systematically analyze how different aggregation semantics affect policy compliance, robustness to predictive overconfidence, sensitivity to prediction errors, and ranking stability within a cooperative decision-making architecture.

The new text added to the Introduction is reproduced below (highlighted in red in the revised manuscript):

"Although the mathematical properties of aggregation operators have been extensively studied in fuzzy logic and MCDM, their role as policy-enforcement mechanisms within hybrid Automated Decision-Making architectures remains largely unexplored. Most existing context-aware frameworks, such as those proposed by Afzal et al. [18] and Shyur [19], either treat aggregation as a static numerical step or embed contextual knowledge directly into the predictive process. Consequently, aggregation is rarely analyzed as an independent design component that governs how predictive evidence and contextual information interact.
This limitation is particularly relevant in cooperative decision-making systems such as CADEMAS. In these architectures, the aggregation operator is not merely a numerical device but an implicit decision policy that determines the extent to which predictive evidence can compensate for contextual constraints, or vice versa. Yet, despite the rich body of aggregation research, most studies focus on proposing new operators or demonstrating their usefulness in specific application domains. Comparatively little attention has been paid to understanding how alternative aggregation semantics affect the behavior of an entire cooperative decision-making framework. Questions such as policy compliance, robustness to predictive overconfidence, sensitivity to localized prediction errors, and ranking stability under different aggregation regimes remain largely unexplored. By explicitly separating evidence generation from contextual assessment, CADEMAS provides an ideal setting to isolate aggregation as a modular policy lever and systematically study its effects, distinguishing this work from traditional comparative applications of aggregation operators such as Blancas and Contreras [24]."

---

### Comments 2: Elementary theory. 
The theoretical analysis is rather elementary. Proposition 1 is obtained by directly rearranging the corresponding score inequalities, while Proposition 2 follows immediately from strict monotonicity. Although these statements are correct, their mathematical novelty is limited. The authors should consider formulating results for broader classes of aggregation operators characterized by properties 

#### Response 2: 
Thank you for pointing this out. We fully agree with this comment.
 
Therefore, we have entirely restructured Section 3.3 to elevate the theoretical analysis. Rather than restricting our proofs to algebraic rearrangements of specific formulas, we now formulate our results around broader axiomatic classes: the Veto family (characterized by zero absorption), the Fully compensatory family, and the Non-linear sub-compensatory family.
 
We have generalized the rank reversal and predictive preservation conditions into formal theorems based on properties such as strict monotonicity and compensability bounds. Specifically, we introduced a new general theorem (Theorem 2) based on the Intermediate Value Theorem to establish rank reversal thresholds for any continuous compensatory regime. The specific linear and geometric formulas are now correctly positioned as corollaries derived from this broader theoretical framework. This change can be found in Section 3.3.

---
### Comments 3: Terminology in Proposition 1. 

The terminology used in Proposition 1 requires correction. The manuscript refers to a “strict conjunctive aggregator” and states that this class includes both the minimum and the algebraic product. In the standard classification of triangular norms, the minimum operator is not a strict t-norm. The authors should instead state the exact assumptions required for the result, for example $A(x,0)=0$ and $A(x,y)>0$ whenever $x>0$ and $y>0$. The proposition should then be formulated in terms of these properties.

#### Response 3: 

Thank you for pointing this out. We fully agree with this comment.

As part of the complete rewrite of Section 3.3, we have corrected this terminology. We removed the incorrect classification of the minimum operator as a strict t-norm. Instead, following your precise suggestion, we now define the non-compensatory "Veto family" explicitly through the exact algebraic assumptions: zero absorption ($A(r,0)=0$) and strict positivity ($A(r,q)>0$ whenever $r,q>0$). Besides, Theorem 1 has been formulated in terms of these exact properties, making the result independent of specific t-norm classifications. This change can be found in Section 3.3.

---

### Comments 4: Weighted geometric boundary cases. 

The definition of the weighted geometric aggregation operator $A_G(R,Q)=R^{\lambda}Q^{1-\lambda},\qquad \lambda\in[0,1]$, requires additional care at the boundary values $\lambda=0$ and $\lambda=1$. If one of the inputs is zero, expressions involving $0^0$ may occur. Moreover, the claimed veto property for $Q=0$ does not hold in the same form when $\lambda=1$. The authors should either restrict the parameter to $\lambda\in(0,1)$ or provide precise conventions and treat the boundary cases separately.

#### Response 4: 
Thank you for pointing this out. We fully agree with this comment.

To avoid the undefined $0^0$ mathematical expressions and to ensure that the strict veto property holds consistently without requiring complex boundary exceptions, we have adopted your suggestion. We have explicitly restricted the parameter domain strictly to $\lambda \in (0,1)$ for the weighted geometric operator throughout the entire manuscript.

---

### Comments 5: Redundant “normalization” property. 
The list of properties imposed on aggregation operators contains a formal redundancy. The statement $A\colon [0,1]^2\longrightarrow[0,1]$ is presented as “normalization”, although it merely specifies the domain and codomain of the operator. The authors should provide an appropriate definition of normalization or remove this item. It would also be useful to state whether continuity, commutativity, idempotency, or strict monotonicity are assumed and which considered operators satisfy these properties.

#### Response 5: 
Thank you for pointing this out. We fully agree with this comment. We have removed the formal redundancy. The mapping specifying the domain and codomain is no longer labeled as "normalization". Instead, we now correctly define normalization through the standard boundary conditions ($A(0,0)=0$ and $A(1,1)=1$). Furthermore, as suggested, we have added a comprehensive breakdown explicitly stating which of the considered operators satisfy continuity, commutativity, idempotency, and strict monotonicity. This detailed classification has been included at the beginning of Section 3."Throughout this work, we assume that all aggregation operators satisfy the following basic properties... Boundary conditions (Normalization): $A(0,0)=0,\quad A(1,1)=1$... The specific operators considered in this paper satisfy the following additional properties: The linear operator $A_L$ is continuous, strictly increasing in both arguments (for $\lambda\in(0,1)$), and idempotent. It is only commutative in the symmetric case ($\lambda=0.5$)... [Followed by the properties for the minimum, algebraic product, and weighted geometric mean]."

---

### Comments 6: Experiments vs. algebraic properties. 
A substantial part of the experimental findings follows directly from the algebraic properties of the selected operators. In particular, the conclusion that the minimum and weighted geometric operators exclude alternatives with $Q_i=0$ follows immediately from their zero-absorption property. The authors should explain what additional information is provided by the Monte Carlo experiments and include experiments involving nonzero intermediate context scores, where the behavior of the operators is less immediate.

#### Response 6: 
Thank you for this highly insightful comment. We fully agree that verifying the $Q_i=0$ boundary experimentally is redundant given the zero-absorption property of the non-linear operators. To address this, we have made two major additions to Section 4:

1. Clarification of the Monte Carlo objectives: We have rewritten the introduction of Section 4 to explicitly state the purpose of the simulations. We clarify that while Section 3 establishes exact algebraic bounds for isolated pairs, the Monte Carlo experiments are designed to evaluate macroscopic, systemic effects. Specifically, they demonstrate how different aggregation regimes absorb or amplify predictive noise, calibration errors, and overconfidence across an entire population to determine the final empirical Top-$K$ selection boundaries.

2. New intermediate context experiment: As suggested, we have introduced a completely new experimental scenario (Section 4.3: Compensation Limits under Intermediate Context). This experiment evaluates a subgroup with a moderate, nonzero context score ($Q_i=0.5$) coupled with an artificially high predictive score ($R_i \sim \mathcal{N}(0.9, 0.05^2)$). This allows us to observe the non-trivial compensation limits of the operators. The results empirically demonstrate our newly established corollaries: while the linear operator allows the inflated predictions to easily override the mediocre context, the geometric operator heavily penalizes the imbalanced scores due to its sub-compensatory semantics, delaying their entry into the Top-$K$ tier.

---

### Comments 7: Experimental protocol. 

The experimental protocol should be strengthened. Only \(N=30\) Monte Carlo trials are used to construct empirical 95% confidence intervals. This number is relatively small for percentile-based interval estimation. The authors should substantially increase the number of trials, explain the precise construction of the confidence intervals, and report the random seeds or all other information required to reproduce the results.

#### Response 7: 

Thank you for this constructive methodological observation. We completely agree that $N=30$ was insufficient for stable percentile-based confidence interval estimation.To strengthen the experimental protocol and ensure statistical robustness, we have re-run all Monte Carlo experiments increasing the number of trials substantially from $N=30$ to $N=1000$. All figures and confidence bands have been updated accordingly. Furthermore, we have updated the text in Section 4 to explicitly detail the construction of these intervals and address reproducibility:

"Each experiment consists of $N=1000$ independent Monte Carlo trials. Reported curves correspond to trial averages, while shaded regions indicate non-parametric 95% confidence intervals. These intervals are constructed precisely by computing the empirical 2.5th and 97.5th percentiles of the evaluation metrics across the $N$ trials at each evaluation point. To ensure full reproducibility of the experimental findings, a global pseudo-random seed was fixed across all simulations, and the implementation code is provided..."

---

8. **Robustness claim (Section 4.3).** The claim that non-linear operators are more robust to contextual uncertainty is based on a narrow experimental setting. The comparison in Section 4.3 uses only one value, \(\lambda=0.35\), one noise distribution, and one synthetic population model. Moreover, the reported difference in Kendall’s rank correlation is moderate. The authors should investigate several values of \(\lambda\), different noise models and population distributions, and possibly additional ranking-stability measures. Otherwise, the conclusions should be restricted to the experimental conditions considered.