---
title: "Calibration by group, error rate parity, sufficiency, and separation"
permalink: /posts/2022/2022-06-06-separation-sufficiency/
date: 2022-06-06
read_time: False
tags:
  - Machine Learning
  - Calibration
  - Algorithmic Fairness
---

In the field of algorithmic fairness, it is well known that there are several definitions of fairness that are impossible to reconcile except in (practically irrelevant) corner cases.
In this context, I have recently tried to wrap my head around why – intuitively – it is impossible for any classifier to achieve *separation* and *sufficiency* at the same time (unless either the classifier is a perfect classifier or there are no base rate differences between groups – we will get to these details in a minute).
Since part of my troubles arose from a misunderstanding of what separation and sufficiency actually mean, let us start by revisiting their definitions.

In the following, assume that we have $n$ groups, $a=1$ , ..., $a=n$, and a binary outcome, $y\in \lbrace \text{True}, \text{False} \rbrace$, and assume that we are analyzing a classifier that returns a *risk score* $r\in [0, 1].$ 

A classifier fulfills **separation** if $R \perp A \vert Y$, i.e., the risk score $R$ is independent of the group assignment $A$ given the (observed) outcome $Y$.
In the binary outcome case, this translates[^1] to *balance of the average score in the positive/negative class* (Kleinberg et a. 2016), i.e.,

$$
\begin{gather}
E_{a=i, y=T}[R] = E_{a=j, y=T}[R] \\
E_{a=i, y=F}[R] = E_{a=j, y=F}[R]
\end{gather}
$$

for all pairs of groups $(i,j)$.
This seems like a reasonable requirement: we would like the classifier to be 'equally sure' about its predictions in all groups; otherwise, something must be off – right? (Wrong! Read on...)

A classifier fulfills **sufficiency** if $Y \perp A \vert R$, i.e., the observed outcome $Y$ is independent of the group assignment $A$ given the risk score $R$.
A slightly stronger requirement (see Barocas et al. for a derivation that this is indeed a stronger requirement) is that the classifier be **calibrated by group**, i.e., that it satisfies

$$
P(T \mid R=r, a=i) = r \quad \forall\, r \in \mathop{supp_i}(R), \quad i \in \{1, \ldots, n\}.
$$

This is a property that certainly is very desirable for any model to be employed in a high-stakes environment, such as healthcare (my main field).
Fortunately, it is also a property that is optimized for by most standard learning procedures, [including maximum likelihood estimation / cross-entropy minimization](https://e-pet.github.io/posts/2022/2022-04-03-maximum-likelihood/).

Now, unfortunately, it is a well-known result (due to Kleinberg et al. 2016) that these two properties cannot hold at the same time *for any classifier* – regardless of model class, how it is constructed, etc. – except if either
<ol type="a">
  <li>the classifier is <i>perfect</i>, i.e., always returns the correct prediction for all samples, or</li>
  <li>the base rates $p(y \mid a)$ are the same in all groups.</li>
</ol>
This sounds like a real bummer, so let us try to understand why it is that the two conditions are incompatible.

### Incompatibility of separation and sufficiency
*This section is very closely based on Kleinberg et al. 2016. I take no credit whatsoever for the ideas presented below.*

Given a group-wise calibrated risk score, we find the average risk score in group $i$ to be

$$
\begin{align}
E_{a=i}[R] &= \int r \cdot P(r \mid a=i) \,\mathrm dr \\
&= \int P(T|R=r, a=i) \cdot P(r \mid a=i) \,\mathrm dr \\
&= \int P(T, r \mid a=i) \,\mathrm dr \\
&= P(T \mid a=i),
\end{align}
$$

i.e., it is equal to that group's _base rate_ (as one might expect!), which we will denote by $P(T \mid a=i) =: p_i$.
Equivalently, we can decompose the average risk score into two terms as

$$
E_{a=i}[R] = p_i = p_i \cdot E_{a=i, y=T}[R] + (1-p_i) \cdot E_{a=i, y=F}[R].
$$

Now set $x_i := E_{a=i, y_i=F}[R]$ and $y_i := E_{a=i, y=T}[R]$, and it becomes apparent that the average scores in the positive / negative classes of each group must satisfy the line equation

$$
	y_i = 1 - \frac{1-p_i}{p_i} x_i
$$

if the risk score is calibrated by group.

<figure style="width: 450px" class="align-center">
  <a href="/images/2022-01-19-Separation-sufficiency.png" title="Visualization of the incompatibility of separation and sufficiency" alt="Visualization of the incompatibility of separation and sufficiency">
  <img src="/images/2022-01-19-Separation-sufficiency.png"></a>
</figure>

To achieve balance of the average scores, we would need equality of $x_i$ and $y_i$ for all $i$, i.e., all lines would have to intersect.
This, however, can only happen if all the base rates $p_i$ are equal (in this case, all lines are identical) or if $y_i=1$ for all $i$, i.e., the classifier is perfect.
In all other cases, separation and sufficiency are not compatible.

The *essential intuition* here is that the average score of a calibrated classifier within each group is equal to the base rate of that group.
From this, it is already almost apparent that equal average risk scores in the positive/negative classes of each group cannot be achieved, if there are base rate differences (and the classifier is not perfect).


### (Limited) compatibility of calibration and error rate balance
Notice that (contrary to claims to the opposite in the literature[^2]) it is possible to construct a classifier that is 
<ol type="a">
  <li>calibrated by group, and</li>
  <li>achieves <i>error rate parity</i>, i.e., equal TPR and FPR across groups.</li>
</ol>

The following example will illustrate this.
Assume we have two classes, $a=1$ and $a=2$, and a binary outcome, $y\in \lbrace \text{True}, \text{False} \rbrace$. The two groups have varying base-rates:
 
Class label $a$ |  $y=\text{True}$ | $y=\text{False}$
--------- |  - | -
  Group 1  | 40 | 10
  Group 2  | 10 | 40

Assume furthermore that we have a classifier with the following confusion table:

Actual outcome \ predicted outcome | True | False
---- | ----- | ----
True | $\mathrm{TP}_1=40$, $\mathrm{TP}_2=10$ | $\mathrm{FN}_1=\mathrm{FN}_2=0$
False | $\mathrm{FP}_1=2, \mathrm{FP}_2=8$ | $\mathrm{TN}_1=8, \mathrm{TN}_2=32$

Thus, the classifier has

$$ \mathrm{TPR} = \frac{\mathrm{TP}}{\mathrm{P}} = 1$$

for both classes and

$$ \mathrm{FPR}_1 = \frac{\mathrm{FP}}{\mathrm{N}} = \frac{2}{10} = 0.2 = \frac{8}{40} = \mathrm{FPR}_2.$$

Thus, we have equal $\mathrm{TPR}$ and $\mathrm{FPR}$ across groups.
This is *not* equivalent to achieving separation, though!

Moreover, we have the positive predictive values

$$ \mathrm{PPV}_1 = \frac{\mathrm{TP}}{\mathrm{TP}+\mathrm{FP}} = \frac{40}{42} \neq \frac{10}{18} = \mathrm{PPV}_2$$

So far, we did not have to take this risk scores $R$ returned by the classifier into account at all!
Up until this point, we only looked at $\mathrm{TPR}$ / $\mathrm{FPR}$ / $\mathrm{PPV}$ etc., all of which are functions of the risk score _and_ the selected classification thresholds.
To achieve **calibration by group**, we simply assign the correct classification probabilities as risk scores:

$$
\begin{align}
&P(T|R=40/42, a=1) = \mathrm{PPV}_1 = \frac{40}{42} \\
&P(T|R=0, a=1) = (1-\mathrm{NPV}_1) = 0 \\
&P(T|R=10/18, a=2) = \mathrm{PPV}_2 = \frac{10}{18} \\
&P(T|R=0, a=2) = (1-\mathrm{NPV}_2) = 0 \\
\end{align}
$$

From Theorem 1.1 in Kleinberg et al. (2016) and our discussion above, we already know that this classifier *cannot* achieve separation. To check this, we simply observe that

$$
\begin{gather}
E_{a=1,y=T}[R] = \frac{40}{42} \neq E_{a=2,y=T}[R] = \frac{10}{18}\\
E_{a=1, y=F}[R] = \frac{2 \cdot 40/42}{10} = \frac{4}{21} \neq E_{a=2, y=F}[R] = \frac{8 \cdot 10/18}{40} = \frac{1}{9}.
\end{gather}
$$

So, in the example above, we *have* error rate ($\mathrm{TPR}$, $\mathrm{FPR}$) balance and calibration by group, but we do *not* have separation (as would be impossible as per the above result).

However, in the general setting, it is unlikely that it will be possible to achieve both perfect calibration by group and error rate balance:
this requires that the ROC curves for the two groups intersect at these error rates.
(And, if we want this to be achievable for _all_ values of the error rates, then the ROC curves must be identical.)
It is highly unlikely that this will hold in any practical scenario:
even disregarding the exact shape of the ROC curve, a weaker requirement would be that the AUROCs for the two groups should be equal, i.e., the discriminative power of the model should be identical for the two groups.
There is no reason to expect this to be true, and the only way of enforcing this would be to actively reduce performance on the better-performing group.
Of course, when there are large discrepancies in discriminative performance for different groups (as indicated, e.g., by widely differing AUROC values), one should, firstly, try to improve performance in the underperforming group (by, e.g., changing the model, gathering more data, or accounting for measurement biases) and, secondly, be transparent about these performance disparities as they will impact downstream applications of the model.
Nevertheless, this will almost surely not lead to identical ROC curves for the different groups, and, thus, calibration by group and (exact) error rate balance will still not be achievable at the same time.

### References
- Kleinberg, Mullainathan, Raghavan (2016) *Inherent Trade-Offs in the Fair Determination of Risk Scores.* [arxiv link](http://arxiv.org/abs/1609.05807v2)
- Barocas, Hardt, Narayanan (2021) *Fairness and Machine Learning*. <https://fairmlbook.org/>. **Note**: The discussion of separation in the book's current version (based on Hardt, Price, Srebro (2016), *Equality of Opportunity in Supervised Learning*) emphasizes assessing separation w.r.t. the classifier's _predictions_ $\hat{y}\in\lbrace 0,1 \rbrace$. This differs from the setting discussed above and by Kleinberg et al. (2016), where we consider separation w.r.t. the _risk scores_ $r\in [0,1]$.
- Alexandra Chouldechova (2017) *Fair prediction with disparate impact: A study of bias in recidivism prediction instruments.* [arxiv link](https://arxiv.org/pdf/1610.07524.pdf)

[^1]: Moritz Hardt very kindly pointed out to me that average score balance as defined in equations (1) and (2) is a *necessary* but not a *sufficient* condition for separation.
[^2]: Chouldechova (2017) writes that "when the [...] prevalence differs between two groups, a [well-calibrated] score [...] cannot have equal false
positive and negative rates across those groups." As the example above demonstrates, this is not true in general. It seems to me that it becomes true when one demands a categorization into just two risk categories (high risk / low risk), like Chouldechova does. (In our example above, the two groups would have different incidence rates in the "high-risk" category.)

-----
