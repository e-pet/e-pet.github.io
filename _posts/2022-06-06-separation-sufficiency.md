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
In this context, I have recently tried to wrap my head around why – intuitively – it is impossible for any classifier to achieve *separation* and *sufficiency* at the same time (unless a) the classifier is a perfect classifier or b) there are no base rate differences between groups – we will get to these details in a minute).
Since part of my troubles arose from a misunderstanding of what separation and sufficiency actually mean, let us start by revisiting their definitions.

In the following, assume that we have $n$ groups, $a=1$ , ..., $a=n$, and a binary outcome, $y\in \{True, False\}$, and assume that we are analyzing a classifier that returns a *risk score* $r\in [0, 1]$. 

A classifier fulfills **separation** if $R⊥A | Y$, i.e., the risk score $R$ is independent of the group assignment $A$ given the (observed) outcome $Y$.
In the binary outcome case, this translates to *balance of the average score in the positive/negative class* (Kleinberg et a. 2016), i.e.,
$$
\begin{gather}
E_{a=i, y=T}[R] = E_{a=j, y=T}[R] \\
E_{a=i, y=F}[R] = E_{a=j, y=F}[R]
\end{gather}
$$
for all pairs of groups $(i,j)$.
This seems like a reasonable requirement: we would like the classifier to be 'equally sure' about its predictions in all groups; otherwise, something must be off – right? (Wrong! Read on...)

A classifier fulfills **sufficiency** if $Y⊥A | R$, i.e., the observed outcome $Y$ is independent of the group assignment $A$ given the risk score $R$.
A slightly stronger requirement (see Barocas et al. for a derivation that this is indeed a stronger requirement) is that the classifier be **calibrated by group**, i.e., that it satisfies
$$
P(T|R=r, a=i) = r \quad \forall\, r, i \in \mathop{supp_i}(R) \times \{1, \ldots, n\}.
$$
This is a property that certainly is very desirable for any model to be employed in a high-stakes environment, such as healthcare (my main field).
Fortunately, it is also a property that is optimized for by most standard learning procedures, [including maximum likelihood estimation / cross-entropy minimization](https://e-pet.github.io/posts/2022/2022-04-03-maximum-likelihood/).

Now, unfortunately, it is a well-known result (due to Kleinberg et al. 2016) that these two properties cannot hold at the same time *for any classifier* – regardless of model class, how it is constructed, etc. – except if either
a) the classifier is *perfect*, i.e., always returns the correct prediction for all samples, or
b) the base rates $p(y|a)$ are the same in all groups.
This sounds like a real bummer, so let us try to understand why it is that the two conditions are incompatible.

### Incompatibility of separation and sufficiency
**Most of the following discussion is very closely based on Kleinberg et al. 2016. I take no credit whatsoever for the ideas presented below.**

Given a group-wise calibrated risk score, we find the average risk score in group $i$ to be
$$
\begin{align}
E_{a=i}[R] &= \int r \cdot P(r|a=i) \,\mathrm dr \\
&= \int P(T|R=r, a=i) \cdot P(r|a=i) \,\mathrm dr \\
&= \int P(T, r | a=i) \,\mathrm dr \\
&= P(T|a=i),
\end{align}
$$
i.e., it is equal to that group's _base rate_ (as one might expect!), which we will denote by $P(T|a=i) \eqqcolon p_i$.
Equivalently, we can decompose the average risk score into two terms as
$$
E_{a=i}[R] = p_i = p_i \cdot E_{a=i, y=T}[R] + (1-p_i) \cdot E_{a=i, y=F}[R].
$$
Now set $x_i\coloneqq E_{a=i, y_i=F}[R]$ and $y_i \coloneqq E_{a=i, y=T}[R]$, and it becomes apparent that the average scores in the positive / negative classes of each group must satisfy the line equation
$$
	y_i = 1 - \frac{1-p_i}{p_i} x_i
$$
if the risk score is calibrated by group.

![](2022-01-19-Separation-sufficiency.png)

To achieve separation, we would need equality of $x_i$ and $y_i$ for all $i$, i.e., all lines would have to intersect.
This, however, can only happen if all the base rates $p_i$ are equal (in this case, all lines are identical) or if $y_i=1$ for all $i$, i.e., the classifier is perfect.
In all other cases, separation and sufficiency are not compatible.

The **essential intuition** here is that the average score of a calibrated classifier within each group is equal to the base rate of that group.
From this, it is already almost apparent that equal average risk scores in the positive/negative classes of each group cannot be achieved, if there are base rate differences (and the classifier is not perfect).


### (Limited) compatibility of calibration and error rate balance
Notice that it is certainly feasible to construct a classifier that is 
a) calibrated by group, and
b) achieves *error rate parity*, i.e., equal TPR and FPR across groups!

The following example will illustrate this.
Assume we have two classes, $a=1$ and $a=2$, and a binary outcome, $y\in \{True, False\}$. The two groups have varying base-rates:
 
Class label $a$ |  $y=True$ | $y=False$
--------- |  - | -
  Group 1  | 40 | 10
  Group 2  | 10 | 40

Assume furthermore that we have a classifier with the following confusion table:

Actual outcome \ predicted outcome | True | False
---- | ----- | ----
True | $TP_1=40$, $TP_2=10$ | $FN_1=FN_2=0$
False | $FP_1=2, FP_2=8$ | $TN_1=8, TN_2=32$

Thus, the classifier has
$$ TPR = \frac{TP}{P} = 1$$
for both classes and
$$ FPR_1 = \frac{FP}{N} = \frac{2}{10} = 0.2 = \frac{8}{40} = FPR_2.$$
Thus, we have equal TPR and FPR across groups.
This is *not* equivalent to achieving separation, though!

Moreover, we have the positive predictive values
$$ PPV_1 = \frac{TP}{TP+FP} = \frac{40}{42} \neq \frac{10}{18} = PPV_2$$

So far, we did not have to take this risk scores $R$ returned by the classifier into account at all!
Up until this point, we only looked at TPR/FPR/PPV etc., all of which are functions of the risk score _and_ the selected classification thresholds.
To achieve **calibration by group**, we simply assign the correct classification probabilities as risk scores:
$$
\begin{align}
&P(T|R=40/42, a=1) = PPV_1 = \frac{40}{42} \\
&P(T|R=0, a=1) = (1-NPV_1) = 0 \\
&P(T|R=10/18, a=2) = PPV_2 = \frac{10}{18} \\
&P(T|R=0, a=2) = (1-NPV_2) = 0 \\
\end{align}
$$
From Theorem 1.1 in Kleinberg et al. (2016) and our discussion above, we already know that this classifier *cannot* achieve separation. To check this, we simply observe that
$$
\begin{gather}
E_{a=1,y=T}[R] = \frac{40}{42} \neq E_{a=2,y=T}[R] = \frac{10}{18}\\
E_{a=1, y=F}[R] = \frac{2 \cdot 40/42}{10} = \frac{4}{21} \neq E_{a=2, y=F}[R] = \frac{8 \cdot 10/18}{40} = \frac{1}{9}.
\end{gather}
$$
So, in the example above, we *have* error rate (TPR, FPR) balance and calibration by group, but we do *not* have separation (as would be impossible as per the above result).

However, in the general setting, it is unlikely that it will be possible to achieve both perfect calibration by group and error rate balance:
this requires that the ROC curves for the two groups intersect at these error rates.
(And, if we want this to be achievable for _all_ values of the error rates, then the ROC curves must be identical.)
It is highly unlikely that this will hold in any practical scenario:
even disregarding the exact shape of the ROC curve, a weaker requirement would be that the AUROCs for the two groups should be equal, i.e., the discriminative power of the model should be identical for the two groups.
There is no reason to expect this to be true, and the only way of enforcing this would be to actively reduce performance on the better-performing group.

### References
- Kleinberg, Mullainathan, Raghavan (2016) *Inherent Trade-Offs in the Fair Determination of Risk Scores.* [arxiv link](http://arxiv.org/abs/1609.05807v2)
- Barocas, Hardt, Narayanan (2021) *Fairness and Machine Learning*. <https://fairmlbook.org/>. **Caution**: while certainly a useful resource, the book is not yet finished, and some proofs and derivations did not seem fully rigorous and convincing to me. In particular, I find the discussion of separation in the current version to be misleading (it is based on Hard, Price, Srebro (2016), *Equality of Opportunity in Supervised Learning*), as it seems to suggest that separation is identical to achieving error rate balance in binary classification. This is only the case when assessing separation w.r.t. the classifier's _predictions_ $\hat{y}\in\{0,1\}$ (which seems like a weird thing to do to me), however, and not when assessing separation w.r.t. the _risk scores_ $r\in [0,1]$ (see above), as discussed above.

-----
