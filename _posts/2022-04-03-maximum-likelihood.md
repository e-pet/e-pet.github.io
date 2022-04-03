---
title: Maximum likelihood, cross-entropy, risk minimization
permalink: /posts/2022/2022-04-03-maximum-likelihood/
date: 2022-04-03
tags:
  - Machine Learning
  - Math
  - Calibration
---

... *really*, yet another post about about maximum likelihood (ML) estimation? Well - yes; I could not find a source that summarized exactly the things I needed to know, so here it is. What will you find?

- A derivation of maximum likelihood estimation
- A derivation of its equivalence to cross-entropy minimization, empirical risk minimization, and least squares estimation
- A summary of some important properties of ML estimation, including why it tends to produce well-calibrated estimators as well as its robustness to model misspecification

The discussion is fully general and applies to both regression and classification settings, i.e., continuous or discrete variables.

Let's go.

### What is maximum likelihood estimation?
The idea is simple: given a model $q(y | x; \theta)$ of the conditional distribution of a target variable $y$ given input data $x$, find the set of parameters $\theta_{ML}$ that maximizes the likelihood of observing the data as they were observed:
$$
\theta_{ML} = \arg\max_{\theta} q(Y | X; \theta),
$$
where $Y$ and $X$ denote vectors (or matrices) representing a given dataset.

If the individual observations $(y_i, x_i)$ are assumed independent of one another, this can be rewritten as 
$$
\begin{align}
\theta_{ML} &= \arg\max_{\theta} \prod_{i=1}^N q(y_i | x_i; \theta), \\
&= \arg\max_{\theta} \ln \left(\prod_{i=1}^N q(y_i | x_i; \theta) \right) \\
&= \arg\max_{\theta} \sum_{i=1}^N \ln q(y_i | x_i; \theta) \\
&= \arg\min_{\theta} - \sum_{i=1}^N \ln q(y_i | x_i; \theta).
\end{align}
$$

Thus, we arrive at the usual formulation of ML estimation as minimizing the *negative log likelihood* (NLL), sometimes also called the *energy* or the *cross-entropy* (the latter will be discussed in more detail below).

### Maximum likelihood estimation as empirical risk minimization
Maximum likelihood estimation can be cast within the extremely broad framework of *empirical risk minimization* (ERM):
$$
\begin{align}
\theta_{ML} &= \arg\min_{\theta} - \sum_{i=1}^N \ln q(y_i | x_i; \theta)\\
&= \arg\min_{\theta} E_{p_{emp}}\left[-\ln q(y|x,\mathbf{\theta})\right],
\end{align}
$$
where $E_p$ is the [expected value](https://en.wikipedia.org/wiki/Expected_value "Expected value") operator with respect to the distribution $p$, and $p_{emp}$ denotes the empirical measure defined by the observed dataset $(X, Y)$.  Thus, likelihood maximization is identical to empirical risk minimization if the *risk* defined as 
$$
\mathcal{R}(x, y, \theta) = -\ln q(y|x,\mathbf{\theta}).
$$

### Maximum likelihood estimation as cross-entropy minimization
The [cross-entropy](https://en.wikipedia.org/wiki/Cross_entropy) of a first distribution $q$ relative to a second distribution $p$ is defined as 
$$ H(p, q) = -E_p[\ln q].$$
Returning to our identification problem,  if we choose $p=p_{emp}(y|x)$ and $q=q(y|x; \theta)$, we observe that maximizing the likelihood $q(Y|X; \theta)$ is identical to minimizing the cross-entropy of the distribution $q(y|x; \theta)$ relative to the empirical distribution $p_{emp}(y|x)$. 

### Maximum likelihood estimation as Kullback-Leibler divergence minimization
The definition of the cross-entropy above can be reformulated in terms of the [Kullback–Leibler divergence](https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence "Kullback–Leibler divergence") (a measure of differences between distributions, also known as the _relative entropy_), since
$$ \begin{align}
H(p,q)&= -E_p\left[\ln q(x)\right] \\
&= -E_p\left[ \ln \frac{p(x) q(x)}{p(x)}\right] \\
&= -E_p\left[\ln p(x) + \ln \frac{q(x)}{p(x)}\right] \\
&= -E_p\left[\ln p(x) - \ln \frac{p(x)}{q(x)}\right] \\
&= -E_p\left[\ln p(x)\right] + E_p\left[\frac{p(x)}{q(x)}\right] \\
&= H(p) + D_{KL}(p||q),
\end{align}$$
where $H(p)$ denotes the [*entropy*](https://en.wikipedia.org/wiki/Entropy_(information_theory)) of the distribution $p$ and $D_{KL}(p||q)$ the Kullback-Leibler divergence.

Again choosing $p=p_{emp}(y|x)$ and $q=q(y|x; \theta)$, and noting that $H(p)$ is independent of our choice of model parameters $\theta$, we observe that maximizing the likelihood of the data is also identical to minimizing the Kullback-Leibler divergence between the empirical distribution $p_{emp}(y|x)$ and the model $q(y|x; \theta)$. (We would, of course, prefer to minimize the divergence with respect to the true, data-generating process $p(y|x)$ instead of the empirical distribution. However, this is obviously infeasible since $p(y|x)$ is unknown.)

### Maximum likelihood and least squares
Known since the eighteenth century, *least-squares estimation* is possibly the single most famous parameter estimation paradigm. It turns out that under mild assumptions, least-squares estimation coincides with maximum likelihood estimation. For an arbitrary, possibly nonlinear regression model $f(x; \theta)$, we have
$$
\theta_{LS} = \arg\min_\theta \sum_{i=1}^N||y_i - f(x_i; \theta)||^2.
$$
If we now assume a Gaussian noise model 
$$
q(y_i|x_i, \theta, \sigma_{\varepsilon}) = \mathcal{N}(f(x_i; \theta), \sigma_{\varepsilon}^2),
$$
we obtain for the maximum likelihood estimator that
$$
\begin{align}
\theta_{ML}, \sigma_{\varepsilon, ML} &= \arg \min_{\theta, \sigma_{\varepsilon}} - \sum_{i=1}^N \ln q(y_i|x_i, \theta, \sigma_{\varepsilon}) \\
&= \arg \min_{\theta, \sigma_{\varepsilon}} - \sum_{i=1}^N \ln \frac{1}{\sqrt{2\pi \sigma_{\varepsilon}^2}} \mathrm{e}^{-\frac{1}{2 \sigma_{\varepsilon}^2} (y_i - f(x_i; \theta))^2} \\
&= \arg \min_{\theta, \sigma_{\varepsilon}} \sum_{i=1}^N (y_i - f(x_i; \theta))^2 + \frac{N}{2} \ln 2 \pi \sigma_{\varepsilon}^2.
\end{align}
$$
Since the optimization with respect to the regression parameters $\theta$ can be carried out independently of the value of $\sigma_{\varepsilon}$, it follows that
$$
\theta_{ML} = \theta_{LS}
$$
for arbitrary functions $f(x; \theta)$. (Again, this relies on the assumption of a Gaussian noise model.)

### Consistency, efficiency, calibration
Maximum likelihood estimation is *asymptotically consistent*: if there is a unique *true* value $\theta^*$ for which $p(y|x) = q(y|x; \theta^*)$ (in other words, there is no *model mismatch* or *model error*), then a maximum likelihood estimator converges towards that value as the number of samples increases. (However, notice that even in the case where there *is* model mismatch, we retain the reassuring property that the ML estimator minimizes the KL divergence between the empirical data distribution and the identified model.)

Moreover, maximum likelihood estimation is *asymptotically efficient*, meaning that for large sample numbers, no consistent estimator achieves a lower mean squared parameter error than the maximum likelihood estimator. (In other words, it [reaches](https://en.wikipedia.org/wiki/Maximum_likelihood_estimation#Efficiency) the [Cramér-Rao lower bound](https://en.wikipedia.org/wiki/Cram%C3%A9r%E2%80%93Rao_bound).)

Finally, ML estimators also tend to be *well-calibrated*, meaning that 
$$
p(y|x, R=r) = r \quad \forall\, r,
$$
where $R$ denotes the (risk score) output of the trained model. This is readily apparent from the fact that an ML-optimal model minimizes the KL divergence from the data-generating distribution, as discussed above: the optimum is only obtained if $p(y|x) = q(y|x; \theta^*)$. (For a more detailed discussion about how maximum likelihood estimation implies calibration, refer to [Liu et al. 2019](https://arxiv.org/pdf/1808.10013.pdf). For the same reason, [the negative log likelihood has also been proposed as a *calibration measure*](https://arxiv.org/pdf/2002.06470.pdf). Notice, however, that it is not a pure measure of calibration; instead, it measures a mixture of calibration and *separation*.)

### Properties of the optimization problem
The likelihood landscape (as a function of the parameters $\theta$ to be optimized) is, in general, non-convex. (It also depends on the way the model is parameterized.) Thus, global optimization strategies are required if local minima are to be escaped. (One of the various benefits of *stochastic* gradient descent is that it is [capable of escaping local minima](https://leon.bottou.org/publications/pdf/nimes-1991.pdf) to some degree. It is, however, of course not a true global optimization strategy.) 

On the positive side, however, the negative log likelihood represents a [_proper scoring rule_](http://www.nowozin.net/sebastian/blog/how-good-are-your-beliefs-part-1-scoring-rules.html) - as opposed to, e.g., classification accuracy, which [is an improper scoring rule and should never be used as an optimization loss function or to drive feature selection and parameter estimation](https://www.fharrell.com/post/class-damage/).

Finally, an interesting remark on the potential for overfitting when doing maximum likelihood estimation, due to Bishop (2006), p. 206:

> It is worth noting that maximum likelihood can exhibit severe over-fitting for data sets that are linearly separable. This arises because the maximum likelihood solution occurs when the hyperplane corresponding to $\sigma = 0.5$, equivalent to $w^T\phi=0$, separates the two classes and the magnitude of $w$ goes to infinity. In this case, the logistic sigmoid function becomes infinitely steep in feature space, corresponding to a Heaviside step function, so that every training point from each class k is assigned a posterior probability $p(C_k|x) = 1$. Furthermore, there is typically a continuum of such solutions because any separating hyperplane will give rise to the same posterior probabilities at the training data points. [...] Maximum likelihood provides no way to favour one such solution over another, and which solution is found in practice will depend on the choice of optimization algorithm and on the parameter initialization. Note that the problem will arise even if the number of data points is large compared with the number of parameters in the model, so long as the training data set is linearly separable. The singularity can be avoided by inclusion of a prior and finding a MAP solution for $w$, or equivalently by adding a regularization term to the error function.

How does this not contradict all the nice properties of maximum likelihood estimation discussed above, such as consistency, efficiency, and calibration? Well, in the case discussed by Bishop, there simply is no unique optimum - instead, there is a manifold of possible solutions. As Bishop remarks, to obtain a specific solution, some prior information must be included about which of the infinitely many solutions of the estimation problem to prefer. Notice that the maximum likelihood solution discussed by Bishop is, in fact, calibrated: it *correctly* assigns high confidence to its predictions.

### The special case of binary classification
For _discrete_ probability distributions $p$ and $q$ with the same support $\mathcal{Y}=\{0, 1\}$, the (binary) cross-entropy simplifies (again assuming $p=p_{\text{emp}}$) to the often-used formulation
$$
\begin{align}
H(p,q) &= -E_p[\ln q] \\
&= -\sum_{y\in\mathcal{Y}} p(y|x) \ln q(y|x)\\
&= -\sum_{i=1}^N y_i \ln q(y_i|x_i) + (1-y_i) \ln (1-q(y_i|x)).
\end{align}
$$

### References
Bottou (1991), Stochastic Gradient Learning in Neural Networks. https://leon.bottou.org/publications/pdf/nimes-1991.pdf

Ljung (1999), System Identification: Theory for the User. Prentice Hall, second edition edition.

Bishop (2006), Pattern Recognition and Machine Learning. Springer.

Nowozin (2015), How good are your beliefs? Part 1: Scoring Rules. http://www.nowozin.net/sebastian/blog/how-good-are-your-beliefs-part-1-scoring-rules.html

Goodfellow, Bengio, Courville (2016), Deep Learning. https://www.deeplearningbook.org/

Guo, Pleiss, Sun, Weinberger (2017), On Calibration of Modern Neural Networks, https://arxiv.org/pdf/1706.04599.pdf

Liu et al. (2018), The implicit fairness criterion of unconstrained learning. https://arxiv.org/pdf/1808.10013.pdf

Harrell (2020), Damage Caused by Classification Accuracy and Other Discontinuous Improper Accuracy Scoring Rules. https://www.fharrell.com/post/class-damage/

Ashukha et al. (2021), Pitfalls of in-domain uncertainty estimation and ensembling in deep learning. https://arxiv.org/pdf/2002.06470.pdf

------
