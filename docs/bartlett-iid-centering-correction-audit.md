# Audit of the Bartlett IID-centering correction

## Question and verdict

`BartlettLongRunCovariance` multiplies the usual sample-mean-centered
Bartlett/Newey--West estimate by $1/r_{N,m}$, where

\[
r_{N,m}
=\frac{1}{N}\left[(N-1)-\frac{2}{N}
  \sum_{k=1}^{m}\left(1-\frac{k}{m+1}\right)(N-k)\right].
\tag{1}
\]

The implementation deserves a qualified rather than an alarming verdict:

1. **The algebra is correct.** For an IID vector history, dividing by
   $r_{N,m}$ makes the finite-bandwidth estimator exactly unbiased for
   \(\operatorname{Cov}(\bar X)=\Lambda/N\), including when
   $m>0$.
2. **It is not part of the cited Bartlett/Newey--West estimator.** The standard
   published estimator centers by the sample mean but does not apply this
   particular scalar correction. Its exact formula is a toolkit derivation
   under IID sampling. However, its leading term agrees with Wolff's published
   correction for sample-centering bias after replacing his rectangular
   window mass by the Bartlett window mass. It is therefore not an arbitrary
   inflation.
3. **It does not spoil the published consistency result in the usual
   small-bandwidth regime.** The multiplier tends to one whenever
   $(m+1)/N\to0$, including the stronger bandwidth conditions used in the
   multivariate spectral-variance consistency theorem cited below.
4. **It is not an exact bias correction or an upper bound for correlated
   histories.** It uniformly inflates the raw positive-semidefinite estimate,
   so it is conservative relative to that estimate in every linear direction,
   but either estimate may still lie above or below the true covariance.

The recommended policy is to keep this scaling available, make it an explicit
named choice, and report it separately from the Bartlett kernel. Given the
project's preference to overestimate rather than underestimate unresolved
uncertainty, keeping `iid-exact-centering-inflation` enabled by default is
defensible: it always inflates the raw PSD estimate, is exact for IID data, and
has leading-order support for correlated data from Wolff. A source-faithful
unscaled Bartlett variation should also be available. The report must not call
the inflation an exact correlated-process correction or a literature upper
bound.

## Literature facts

Newey and West define the triangular weights

\[
w_k=1-\frac{k}{m+1}
\]

and form a lag-weighted covariance estimate with a common sample-size
normalization. Their Eq. (5) is positive semidefinite and their theorems give
consistency under stated asymptotic conditions. They do not give Eq. (1):
[Newey and West, *Econometrica* 55 (1987), 703--708,
DOI 10.2307/1913610](https://doi.org/10.2307/1913610), with an
[author-hosted/NBER working-paper copy](https://users.ssc.wisc.edu/~behansen/718/NeweyWest1987.pdf).

For the MCMC vector-mean problem, Vats, Flegal, and Jones define multivariate
sample autocovariances using $Y_t-\bar Y_n$, then sum them with a lag window.
Their Theorem 1 proves strong consistency under their Conditions 1--4. For the
modified Bartlett window, their lag-window discussion gives sufficient
conditions including $b_n^2/n\to0$, where $b_n$ is the truncation point.
They likewise do not apply Eq. (1): [Vats, Flegal, and Jones, *Bernoulli* 24
(2018), 1860--1909, DOI 10.3150/16-BEJ914](https://doi.org/10.3150/16-BEJ914),
[author manuscript arXiv:1507.08266v3](https://arxiv.org/abs/1507.08266v3).

These two sources establish the standard estimator, sample-mean centering,
positive semidefiniteness, and asymptotic consistency. They are not evidence
for the toolkit's scalar finite-sample multiplier.

Wolff analyzes precisely the sample-centering bias that motivates such a
multiplier. His Eq. (32) gives, to leading order,

\[
\mathbb E[\widehat\Gamma_{\alpha\beta}(t)]
-\Gamma_{\alpha\beta}(t)
\simeq-\frac{C_{\alpha\beta}}{N},
\tag{2}
\]

where $C_{\alpha\beta}$ is the full autocorrelation sum. For a rectangular
sum from $-W$ through $W$, his Eq. (49) therefore multiplies the result by

\[
1+\frac{2W+1}{N}.
\tag{3}
\]

Wolff explicitly describes this as cancellation of the leading centering bias,
not as an exact finite-sample identity or an upper-bound theorem:
[Wolff, *Computer Physics Communications* 156 (2004), 143--153,
DOI 10.1016/S0010-4655(03)00467-3](https://doi.org/10.1016/S0010-4655(03)00467-3),
[author manuscript arXiv:hep-lat/0306017v4](https://arxiv.org/abs/hep-lat/0306017v4).

For Bartlett weights, the total two-sided lag-window mass is

\[
1+2\sum_{k=1}^{m}\left(1-\frac{k}{m+1}\right)=m+1=b.
\tag{4}
\]

Thus the Wolff-style leading correction for a Bartlett window is $1+b/N$.
As shown below, the toolkit's exact-IID multiplier satisfies

\[
\frac1{r_{N,m}}
=1+\frac bN+O\!\left(\frac{b^2}{N^2}\right).
\tag{5}
\]

This is meaningful literature support for its *leading correlated-process
behavior*. It does not turn the toolkit's all-orders IID formula into a
published exact correction for correlated histories.

## Independent IID derivation

Let $X_1,\ldots,X_N\in\mathbb R^p$ be IID with mean $\mu$ and covariance
$\Lambda$. Define $Z_t=X_t-\bar X$ and let $Z$ be the $N\times p$
matrix with rows $Z_t^{\mathsf T}$. Let $K$ be the symmetric Bartlett
weight matrix

\[
K_{ts}=\begin{cases}
1-|t-s|/(m+1),&|t-s|\le m,\\
0,&|t-s|>m.
\end{cases}
\]

The uncorrected covariance-of-the-mean estimate implemented by the rolling-sum
factor is

\[
\widehat V_{\rm raw}=\frac{1}{N^2}Z^{\mathsf T}KZ.
\tag{6}
\]

Write $M=I-\mathbf1\mathbf1^{\mathsf T}/N$. If $E$ contains the
uncentered deviations $X_t-\mu$, then $Z=ME$. IID sampling gives

\[
\mathbb E[E_tE_s^{\mathsf T}]=\delta_{ts}\Lambda.
\]

Consequently,

\[
\mathbb E[Z^{\mathsf T}KZ]
=\operatorname{tr}(MKM)\Lambda
=\operatorname{tr}(KM)\Lambda,
\tag{7}
\]

because $M=M^{\mathsf T}=M^2$. Equivalently, the individual centered
cross-products satisfy

\[
\mathbb E[Z_tZ_t^{\mathsf T}]
=\left(1-\frac1N\right)\Lambda,
\qquad
\mathbb E[Z_tZ_s^{\mathsf T}]
=-\frac1N\Lambda\quad(t\ne s).
\tag{8}
\]

Let

\[
S_{N,m}=\sum_{k=1}^{m}w_k(N-k).
\]

Since $K$ has $N$ unit diagonal entries and $2S_{N,m}$ weighted
off-diagonal entries,

\[
\operatorname{tr}(KM)
=N-\frac1N\mathbf1^{\mathsf T}K\mathbf1
=(N-1)-\frac{2S_{N,m}}{N}.
\tag{9}
\]

Substitution of Eq. (9) into Eq. (6) yields

\[
\mathbb E[\widehat V_{\rm raw}]
=r_{N,m}\frac{\Lambda}{N},
\]

with exactly the $r_{N,m}$ in Eq. (1). Therefore

\[
\mathbb E\left[\frac{\widehat V_{\rm raw}}{r_{N,m}}\right]
=\operatorname{Cov}(\bar X)
=\frac{\Lambda}{N}.
\tag{10}
\]

This proves exact IID unbiasedness for the entire covariance matrix, not only
for its diagonal. No Gaussian assumption is used; finite second moments and
IID sampling suffice.

The weighted-pair sum can also be evaluated in closed form. With $b=m+1$,

\[
r_{N,m}
=1-\frac{b}{N}+\frac{b^2-1}{3N^2}.
\tag{11}
\]

The implementation requires $0\le m<N$, hence $1\le b\le N$. Equation
(11) is strictly positive in that domain, so the square root used to construct
the factor is defined. The multiplier $1/r_{N,m}$ is greater than one. It
ranges from the ordinary $N/(N-1)$ Bessel multiplier at $m=0$ to

\[
\frac{3N^2}{N^2-1}
\]

at $m=N-1$ (equal to 4 for the smallest permitted history, $N=2$, and
approaching 3 for large $N$).

## Why the correlated case is different

For a stationary correlated history, define

\[
\Gamma_h
=\mathbb E[(X_t-\mu)(X_{t+h}-\mu)^{\mathsf T}].
\]

Sample centering then gives

\[
\begin{aligned}
\mathbb E[Z_tZ_s^{\mathsf T}]
={}&\Gamma_{s-t}
-\frac1N\sum_a\Gamma_{a-t}
-\frac1N\sum_a\Gamma_{s-a}
+\frac1{N^2}\sum_{a,c}\Gamma_{c-a}.
\end{aligned}
\tag{12}
\]

Unlike Eq. (8), Eq. (12) is not a single scalar times one common matrix. The
finite-sample effect depends on the full autocovariance sequence, boundaries,
bandwidth, and kernel. Therefore dividing by the IID scalar $r_{N,m}$ cannot
be an exact general correction for a correlated process.

The correction nevertheless has three limited, provable properties:

- Since the raw Bartlett estimate is positive semidefinite and
  $1/r_{N,m}>1$, the corrected estimate is larger than the raw estimate in
  the Loewner order. Every projected variance $a^{\mathsf T}\widehat Va$ is
  inflated by the same factor.
- Its leading $1+b/N$ behavior agrees with the correction obtained by
  applying Wolff's Eq. (49) reasoning to Bartlett's total lag-window mass.
  Wolff's approximation requires a window small relative to the history and
  replacement of the truncated covariance by the full long-run covariance;
  it does not justify the higher-order terms of $1/r_{N,m}$ for a correlated
  process.
- If $b/N\to0$, Eq. (11) gives $r_{N,m}\to1$. The correction is therefore
  asymptotically negligible. In particular, $b_n^2/n\to0$, one sufficient
  Bartlett condition in Vats--Flegal--Jones, implies $b_n/n\to0$. Multiplying
  their strongly consistent estimator by $1/r_{N,m}\to1$ preserves strong
  consistency.

Neither property makes the corrected estimate a finite-sample upper bound on
the true covariance. A missing positive correlation tail can still cause
underestimation after inflation. For a process with negative correlations, or
when the raw estimate is already high, the multiplier can worsen
overestimation. It may also worsen mean-squared error even in the IID case:
unbiasedness alone is not an MSE or coverage guarantee.

If $b/N\to c>0$, Eq. (11) instead gives

\[
r_{N,m}\to 1-c+\frac{c^2}{3},
\]

so the multiplier does not disappear. That fixed-bandwidth regime is outside
the small-bandwidth consistency conditions cited above and must not inherit
their interpretation.

## Implementation and test audit

`src/gfrgtoolkit/stages/statistics/bartlett.py` computes Eq. (1) literally as
`iid_bias_factor` and divides every rolling-sum factor row by
`sqrt(iid_bias_factor)`. Its evidence records the applied multiplier as
`iid_centering_correction = 1.0 / iid_bias_factor`. The production algebra
matches the derivation above.

Current tests provide two relevant pieces of evidence:

- `test_zero_lag_bartlett_is_unbiased_for_iid_history_covariance` verifies the
  $m=0$ Bessel special case through the public processing interface.
- `test_bartlett_recovers_known_var1_long_run_covariance` checks a long
  correlated bivariate VAR(1) history against its analytic asymptotic
  covariance. At that sample size and bandwidth, the correction is close to
  one, so this principally validates asymptotic Bartlett behavior.

There is no committed test that independently checks exact IID unbiasedness
for $m>0$, the closed form in Eq. (11), positivity over every allowed
$(N,m)$, or the fact that the correction uniformly inflates the raw
estimate. Those are the most valuable additions if this policy remains.

## Recommended disposition

Do not delete the correction as an algebraic mistake: it is not one. Instead:

1. expose a named finite-sample scaling policy, distinguishing the published
   unscaled Bartlett estimator from `iid-exact-centering-inflation`; retaining
   the inflation as the conservative project default is defensible, while
   strict literature reproduction should select the unscaled variant;
2. report the Bartlett publication and toolkit derivation as separate
   provenance entries;
3. describe the latter as exact under IID sampling, matching Wolff's leading
   correlated-process correction, asymptotically negligible under small
   bandwidth, and only directionally conservative relative to the raw
   estimate;
4. add the missing $m>0$ IID and policy-property tests listed above; and
5. warn or mark the estimator unresolved when the declared bandwidth consumes
   a substantial fraction of the history, using a separately justified policy
   rather than inventing a literature threshold.

This disposition keeps the useful conservative tendency visible without
misrepresenting it as part of Newey--West or as a guaranteed covariance upper
bound.
