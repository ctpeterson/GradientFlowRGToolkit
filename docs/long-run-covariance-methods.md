---
title: Long-Run Covariance Methods in GradientFlowRGToolkit
subtitle: A sourced guide to the implementation and its limits
author: GradientFlowRGToolkit
date: 31 August 2026
geometry: margin=1in
fontsize: 11pt
colorlinks: true
---

# Implemented policy

Aligned flow times and measurement channels are estimated jointly.
Bartlett/Newey--West is the primary positive-semidefinite full-matrix
estimator. Multivariate batch means and scalar projected Wolff calculations
are independent checks. Over-lugsail batch means is a named
positive-leading-bias variation. The inherited rectangular estimator is
retained only under an explicitly experimental name.

# The target

Let \(Y_t\in\mathbb R^p\), for \(t=1,\ldots,N\), contain every aligned
measurement made on configuration \(t\). Define

\[
\Gamma(k)=\operatorname{Cov}(Y_t,Y_{t+k}),\qquad
\Gamma(-k)=\Gamma(k)^{\mathsf T}.
\]

The long-run covariance is

\[
\Sigma=\sum_{k=-\infty}^{\infty}\Gamma(k).
\]

Under a multivariate Markov-chain central limit theorem,

\[
\sqrt N(\overline Y-\mu)\Longrightarrow\mathcal N_p(0,\Sigma),
\qquad
\operatorname{Cov}(\overline Y)\simeq\frac{\Sigma}{N}.
\]

Wolff defines this full matrix and the covariance of the means in Eqs. (2),
(4), and (12)--(13) of [Wolff (2004)](https://arxiv.org/abs/hep-lat/0306017).
[Vats, Flegal, and Jones (2018)](https://arxiv.org/abs/1507.08266) use the
same target in multivariate spectral-variance theory.

If plaquette, Symanzik, and clover measurements come from the same
configurations, they are coordinates of one \(Y_t\). The implementation
concatenates their histories, estimates once, and only then returns channel
views. Cross-channel covariance is therefore retained.

# Primary estimator: Bartlett/Newey--West

For \(Z_t=Y_t-\overline Y\), define

\[
\widehat\Gamma(k)=
\frac1N\sum_{t=1}^{N-k}Z_tZ_{t+k}^{\mathsf T}.
\]

With largest included lag \(m\), the implementation forms

\[
\widehat\Sigma_{\mathrm B}(m)
=\widehat\Gamma(0)
+\sum_{k=1}^{m}
\left(1-\frac{k}{m+1}\right)
\left[\widehat\Gamma(k)+\widehat\Gamma(k)^{\mathsf T}\right].
\tag{1}
\]

The covariance attached to the sample mean is
\(\widehat\Sigma_{\mathrm B}/N\).

[Newey and West (1987), Eq. (5) and Theorems 1--2](https://doi.org/10.2307/1913610)
give the triangular weights and establish positive semidefiniteness and
consistency under their assumptions. Vats, Flegal, and Jones identify the
same modified Bartlett window as satisfying their multivariate conditions
under stated bandwidth rates.

## PSD without projection

Set \(b=m+1\). The triangular kernel is the autocorrelation of a length-\(b\)
box:

\[
1-\frac{|k|}{b}
=
\frac{\text{overlap of two length-\(b\) boxes separated by \(k\)}}{b}.
\]

The estimate therefore has a moving-sum factorization

\[
\widehat{\operatorname{Cov}}(\overline Y)=F^{\mathsf T}F.
\]

The code constructs \(F\) directly. PSD is structural, no eigenvalue clipping
is used, and the intentional low rank when \(p>N\) is preserved.

## Exact IID centering correction

Subtracting the sample mean biases empirical lag products downward. For IID
vector observations and fixed Bartlett weights \(w_k=1-k/(m+1)\), the
uncorrected covariance of the mean has the exact multiplicative expectation

\[
r_{N,m}
=
\frac1N\left[
(N-1)-\frac2N\sum_{k=1}^{m}w_k(N-k)
\right].
\tag{2}
\]

The implementation divides by \(r_{N,m}\). At \(m=0\), this is the ordinary
\(N/(N-1)\) correction.

Equation (2) is a transparent toolkit derivation recorded as
**iid_centering_correction**. It is not attributed to Newey and West. It
removes the exact IID centering bias, but not every finite-sample bias for a
correlated history. Wolff independently derives his leading scalar
rectangular-window correction in Eq. (49).

# Bandwidth and unresolved autocorrelation

The required **maximum_lag** is the \(m\) in Eq. (1). It is not a hidden
multiple of a diagonal autocorrelation estimate. Consistency results require,
among other assumptions,

\[
b_N\longrightarrow\infty,\qquad N/b_N\longrightarrow\infty,
\]

in a fixed-dimensional regime. A rule such as \(N^{1/3}\) is only a rate; its
MSE-optimal coefficient contains unknown process-dependent quantities
([Liu, Vats, and Flegal (2022)](https://arxiv.org/abs/1804.05975)).

The optional **BandwidthStabilityCheck** compares every coordinate's variance
at the selected lag with every declared lower comparison lag. Every comparison
is retained as typed evidence, and the worst relative change drives the
aggregate status. If that change exceeds the declared tolerance, processing records
typed **Unresolved** autocorrelation evidence while still returning the
estimate. A strict workflow can request **UnresolvedAutocorrelationAction.Raise**.

    method = BartlettLongRunCovariance(
        maximum_lag=64,
        stability=BandwidthStabilityCheck(
            comparison_lags=(16, 32),
            relative_tolerance=0.10,
        ),
    )

    # Optional publication or CI gate:
    strict_method = BartlettLongRunCovariance(
        maximum_lag=64,
        stability=BandwidthStabilityCheck(
            comparison_lags=(16, 32),
            relative_tolerance=0.10,
        ),
        on_unresolved=UnresolvedAutocorrelationAction.Raise,
    )

This guards one observable failure mode. It is not proof that every possible
linear combination has resolved its slowest mode.

# Why a finite-data upper bound is impossible in general

Preferring overestimation to underestimation is sensible, but cannot be
guaranteed without assumptions on the unseen correlation tail. Consider

\[
Y_t=\epsilon_t+\theta\epsilon_{t-L},\qquad \theta>0,
\]

where \(\epsilon_t\) are IID with variance \(s^2\). Then

\[
\Gamma(0)=(1+\theta^2)s^2,\qquad
\Gamma(\pm L)=\theta s^2,\qquad
\Sigma=(1+\theta)^2s^2.
\]

Every lag below \(L\) appears uncorrelated, but omitting lag \(L\) misses
\(2\theta s^2\). A finite history that cannot see \(L\) cannot infer a
distribution-free upper covariance. Wolff similarly warns that the run must
be much longer than the decay scale for reliable error estimation.

The general safety policy is therefore typed unresolved evidence, not an
arbitrary inflation. Exploratory analysis remains available; strict rejection
is an explicit workflow choice.

# Positive-leading-bias over-lugsail variation

[Vats and Flegal (2022)](https://arxiv.org/abs/1809.04541) define

\[
k_L(x)=\frac{k(x)}{1-c}-\frac{c\,k(rx)}{1-c}.
\tag{3}
\]

For strongly positive persistence and a first-order Bartlett kernel, their
recommended over-lugsail parameters are \(r=3\) and \(c=1/2\). For batch
means, the implementation uses

\[
\widehat\Sigma_L
=\frac{\widehat\Sigma_b}{1-c}
-\frac{c\,\widehat\Sigma_{\lfloor b/r\rfloor}}{1-c}.
\tag{4}
\]

This reverses the leading asymptotic bias under the paper's assumptions.
It is not a finite-sample guarantee and not Loewner domination of the true
matrix. It may be inappropriate for anti-persistent histories. A lugsail
estimate is a difference of covariance estimates and may be indefinite, so
it is a systematic variation and any nearest-PSD projection is recorded. The
implementation performs the signed eigensystem calculation in the
sample-supported row space and returns a projected low-rank factor; it does
not materialize a dense \(p\times p\) covariance.

# Multivariate batch-means validator

Let \(N=ab\), with \(a\) batches of size \(b\), and let
\(\overline Y_j(b)\) be the batch means. The covariance of the overall mean is

\[
\widehat{\operatorname{Cov}}(\overline Y)
=
\frac{1}{a(a-1)}
\sum_{j=1}^{a}
(\overline Y_j-\overline Y)
(\overline Y_j-\overline Y)^{\mathsf T}.
\tag{5}
\]

[Vats, Flegal, and Jones (2019), Eq. (10) and Theorem 2](https://arxiv.org/abs/1512.07713)
give the multivariate construction and consistency conditions. The
implementation records batch size, batch count, and discarded tail count,
and constructs a PSD factor directly.

Both \(a\) and \(b\) must grow in the cited theorem. Batch-size sweeps are
evidence. Batch means is an independent implementation comparator, not a
finite-sample conservative guarantee.

# Projected Wolff validator

Wolff's practical method is applied to each scalar coordinate and to any
explicitly declared linear combinations as a validator.
It uses his lag-dependent denominator

\[
\widehat\Gamma_F(t)
=\frac1{N-t}\sum_{i=1}^{N-t}z_i z_{i+t},
\tag{6}
\]

the scalar rectangular sum, Eq. (49)'s finite-\(N\) correction, and

\[
\frac{S}{\overline\tau(W)}
=
\log\left(
\frac{2\widehat\tau_{\rm int}(W)+1}
     {2\widehat\tau_{\rm int}(W)-1}
\right),
\]

\[
g(W)=
e^{-W/\overline\tau(W)}
-\frac{\overline\tau(W)}{\sqrt{WN}}.
\tag{7}
\]

The first sign change to \(g(W)<0\) selects the window; these are Wolff's
Eqs. (31), (35), and (49)--(52). Reaching the declared lag cap records the
unresolved coordinate or projection count. The enclosing Bartlett method
applies its declared record-or-raise policy. An optional relative-variance
tolerance can apply that policy to a fully resolved but excessive
Bartlett--Wolff disagreement as well.

This validates scalar variances. It is not used to assemble a matrix from
different entrywise windows, because that would not be one coherent
multivariate estimator.

# High dimension and rank

Every sample covariance here can be written

\[
Z^{\mathsf T}KZ,\qquad Z\in\mathbb R^{N\times p}.
\]

After centering, its rank is at most \(N-1\); batch means has rank at most
\(a-1\). Thus a valid covariance is necessarily singular when \(p\geq N\).
PSD projection cannot manufacture information in the null space.

The evidence records numerical rank and a scale-aware tolerance. Future
inverse-covariance fits must declare dimension reduction, flow-time selection,
a low-rank formulation, or separately sourced regularization.

# Application evidence policy

Changing external datasets are integration evidence, not estimator contracts,
acceptance thresholds, or stable regression targets. Production analysis must
establish stability for its scientifically relevant projections or report
unresolved autocorrelation. Dataset-derived diagnostics belong in the private
analysis record until an explicitly approved public dataset revision exists.

# Public configuration and evidence

    # Primary PSD full matrix
    BartlettLongRunCovariance(maximum_lag=64)

    # Independent PSD comparator
    MultivariateBatchMeans(batch_size=32)

    # Positive-leading-bias systematic variation
    LugsailBatchMeans(
        batch_size=60,
        lugsail_scale=3,
        lugsail_weight=0.5,
    )

    # Legacy differential comparison only
    ExperimentalRectangularLongRunCovariance(
        window_factor=3.0
    )

Evidence records estimator identity and source, history length, dimension,
rank, finite-sample correction, bandwidth or batching policy, projection
diagnostics, and requested validator results.

# Validation currently in the test suite

Tests through **RunningCoupling.process(...)** cover:

- exact cross-channel covariance preservation;
- exact IID zero-lag covariance of the mean;
- an independently worked multivariate batch-means example;
- the rank bound when \(p>N\);
- an AR(1) case where over-lugsail offsets ordinary downward bias;
- non-blocking unresolved evidence and explicit strict rejection for a
  non-plateauing bandwidth;
- a two-dimensional VAR(1) process with known full long-run covariance.

These tests support the implementation; they do not prove that a particular
real history has resolved its longest autocorrelation scale.

# Primary references

- U. Wolff, [Monte Carlo errors with less errors](https://arxiv.org/abs/hep-lat/0306017),
  Computer Physics Communications 156 (2004) 143--153.
- W. K. Newey and K. D. West,
  [A simple, positive semi-definite, heteroskedasticity and autocorrelation
  consistent covariance matrix](https://doi.org/10.2307/1913610),
  Econometrica 55 (1987) 703--708.
- D. Vats, J. M. Flegal, and G. L. Jones,
  [Strong consistency of multivariate spectral variance estimators in Markov
  chain Monte Carlo](https://arxiv.org/abs/1507.08266),
  Bernoulli 24 (2018) 1860--1909.
- D. Vats, J. M. Flegal, and G. L. Jones,
  [Multivariate output analysis for Markov chain Monte
  Carlo](https://arxiv.org/abs/1512.07713),
  Biometrika 106 (2019) 321--337.
- D. Vats and J. M. Flegal,
  [Lugsail lag windows for estimating time-average covariance
  matrices](https://arxiv.org/abs/1809.04541),
  Biometrika 109 (2022) 735--750.
- Y. Liu, D. Vats, and J. M. Flegal,
  [Batch size selection for variance estimators in
  MCMC](https://arxiv.org/abs/1804.05975),
  Methodology and Computing in Applied Probability 24 (2022) 65--93.
