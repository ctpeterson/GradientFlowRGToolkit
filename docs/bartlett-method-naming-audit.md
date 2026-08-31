# Naming audit for the Bartlett long-run covariance estimator

## Question

What is the most precise literature name for the estimator implemented by
`BartlettLongRunCovariance`?

## Finding

The best domain-specific name is:

> **modified-Bartlett multivariate spectral variance estimator for the
> covariance of a Monte Carlo sample mean**

This identifies both the estimator family and the particular lag window. In
short prose, **modified-Bartlett spectral variance estimator** is sufficient
when the multivariate Monte Carlo context is already clear.

The implementation starts from a vector Monte Carlo history, forms sample
cross-autovariance matrices at positive and negative lags, and combines them
with

\[
w(k)=1-\frac{|k|}{b},\qquad |k|<b.
\]

Vats, Flegal, and Jones call a weighted, truncated sum of multivariate sample
autocovariances a **multivariate spectral variance estimator** (MSVE) in their
Eq. (2.2). In their lag-window catalogue, the formula above is explicitly the
**modified Bartlett window** when the Parzen-family exponent is one. This is
the closest primary-source terminology to the toolkit's vector-valued MCMC
use case [Vats--Flegal--Jones 2018, Secs. 2.1 and
2.2.4](https://arxiv.org/html/1507.08266v3#S2.SS1).

The code reports the covariance of the sample mean rather than the unscaled
central-limit-theorem covariance. It also applies a toolkit-specific exact IID
sample-centering correction. Those scale and correction choices should remain
separate evidence; neither changes the name of the underlying lag-window
estimator, and neither should be attributed to the cited papers.

## Why the nearby names differ

### “Bartlett's method”

Avoid this as the formal name. Bartlett's original spectral work concerns
smoothing spectra by averaging periodograms from contiguous sections of a time
series, with a related correlogram truncation interpretation
[Bartlett 1948](https://doi.org/10.1038/161686a0) and
[Bartlett 1950](https://doi.org/10.1093/biomet/37.1-2.1). Consequently,
“Bartlett's method” can refer to an averaged-periodogram spectrum estimator,
not specifically to the zero-frequency long-run covariance calculation in
this toolkit.

“Bartlett window” is much less ambiguous because it names the triangular
weights. For complete fidelity to the sources used here, “modified Bartlett
window” is preferable.

### “Newey--West estimator” or “HAC estimator”

Newey and West describe a positive-semidefinite **heteroskedasticity- and
autocorrelation-consistent covariance matrix**. Their Eq. (5) uses the same
linearly declining weights and calls them modified Bartlett weights. Thus
**Newey--West HAC estimator** is mathematically recognizable shorthand for the
kernel construction
[Newey--West 1987](https://doi.org/10.2307/1913610); the authors' openly listed
working-paper record also describes the estimator as PSD by construction and
consistent under its assumptions
[NBER Technical Working Paper 55](https://www.nber.org/papers/t0055).

However, “HAC” and “Newey--West” foreground the econometric GMM/regression
setting. This toolkit estimates the Monte Carlo error of a vector sample mean,
for which the MCMC literature's “multivariate spectral variance estimator” is
more direct. “Newey--West” is useful as a cross-disciplinary alias, not as the
primary report name.

### “Long-run covariance estimator”

This is an accurate domain role, but it does not identify the numerical
method. It should be paired with the window name: **modified-Bartlett long-run
covariance estimator**. When mathematical precision matters, state whether the
reported object is the long-run covariance itself or that covariance divided
by the configuration count to estimate the covariance of the sample mean.

## Recommendation by interface

| Surface | Recommended name | Rationale |
| --- | --- | --- |
| Scientific prose | modified-Bartlett multivariate spectral variance estimator for the covariance of the Monte Carlo sample mean | Most precise about family, window, dimension, and reported scale |
| Compact result/report label | `modified-bartlett-msve` | Unambiguous and compact; expand “MSVE” on first use |
| Python class | `ModifiedBartlettLongRunCovariance` | Keeps the current domain role while naming the exact window; `BartlettLongRunCovariance` remains understandable if API stability is preferred |
| Cross-disciplinary alias | Newey--West HAC estimator with modified Bartlett weights | Correct aid for econometrics readers, but not the primary MCMC name |

Do not call the full implementation simply “Bartlett's method,” and do not
present its toolkit-specific finite-sample correction as part of the
Newey--West or Vats--Flegal--Jones literature estimator.

## Primary sources

1. D. Vats, J. M. Flegal, and G. L. Jones, “Strong consistency of
   multivariate spectral variance estimators in Markov chain Monte Carlo,”
   *Bernoulli* **24** (2018), 1860--1909.
   DOI: [10.3150/16-BEJ914](https://doi.org/10.3150/16-BEJ914).
   [Author manuscript](https://arxiv.org/abs/1507.08266).
2. W. K. Newey and K. D. West, “A simple, positive semi-definite,
   heteroskedasticity and autocorrelation consistent covariance matrix,”
   *Econometrica* **55** (1987), 703--708.
   DOI: [10.2307/1913610](https://doi.org/10.2307/1913610).
   [NBER working-paper record](https://www.nber.org/papers/t0055).
3. M. S. Bartlett, “Smoothing periodograms from time-series with continuous
   spectra,” *Nature* **161** (1948), 686--687.
   DOI: [10.1038/161686a0](https://doi.org/10.1038/161686a0).
4. M. S. Bartlett, “Periodogram analysis and continuous spectra,”
   *Biometrika* **37** (1950), 1--16.
   DOI: [10.1093/biomet/37.1-2.1](https://doi.org/10.1093/biomet/37.1-2.1).
