# Covariance-estimator implementation source audit

**Audit date:** 31 August 2026

**Scope:** `src/gfrgtoolkit/stages/statistics/` and the joint-history assembly
in `src/gfrgtoolkit/stages/process.py`

**Authority policy:** published papers and their authors' preprints are the
scientific authorities; the current source tree is the implementation being
audited.

## Closeout status

This file preserves the findings from the pre-closeout audit. The current
implementation has resolved its three actionable findings:

1. every declared Bartlett comparison lag is evaluated and retained as
   `BandwidthComparisonEvidence`;
2. lugsail covariance clipping is explicitly identified as toolkit policy and
   is now performed in a sample-supported low-rank factor space; and
3. projected Wolff validation accepts declared linear combinations and an
   optional relative-variance threshold that participates in the existing
   record-or-raise unresolved policy.

The detailed sections below remain useful as the derivation and audit trail,
but statements phrased as “current” describe the audited pre-closeout
revision unless the closeout status says otherwise.

## Executive conclusion

The scientific core of the three named matrix estimators is recognizable and
mostly faithful to its cited source:

| Implementation | Audit result |
|---|---|
| `BartlettLongRunCovariance` | Implements the modified-Bartlett/Newey--West lag window and is PSD by an exact moving-sum factorization. Its finite-sample IID centering correction and bandwidth-stability diagnostic are toolkit policies, not results attributed to Newey and West. |
| `MultivariateBatchMeans` | Implements the multivariate batch-means estimator, divided by the used sample count so that the returned object is the covariance of the sample mean rather than the CLT time-average covariance. |
| `LugsailBatchMeans` | Implements the paper's lugsail linear combination with the paper's default over-lugsail parameters. Its direct covariance eigenvalue clipping is **not** the positive-definite correlation-matrix adjustment published in the paper. |
| `ProjectedWolffValidation` | Closely implements Wolff's scalar automatic window calculation, separately for each coordinate. It is a validator of marginal variances, not a full-covariance Gamma-method estimator. |
| `ExperimentalRectangularLongRunCovariance` | Contains a standard-looking rectangular lag sum surrounded by an unsourced window-selection heuristic and an ad hoc nearest-PSD repair. Its `source=None` and experimental name are scientifically appropriate. It must not be described as Wolff's Gamma method. |

Three follow-up issues deserve explicit attention:

1. `BandwidthStabilityCheck.comparison_lags` is documented as a grid, but the
   estimator calculates only `comparison_lags[-1]`; the earlier entries are
   merely copied into evidence.
2. The lugsail evidence now cites publication DOI
   [10.1093/biomet/asab049](https://doi.org/10.1093/biomet/asab049); its PSD
   projection must still not be mistaken for the adjustment in that paper.
3. A successfully selected Wolff window marks the diagnostic resolved even
   when the Wolff variance and selected Bartlett variance differ greatly;
   that difference is recorded but has no configured acceptance threshold.

## 1. Common statistical target and units

Let the aligned measurement on Monte Carlo configuration \(t\) be the vector

\[
Y_t\in\mathbb R^p,\qquad t=1,\ldots,N,
\]

with stationary mean \(\mu\) and lag covariance

\[
R(k)=\operatorname{Cov}(Y_t,Y_{t+k}),\qquad
R(-k)=R(k)^{\mathsf T}.
\]

The time-average, long-run, or Markov-chain-CLT covariance is

\[
\Sigma=\sum_{k=-\infty}^{\infty}R(k).
\]

The multivariate CLT is conventionally written

\[
\sqrt N(\bar Y_N-\mu)\Rightarrow\mathcal N_p(0,\Sigma).
\]

Consequently, the covariance needed for uncertain values representing the
sample mean is

\[
\operatorname{Cov}(\bar Y_N)\simeq\frac{\Sigma}{N}.
\tag{1}
\]

This distinction is essential when comparing the code to the papers. Vats,
Flegal, and Jones state the multivariate CLT in Eq. (1) of
[Multivariate Output Analysis for Markov Chain Monte Carlo](https://arxiv.org/html/1512.07713v4),
DOI [10.1093/biomet/asz002](https://doi.org/10.1093/biomet/asz002).
Their estimators are written as estimators of \(\Sigma\). The toolkit returns an
estimate of \(\Sigma/N\) because that is the covariance to attach to `means`.

### Full-vector assembly occurs before estimation

`process.py` first concatenates all selected observables along the value axis:

```python
joint_histories = np.concatenate(
    [
        histories[observable]
        for observable in configuration.dataset.observables
    ],
    axis=1,
)
joint_estimate = configuration.averaging.method.estimate(joint_histories)
joint_energy = joint_estimate.values.reshape(observable_count, time_count)
```

If each observable history has shape `(N, T)`, the estimator receives one
matrix of shape `(N, observable_count * T)`. Therefore every implemented
matrix estimator sees cross-flow-time and cross-observable products before
the result is reshaped into channel views. This implements the project's
correlation-preservation invariant; it is not a diagonal-only analysis.

## 2. Shared implementation contract

All estimators accept a finite two-dimensional history matrix. `core.py`
rejects non-matrices, fewer than two configurations, zero columns, and any
non-finite value. The protocol returns:

- correlated means in `LongRunCovarianceEstimate.values`; and
- method, source, dimensions, rank, projection, bandwidth/batch, and
  autocorrelation evidence in `LongRunCovarianceEstimate.evidence`.

### Covariance factors

For the estimators that are PSD by construction, the code works with a factor
\(F\in\mathbb R^{q\times p}\) such that

\[
\widehat{\operatorname{Cov}}(\bar Y)=F^{\mathsf T}F.
\tag{2}
\]

`correlated_values_from_factor` forms the dense covariance only when the
number of values does not exceed the number of factor rows. Otherwise it
creates \(q\) independent unit-variance latent `gvar` values \(\eta\) and returns

\[
\bar Y+F^{\mathsf T}\eta,
\]

whose covariance is exactly Eq. (2). This avoids forcing a dense \(p\times p\)
matrix merely to represent an intentionally low-rank estimate.

`factor_spectrum` applies an SVD to \(F\), using

\[
\text{tol}_{\rm singular}
=\max(q,p)\,\epsilon_{\rm machine}\,s_{\max}
\]

and records the covariance-scale tolerance
`tol_singular**2`. This rank threshold is a numerical policy, not part of any
of the estimator papers.

## 3. Bartlett/Newey--West estimator

### 3.1 Published formula

For centered observations \(Z_t=Y_t-\bar Y\), define the sample lag product

\[
\widehat R(k)=\frac1N\sum_{t=1}^{N-k}Z_tZ_{t+k}^{\mathsf T}.
\]

With maximum included lag \(m\), the modified-Bartlett estimator is

\[
\widehat\Sigma_{B,m}
=\widehat R(0)+\sum_{k=1}^{m}
\left(1-\frac{k}{m+1}\right)
\left[\widehat R(k)+\widehat R(k)^{\mathsf T}\right].
\tag{3}
\]

This is Eq. (5) of Newey and West,
[A Simple, Positive Semi-Definite, Heteroskedasticity and Autocorrelation
Consistent Covariance Matrix](https://doi.org/10.2307/1913610),
*Econometrica* 55 (1987), 703--708. The authors' NBER working-paper record is
[Technical Working Paper 55](https://www.nber.org/papers/t0055), DOI
[10.3386/t0055](https://doi.org/10.3386/t0055). The paper proves positive
semidefiniteness and consistency under its stated asymptotic conditions.

The toolkit must attach \(\Sigma/N\) to the sample mean, so its uncorrected target is

\[
\widehat V_{B,m}=\frac{\widehat\Sigma_{B,m}}{N}
=\frac1{N^2}\left\{
\sum_t Z_tZ_t^{\mathsf T}
+\sum_{k=1}^m w_k\sum_{t=1}^{N-k}
(Z_tZ_{t+k}^{\mathsf T}+Z_{t+k}Z_t^{\mathsf T})
\right\},
\tag{4}
\]

where \(w_k=1-k/(m+1)\).

### 3.2 Moving-sum factorization in the code

Set \(b=m+1\). Pad the centered history with \(b-1\) zero rows at both ends
and let \(S_j\) be every length-\(b\) rolling sum. A pair \(Z_t,Z_{t+k}\)
appears together in exactly \(b-k\) rolling windows. Therefore

\[
\frac1b\sum_jS_jS_j^{\mathsf T}
=\sum_tZ_tZ_t^{\mathsf T}
+\sum_{k=1}^{b-1}\left(1-\frac{k}{b}\right)
\sum_{t=1}^{N-k}
(Z_tZ_{t+k}^{\mathsf T}+Z_{t+k}Z_t^{\mathsf T}).
\tag{5}
\]

This is Eq. (4)'s numerator and is PSD because it is a sum of outer products.
The implementation constructs these rolling sums by cumulative subtraction:

```python
bandwidth = maximum_lag + 1
padded = np.pad(
    centered,
    ((bandwidth - 1, bandwidth - 1), (0, 0)),
)
cumulative = np.vstack(
    (np.zeros((1, value_count)), np.cumsum(padded, axis=0))
)
rolling_sums = cumulative[bandwidth:] - cumulative[:-bandwidth]
factor = rolling_sums / np.sqrt(
    bandwidth
    * configuration_count
    * configuration_count
    * iid_bias_factor
)
```

Ignoring `iid_bias_factor` for the moment, Eq. (5) shows directly that
`factor.T @ factor` equals Eq. (4). This is a faithful implementation of the
Newey--West triangular kernel, with no eigenvalue projection.

### 3.3 Exact IID centering correction is a toolkit derivation

Because \(Z_t=Y_t-\bar Y\), sample lag products are biased even for IID data.
If the population covariance of each IID \(Y_t\) is \(\Lambda\), then

\[
E[Z_tZ_t^{\mathsf T}]=\left(1-\frac1N\right)\Lambda,
\]

and for \(t\ne s\),

\[
E[Z_tZ_s^{\mathsf T}]=-\frac1N\Lambda.
\]

Taking the expectation of Eq. (4) gives

\[
E[\widehat V_{B,m}]
=r_{N,m}\frac{\Lambda}{N},
\]

with

\[
r_{N,m}=\frac1N\left[
(N-1)-\frac2N\sum_{k=1}^m
\left(1-\frac{k}{m+1}\right)(N-k)
\right].
\tag{6}
\]

The code computes exactly this \(r_{N,m}\) as `iid_bias_factor` and divides
the factor by \(\sqrt{r_{N,m}}\). At \(m=0\), \(r_{N,0}=(N-1)/N\), so this
reduces to the ordinary Bessel correction for the covariance of an IID mean.

This correction is mathematically justified for IID histories. It is not
Eq. (5) of Newey--West, and it does not remove all finite-\(N\) bias for a
correlated process. The evidence field `iid_centering_correction` correctly
records its applied multiplier \(1/r_{N,m}\), but the single `source` URL only
supports the Bartlett kernel, not this toolkit-specific derivation.

### 3.4 Bandwidth choice and stability evidence

The source consistency result is asymptotic: the bandwidth must grow with
sample size while remaining small relative to sample size under the relevant
regularity conditions. A fixed configured `maximum_lag` is therefore a
finite-sample analysis choice, not automatically a consistent bandwidth
sequence.

The optional toolkit stability check compares the diagonal variances at the
selected \(m\) with a lower lag \(m_0\):

\[
\Delta_{\max}=\max_i
\frac{|\widehat V_{ii}(m)-\widehat V_{ii}(m_0)|}
{\max(|\widehat V_{ii}(m)|,|\widehat V_{ii}(m_0)|,\mathrm{tiny})}.
\tag{7}
\]

It records or raises if Eq. (7) exceeds a declared tolerance. This is a
sensible analysis diagnostic, but it has no claimed Newey--West provenance
and it only tests coordinate variances, not arbitrary linear combinations.

**Implementation discrepancy:** despite the field name and docstring
`comparison_lags` / "comparison grid", the implementation executes

```python
comparison_lag = method.stability.comparison_lags[-1]
comparison_factor, _ = factor_at(comparison_lag)
```

Only the largest lower lag is calculated. Earlier configured lags are
included in `bandwidth_scan_lags` evidence but do not participate in the
diagnostic. Either the interface/documentation should say "largest lower
comparison lag" or the implementation should actually scan the full grid.

### 3.5 Bartlett audit verdict

- **Formula:** aligned with Newey--West Eq. (5).
- **PSD claim:** justified structurally by the moving-sum factor.
- **Returned scale:** correctly estimates covariance of the mean, \(\Sigma/N\).
- **IID correction:** exact under IID sampling, but toolkit-derived.
- **Bandwidth adequacy:** never guaranteed by the source citation; evidence
  and user judgement remain necessary.
- **Source field:** DOI 10.2307/1913610 is appropriate for the kernel.

## 4. Multivariate non-overlapping batch means

### 4.1 Published formula and conversion to covariance of the mean

Suppose \(N=ab\), with \(a\) complete batches of \(b\) configurations, and

\[
\bar Y_j(b)=\frac1b\sum_{t=1}^bY_{jb+t}.
\]

Eq. (10) of Vats, Flegal, and Jones defines the estimator of the CLT
covariance \(\Sigma\) as

\[
\widehat\Sigma_b
=\frac{b}{a-1}\sum_{j=0}^{a-1}
(\bar Y_j-\bar Y)(\bar Y_j-\bar Y)^{\mathsf T}.
\tag{8}
\]

The primary publication is
[Multivariate Output Analysis for Markov Chain Monte Carlo](https://doi.org/10.1093/biomet/asz002),
*Biometrika* 106 (2019), 321--337; the openly accessible author preprint
contains [Eq. (10)](https://arxiv.org/html/1512.07713v4).

Dividing Eq. (8) by \(N=ab\) gives the covariance estimator for the overall
mean:

\[
\widehat V_b=\frac{1}{a(a-1)}\sum_{j=0}^{a-1}
(\bar Y_j-\bar Y)(\bar Y_j-\bar Y)^{\mathsf T}.
\tag{9}
\]

The code matches Eq. (9):

```python
batch_means = used_values.reshape(
    batch_count,
    method.batch_size,
    value_count,
).mean(axis=1)
means = batch_means.mean(axis=0)
centered_batch_means = batch_means - means[np.newaxis, :]
factor = centered_batch_means / np.sqrt(
    batch_count * (batch_count - 1.0)
)
```

Thus `factor.T @ factor` is Eq. (9). The returned units are correct even
though the paper states Eq. (8) in CLT-covariance units.

### 4.2 Tail handling

When \(N\) is not divisible by \(b\), the code sets

\[
a=\lfloor N/b\rfloor,\qquad N_{\rm used}=ab,
\]

and discards the final \(N-N_{\rm used}\) configurations. Both the returned
mean and covariance are based on `used_values`, not on the full history.
This makes the source assumption \(N=ab\) exact for the analyzed prefix and
is recorded as `discarded_configuration_count`. It also means that changing
the batch size can change the point estimate when a tail is discarded; a
method comparison is not purely a covariance comparison in that case.

### 4.3 Asymptotic caveat

The cited consistency theory does not justify an arbitrary fixed batch size.
Both \(b\to\infty\) and \(a=N/b\to\infty\) are required along the asymptotic
sequence. The implementation validates only the finite computability
condition \(a\ge2\). Batch-size adequacy therefore remains analysis evidence,
not a construction invariant.

### 4.4 Batch-means audit verdict

- **Formula:** exact match to published multivariate batch means after the
  necessary division by \(N\).
- **PSD claim:** exact, because the estimate is an outer-product factor.
- **Cross-covariance:** retained by vector batch means.
- **Tail policy:** explicit and recorded, but can alter the mean.
- **Source field:** DOI 10.1093/biomet/asz002 is appropriate.

## 5. Over-lugsail batch means

### 5.1 Published formula

Vats and Flegal define a lugsail modification of any lag window by

\[
k_L(x)=\frac{k(x)}{1-c}-\frac{c\,k(rx)}{1-c}.
\tag{10}
\]

For the Bartlett/batch-means case their Eq. (7) becomes

\[
\widehat\Sigma_L
=\frac{\widehat\Sigma_b}{1-c}
-\frac{c\,\widehat\Sigma_{b/r}}{1-c}.
\tag{11}
\]

See [Lugsail Lag Windows for Estimating Time-Average Covariance
Matrices](https://arxiv.org/html/1809.04541v3), Eqs. (2) and (7), published in
*Biometrika* 109 (2022), 735--750, DOI
[10.1093/biomet/asab049](https://doi.org/10.1093/biomet/asab049).

For a first-order kernel, the paper gives leading bias

\[
\operatorname{Bias}(\widehat\Sigma_L)
=\frac{\Gamma}{b}\frac{1-rc}{1-c}+o(b^{-1}).
\tag{12}
\]

For the common positively persistent case, ordinary Bartlett/BM has negative
leading diagonal bias; choosing \(rc>1\) reverses that leading term. The
paper's recommended over-lugsail values for this first-order setting are
\(r=3,c=1/2\).

### 5.2 Formula-to-code map

The implementation requires integer \(r\ge2\), divisibility of \(b\) by
\(r\), \(0<c<1\), and \(c>1/r\). The final condition is precisely \(rc>1\),
the over-lugsail regime in Eq. (12).

`covariance_for_batch_size` returns Eq. (9), the covariance of the mean, for
both \(b\) and \(b/r\). Since both are divided by the same used \(N\), the
same linear combination as Eq. (11) is valid:

```python
large = covariance_for_batch_size(method.batch_size)
small = covariance_for_batch_size(smaller_batch_size)
covariance = (
    large / (1.0 - method.lugsail_weight)
    - method.lugsail_weight
    * small
    / (1.0 - method.lugsail_weight)
)
```

The helper centers both batch scales on the same `means = used_values.mean`,
as the paper's common \(\bar Y\) requires. The large-batch prefix length is divisible
by both \(b\) and \(b/r\), so the two estimates use the same configurations.

### 5.3 What “conservative” does and does not mean

The over-lugsail setting reverses a leading asymptotic bias under assumptions
about the process and correlation structure. It does **not** prove any of the
following finite-sample matrix inequalities:

\[
\widehat V_L\succeq V,
\qquad
\widehat V_L\succeq\widehat V_b,
\qquad
\operatorname{diag}(\widehat V_L)
\ge\operatorname{diag}(V)
\quad\text{for every process}.
\]

The paper explicitly treats extra variability as the price of correcting
negative bias and recommends the over-lugsail for high to extreme positive
correlation. Anti-persistent or mixed-sign processes do not inherit the
simple “overestimate” interpretation. The evidence string
`positive-for-positive-correlation` should be read as a conditional
first-order statement, not a guarantee.

### 5.4 PSD adjustment differs from the publication

A difference of covariance matrices need not be PSD. The implementation
symmetrizes the estimate, diagonalizes the covariance itself, and, when an
eigenvalue is below a scale-aware negative tolerance, clips all negative
covariance eigenvalues to zero:

```python
eigenvalues, eigenvectors = np.linalg.eigh(covariance)
substantive_negative = eigenvalues < -rank_tolerance
if np.count_nonzero(substantive_negative):
    projected_eigenvalues = np.maximum(eigenvalues, 0.0)
    covariance = (
        eigenvectors * projected_eigenvalues
    ) @ eigenvectors.T
```

For a symmetric matrix, this is the nearest PSD matrix under the Frobenius
norm. It is a defensible declared numerical policy, and the code records the
number of substantive negative modes and relative Frobenius adjustment.

It is **not** the adjustment published by Vats and Flegal. Their Section 5
forms the estimated **correlation** matrix, floors its eigenvalues at a
positive \(\epsilon N^{-u}\), and transforms back with the estimated marginal
variances. That adjustment aims for positive definiteness and preserves the
variance scaling through the correlation construction. The toolkit instead
projects directly in covariance units to the PSD boundary and can change the
diagonal. Therefore the paper supports the lugsail estimator, but not this
exact projection rule.

There is also a small implementation nuance: if all negative eigenvalues are
within `rank_tolerance`, `projected_mode_count` is zero and `covariance` is
not replaced by the clipped matrix, although `numerical_rank` is computed
from clipped eigenvalues. Thus evidence may call the matrix unprojected while
the rank calculation treats tiny negative modes as zero.

### 5.5 Lugsail audit verdict

- **Linear combination:** exact match to published Eq. (7), in covariance-of-
  mean units.
- **Default \(r=3,c=1/2\):** matches the paper's first-order over-lugsail
  recommendation.
- **Conservativeness:** conditional asymptotic bias direction, not a bound.
- **Projection:** declared and measured, but not the paper's adjustment.
- **Source field:** publication DOI 10.1093/biomet/asab049 and Eq. (7) are
  now reported.

## 6. Projected scalar Wolff validation

### 6.1 Published scalar Gamma-method calculation

For one centered scalar history \(z_t\), Wolff's single-replica form of Eq.
(31) is

\[
\widehat\Gamma(k)
=\frac1{N-k}\sum_{t=1}^{N-k}z_tz_{t+k}.
\tag{13}
\]

At candidate window \(W\), Eqs. (34), (35), and (41) give

\[
\widehat C(W)=\widehat\Gamma(0)
+2\sum_{k=1}^W\widehat\Gamma(k),
\qquad
\widehat\tau_{\rm int}(W)
=\frac{\widehat C(W)}{2\widehat\Gamma(0)}.
\tag{14}
\]

Wolff's Eq. (51) converts the integrated time to an effective exponential
decay time \(\bar\tau\):

\[
\bar\tau(W)=
\frac{S}{\log\left[
(2\widehat\tau_{\rm int}(W)+1)/
(2\widehat\tau_{\rm int}(W)-1)
\right]}.
\tag{15}
\]

The first \(W\) for which Eq. (52)

\[
g(W)=e^{-W/\bar\tau(W)}
-\frac{\bar\tau(W)}{\sqrt{WN}}
\tag{16}
\]

is negative is selected. Wolff recommends \(S\in[1,2]\), with 1.5 the
reference implementation's default. His Eq. (49) corrects the leading
mean-subtraction bias by multiplying \(\widehat C(W)\) by

\[
1+\frac{2W+1}{N}.
\tag{17}
\]

The source is U. Wolff,
[Monte Carlo Errors with Less Errors](https://arxiv.org/html/hep-lat/0306017v4),
*Computer Physics Communications* 156 (2004), 143--153, DOI
[10.1016/S0010-4655(03)00467-3](https://doi.org/10.1016/S0010-4655(03)00467-3),
with erratum DOI
[10.1016/j.cpc.2006.12.001](https://doi.org/10.1016/j.cpc.2006.12.001).

### 6.2 Formula-to-code map

`validate_projected_wolff` vectorizes the scalar calculation across columns:

```python
gamma_zero = np.sum(centered * centered, axis=0) / configuration_count
covariance_sums = gamma_zero.copy()

for lag in range(1, maximum_lag + 1):
    lag_covariance = (
        np.sum(centered[:-lag] * centered[lag:], axis=0)
        / (configuration_count - lag)
    )
    covariance_sums += 2.0 * lag_covariance
```

This matches Eqs. (13)--(14). The `effective_times` expression is Eq. (15),
the `window_function` is Eq. (16), and the stored `wolff_variances` are

```python
covariance_sums[newly_selected]
* (1.0 + (2.0 * lag + 1.0) / configuration_count)
/ configuration_count
```

which applies Eq. (17) and then Eq. (1), producing variance of the mean.

For \(\widehat\tau_{\rm int}\le 1/2\), Wolff prescribes a tiny positive effective decay time; the
implementation does the same. A zero-variance coordinate is a toolkit edge
case: it is assigned window zero and variance zero because \(\tau_{\rm int}\) is undefined
there.

### 6.3 “Projected” means coordinate projections, not a matrix method

Each input column is treated as one scalar projection. The routine does not
estimate off-diagonal lag covariance, choose a common multivariate window,
or assemble a covariance matrix. It compares its coordinate variances to
the diagonal of the encompassing Bartlett estimate. That design avoids the
scientifically unsafe operation of filling different covariance entries from
incompatible automatically selected windows.

The method is therefore a faithful **coordinate-wise Wolff validator**. It is
not a published full-covariance extension of Wolff's method, and the source
does not justify calling the old rectangular matrix estimator a Gamma method.

### 6.4 Resolution semantics

If no sign change occurs before the configured lag cap, that coordinate is
counted as unresolved. The enclosing Bartlett estimator can record the
problem and continue or raise `UnresolvedAutocorrelation`.

When every coordinate selects a window, the code records the largest
relative variance difference between Wolff and Bartlett. It does **not**
compare that difference to a tolerance. Consequently:

- `Resolved` means the automatic Wolff window condition was met;
- it does not mean Bartlett and Wolff agree;
- a large `maximum_relative_variance_difference` is evidence the caller must
  inspect, not currently a typed failure.

Wolff himself recommends visual inspection of representative autocorrelation
and integrated-time plateaus. The code does not reproduce his error-of-error
bars, plateau plots, multiple-replica compatibility test, or derived-observable
gradient machinery.

### 6.5 Wolff audit verdict

- **Scalar equations:** close match to Eqs. (31), (35), and (49)--(52).
- **Scope:** coordinate-wise validation only.
- **Unresolved cap:** explicit and correctly surfaced.
- **Agreement criterion:** recorded but not enforced.
- **Source field:** publication DOI is appropriate.

## 7. Experimental rectangular estimator

### 7.1 What the code computes

The estimator first constructs coordinate variances with denominator \(N\).
For each lag, it computes diagonal normalized lag products

\[
\widehat\rho_i(k)=
\frac{\sum_{t=1}^{N-k}Z_{t,i}Z_{t+k,i}}
{N\,\widehat R_{ii}(0)}.
\tag{18}
\]

Each coordinate starts at \(\widehat\tau_i=1/2\). A positive Eq. (18) is added only
while \(k<c\widehat\tau_i\), where `c = window_factor`. The probe stops when
no coordinate is active or when it reaches `min(N // 4, 1000)`. A common
matrix window is then

\[
W=\min\left(
\max(4,\lceil c\max_i\widehat\tau_i\rceil),
\lfloor N/4\rfloor
\right).
\tag{19}
\]

At that window the raw matrix is

\[
\widehat V_{\rm rect}(W)=\frac1{N^2}\left[
Z^{\mathsf T}Z+
\sum_{k=1}^W
(Z_{1:N-k}^{\mathsf T}Z_{1+k:N}
+Z_{1+k:N}^{\mathsf T}Z_{1:N-k})
\right].
\tag{20}
\]

Equation (20) is a rectangular, equally weighted lag sum in covariance-of-
mean units. It resembles Wolff's scalar Eq. (35) after extending lag products
to matrices, but its denominator, window rule, and matrix repair are not
Wolff's algorithm.

### 7.2 Why it is not Wolff's Gamma method

The discrepancies are substantive:

1. Wolff uses \(1/(N-k)\) in the lag covariance; Eq. (18) effectively uses
   \(1/N\).
2. Wolff accumulates the signed autocovariance sum and applies Eqs. (15)--(16);
   this heuristic accumulates only currently positive diagonal correlations
   under a different self-consistency inequality.
3. Wolff chooses a scalar window for a declared scalar projection; this
   heuristic takes the maximum coordinate-derived scale and uses it for a
   full matrix.
4. Wolff applies the explicit finite-\(N\) correction in Eq. (17); this
   estimator does not.
5. A rectangular lag window is not guaranteed PSD. The code clips all
   negative covariance eigenvalues to zero after estimation, a step absent
   from Wolff's scalar method.

The transitional alias

```python
GammaMethod = ExperimentalRectangularLongRunCovariance
```

is explicitly accompanied by the source comment “This is not Wolff's
method.” The evidence uses `estimator="experimental-rectangular-lag-sum"`
and `source=None`. Those labels accurately reflect the audit: the method is
an agent-generated legacy heuristic retained for differential comparison,
not a literature-backed production estimator.

### 7.3 Projection semantics

Unlike lugsail, this implementation always reconstructs the covariance from
`maximum(eigenvalue, 0)`. `projected_mode_count`, however, counts every
strictly negative eigenvalue without a scale tolerance. A tiny roundoff-level
negative mode therefore counts as a projection. The relative Frobenius
adjustment is appropriately recorded.

### 7.4 Experimental estimator audit verdict

- **Rectangular sum:** mathematically legible but not PSD by construction.
- **Window selection:** no identified canonical source.
- **Gamma-method name:** scientifically misleading except as a transitional
  compatibility alias.
- **Production status:** quarantine and `source=None` are justified.

## 8. Cross-method matrix and rank implications

For centered \(N\times p\) data, a sample-supported covariance generally has
rank no larger than \(N-1\). Batch means has rank no larger than \(a-1\).
Consequently, singularity for \(p\ge N\) is expected evidence, not a defect.

The Bartlett and ordinary batch-means implementations preserve this rank
honestly through factors. Lugsail and experimental rectangular methods form
dense matrices and diagonalize them. Their PSD projections can remove
negative directions but cannot supply independently observed information in
the sample null space.

This matters for downstream fits: none of the cited covariance-estimation
papers licenses silently inverting a rank-deficient matrix. Any dimension
reduction, pseudoinverse, shrinkage, or regularization is a separate method
choice requiring its own evidence and authority.

## 9. Source and evidence audit

| Evidence estimator | Current source value | Audit |
|---|---|---|
| `bartlett-newey-west` | `https://doi.org/10.2307/1913610#equation-5` | Correct publication DOI and equation attribution for the triangular kernel. Does not by itself source the toolkit IID correction or stability check. |
| `multivariate-batch-means` | `https://doi.org/10.1093/biomet/asz002#equation-10` | Correct publication DOI and equation attribution. The code's extra division by \(N\) is the necessary conversion to covariance of the mean. |
| `over-lugsail-batch-means` | `https://doi.org/10.1093/biomet/asab049#equation-7` | Correct publication DOI and equation attribution for the lugsail combination. The source does not describe the code's direct nearest-PSD projection. |
| projected Wolff evidence | `https://doi.org/10.1016/S0010-4655(03)00467-3` | Correct publication DOI. The 2007 erratum DOI may also be worth retaining in extended documentation. |
| `experimental-rectangular-lag-sum` | `None` | Correct: no canonical source has been established for the complete heuristic. |

## 10. Recommended interpretation in reports

Reports should distinguish three categories explicitly:

1. **Literature estimator:** Bartlett/Newey--West, multivariate batch means,
   or over-lugsail batch means, with publication DOI and equation.
2. **Toolkit policy layered on the estimator:** IID centering correction,
   chosen bandwidth/batch size, record-versus-raise behavior, rank tolerance,
   and any PSD projection.
3. **Diagnostic or experimental comparator:** projected scalar Wolff evidence
   or the unsourced rectangular heuristic.

In particular, a report should not imply that:

- Newey and West selected the configured bandwidth or derived the toolkit's
  exact IID correction;
- Vats and Flegal published the toolkit's direct covariance eigen-clipping;
- a Wolff window sign change proves agreement with the matrix estimator; or
- `GammaMethod` denotes a validated full-covariance implementation of Wolff.

## Primary references

- W. K. Newey and K. D. West,
  [A Simple, Positive Semi-Definite, Heteroskedasticity and Autocorrelation
  Consistent Covariance Matrix](https://doi.org/10.2307/1913610),
  *Econometrica* 55 (1987), 703--708. Working-paper DOI:
  [10.3386/t0055](https://doi.org/10.3386/t0055).
- D. Vats, J. M. Flegal, and G. L. Jones,
  [Strong Consistency of Multivariate Spectral Variance Estimators in Markov
  Chain Monte Carlo](https://arxiv.org/html/1507.08266v3),
  *Bernoulli* 24 (2018), 1860--1909, DOI
  [10.3150/16-BEJ914](https://doi.org/10.3150/16-BEJ914).
- D. Vats, J. M. Flegal, and G. L. Jones,
  [Multivariate Output Analysis for Markov Chain Monte
  Carlo](https://arxiv.org/html/1512.07713v4),
  *Biometrika* 106 (2019), 321--337, DOI
  [10.1093/biomet/asz002](https://doi.org/10.1093/biomet/asz002).
- D. Vats and J. M. Flegal,
  [Lugsail Lag Windows for Estimating Time-Average Covariance
  Matrices](https://arxiv.org/html/1809.04541v3),
  *Biometrika* 109 (2022), 735--750, DOI
  [10.1093/biomet/asab049](https://doi.org/10.1093/biomet/asab049).
- U. Wolff,
  [Monte Carlo Errors with Less Errors](https://arxiv.org/html/hep-lat/0306017v4),
  *Computer Physics Communications* 156 (2004), 143--153, DOI
  [10.1016/S0010-4655(03)00467-3](https://doi.org/10.1016/S0010-4655(03)00467-3),
  erratum DOI
  [10.1016/j.cpc.2006.12.001](https://doi.org/10.1016/j.cpc.2006.12.001).
