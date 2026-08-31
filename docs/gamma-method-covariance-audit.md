# Audit of the full-covariance Gamma-method estimator

Status: historical methodology audit of the removed experimental method.

## Executive conclusion

The core idea is not an unsupported agent invention.  For a vector-valued
stationary Markov-chain history, the covariance of the sample-mean vector is
governed by the sum of **all lagged cross-covariance matrices**, not merely by
autocorrelation corrections to the diagonal.  Wolff writes this matrix result
explicitly in Eqs. (11)--(13), and modern MCMC literature calls the corresponding
sample construction a multivariate spectral-variance estimator (MSVE).  The
`N**2` normalization in both local implementations is consistent with the MSVE
definition.[^wolff][^vats-msve]

The present implementation as a whole is nevertheless **not justified as
"Wolff's Gamma method"**.  It combines an established rectangular multivariate
lag sum with an implementation-specific window heuristic and a nearest-PSD
projection.  In particular:

- the window loop is neither Wolff's automatic window nor Sokal's stated
  self-consistent window;
- choosing the common window from the maximum of the coordinate-wise diagonal
  autocorrelation times has no general guarantee for arbitrary linear
  combinations of those coordinates;
- the rectangular lag window can be indefinite, while clipping its negative
  eigenvalues changes the statistical estimator rather than merely changing its
  numerical representation;
- when the number of flow-time values `M` exceeds the number of configurations
  `N`, any covariance built from the centered `N x M` history is necessarily
  singular, and the fixed-dimension asymptotic theory cited here does not justify
  this high-dimensional regime; and
- estimating each measurement channel separately discards cross-channel blocks
  that are required by any later joint comparison, combination, or fit.

The scientifically defensible next step is therefore not to discard full
covariance.  It is to retain the multivariate formulation, replace the current
hybrid policy with a named estimator, and validate it independently on IID and
multivariate autoregressive processes with known long-run covariance.

## Code under audit

The audited toolkit implementation was removed after this review established
that it had no defensible scientific role.
The legacy implementation is
[`../ContinuousBetaFunction/src/betafn/processing/gamma.py`](../../ContinuousBetaFunction/src/betafn/processing/gamma.py).

Both compute the same unprojected covariance.  If

\[
Z_i = X_i-\bar X, \qquad i=1,\ldots,N,
\]

is an `M`-component centered history, they form

\[
\widehat V_W(\bar X)
= \frac{1}{N^2}
  \left[
    \sum_{i=1}^N Z_i Z_i^T
    + \sum_{k=1}^W \sum_{i=1}^{N-k}
      \left(Z_i Z_{i+k}^T + Z_{i+k}Z_i^T\right)
  \right].
\tag{1}
\]

The new implementation then replaces negative eigenvalues of (1) by zero before
constructing correlated values.  The legacy implementation passes (1) directly
to `gvar`.

## 1. Established results

### 1.1 The target really is a full long-run covariance matrix

For the equilibrium lag cross-covariance

\[
\Gamma_{ab}(k)
= \operatorname{Cov}(X_{i,a},X_{i+k,b}),
\qquad
\Gamma_{ab}(-k)=\Gamma_{ba}(k),
\]

define the long-run covariance

\[
C = \sum_{k=-\infty}^{\infty}\Gamma(k).
\tag{2}
\]

Under the usual stationarity, summability, and Markov-chain central-limit
conditions,

\[
\sqrt N(\bar X-\mu) \Rightarrow \mathcal N_M(0,C),
\qquad
\operatorname{Cov}(\bar X) \simeq \frac{C}{N}.
\tag{3}
\]

Wolff gives (2)--(3) component by component in Eqs. (11)--(13).  He also defines
the cross-lag relation \(\Gamma_{ab}(-k)=\Gamma_{ba}(k)\) in Eq. (4).[^wolff]
Vats, Flegal, and Jones formulate the same object as the covariance matrix in a
multivariate Markov-chain CLT and define an MSVE as a weighted, truncated sum of
sample autocovariance matrices in Eqs. (2.1)--(2.2).[^vats-msve]

Consequently, correcting only the diagonal variances is generally insufficient.
Each off-diagonal entry has its own lagged cross-covariance sum.  This part of the
current method is established statistical methodology.

### 1.2 The `N**2` normalization is correct for the estimator actually coded

Vats, Flegal, and Jones define

\[
\widehat\Gamma_N(k)
= \frac{1}{N}\sum_{i=1}^{N-k} Z_i Z_{i+k}^T.
\tag{4}
\]

With a rectangular lag window, the estimated long-run covariance is

\[
\widehat C_W
= \widehat\Gamma_N(0)
  +\sum_{k=1}^W
   \left[\widehat\Gamma_N(k)+\widehat\Gamma_N(k)^T\right].
\tag{5}
\]

Dividing (5) by `N` to estimate the covariance of the mean produces exactly
Eq. (1), hence the code's division by `N**2` is correct.[^vats-msve]

The use of `N`, rather than `N-k`, in Eq. (4) also retains the triangular
finite-history factor.  With the *true* mean and autocovariances, the exact
finite-`N` identity is

\[
\operatorname{Cov}(\bar X)
=\frac{1}{N}\left[
\Gamma(0)+\sum_{k=1}^{N-1}
\left(1-\frac{k}{N}\right)
\left(\Gamma(k)+\Gamma(k)^T\right)
\right].
\tag{6}
\]

Equation (1) is still an estimated and truncated version of (6); the legacy
docstring's description of integrating the "exact" cross-correlation functions
is therefore too strong.

### 1.3 Centering by the sample mean creates a known finite-sample bias

The local code centers every history with the same estimated sample mean.  Wolff
derives the resulting leading bias in the estimated autocovariance in Eq. (32)
and applies the leading correction

\[
\widehat C_W \longmapsto
\widehat C_W\left(1+\frac{2W+1}{N}\right)
\tag{7}
\]

in Eq. (49).[^wolff]  Neither local implementation applies Eq. (7).

For an IID scalar history of variance \(\sigma^2\), Eq. (1) makes the omitted
bias especially transparent:

\[
\mathbb E[\widehat V_W(\bar X)]
= \frac{\sigma^2}{N}
\left[
1-\frac{2W+1}{N}+\frac{W(W+1)}{N^2}
\right].
\tag{8}
\]

Thus even for independent data the mandatory minimum window of four introduces
about a \(9/N\) relative downward bias before higher-order terms.  This is not a
normalization error in Eq. (1); it is a finite-sample centering bias.  An
asymptotic MSVE may legitimately leave it uncorrected, but then it should not be
described as a faithful implementation of Wolff's finite-sample procedure.

An independent local simulation performed during this audit makes the effect
large enough to see directly.  Across 20,000 IID unit-variance repetitions, the
mean normalized estimates `N * V_hat` (whose target is one) were approximately
`0.621`, `0.823`, `0.915`, and `0.975` for `N = 20`, `50`, `100`, and `400`,
respectively; the selected window was essentially always four.  The
corresponding unprojected scalar estimates were negative in approximately
`14.68%`, `2.395%`, `0.155%`, and `0%` of repetitions.  Eigenvalue clipping turns
those negative scalar estimates into zero, so it changes the bias distribution
rather than diagnosing or correcting Eq. (8).  The simulated means track Eq.
(8), including the expected recovery only as `N` becomes large relative to the
forced window.

## 2. Implementation-specific choices

### 2.1 One common window from the maximum diagonal time

Both implementations estimate one autocorrelation time per coordinate, set

\[
W=\left\lceil c\max_a\widehat\tau_{\mathrm{int},a}\right\rceil,
\tag{9}
\]

and use that `W` for every matrix entry.  A common lag window is coherent: it
applies one matrix-valued spectral estimator instead of constructing entries
with unrelated truncations.  Taking the maximum coordinate time is also an
understandable conservative heuristic relative to analyzing each coordinate
alone.

It is not, however, a general bound on the autocorrelation time of every linear
combination \(u^T X\).  Fast modes can cancel in a linear combination and expose
a slow mode that is weak in each coordinate's normalized diagonal
autocorrelation.  Wolff addresses a requested derived quantity by projecting the
primary histories onto that quantity's gradient and analyzing the projected
autocorrelation directly (Eqs. (33) and (37)).[^wolff]  No primary source found
in this audit establishes Eq. (9) as a sufficient window for an entire
covariance matrix.

This choice is therefore a toolkit policy, not a consequence of the Gamma
method.  It requires a sensitivity study or a separate bandwidth justification.

### 2.2 The loop is neither Wolff's nor Sokal's automatic window

The code starts every coordinate at \(\widehat\tau=1/2\).  At lag `k`, it adds a
correlation only when both

```text
rho_a(k) > 0
k < c * tau_a(current)
```

hold.  It stops the entire scan whenever no coordinate is active at one lag, and
finally applies Eq. (9).

Sokal's rectangular self-consistent rule instead computes the **signed** partial
sum through candidate `M` and chooses the smallest `M` satisfying

\[
M \ge c\,\widehat\tau_{\mathrm{int}}(M).
\tag{10}
\]

Sokal recommends approximately `c = 4` for nearly exponential decay and at
least `6`, possibly `10`, when decay can be slower.[^sokal]  The legacy
docstring's statement that Sokal recommended `1.5` is incorrect.

Wolff's procedure is different again.  It assumes an exponential scale
\(\tau\simeq S\tau_{\mathrm{int}}\), computes an effective \(\bar\tau(W)\), and
chooses the first sign change of his function

\[
g(W)=\exp[-W/\bar\tau(W)]-\bar\tau(W)/\sqrt{WN}
\tag{11}
\]

in Eqs. (50)--(52).  Wolff suggests `S = 1...2`, with `1.5` the software
default.[^wolff]  This `S` is not Sokal's multiplier `c` and should not be given
the same configuration name.

The local positive-only accumulation is thus an undocumented hybrid.  It also
has concrete edge cases:

- negative correlations are omitted from the estimated diagonal time even
  though they are included in the final covariance sum;
- a lag at which every coordinate happens to be nonpositive terminates the scan,
  so later positive lobes cannot be detected;
- with a configured factor `c <= 2`, lag one cannot pass
  `1 < c * 0.5`, so the scan never begins; and
- the hard caps `N//4` and `1000`, and the mandatory minimum of four before the
  `N//4` cap, are implementation policies with no recorded statistical
  rationale.

### 2.3 Fixed-window rectangular truncation lacks the cited consistency result

The consistency theorem for MSVEs requires, among other conditions, a bandwidth
\(b_N\) that grows while \(b_N/N\) vanishes.  Vats, Flegal, and Jones explicitly
state \(b_N\to\infty\) and \(N/b_N\to\infty\).  They also show that simple
rectangular truncation fails one of their sufficient lag-window conditions,
whereas their modified Bartlett and Tukey--Hanning examples satisfy the stated
conditions under suitable bandwidth growth.[^vats-msve]

For a process with a fixed autocorrelation scale, Eq. (9) can converge to a fixed
`W` as `N` grows.  Its omitted tail then need not vanish.  This does not make a
finite-window estimate useless, but it means the strong-consistency theorem must
not be cited as though it covered the current data-dependent rule.

## 3. PSD behavior and high dimension

### 3.1 Why the rectangular estimate can be indefinite

Let `Z` be the centered `N x M` history matrix.  Equation (1) can be written

\[
\widehat V_W(\bar X)=\frac{1}{N^2}Z^T K_W Z,
\tag{12}
\]

where \((K_W)_{ij}=1\) for \(|i-j|\le W\) and zero otherwise.  The rectangular
Toeplitz matrix \(K_W\) is not generally positive semidefinite.  Therefore Eq.
(12) need not be PSD even though the population long-run covariance is PSD.
This is a property of the estimator, not evidence that the original measurements
have an invalid covariance.

### 3.2 A standard PSD-preserving alternative

The Bartlett/Newey--West lag weights

\[
w_k=1-\frac{|k|}{W+1},\qquad |k|\le W,
\tag{13}
\]

replace `K_W` by a PSD Toeplitz kernel, making `Z.T @ K @ Z` PSD by construction.
Newey and West introduced this positive-semidefinite heteroskedasticity- and
autocorrelation-consistent construction and proved consistency under their
conditions.[^newey-west]  In the MCMC setting, Vats, Flegal, and Jones call
Eq. (13) the modified Bartlett window and include it in their strongly consistent
MSVE class under suitable bandwidth growth.[^vats-msve]

Data-dependent bandwidth selection for HAC estimators has its own established
literature; for example, Andrews derives automatic bandwidth procedures rather
than identifying the bandwidth with a coordinate-wise maximum autocorrelation
time.[^andrews]  A bandwidth policy still has to be checked for the assumptions
and the target high-dimensional regime.

Multivariate batch means is another PSD-by-construction comparator with
established MCMC consistency results, and is useful as an independent numerical
oracle even if a spectral estimator remains the production choice.[^vats-output]

These alternatives do not eliminate bandwidth bias or bandwidth selection.
They avoid manufacturing negative eigenvalues through the lag kernel.

### 3.3 What nearest-PSD projection does and does not justify

For a symmetric matrix, replacing negative eigenvalues by zero is the unique
nearest PSD matrix in Frobenius norm.[^higham]  The code's matrix-nearness claim
is therefore mathematically correct.

That theorem does **not** show that the projected matrix is the best statistical
estimate of the long-run covariance.  Projection changes variances and
correlations, discards information about the magnitude and directions of the
estimator's failure, and creates zero-variance directions.  It should be treated
as a declared regularization or fallback, not as validation of the rectangular
estimator.

The current diagnostic counts every eigenvalue strictly below floating-point
zero.  It has no scale-aware tolerance and does not distinguish a substantively
negative mode from roundoff in an exact null space.  Consequently,
"projected mode count" is not presently a scientifically interpretable rank or
failure diagnostic.

### 3.4 When `M > N`

Because `Z` is centered, `rank(Z) <= N-1`.  Equation (12), and its
eigenvalue-clipped projection, therefore satisfy

\[
\operatorname{rank}(\widehat V_W)\le N-1.
\tag{14}
\]

If `M > N-1`, the estimated covariance must be singular regardless of whether
the lag kernel is rectangular or Bartlett.  Nearest-PSD projection cannot make
it invertible without adding a separate positive regularizer.

This matters directly to the current use case: hundreds of configurations and
roughly eight hundred flow-time coordinates cannot support an unconstrained,
full-rank covariance over all flow times.  Correlated linear propagation remains
possible, but an inverse-covariance fit in the full coordinate space is
underdetermined.  Dimension reduction, a physically declared covariance model,
or fit-local regularization would be additional scientific choices.

The Vats--Flegal--Jones theory is formulated for a vector in
\(\mathbb R^p\) with fixed `p` and `p x p` limiting matrices; it provides no
high-dimensional theorem with `p` comparable to or larger than `N`.[^vats-msve]
The current `M > N` application must therefore be validated as its own regime,
not justified merely by citing fixed-dimensional MSVE consistency.

## 4. Cross-channel covariance

The new processing code forms configured linear combinations from raw histories
before averaging, which correctly retains covariance among the source histories
inside that combination.  It then calls `gamma_estimate` separately for each
measurement channel.  The legacy implementation likewise groups by flow and
observable type and estimates each group independently.

Separate calls create independent correlated-value blocks.  They preserve:

- covariance among flow times within one channel; and
- source covariance inside a raw-history linear combination.

They discard:

- covariance between the final plaquette, combined Symanzik, and clover
  channels, even when measured on the same gauge configurations; and
- covariance between flow definitions when their configuration identities are
  aligned.

The multivariate CLT and MSVE do not privilege "flow time" over "operator" as a
vector coordinate.  If two outputs will enter any joint difference, average,
model comparison, or fit, their aligned histories should be stacked and their
cross-channel blocks estimated jointly.[^wolff][^vats-msve]  If channels are
guaranteed never to enter joint arithmetic, separate estimation can be an
explicit resource-saving policy, but the omitted cross-covariances must remain
visible in provenance.  The legacy comment that no downstream code currently
fits channels jointly is an implementation fact, not a statistical independence
argument.

Histories with different configuration sets cannot simply be stacked by row.
Joint estimation then requires semantic configuration identities and an explicit
missing-data/replica policy; guessing alignment from equal lengths is invalid.

## 5. Findings classified

### Established and correctly represented

- The covariance of a vector sample mean is the full integrated lagged
  cross-covariance matrix divided by `N`.
- Symmetrizing positive and negative lags as
  `Gamma(k) + Gamma(k).T` is correct without assuming a reversible chain.
- The code's `N**2` normalization matches the standard sample-autocovariance
  convention used by an MSVE.
- Raw-history combinations before averaging preserve the source covariance to
  which that combination is entitled.
- Eigenvalue clipping is the nearest-PSD matrix in Frobenius norm.

### Implementation-specific, not established by the cited Gamma-method papers

- selecting one matrix window from the maximum diagonal estimated time;
- adding only positive diagonal lag correlations while choosing that window;
- the minimum window `4`, maximum `N//4`, and probe cap `1000`;
- always applying nearest-PSD projection; and
- estimating each measurement channel as an independent covariance block.

### Incorrect descriptions or unsupported claims

- The code is not a faithful implementation of Wolff's automatic-window
  procedure.
- It is not Sokal's self-consistent window procedure either.
- `1.5` is Wolff's default scale-ratio parameter, not Sokal's recommended
  self-consistent multiplier.
- A rectangular finite-window spectral estimate is not inherently indefinite
  "especially when there are more values than configurations."  `M > N` forces
  singularity; rectangular weighting can cause substantive indefiniteness.
  Those are different phenomena.
- PSD projection is not merely a numerical repair when genuinely negative
  modes exist.
- The current projection count, with a strict zero threshold, cannot distinguish
  roundoff-split null modes from negative statistical modes.
- Separate channel estimation does lose real covariance whenever the channels
  share configurations and are later used jointly.

### Risks requiring evidence before scientific use

- downward finite-sample bias from global centering, including for IID data;
- missed long autocorrelation in a linear combination not represented by a
  coordinate-wise diagonal time;
- missed delayed or oscillatory correlations due to positive-only early
  stopping;
- sensitivity of results to window/bandwidth policy;
- singular covariance and unstable inverse fits when `M > N`; and
- distortion of low-variance directions by projection, even when its global
  Frobenius adjustment is small.

## 6. Recommendations

1. **Keep the full-covariance objective.**  It is the correct target and has
   primary-source support.  Do not revert to independently inflated diagonal
   errors.
2. **Do not freeze the current estimator as an oracle yet.**  Label existing
   outputs experimental until the choices below are resolved.
3. **Name the estimator by what it does.**  For example,
   "multivariate rectangular spectral-variance estimator with [named] bandwidth
   rule and nearest-PSD projection."  Reserve "Wolff" for an implementation of
   Eqs. (31), (35), (49), and (50)--(52), including documented deviations.
4. **Prototype a Bartlett MSVE.**  Use one explicit bandwidth for the joint
   vector and PSD-by-construction lag weights.  Retain rectangular/Wolff-style
   estimates as sensitivity variants rather than silently projecting them.
5. **Make bandwidth evidence first-class.**  Record the rule, candidate
   bandwidths, signed autocorrelation summaries, truncation/probe caps, and
   stability of scientifically relevant linear combinations.
6. **Estimate channels jointly when downstream use is joint.**  Stack aligned
   raw histories once, apply observable combinations as linear maps, and then
   extract channel blocks.  This guarantees algebraic consistency between a
   direct raw-history combination and covariance propagation from the joint
   estimate.
7. **Treat high dimension explicitly.**  Report numerical rank with a
   scale-aware tolerance, the `M/N` ratio, and whether later operations require
   inversion.  Do not present eigenvalue clipping as solving rank deficiency.
8. **Add independent numerical oracles before choosing a production policy:**
   - IID multivariate Gaussian histories, checking Eq. (8) and the selected
     finite-sample correction;
   - scalar AR(1), including positive and negative coefficients;
   - a VAR(1) process with analytically known long-run covariance, as in Eq.
     (3.2) of Vats, Flegal, and Jones;
   - a two-coordinate construction where a slow mode is exposed only by a
     linear combination;
   - exact agreement between joint-covariance propagation and combining aligned
     raw histories first; and
   - `M > N` cases checking rank, PSD tolerance, and rejection of undeclared
     inverse-covariance use.
9. **Compare against an independent estimator.**  Multivariate batch means is a
   useful PSD comparator.  Agreement across justified bandwidth/batch-size
   ranges is stronger evidence than agreement with the legacy agent-written
   code.
10. **Separate methodology decisions.**  Window choice, lag kernel,
    finite-sample bias correction, high-dimensional regularization, and
    cross-channel scope are distinct choices.  They should not be bundled into
    one opaque `GammaMethod(window=3.0)` setting.

## Sources

[^wolff]: U. Wolff, ["Monte Carlo errors with less errors," *Computer Physics
    Communications* 156 (2004) 143--153](https://doi.org/10.1016/S0010-4655(03)00467-3),
    especially Eqs. (4), (11)--(13), (31)--(37), (49), and (50)--(52).  The
    [author manuscript is available on arXiv](https://arxiv.org/abs/hep-lat/0306017).

[^sokal]: A. D. Sokal, ["Monte Carlo Methods in Statistical Mechanics:
    Foundations and New Algorithms," in *Functional Integration: Basics and
    Applications* (1997), pp. 131--192](https://doi.org/10.1007/978-1-4899-0319-8_6),
    Section 3.3, especially Eqs. (3.16)--(3.19) and the automatic-window
    discussion.

[^vats-msve]: D. Vats, J. M. Flegal, and G. L. Jones, ["Strong consistency of
    multivariate spectral variance estimators in Markov chain Monte Carlo,"
    *Bernoulli* 24 (2018) 1860--1909](https://doi.org/10.3150/16-BEJ914),
    especially Eqs. (1.3), (2.1)--(2.2), Conditions 1--4, Theorem 1, and Remark
    7.  An [open journal PDF](https://imstat.org/publications/bej/bej_24_3/bej_24_3.pdf)
    is available from the Institute of Mathematical Statistics.

[^newey-west]: W. K. Newey and K. D. West, ["A Simple, Positive
    Semi-definite, Heteroskedasticity and Autocorrelation Consistent Covariance
    Matrix," *Econometrica* 55 (1987) 703--708](https://doi.org/10.2307/1913610).
    The [NBER technical-paper version](https://www.nber.org/papers/t0055) states
    the PSD-by-construction and consistency results.

[^andrews]: D. W. K. Andrews, ["Heteroskedasticity and Autocorrelation
    Consistent Covariance Matrix Estimation," *Econometrica* 59 (1991)
    817--858](https://doi.org/10.2307/2938229), especially Sections 5--6 on
    automatic bandwidth selection.

[^higham]: N. J. Higham, ["Computing a nearest symmetric positive semidefinite
    matrix," *Linear Algebra and its Applications* 103 (1988)
    103--118](https://doi.org/10.1016/0024-3795(88)90223-6), especially Theorem
    2.1.  For symmetric `A`, the result reduces to replacing each negative
    eigenvalue by zero.

[^vats-output]: D. Vats, J. M. Flegal, and G. L. Jones, ["Multivariate output
    analysis for Markov chain Monte Carlo," *Biometrika* 106 (2019)
    321--337](https://doi.org/10.1093/biomet/asz002), especially Section 4 on
    the multivariate batch-means estimator and its strong consistency.
