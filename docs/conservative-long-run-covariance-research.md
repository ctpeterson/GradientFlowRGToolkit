# Long-run covariance estimation: sourced implementation guidance

Status: research note, not a statement of implemented behavior.

This note identifies a defensible replacement for the current rectangular
lag-sum estimator and answers a narrower question: can the toolkit arrange to
overestimate uncertainty when an autocorrelation time is not resolved? The
short answer is:

> Bartlett/Newey--West gives a positive-semidefinite, consistent estimator, but
> it does not guarantee conservative finite-sample uncertainty. Some methods
> deliberately reduce or reverse the usual downward bias, but none found here
> gives a distribution-free finite-sample matrix upper bound from one finite
> history. If the autocorrelation tail is unresolved, the scientifically safe
> behavior is an explicit unresolved-autocorrelation failure, not an asserted
> covariance.

That conclusion is compatible with Wolff's own warning immediately after his
Eqs. (12)--(13): the run lengths must be much larger than the decay scale, and
if that condition is violated, reliable error estimation is hardly possible
([Wolff 2004, pp. 3--4](https://arxiv.org/pdf/hep-lat/0306017)).

## 1. Quantity being estimated

Let the aligned, stationary, vector-valued Monte Carlo history be

\[
Y_t\in\mathbb R^p,\qquad t=1,\ldots,N,
\]

with mean \(\mu\) and lag covariance

\[
\Gamma(k)=\operatorname{Cov}(Y_t,Y_{t+k}),\qquad
\Gamma(-k)=\Gamma(k)^T.
\]

The long-run covariance (also called the time-average covariance or the
spectral density at zero, up to convention-dependent constants) is

\[
\Sigma=\sum_{k=-\infty}^{\infty}\Gamma(k).
\]

Under the corresponding multivariate central limit theorem,

\[
\sqrt N(\bar Y-\mu)\Rightarrow\mathcal N_p(0,\Sigma),
\qquad
\operatorname{Cov}(\bar Y)\simeq \frac{\Sigma}{N}.
\]

The distinction must be explicit in code and evidence:

- the lag-window estimator estimates \(\Sigma\);
- the covariance attached to the sample mean is \(\widehat\Sigma/N\).

Wolff gives the matrix-valued definitions in Eqs. (2), (4), and (12), followed
by \(\operatorname{Cov}(\bar Y)=\Sigma/N\) in Eq. (13). Vats, Flegal, and Jones
give the multivariate CLT and weighted matrix lag sum in Eqs. (1.3), (2.1), and
(2.2) of their primary treatment
([Wolff 2004](https://arxiv.org/pdf/hep-lat/0306017);
[Vats, Flegal, and Jones 2018](https://arxiv.org/pdf/1507.08266)).

All measurement channels observed on the same configurations belong in one
\(Y_t\). Separate channel calculations silently set cross-channel covariance to
zero. Raw-history combinations must be made before this joint vector is
estimated, in accordance with project invariants 5 and 6.

## 2. Bartlett/Newey--West estimator

Define centered observations \(Z_t=Y_t-\bar Y\) and the denominator-\(N\)
empirical lag covariance

\[
\widehat\Gamma(k)=\frac1N\sum_{t=1}^{N-k}Z_tZ_{t+k}^T,
\qquad k\geq0.
\]

For a largest included lag \(m\), the Bartlett/Newey--West estimator is

\[
\widehat\Sigma_{\mathrm B}(m)
=\widehat\Gamma(0)
+\sum_{k=1}^{m}\left(1-\frac{k}{m+1}\right)
 \left[\widehat\Gamma(k)+\widehat\Gamma(k)^T\right].
\tag{1}
\]

An equivalent convention writes \(b=m+1\), uses
\(w_b(k)=1-|k|/b\) for \(|k|<b\), and calls \(b\) the bandwidth. The API must
choose one convention and record both `largest_included_lag` and `bandwidth`
to remove this common off-by-one ambiguity.

Newey and West define the triangular weights in their Eq. (5) and prove in
Theorem 1 that the resulting matrix is positive semidefinite. They also prove
consistency under their stated conditions
([Newey and West 1987, pp. 704--707](https://users.ssc.wisc.edu/~behansen/718/NeweyWest1987.pdf),
[NBER record](https://www.nber.org/papers/t0055)). Vats, Flegal, and Jones call
the same \(q=1\) kernel the modified Bartlett/Parzen window; Remark 7(3) shows
that it meets their lag-window conditions under the stated bandwidth rates
([Vats, Flegal, and Jones 2018, pp. 7--12](https://arxiv.org/pdf/1507.08266)).

### What PSD does and does not mean

Positive semidefinite means

\[
u^T\widehat\Sigma_{\mathrm B}u\geq0
\quad\text{for every }u.
\]

It does **not** mean

\[
\widehat\Sigma_{\mathrm B}-\Sigma\succeq0,
\]

and therefore does not mean that every uncertainty is overestimated. PSD is
an algebraic validity guarantee; conservatism is a separate statistical
claim.

The estimator is linearly equivariant when the same bandwidth is used: for a
fixed matrix \(A\), estimating the transformed history \(AY_t\) gives

\[
\widehat\Sigma_{AY}=A\widehat\Sigma_YA^T.
\]

This is an important property test and a reason not to choose a different
bandwidth independently for every matrix entry.

### Consistency is asymptotic and fixed-dimensional

The multivariate strong-consistency theorem of Vats, Flegal, and Jones assumes
a strong invariance principle, moment/mixing conditions, and bandwidth growth.
In particular, their Conditions 1--4 require, among more technical rates,

\[
b_N\to\infty,
\qquad
N/b_N\to\infty.
\]

Their Theorem 1 then gives
\(\widehat\Sigma_{\mathrm B}\to\Sigma\) almost surely. The theorem is for a
fixed \(p\); it is not a high-dimensional \(p/N\) theorem
([Vats, Flegal, and Jones 2018, pp. 7--10](https://arxiv.org/pdf/1507.08266)).

By contrast, their Remark 7(1) shows that the rectangular/simple-truncation
window fails one of the conditions used by that theorem. That does not prove
every rectangular estimator inconsistent, but it prevents citing this theorem
as justification for the previous implementation
([same paper, p. 11](https://arxiv.org/pdf/1507.08266)).

## 3. Bandwidth selection

Bandwidth controls a bias--variance trade-off:

- too small: correlated tail terms are omitted, commonly causing downward
  bias for positively persistent histories;
- too large: noisy lag estimates dominate, increasing estimator variance;
- no finite data-dependent choice can reveal a correlation mode whose decay
  lies beyond the informative length of the history without additional model
  assumptions.

Three different statements must not be conflated.

### 3.1 Explicit bandwidth

An explicitly supplied \(m\) is the narrowest reproducible first interface.
It is a scientific analysis choice, not an implementation constant. A result
should record:

- the largest included lag and triangular weights;
- the history length and any independent-replica boundaries;
- the values of relevant projected variances over a declared bandwidth grid;
- whether the chosen value is internal to a stable region or lies at a
  diagnostic cap.

This does not make the choice correct by declaration. It makes the assumption
inspectable and permits systematic variations.

### 3.2 Power-law rules

Rules such as \(b_N=\lfloor N^{1/3}\rfloor\) or
\(\lfloor N^{1/2}\rfloor\) are policies, not universal estimates of the
autocorrelation scale. The consistency literature permits broad power-law
sequences subject to its rates. Mean-squared-error calculations for Bartlett
batch/spectral estimators often give

\[
b_{\mathrm{opt}}=C_{\mathrm{process}}N^{1/3},
\]

but the coefficient contains unknown long-run quantities. Liu, Vats, and
Flegal give, for one overlapping-Bartlett matrix entry,

\[
b_{\mathrm{opt},ij}
=\left[
\frac{3\,\Gamma_{ij}^{(1)2}N}
{\Sigma_{ii}\Sigma_{jj}+\Sigma_{ij}^2}
\right]^{1/3}
\tag{2}
\]

in their Eq. (3) specialized on p. 6. Their Eq. (2) displays the leading
squared-bias term proportional to \(b^{-2}\) and variance term proportional to
\(b/N\)
([Liu, Vats, and Flegal 2022, pp. 5--7](https://arxiv.org/pdf/1804.05975)).

Thus `N**(1/3)` alone drops the process-dependent coefficient. A larger
\(N^{1/2}\) rule is often used to reduce truncation bias, particularly for slow
mixing, but the same paper calls it MSE-suboptimal and does not prove it
conservative ([pp. 1--2](https://arxiv.org/pdf/1804.05975)).

### 3.3 Plug-in rules

Andrews derives asymptotic truncated-MSE-optimal kernels and data-dependent
plug-in bandwidths; these optimize an asymptotic two-sided loss, not the risk
of underestimating uncertainty
([Andrews 1991](https://personal.utdallas.edu/~d.sul/Econo2/andrews91.pdf),
especially Sections 4--6).

For MCMC, Liu, Vats, and Flegal propose a lower-variability parametric pilot:
fit an AR(\(r\)) model to each marginal, choose its order by AIC, estimate the
unknown \(\Sigma_{ii}\) and first bias moment, then combine marginal
coefficients into one bandwidth. Their Eqs. (3) and the formulas on pp. 7--9
specify the calculation
([Liu, Vats, and Flegal 2022](https://arxiv.org/pdf/1804.05975)).

This is a real sourced automatic policy, but it adds assumptions:

- marginal AR approximations may not expose a slow *linear combination*;
- AIC order selection and pilot failures need diagnostics;
- the procedure is optimized for MSE, not guaranteed overcoverage;
- it has not been justified here for \(p>N\) flow-time vectors.

It should therefore be a named policy and a systematic variation, not an
unrecorded default.

## 4. Can unresolved autocorrelation be made conservative?

### 4.1 No general finite-sample matrix guarantee

None of the primary sources reviewed here proves a distribution-free,
finite-\(N\) guarantee

\[
\widehat\Sigma_N\succeq\Sigma

\]

for an arbitrary stationary correlated process. This is stronger than PSD and
stronger than positive bias of individual entries. An observed finite prefix
cannot, without restrictions on the tail, rule out a much slower mode that
would enlarge the infinite covariance sum.

A scalar counterexample makes the unresolved-tail problem exact. Let

\[
Y_t=\epsilon_t+\theta\epsilon_{t-L},
\qquad \theta>0,
\]

where the \(\epsilon_t\) are IID with variance \(s^2\). Then

\[
\Gamma(0)=(1+\theta^2)s^2,
\qquad
\Gamma(\pm L)=\theta s^2,
\qquad
\Sigma=(1+\theta)^2s^2.
\]

For a known mean and any Bartlett bandwidth that ends before lag \(L\), all
retained nonzero-lag population covariances vanish. Its expectation therefore
targets \((1+\theta^2)s^2\), below the truth by exactly \(2\theta s^2\). No
choice rule that certifies a window before the unresolved lag can have a
general expected-overestimation guarantee.

There is also a self-contained impossibility argument stronger than this
fixed-window example. Fix a history length \(N\), choose an integer \(M\geq N\),
and consider zero-mean stationary Gaussian processes with spectral density

\[
f_K(\lambda)=
\frac{\exp[K\cos(M\lambda)]}{2\pi I_0(K)},
\qquad K>0,
\]

where \(I_j\) is a modified Bessel function. Their autocovariances satisfy
\(\gamma_0=1\), \(\gamma_h=0\) for \(0<|h|<M\), and
\(\gamma_{jM}=I_j(K)/I_0(K)\). Consequently every observed length-\(N\) vector
has exactly the same \(\mathcal N(0,I_N)\) distribution for every \(K\), while

\[
\Sigma_K=2\pi f_K(0)=\frac{e^K}{I_0(K)}
\sim\sqrt{2\pi K}
\]

is unbounded. Thus the finite data distribution cannot distinguish members of
this family, and no finite-mean statistic of those data can be conservative in
expectation uniformly over the family. This is a derivation included here,
not a theorem attributed to one of the cited papers; it explains why a tail or
mixing assumption is logically necessary.

The practical consequence is important: when the data do not resolve the
autocorrelation scale, multiplying the observed covariance by a plausible factor
does not turn ignorance into a proven upper bound. The toolkit should expose
that as `UnresolvedAutocorrelation` (or equivalent typed evidence/failure).

### 4.2 Lugsail estimators: useful bias direction, not a bound

Vats and Flegal define the lugsail transformation of a lag window

\[
k_L(x)=\frac{1}{1-c}k(x)-\frac{c}{1-c}k(rx),
\qquad r\geq1,\quad0\leq c<1.
\tag{3}
\]

It deliberately permits weights above one. Their Theorem 1 establishes that
the lugsail spectral estimator inherits consistency. Corollary 1 shows that,
for a kernel of order \(q\), choosing \(c>r^{-q}\) reverses the sign of the
usual leading bias term under the paper's positive-persistence setting. For
high-to-extreme positive correlation they recommend the “over lugsail” choice
\(r=3\) and

\[
c=\frac{2}{1+r^q};
\]

for Bartlett, \(q=1\), hence \(c=1/2\)
([Vats and Flegal 2022, Eqs. (2), Corollary 1, and Table 1](https://arxiv.org/pdf/1809.04541),
pp. 4--6 and 10--11).

This is the closest sourced method to the requested preference, but its scope
must be stated exactly:

- it controls the **first-order asymptotic bias direction**, not the realized
  finite-sample error;
- it is motivated by positive persistence and can be counterproductive for
  anti-persistent processes (paper pp. 1--2);
- positive diagonal bias is not a Loewner-order matrix upper bound;
- the lugsail estimator is a difference of two covariance estimates and can
  have negative eigenvalues. The authors explicitly say lugsail estimates are
  not guaranteed PSD and propose a separate eigenvalue adjustment (paper
  p. 11).

Consequently an over-lugsail Bartlett estimate is well justified as a named
systematic variation and warning diagnostic. It should not silently replace
the PSD Bartlett primary estimate. If projected to PSD, that projection is
again an explicit covariance policy with recorded adjustment, not a theorem
that the result is conservative.

### 4.3 Scalar initial-sequence estimators

For a **stationary reversible** Markov chain and one scalar functional, Geyer
pairs adjacent autocovariances,

\[
G_i=\gamma(2i)+\gamma(2i+1).
\]

The population sequence is positive, decreasing, and convex. The initial
positive sequence estimator truncates before the first nonpositive empirical
pair; the initial monotone and initial convex variants enforce more of the
population shape. Geyer's Theorem 3.2 proves the asymptotic statement

\[
\liminf_{N\to\infty}\widehat\sigma^2_{\mathrm{IS},N}
\geq \sigma^2
\qquad\text{almost surely}.
\tag{4}
\]

This is “consistent overestimation” in Geyer's terminology, but it is not a
finite-sample guarantee. It also depends essentially on reversibility; generic
stationarity is insufficient
([Geyer 1992, Theorems 3.1--3.2, pp. 477--478](https://www2.stat.duke.edu/homeweb/scs/Courses/Stat376/Papers/GeyerStatSci1992.pdf)).

This is valuable for selected projected quantities if the sampler's detailed
balance and history ordering are part of provenance. It cannot be assumed from
the numerical histories alone.

### 4.4 Multivariate initial sequence

Dai and Jones extend the reversible-chain construction. With symmetric
empirical lag covariances they form

\[
G_{N,i}=\widetilde\Gamma_N(2i)+
        \widetilde\Gamma_N(2i+1),
\qquad
S_{N,m}=-\widehat\Gamma_N(0)+2\sum_{i=0}^{m}G_{N,i}.
\]

Their multivariate initial-sequence estimator starts at the first positive-
definite \(S_{N,m}\) and stops at the first local maximum of its determinant.
Theorem 2 proves

\[
\liminf_{N\to\infty}
\det(\widehat\Sigma_{\mathrm{mIS},N})
\geq\det(\Sigma)
\qquad\text{almost surely}.
\tag{5}
\]

Their adjusted variant replaces negative eigenvalues of each added paired-lag
matrix by zero; Theorem 3 gives the same asymptotic determinant result
([Dai and Jones 2017, Eqs. (4)--(6), Theorems 1--3](https://arxiv.org/pdf/1706.00853),
pp. 2--4).

Equation (5) concerns **generalized variance** (confidence-ellipsoid volume),
not every diagonal and not Loewner domination. The method assumes detailed
balance/reversibility, a multivariate CLT, finite second moments, and fixed
dimension. It also requires positive-definite empirical partial sums.

That last requirement makes it inapplicable to the current full flow-time
vector when \(p>N\): the sample-based covariance has rank at most \(N-1\), so
no such \(p\times p\) matrix can be positive definite and its determinant is
zero. Multivariate initial sequence is a candidate only for a small,
predeclared vector of derived quantities, not for thousands of joint
channel/flow-time coordinates.

## 5. Multivariate batch means as a validator

Let \(N=ab\), where \(a\) is the number of nonoverlapping batches and \(b\)
their size. With batch means

\[
\bar Y_j(b)=\frac1b\sum_{t=1}^{b}Y_{jb+t},
\]

the multivariate batch-means estimator is

\[
\widehat\Sigma_{\mathrm{BM}}(b)
=\frac{b}{a-1}\sum_{j=0}^{a-1}
  (\bar Y_j(b)-\bar Y)(\bar Y_j(b)-\bar Y)^T.
\tag{6}
\]

This is PSD by construction. Vats, Flegal, and Jones give Eq. (6) as their Eq.
(10) and prove strong consistency in Theorem 2 under their strong-invariance,
moment, and polynomial-ergodicity conditions, with both

\[
b\to\infty,
\qquad a=N/b\to\infty.
\]

They explicitly note that it is singular unless \(a>p\)
([Vats, Flegal, and Jones 2019, Eq. (10), Condition 2, Theorem 2](https://arxiv.org/pdf/1512.07713),
pp. 14--16).

For the current \(p>N\) case, \(a\leq N<p\), so a full mBM estimate is
necessarily singular. More generally,

\[
\operatorname{rank}(\widehat\Sigma_{\mathrm{BM}})\leq a-1,
\]

which can be much lower than the rank of Bartlett/Newey--West. This is not a
reason to reject mBM as a validator of diagonals or selected linear
projections; it is a reason never to advertise it as invertible.

Liu, Vats, and Flegal show that Bartlett overlapping batch means is
asymptotically equivalent to the Bartlett spectral estimator. Nonoverlapping
mBM remains a useful independent implementation path, but agreement is
cross-implementation evidence within the same long-run-covariance theory, not
an independent physical oracle
([Liu, Vats, and Flegal 2022, pp. 5--7](https://arxiv.org/pdf/1804.05975)).

Batch-size implications:

- \(N^{1/3}\) is the MSE-optimal *rate* for the Bartlett estimators studied,
  with a process-dependent coefficient;
- larger batches reduce truncation bias but leave fewer batches and increase
  variability;
- both batch count and batch size must grow for the consistency theorem;
- a batch-size sweep is evidence, not optional plotting decoration.

Standard batch means itself commonly has negative finite-sample bias under
positive persistence. Lugsail batch means has a positive first-order-bias
option, but, like lugsail spectral estimation, is not a finite-sample upper
bound and can require an explicit PSD adjustment
([Vats and Flegal 2022, Eqs. (7), Theorems 3--4](https://arxiv.org/pdf/1809.04541)).

## 6. Faithful projected Wolff Gamma method

Wolff's practical Gamma method is most directly justified for a declared
scalar derived quantity

\[
F=f(A_1,\ldots,A_p).
\]

Let \(f_\alpha=\partial f/\partial A_\alpha\), evaluated at the observed means.
The delta-method projected lag covariance is

\[
\widehat\Gamma_F(t)
=\sum_{\alpha,\beta}f_\alpha f_\beta
 \widehat\Gamma_{\alpha\beta}(t).
\tag{7}
\]

Equivalently, form the projected centered history
\(z_t=\sum_\alpha f_\alpha(Y_{t\alpha}-\bar Y_\alpha)\) and calculate its scalar
autocovariance. These two calculations must agree numerically.

A faithful one-replica implementation uses the lag-dependent denominator in
Wolff Eq. (31),

\[
\widehat\Gamma_F(t)
=\frac1{N-t}\sum_{i=1}^{N-t}z_i z_{i+t},
\tag{8}
\]

then the rectangular scalar sum in Eq. (35),

\[
\widehat C_F(W)=
\widehat\Gamma_F(0)+2\sum_{t=1}^{W}\widehat\Gamma_F(t),
\qquad
\widehat\tau_{\mathrm{int},F}(W)
=\frac{\widehat C_F(W)}{2\widehat\Gamma_F(0)}.
\tag{9}
\]

Because subtracting the sample mean biases every lag estimate by
approximately \(-C_F/N\), Wolff applies Eq. (49):

\[
\widehat C_F(W)\leftarrow
\widehat C_F(W)\left(1+\frac{2W+1}{N}\right).
\tag{10}
\]

The variance of the reported mean is then \(\widehat C_F(W)/N\).

Wolff's automatic window is not \(W=\lceil S\tau_{\rm int}\rceil\). For each
candidate \(W\), Eqs. (50)--(51) solve for an effective exponential decay
time \(\bar\tau(W)\):

\[
\frac{S}{\bar\tau(W)}
=\log\!\left(
\frac{2\widehat\tau_{\mathrm{int},F}(W)+1}
     {2\widehat\tau_{\mathrm{int},F}(W)-1}
\right).
\tag{11}
\]

If \(\widehat\tau_{\mathrm{int},F}\leq1/2\), Wolff substitutes a tiny positive
\(\bar\tau\). Equation (52) defines

\[
g(W)=e^{-W/\bar\tau(W)}-
\frac{\bar\tau(W)}{\sqrt{WN}}.
\tag{12}
\]

The first sign change to \(g(W)<0\) selects the window. Wolff describes
\(S\in[1,2]\) as reasonable and explicitly requires visual confirmation that
the estimated integrated autocorrelation time has a plateau near the selected
window. Equations (31), (33), (35), (41), and (49)--(52), including that
plateau requirement, are the implementation authority
([Wolff 2004, pp. 6--12](https://arxiv.org/pdf/hep-lat/0306017)).

This method should be used as a validator on selected scientific projections.
It should not be relabelled as a full-matrix PSD method: its rectangular sum is
scalar and its window is specific to \(F\). Different scalar windows cannot be
assembled entry by entry into a coherent covariance matrix.

## 7. High-dimensional limitations in the target use case

The current flow-time vector has more coordinates than configurations in at
least some ensembles. This changes representation and downstream fitting, not
the definition of \(\Sigma\).

Any estimator expressible as

\[
\widehat\Sigma=Z^T K Z,
\qquad Z\in\mathbb R^{N\times p},
\]

has rank at most \(N-1\) after centering. Therefore:

- Bartlett/Newey--West is PSD but singular when \(p\geq N\);
- mBM has the stronger rank bound \(a-1\);
- multivariate initial sequence cannot find a positive-definite partial sum
  when \(p>N\);
- no PSD projection can manufacture empirical information in the null space;
- inverse-covariance fits need declared dimension reduction, flow-time
  selection, a low-rank formulation, or separately sourced regularization.

The cited strong-consistency theorems hold for fixed \(p\). They do not justify
treating a singular \(p>N\) estimate as if it came with large-dimensional
operator-norm guarantees. Covariance storage should preferably preserve a
factor/low-rank representation and expose rank as evidence.

## 8. Recommended implementation contract

### Primary estimator

Implement one joint `BartlettLongRunCovariance` scientific value with:

1. all aligned channels and flow times concatenated before estimation;
2. Eq. (1), denominator \(N\), and no after-the-fact clipping in the ordinary
   path;
3. covariance-of-mean returned explicitly as
   \(\widehat\Sigma_{\mathrm B}/N\);
4. a required, named bandwidth policy;
5. immutable evidence containing history length, dimension, rank, weights,
   selected lag, policy identity, and projection diagnostics;
6. exact preservation of independent-replica boundaries if replicas exist.

For the first vertical slice, an explicitly supplied bandwidth is more honest
than an automatic default whose model has not yet been independently validated. An
AR-pilot policy from Liu--Vats--Flegal can be added as a named alternative.

### Resolution policy

Predeclare a small set of scientifically relevant projections \(u_j\), such as
individual retained flow times, channel combinations used downstream, and
derivative stencils. Record

\[
v_j(m)=u_j^T\widehat\Sigma_{\mathrm B}(m)u_j
\]

over a bandwidth grid that includes values below and above the selected one.
Also compute nonoverlapping batch-means and faithful projected Wolff estimates
where their assumptions apply.

The exact numerical plateau tolerance and cap are toolkit policies that need
calibration on synthetic processes; they are not supplied by the papers. The
essential semantics should be:

- stable before the maximum informative bandwidth: return an estimate and all
  sensitivity evidence;
- selected at the cap, no plateau, validator disagreement beyond calibrated
  tolerance, or estimated decay comparable to the run: return/record
  `UnresolvedAutocorrelation` rather than label an inflation as conservative.

This is the only generally defensible way to ensure that an unresolved long
mode does not silently produce an underestimated reported covariance.

### Conservative variations

Retain, but do not silently substitute:

- enlarged-bandwidth Bartlett estimates;
- over-lugsail Bartlett (\(r=3,c=1/2\)) for positive-persistence scenarios,
  with any negative eigenvalues and PSD adjustment recorded;
- scalar Geyer initial-monotone/convex estimates when reversibility is part of
  provenance;
- low-dimensional Dai--Jones mIS/mISadj estimates when \(p<N\).

No result should call these a “guaranteed upper covariance.” The evidence must
say whether the claim is PSD, consistency, positive leading bias, or
asymptotic generalized-variance overestimation.

## 9. Independent validation requirements

Before promotion into an acceptance workflow, test through the public processing
interface:

1. **Vector IID normal:** recover \(\Sigma/N\) within sampling tolerance; no
   mandatory nonzero correlation window; PSD and expected rank.
2. **Vector VAR(1):** use the analytic
   \[
   \Sigma=(I-\Phi)^{-1}V+V(I-\Phi^T)^{-1}-V
   \]
   from Vats, Flegal, and Jones Eq. (3.2), covering positive, negative, and
   mixed cross-correlation
   ([paper pp. 12--14](https://arxiv.org/pdf/1507.08266)).
3. **Hidden slow mode:** combine fast coordinates so one linear combination
   has a much slower decay; verify marginal pilot rules cannot falsely certify
   resolution and the public result diagnoses the issue.
4. **Two-scale scalar history:** test Wolff's window, correction, and plateau
   against a known autocorrelation sum.
5. **Linear equivariance:** estimate \(AY\) directly and compare with
   \(A\widehat\Sigma A^T\) under the same policy.
6. **Joint channels:** construct known cross-channel covariance and verify it
   survives combinations and estimation.
7. **Bandwidth sensitivity:** verify evidence ordering and that a cap/no-
   plateau case cannot report `resolved`.
8. **Coverage simulation:** measure one-dimensional and joint confidence
   coverage for base Bartlett, enlarged bandwidth, over-lugsail, batch means,
   and projected Wolff. Do not choose tolerances merely to preserve legacy
   output.
9. **High dimension:** test \(p>N\), reported rank, singular representation,
   and rejection of an undeclared inverse.
10. **Reversibility restriction:** initial-sequence policies must reject when
    sampler reversibility is absent from provenance.

## 10. Source map and strength of claims

| Source | Exact implementation authority | What it does not prove |
|---|---|---|
| [Newey & West (1987)](https://users.ssc.wisc.edu/~behansen/718/NeweyWest1987.pdf), Eq. (5), Thms. 1--2 | Bartlett weights; PSD; consistency under stated conditions | Finite-sample overestimation |
| [Vats, Flegal & Jones (2018)](https://arxiv.org/pdf/1507.08266), Eq. (2.2), Conditions 1--4, Thms. 1--3, Remark 7 | Full multivariate lag-window estimator; fixed-\(p\) strong consistency; Bartlett qualification; rectangular-window limitation | \(p>N\) accuracy; finite-sample upper bound |
| [Andrews (1991)](https://personal.utdallas.edu/~d.sul/Econo2/andrews91.pdf), Sections 4--6 | Asymptotic MSE and data-dependent plug-in bandwidths | One-sided conservative uncertainty |
| [Liu, Vats & Flegal (2022)](https://arxiv.org/pdf/1804.05975), Eqs. (2)--(3), Sections 2--3 | \(N^{1/3}\) rate with process coefficient; AR-pilot batch-size method | Universal coefficient; guaranteed slow-mode detection |
| [Vats & Flegal (2022)](https://arxiv.org/pdf/1809.04541), Eq. (2), Thms. 1--4, Cor. 1, Table 1 | Lugsail consistency and leading-bias control; over-lugsail parameters | PSD for lugsail; finite-sample or Loewner upper bound |
| [Geyer (1992)](https://www2.stat.duke.edu/homeweb/scs/Courses/Stat376/Papers/GeyerStatSci1992.pdf), Thms. 3.1--3.2 | Reversible scalar initial-sequence structure and asymptotic liminf overestimate | Nonreversible chains; finite-\(N\) guarantee; full high-dimensional matrix |
| [Dai & Jones (2017)](https://arxiv.org/pdf/1706.00853), Eqs. (4)--(6), Thms. 1--3 | Reversible multivariate initial sequence; asymptotic determinant overestimate | Loewner/diagonal upper bound; \(p>N\) applicability |
| [Vats, Flegal & Jones (2019)](https://arxiv.org/pdf/1512.07713), Eq. (10), Condition 2, Thm. 2 | Multivariate batch-means formula and strong consistency; singularity when batches are too few | Finite-sample conservatism |
| [Wolff (2004)](https://arxiv.org/pdf/hep-lat/0306017), Eqs. (31), (33), (35), (41), (49)--(52) | Projected scalar Gamma method, centering-bias correction, and automatic window | A PSD full-matrix estimator or reliable inference when \(N\) is not much larger than the decay scale |

## Conclusion

The appropriate primary replacement is joint Bartlett/Newey--West because its
target, matrix construction, PSD property, and asymptotic conditions are
clear. Multivariate batch means and faithful projected Wolff calculations
provide complementary checks. Over-lugsail and reversible initial-sequence
methods are justified conservative *variations* in their narrower senses.

The user's desired asymmetry should be implemented chiefly as a failure
policy: if the tail cannot be resolved, do not issue a plausible-looking small
covariance. Positive-leading-bias methods can reduce the chance of
underestimation when their assumptions apply, but the literature does not
support presenting them as finite-sample covariance upper bounds.
