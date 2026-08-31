# Tree-level and finite-volume normalization corrections

This note records the scientific authority and implementation meaning of the
finite-volume normalization (FVN) and finite-volume tree-level normalization
(TLN) corrections in GradientFlowRGToolkit. Literature formulas are separated
from extensions and numerical choices inherited from
../ContinuousBetaFunction.

## Current implementation note

The historical implementation discussion below records what was inherited
and why it was questioned. The closeout implementation no longer truncates
FVN at \(|n|\leq2\), and TLN no longer uses a uniform grid, recurrence, or
cubic spline. FVN uses a tolerance-controlled direct/modular Jacobi series;
TLN evaluates the cached collapsed spectrum directly at requested times with
a fixed-order inner reduction. See
`tree-level-correction-implementation.md` for the current line-by-line
contract.

## Primary sources

1. Z. Fodor, K. Holland, J. Kuti, D. Nogradi, and C. H. Wong, “The
   Yang–Mills gradient flow in finite volume,” *JHEP* **11** (2012) 007,
   DOI [10.1007/JHEP11(2012)007](https://doi.org/10.1007/JHEP11(2012)007),
   [arXiv:1208.1051v2](https://arxiv.org/abs/1208.1051). This is the
   authority for the continuum finite-volume normalization.
2. Z. Fodor, K. Holland, J. Kuti, S. Mondal, D. Nogradi, and C. H. Wong,
   “The lattice gradient flow at tree-level and its improvement,” *JHEP*
   **09** (2014) 018,
   DOI [10.1007/JHEP09(2014)018](https://doi.org/10.1007/JHEP09(2014)018),
   [arXiv:1406.0827v2](https://arxiv.org/abs/1406.0827). This is the
   authority for the finite-spacing kernels and the combined finite-volume,
   finite-spacing normalization.

The legacy files
[processing/finite_volume.py](../../ContinuousBetaFunction/src/betafn/processing/finite_volume.py)
and
[processing/tln.py](../../ContinuousBetaFunction/src/betafn/processing/tln.py)
are implementation references only. They cannot override the papers.

## Shared convention and correction direction

Both papers use

\[
E(t)=-\frac12\operatorname{Tr}F_{\mu\nu}F_{\mu\nu}(t).
\]

At leading order in infinite volume,

\[
\left\langle t^2E(t)\right\rangle
=\frac{3(N^2-1)}{128\pi^2}g^2+O(g^4).
\]

Writing a finite-volume or finite-spacing normalization as \(C\) changes the
leading term to

\[
\left\langle t^2E(t)\right\rangle
=\frac{3(N^2-1)}{128\pi^2}g^2C.
\]

The corrected coupling is therefore obtained by **dividing** by \(C\):

\[
g_{\mathrm{corrected}}^2
=\frac{128\pi^2}{3(N^2-1)}
\frac{t^2\langle E(t)\rangle}{C}.
\]

The toolkit delta-factor convention is

\[
\Delta=C-1,\qquad C=1+\Delta.
\]

This agrees with Eq. (4.1) of
[Fodor et al. (2012)](https://arxiv.org/html/1208.1051v2#S4.E32) and the
explicit division described in Sec. 7 of
[Fodor et al. (2014)](https://arxiv.org/html/1406.0827v2#S7).

## FVN: continuum finite-volume normalization

### Paper formula

On a periodic hypercubic four-torus of extent \(L\), Eqs. (1.2)–(1.3) of
[Fodor et al. (2012)](https://arxiv.org/html/1208.1051v2#S1.E2) give

\[
\left\langle t^2E(t)\right\rangle
=g_R^2(\mu)\frac{3(N^2-1)}{128\pi^2}(1+\delta),
\]

with

\[
\delta=\delta_a+\delta_e,\qquad
\delta_a=-\frac{64\pi^2t^2}{3L^4},
\]

\[
\delta_e
=\vartheta_3^4\!\left(\exp\!\left[-\frac{L^2}{8t}\right]\right)-1.
\]

For \(c=\sqrt{8t}/L\), the equivalent Eq. (3.9) form is

\[
C_{\mathrm{FVN}}(c)=1+\delta(c)
=\vartheta_3^4(e^{-1/c^2})-\frac{\pi^2}{3}c^4,
\]

where the paper defines

\[
\vartheta_3(q)=\sum_{n=-\infty}^{\infty}q^{n^2}.
\]

The algebraic term is the gauge-zero-mode contribution. The theta-function
term is the non-zero-mode momentum sum. Zero modes are treated exactly and
non-zero modes perturbatively. The paper reports
\(|\delta(c)|<10\%\) for \(0\le c\le1/2\). Direct evaluation at \(c=0.3\)
gives \(\delta\simeq-0.0265\).

This is a leading-order continuum result on a four-dimensional torus with
periodic gauge fields. The displayed formula is for a hypercube with one
extent \(L\). The paper allows massless fermions in arbitrary representations,
but fermions do not alter this displayed leading normalization.

### Behavior inherited from ContinuousBetaFunction

The legacy implementation generalizes the hypercubic formula to possibly
unequal extents \(L_\mu\):

\[
\delta_a^{\mathrm{legacy}}
=-\frac{64\pi^2t^2}{3\prod_{\mu=1}^4L_\mu},
\]

\[
\delta_e^{\mathrm{legacy}}
=\prod_{\mu=1}^4
\left(1+2e^{-L_\mu^2/(8t)}+2e^{-L_\mu^2/(2t)}\right)-1.
\]

The second formula truncates each one-dimensional theta sum after
\(n=\pm2\):

\[
\vartheta_3(e^{-L_\mu^2/(8t)})
\simeq1+2e^{-L_\mu^2/(8t)}+2e^{-4L_\mu^2/(8t)}.
\]

For an isotropic box it approaches the paper formula when the omitted
\(|n|\ge3\) terms are negligible. At \(c\le1/2\), the first omitted
single-direction term is at most \(2e^{-9/c^2}\le2e^{-36}\). The
truncation should nevertheless be named and tested rather than called the
exact theta function.

Neither canonical paper explicitly gives the unequal-extent product. It
follows naturally for the separable free non-zero-mode sum, and the legacy
zero-mode term replaces \(L^4\) by the four-volume, but this rectangular-torus
extension is a derived implementation choice. A paper-faithful contract
should initially restrict FVN to \(L^4\), or identify and independently test
the rectangular extension.

FVN has no flow-action, simulation-action, or energy-operator parameter:
those discretization distinctions disappear in the continuum tree-level
normalization. Its direct source DOI is 10.1007/JHEP11(2012)007.

## TLN: finite-volume, finite-spacing tree-level normalization

### Three independent discretizations

Fodor et al. (2014) distinguish:

1. the lattice action used by the gradient flow, with kernel
   \(\mathcal S^f\);
2. the dynamical gauge action used to generate the ensemble, with kernel
   \(\mathcal S^g\); and
3. the measured energy-density discretization, with kernel
   \(\mathcal S^e\).

Their abbreviations are ordered **Flow–Action–Observable**. W denotes the
Wilson coefficient \(c=0\), S denotes tree-level Symanzik
\(c=-1/12\), and C denotes the clover energy operator. Thus SSC means
Symanzik flow, Symanzik dynamical action, and clover observable. See the text
following Eqs. (3.5)–(3.7) in
[the 2014 paper](https://arxiv.org/html/1406.0827v2#S3.p5).

Equation (3.4) defines

\[
\widehat p_\mu=\frac2a\sin\left(\frac{ap_\mu}{2}\right),\qquad
\widetilde p_\mu=\frac1a\sin(ap_\mu).
\]

The action-like kernel in Eq. (3.5) is

\[
\begin{aligned}
S_{\mu\nu}(c)={}&
\delta_{\mu\nu}\left(
\widehat p^2-a^2c\sum_\rho\widehat p_\rho^4
-a^2c\widehat p_\mu^2\widehat p^2
\right)\\
&-\widehat p_\mu\widehat p_\nu
\left[1-a^2c(\widehat p_\mu^2+\widehat p_\nu^2)\right].
\end{aligned}
\]

The clover observable kernel in Eq. (3.6) is

\[
K_{\mu\nu}=
\left(\delta_{\mu\nu}\widetilde p^2
-\widetilde p_\mu\widetilde p_\nu\right)
\cos\left(\frac{ap_\mu}{2}\right)
\cos\left(\frac{ap_\nu}{2}\right).
\]

Equation (3.10) adds the gauge-fixing kernel

\[
\mathcal G_{\mu\nu}=\alpha^{-1}\widehat p_\mu\widehat p_\nu.
\]

The authors use \(\alpha=1\) for convenience and verify gauge-parameter
independence of the final expansion coefficients.

### Master finite-volume expression

Equation (6.1) defines

\[
C(a^2/t,\sqrt{8t}/L)
=1+\delta(\sqrt{8t}/L,a/L).
\]

For periodic gauge fields on the four-torus, Eq. (6.2) is

\[
\begin{aligned}
C(a^2/t,\sqrt{8t}/L)
={}&\frac{128\pi^2t^2}{3L^4}\\
&+\frac{64\pi^2t^2}{3L^4}
\sum_{p\ne0}\operatorname{Tr}\left[
e^{-t(\mathcal S^f+\mathcal G)}
(\mathcal S^g+\mathcal G)^{-1}
e^{-t(\mathcal S^f+\mathcal G)}
\mathcal S^e
\right],
\end{aligned}
\]

with \(p_\mu=2\pi n_\mu/L\) on the finite Fourier grid. The first term is
the zero mode and is identical at finite and zero lattice spacing. Only the
non-zero-mode sum has discretization effects. The paper also stresses that
the two exponentials may not be combined in general because the flow and
dynamical-action kernels need not commute. See Eq. (3.15) and the
[HTML form of Eq. (6.2)](https://arxiv.org/html/1406.0827v2#S6.E44).

This is tree-level improvement, not an all-orders correction. The paper says
it reduces cutoff effects without changing the continuum-extrapolated result;
one-loop finite-spacing terms are outside its scope. Its direct source DOI is
10.1007/JHEP09(2014)018. Its independent continuum finite-volume limit is
sourced by 10.1007/JHEP11(2012)007.

## What tln.py computes

The present [tln.py](../src/gfrgtoolkit/stages/tree_level/tln.py) numerically
evaluates Eq. (6.2), with the following specialization.

### Applicability contract

- Internal lattice units set \(a=1\), so the flow-time argument is
  \(t/a^2\).
- The flow kernel is hard-coded to Wilson flow, \(c_f=0\).
- The dynamical gauge kernel supports Wilson (w, \(c_g=0\)) and tree-level
  Symanzik (s, \(c_g=-1/12\)) actions.
- The energy kernel supports Wilson plaquette (p, \(c_e=0\)), action-like
  tree-level Symanzik (s, \(c_e=-1/12\)), and clover (c, the separate
  \(K_{\mu\nu}\) branch).
- The geometry requires three equal spatial extents \(N_s\), but allows a
  different temporal extent \(N_t\). This useful \(N_s^3N_t\) extension goes
  beyond the \(L^4\) geometry displayed in the canonical equations.
- Requested flow times must be finite and positive. The generated grid ends
  near \(t=N_s^2/32\), corresponding to
  \(\sqrt{8t}/N_s=1/2\).

The implementation is therefore W–(W or S)–(W, S, or C), not
flow-independent. A new non-Wilson flow requires its own
\(\mathcal S^f\) kernel and applicability case.

### Momentum-orbit reduction

The function _momentum_orbits enumerates nonnegative momentum
representatives. Reflections provide multiplicities in each direction, and
permutations of the three equal spatial axes provide the remaining spatial
multiplicity. The zero mode is removed. The implementation verifies that the
multiplicities sum to

\[
N_s^3N_t-1,
\]

the number of non-zero points on the full finite Fourier grid. This orbit
decomposition is an implementation optimization; the paper states the full
sum.

### Kernel construction

For each representative, tln.py constructs

\[
p_\mu=\frac{2\pi n_\mu}{N_\mu},\qquad
\widehat p_\mu=2\sin(p_\mu/2),\qquad
\widetilde p_\mu=\sin(p_\mu),
\]

then evaluates the paper’s kernels with \(a=1\). It adds the gauge-fixing
matrix \(\widehat p_\mu\widehat p_\nu\), corresponding to \(\alpha=1\), to
the flow and propagator kernels but not to the transverse energy kernel.

### Spectral reduction

Let

\[
F=\mathcal S^f+\mathcal G,\qquad
P=\mathcal S^g+\mathcal G,\qquad
O=\mathcal S^e.
\]

Because \(F\) is real symmetric, numpy.linalg.eigh gives

\[
F=V\operatorname{diag}(\lambda_i)V^T.
\]

For

\[
A=V^TP^{-1}V,\qquad B=V^TOV,
\]

the trace in Eq. (6.2) becomes

\[
\operatorname{Tr}(e^{-tF}P^{-1}e^{-tF}O)
=\sum_{ij}A_{ij}B_{ji}e^{-t(\lambda_i+\lambda_j)}.
\]

This identity lets _spectral_terms replace each momentum’s matrix expression
by scalar coefficient/exponent pairs. _collapse_terms sorts the exponents and
merges numerically equal values to avoid redundant work.

### Uniform grid and exponential recurrence

On a uniform grid \(t_k=k\epsilon\),

\[
e^{-t_k\mu}=(e^{-\epsilon\mu})^k.
\]

The function _sum_exponentials evaluates one exponential per spectral term
and uses multiplication for later grid points. Work is split among a number of
chunks derived from the available Numba threads. After adding partial sums,
_tree_level_grid applies

\[
f(t)=\frac{64\pi^2t^2}{3N_s^3N_t}
\]

and returns

\[
C_{\mathrm{TLN}}(t)
=2f(t)+f(t)\sum_{p\ne0}\operatorname{Tr}(\cdots).
\]

The public tree_level_delta operation fits a cubic spline to this grid and
returns \(C_{\mathrm{TLN}}-1\). Its LRU cache is keyed by geometry, dynamical
action coefficient, energy coefficient, and grid spacing. The flow
coefficient is absent only because Wilson flow is hard-coded.

The spectral decomposition, exponent-merging tolerance, grid recurrence,
Numba fast-math mode, thread-derived chunking, cubic interpolation, and cache
are numerical implementation choices rather than claims in the paper. They
need independent accuracy and determinism tests. A published formula alone
does not test spline or summation error.

## Correct relationship between FVN and TLN

Define

\[
x=\frac{a^2}{t},\qquad
c=\frac{\sqrt{8t}}{L},\qquad
c^2=\frac{8(t/a^2)}{(L/a)^2}.
\]

The present TLN is the finite-volume factor

\[
C_{\mathrm{TLN}}(x,c)=C(x,c)
\]

from Eq. (6.2), while FVN is its continuum finite-volume limit:

\[
C_{\mathrm{FVN}}(c)=C(0,c)=1+\delta(c).
\]

Therefore, for every supported discretization,

\[
\boxed{
\lim_{x\to0\ \mathrm{at\ fixed}\ c}C_{\mathrm{TLN}}(x,c)
=C_{\mathrm{FVN}}(c)
}.
\]

“Sufficiently large \(t/a^2\)” is meaningful here only together with the
quantity held fixed. At fixed \(c\), increasing \(t/a^2\) requires increasing
\(L/a\) as well. Increasing \(t/a^2\) on one fixed lattice changes \(c\) and
does not isolate cutoff effects.

Three limits provide distinct tests:

1. **Continuum at fixed finite-volume scheme:** hold \(c\) fixed and increase
   \(L/a\). Then \(t/a^2=(cL/a)^2/8\to\infty\), and TLN must approach FVN.
2. **Infinite volume at fixed lattice resolution:** hold \(a^2/t\) fixed and
   increase \(L/a\), so \(c\to0\). The finite-volume momentum sum must approach
   the 2014 paper’s infinite-volume lattice normalization \(C(a^2/t)\).
3. **Both limits:** only when \(a^2/t\to0\) and \(c\to0\) do both
   normalizations approach one.

An acceptance test should use case 1, preferably on an \(L^4\) box directly
covered by the papers, at fixed \(c\le1/2\). It should compare different
flow/action/operator triplets to the same FVN limit. Its tolerance must
separately cover the FVN theta truncation, TLN grid interpolation, floating
summation, and expected leading \(C_2a^2/t\) cutoff term. Agreement on one
lattice is regression evidence, not a convergence test.

## Source metadata for evidence and reports

Correction evidence should identify method, applicability, and source:

| Method | Source DOI | Equation authority |
|---|---|---|
| Continuum finite-volume normalization | 10.1007/JHEP11(2012)007 | Eqs. (1.2), (1.3), and (3.9) |
| Finite-volume, finite-spacing TLN | 10.1007/JHEP09(2014)018 | Eqs. (3.4)–(3.10), (3.15), and (6.2) |

For TLN, the 2012 DOI is also useful as a second source for the independent
continuum finite-volume limit. Reports should print the correction method and
source DOI, not leave the correction encoded only in an enum. They should
also print the actual Flow–Action–Observable triple, geometry, lattice-unit
convention, numerical grid spacing, and whether the rectangular-torus
extension was used; these choices change the finite-lattice correction even
when the DOI is unchanged.
