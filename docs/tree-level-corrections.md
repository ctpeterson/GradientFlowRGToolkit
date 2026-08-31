---
title: Finite-Volume and Tree-Level Normalization Corrections
subtitle: A sourced implementation guide to fvn.py and tln.py
author: GradientFlowRGToolkit
date: 31 August 2026
geometry: margin=1in
fontsize: 11pt
colorlinks: true
---

# Implemented contract

Both correction modules return a dimensionless quantity

\[
\Delta(t)=C(t)-1,
\]

and processing defines the corrected gradient-flow coupling by dividing its
usual normalization by \(1+\Delta=C\). Flow time is represented in lattice
units \(t/a^2\), and lattice extents are counts of sites.

The two methods remove different effects:

| Method | Quantity | Discretization dependence | Publication source |
|---|---|---|---|
| FVN | continuum finite-volume factor | none | [Fodor et al. (2012)](https://doi.org/10.1007/JHEP11(2012)007) |
| TLN | finite-volume, finite-spacing tree-level factor | flow, gauge action, and energy operator | [Fodor et al. (2014)](https://doi.org/10.1007/JHEP09(2014)018) |

Each processed channel records the method, DOI, volume, every applicable
discretization, validity domain, numerical tolerance, and implementation notes
in `CorrectionEvidence`. The processing report prints these method details.

# Continuum finite-volume normalization (`fvn.py`)

## Paper formula

For a periodic hypercubic four-torus of extent \(L\), published
Eqs. (1.2)--(1.3) of
[The Yang--Mills gradient flow in finite volume](https://arxiv.org/abs/1208.1051)
give

\[
\left\langle t^2 E(t)\right\rangle
=g_R^2\frac{3(N^2-1)}{128\pi^2}\,[1+\delta(t,L)],
\]

with algebraic zero-mode and exponential nonzero-mode terms

\[
\delta_a=-\frac{64\pi^2t^2}{3L^4},
\qquad
\delta_e=\vartheta_3\!\left(e^{-L^2/(8t)}\right)^4-1.
\tag{1}
\]

The exact Jacobi series is

\[
\vartheta_3(q)=1+2\sum_{n=1}^{\infty}q^{n^2}.
\]

## Complete rectangular-torus evaluation

For rectangular extents \(L_\mu\), `fvn.py` uses the product extension

\[
\Delta_a=-\frac{64\pi^2t^2}{3\prod_\mu L_\mu},
\]

and the complete one-dimensional theta functions:

\[
\Delta_e=
\prod_{\mu=1}^{4}
\left[
1+2\sum_{n=1}^{\infty}e^{-n^2L_\mu^2/(8t)}
\right]-1.
\tag{2}
\]

The paper displays a hypercube; the rectangular product is a declared toolkit
extension. It is not inferred from the changing analysis dataset.

For \(x=L_\mu^2/(8t)\geq\pi\), `_theta3_exp_minus` sums

\[
1+2\sum_{n=1}^{\infty}e^{-xn^2}
\]

directly. For \(x<\pi\), it uses the Jacobi modular transformation

\[
\vartheta_3(e^{-x})=
\sqrt{\frac{\pi}{x}}\,
\vartheta_3(e^{-\pi^2/x}),
\]

so the evaluated series always decays rapidly. Terms stop only after every
requested value satisfies the declared relative tolerance (default
`1e-15`); a hard iteration guard raises a typed error instead of silently
returning a truncation.

`finite_volume_correction` validates a positive finite one-dimensional time
grid and the exact canonical geometry syntax `lN lN lN tN` without spaces,
evaluates (2) vectorially, and returns both the correction and its source
evidence. It is independent of lattice discretization because it is the
continuum finite-volume factor.

# Finite-lattice tree-level normalization (`tln.py`)

## Paper formula and code convention

Section 6 and published Eq. (6.2) of
[The lattice gradient flow at tree-level and its improvement](https://arxiv.org/html/1406.0827v2)
combine finite volume with finite lattice spacing. In lattice units \(a=1\),
the code evaluates

\[
C(t,L)=
\frac{128\pi^2t^2}{3V}
+\frac{64\pi^2t^2}{3V}
\sum_{p\ne0}
\operatorname{Tr}\!\left[
e^{-t(S_f+G)}(S_g+G)^{-1}
e^{-t(S_f+G)}S_e
\right],
\tag{3}
\]

where \(V=N_s^3N_t\). The first term is the zero-mode contribution. The sum
contains every nonzero periodic lattice momentum
\(p_\mu=2\pi n_\mu/N_\mu\). The module returns \(C-1\).

The paper permits separate flow, gauge-action, and energy-operator
discretizations. This implementation fixes Wilson flow, \(c_f=0\), while
supporting Wilson or tree-level Symanzik gauge action and plaquette,
tree-level Symanzik, or clover energy density.

## Momentum and kernel construction

For every momentum, `tln.py` forms

\[
\hat p_\mu=2\sin(p_\mu/2),
\qquad
\widetilde p_\mu=\sin(p_\mu).
\]

For a Symanzik-family coefficient \(c\), `_action_kernel` implements published
Eq. (3.5):

\[
S_{\mu\nu}(c)=
\delta_{\mu\nu}
\left[
\hat p^2-c\sum_\rho\hat p_\rho^4
-c\hat p_\mu^2\hat p^2
\right]
-\hat p_\mu\hat p_\nu
\left[1-c(\hat p_\mu^2+\hat p_\nu^2)\right].
\tag{4}
\]

The coefficient is zero for Wilson and \(-1/12\) for tree-level Symanzik.
For the clover observable, `_clover_kernel` implements published Eq. (3.6):

\[
K_{\mu\nu}=
(\delta_{\mu\nu}\widetilde p^2-
\widetilde p_\mu\widetilde p_\nu)
\cos(p_\mu/2)\cos(p_\nu/2).
\tag{5}
\]

The private value `999.0` is only a dispatch sentinel selecting (5); it is not
a physical improvement coefficient. Gauge fixing uses \(\alpha=1\),
\(G_{\mu\nu}=\hat p_\mu\hat p_\nu\). The final expression is gauge-parameter
independent, while adding \(G\) makes the matrices invertible away from the
removed zero mode.

## Exact momentum-orbit reduction

`_momentum_orbits` exploits symmetries of an isotropic spatial lattice:

1. reflection sends each component to the half interval
   \(0\le n_\mu\le N_\mu/2\);
2. permutations of the three equal spatial axes permit representatives with
   \(n_x\le n_y\le n_z\);
3. reflection and permutation counts become one multiplicity per
   representative.

The first representative is the zero mode and is removed. A runtime invariant
requires the remaining multiplicities to sum to \(N_s^3N_t-1\). The reduction
therefore changes computational cost, not Eq. (3).

## Spectral reduction

For each momentum, `_spectral_terms` diagonalizes the real symmetric flow
kernel once:

\[
S_f+G=V\,\operatorname{diag}(\lambda_i)V^{\mathsf T}.
\]

It then forms

\[
A=V^{\mathsf T}(S_g+G)^{-1}V,
\qquad
B=V^{\mathsf T}S_eV.
\]

The trace in Eq. (3) becomes

\[
\sum_{ij} A_{ij}B_{ji}
e^{-t(\lambda_i+\lambda_j)}.
\tag{6}
\]

Thus every orbit contributes coefficient/exponent pairs
\((wA_{ij}B_{ji},\lambda_i+\lambda_j)\). Processing in blocks limits temporary
memory. `_collapse_terms` stably sorts exponents and sums coefficients whose
exponents agree within `1e-11`; this substantially reduces repeated work for
Wilson flow.

## Direct requested-time evaluation and deterministic reduction

The collapsed coefficient/exponent arrays are cached by geometry,
discretizations, and the declared exponent-collapse tolerance. For every
requested time, `_evaluate_spectrum` computes Eq. (6) directly:

```python
for time_index in prange(flow_times.shape[0]):
    total = 0.0
    for term_index in range(coefficients.shape[0]):
        total += coefficients[term_index] * np.exp(
            -flow_times[time_index] * exponents[term_index]
        )
    values[time_index] = total
```

Parallelism is only across times. Each time uses the same sequential term
order, and fast math is disabled. Results are therefore independent of the
Numba thread count. There is no interpolation grid, spline, or recurrence,
so an arbitrary requested time is evaluated as the published momentum sum.

Before calculation, the public method enforces the exact domain

\[
0 < t\leq N_s^2/32,
\qquad \sqrt{8t}/N_s\leq1/2.
\]

There is no rounded-grid boundary through which an out-of-domain time can
leak. A nonpositive, nonfinite, empty, or out-of-domain request raises a typed
error before the expensive spectrum calculation.

# Why FVN and TLN converge

The continuum comparison must hold the finite-volume scheme parameter

\[
c=\sqrt{8t}/L
\]

fixed while increasing \(L/a\). Then

\[
\frac{t}{a^2}=\frac{c^2}{8}\left(\frac{L}{a}\right)^2\longrightarrow\infty,
\qquad
\frac{a^2}{t}\longrightarrow0.
\]

Under this limit, published Eq. (6.1) of the 2014 paper states that the finite-lattice
factor approaches the continuum finite-volume factor of the 2012 paper.
Increasing \(t\) on one fixed lattice is not this test because it changes
\(c\) at the same time.

The convergence test is analytic/synthetic and does not derive a tolerance
from an external analysis dataset. Independent tiny-lattice tests also
enumerate the full momentum grid and evaluate the matrix-exponential formula
directly for all six supported gauge-action/energy-operator combinations.

# Applicability and limitations

- Gauge fields are periodic in all four directions.
- TLN requires three equal spatial extents; its temporal extent may differ.
- TLN currently supports Wilson flow only.
- TLN's gauge action and energy-density operator must match how configurations
  and measurements were produced.
- TLN is a tree-level correction. It does not remove higher-order lattice
  artifacts.
- FVN evaluates the complete theta function to a declared tolerance; its
  rectangular-product extension is recorded as toolkit policy.
- FVN and TLN are not interchangeable at coarse \(t/a^2\). Their convergence
  is a continuum-limit check, not an equality expected on every ensemble.

# Primary references

- Z. Fodor, K. Holland, J. Kuti, D. Nogradi, and C. H. Wong,
  [The Yang--Mills gradient flow in finite volume](https://arxiv.org/abs/1208.1051),
  JHEP 11 (2012) 007,
  [doi:10.1007/JHEP11(2012)007](https://doi.org/10.1007/JHEP11(2012)007).
- Z. Fodor, K. Holland, J. Kuti, S. Mondal, D. Nogradi, and C. H. Wong,
  [The lattice gradient flow at tree-level and its improvement](https://arxiv.org/html/1406.0827v2),
  JHEP 09 (2014) 018,
  [doi:10.1007/JHEP09(2014)018](https://doi.org/10.1007/JHEP09(2014)018).
