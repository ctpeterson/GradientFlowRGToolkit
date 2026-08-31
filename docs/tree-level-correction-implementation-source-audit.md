# Finite-volume and tree-level correction implementation source audit

**Audit date:** 31 August 2026

**Scope:** `src/gfrgtoolkit/stages/tree_level/fvn.py`,
`src/gfrgtoolkit/stages/tree_level/tln.py`, their shared evidence values, the
application of corrections in `stages/process.py`, and current tests.

**Authority policy:** the two Fodor et al. papers are the scientific
authorities. The legacy ContinuousBetaFunction files and frozen numerical
values are implementation references, not authorities.

## Closeout status

This file preserves the findings from the pre-closeout audit. The current
implementation has addressed its actionable numerical and contract gaps:

- FVN evaluates the complete Jacobi theta function using direct/modular
  series to a declared relative tolerance;
- TLN evaluates the collapsed spectrum directly at requested times, without a
  grid, recurrence, spline, fast math, or thread-dependent inner reduction;
- the exact \(\sqrt{8t}/N_s\leq1/2\) boundary, nonempty time grids, canonical
  geometry syntax, positive normalization, and a nonzero momentum domain are
  enforced with typed failures;
- independent tiny full-momentum matrix-exponential tests cover all six
  supported gauge-action/energy-operator pairs; and
- reports carry the DOI, domain, tolerance, discretizations, and numerical
  implementation notes.

The detailed sections below are retained as the audit trail. Descriptions of
truncation, interpolation, or missing tests refer to the audited pre-closeout
revision, not the current implementation.

## Executive conclusion

The central formulas are implemented correctly for the paper's periodic,
hypercubic setup:

- `fvn.py` reproduces the 2012 continuum finite-volume normalization after
  replacing the exact Jacobi theta function by the legacy series truncated at
  $|n|\le2$.
- `tln.py` implements the 2014 finite-volume, finite-spacing momentum sum,
  including the action and clover kernels, gauge fixing, two flow
  exponentials, the nonzero-mode sum, and the analytic zero-mode term.
- processing applies both in the correct direction by dividing the usual
  coupling normalization by $C=1+\Delta$.

The audit also found limitations that the DOI alone does not justify:

1. Both implementations extend the papers' displayed $L^4$ formulas to a
   rectangular four-torus. This is a natural separable/free-field extension,
   but it is not explicitly derived in either cited paper.
2. FVN has no applicability guard on $c=\sqrt{8t}/L$. The $|n|\le2$
   truncation is effectively exact for the intended $c\le1/2$ range, but can
   become materially wrong at much larger $c$.
3. TLN records neither its rectangular-torus extension nor its numerical
   exponent-collapse/fast-math policies in evidence.
4. The TLN grid is rounded *up*, so the public check admits a small interval
   beyond the documented $\sqrt{8t}/N_s=1/2$ boundary.
5. Cubic-spline interpolation has no accuracy test or lower flow-time guard.
   For $t$ much smaller than the interpolation spacing, relative error can be
   large even though the request is accepted.
6. The exponential sum is partitioned according to Numba's process-global
   thread count and compiled with `fastmath=True`. The thread count is neither
   a cache key nor evidence, so bitwise results can depend on hidden numerical
   state.
7. The current fixed-$c$ TLN-to-FVN convergence test is scientifically sound,
   but covers only Wilson flow, Wilson gauge action, and plaquette energy.
   The frozen TLN regression values do not identify an independent authority.
8. The implementation manual is correct on two easily confused attribution
   points: $N_s^3N_t$ is a derived rectangular-time extension of the paper's
   displayed $L^4$ result, and the fixed-$c$ continuum limit is FVN, not one.
   The factor tends to one only in the additional $c\to0$ limit.

## 1. Shared normalization convention

Both papers define

\[
E(t)=-\frac12\operatorname{Tr}F_{\mu\nu}F_{\mu\nu}(t).
\]

At leading order in infinite volume,

\[
\left\langle t^2E(t)\right\rangle
=g^2\frac{3(N_c^2-1)}{128\pi^2}+O(g^4).
\tag{1}
\]

At finite volume or finite lattice spacing, write the tree-level
normalization as $C(t)$:

\[
\left\langle t^2E(t)\right\rangle
=g^2\frac{3(N_c^2-1)}{128\pi^2}C(t)+O(g^4).
\tag{2}
\]

The coupling corrected at tree level is therefore

\[
g_{\rm GF}^2(t)
=\frac{128\pi^2}{3(N_c^2-1)}
\frac{t^2\langle E(t)\rangle}{C(t)}.
\tag{3}
\]

The toolkit stores

\[
\Delta(t)=C(t)-1,
\qquad C(t)=1+\Delta(t).
\tag{4}
\]

This agrees with Eq. (4.1) of Fodor et al. (2012) and the explicit
tree-level division in Section 7 of Fodor et al. (2014). The primary sources
are:

- Z. Fodor et al., [The Yang--Mills Gradient Flow in Finite
  Volume](https://arxiv.org/html/1208.1051v2), JHEP 11 (2012) 007,
  DOI [10.1007/JHEP11(2012)007](https://doi.org/10.1007/JHEP11(2012)007).
- Z. Fodor et al., [The Lattice Gradient Flow at Tree-Level and Its
  Improvement](https://arxiv.org/html/1406.0827v2), JHEP 09 (2014) 018,
  DOI [10.1007/JHEP09(2014)018](https://doi.org/10.1007/JHEP09(2014)018).

### Formula-to-code map in processing

`process.py` obtains either correction and evaluates:

```python
normalization = (
    128.0 * np.pi * np.pi / (3.0 * (nc * nc - 1.0))
    * selected_times
    * selected_times
    / (1.0 + correction.delta)
)
coupling = normalization * energy
```

This is exactly Eq. (3) with Eq. (4). There is no sign inversion: a
normalization factor $C>1$ reduces the corrected coupling, while $C<1$
increases it.

## 2. Continuum finite-volume normalization (`fvn.py`)

### 2.1 Published hypercubic formula

For a periodic four-torus with common extent $L$, Eqs. (1.2)--(1.3) of the
2012 paper give

\[
\Delta(t,L)=\Delta_a+\Delta_e,
\]

\[
\Delta_a=-\frac{64\pi^2t^2}{3L^4},
\qquad
\Delta_e=
\vartheta_3\!\left(e^{-L^2/(8t)}\right)^4-1.
\tag{5}
\]

The Jacobi function is

\[
\vartheta_3(q)=\sum_{n=-\infty}^{\infty}q^{n^2}
=1+2\sum_{n=1}^{\infty}q^{n^2}.
\tag{6}
\]

With the finite-volume scheme parameter

\[
c=\frac{\sqrt{8t}}{L},
\]

Eq. (3.9) of the paper is equivalently

\[
C_{\rm FVN}(c)
=\vartheta_3(e^{-1/c^2})^4-\frac{\pi^2}{3}c^4.
\tag{7}
\]

The algebraic part comes from the constant gauge mode, which the paper treats
exactly. The theta function represents the perturbative nonzero modes. The
paper displays the isotropic $L^4$ geometry and notes that the finite-volume
correction stays below ten percent for $0\le c\le1/2$.

### 2.2 Rectangular extension and theta truncation

The implementation accepts four possibly unequal extents $L_\mu$ and uses

\[
\Delta_a^{\rm impl}
=-\frac{64\pi^2t^2}{3\prod_{\mu=1}^4L_\mu},
\tag{8}
\]

\[
\Delta_e^{\rm impl}
=\prod_{\mu=1}^4
\left[
1+2e^{-L_\mu^2/(8t)}+2e^{-L_\mu^2/(2t)}
\right]-1.
\tag{9}
\]

Equation (9) is the direct product of Eq. (6) truncated after $n=2$ in each
direction:

\[
\vartheta_3(e^{-L_\mu^2/(8t)})
\approx
1+2e^{-L_\mu^2/(8t)}
+2e^{-4L_\mu^2/(8t)}.
\]

The code's algebraic loop is Eq. (8). It starts with $-64\pi^2/3$ and, for
each extent, divides by

\[
\sqrt{L_\mu^2/t}=L_\mu/\sqrt t.
\]

After four directions this supplies $t^2/\prod_\mu L_\mu$:

```python
algebraic = np.full_like(times, -64.0 * np.pi**2 / 3.0)
for extent in extents:
    ratio = extent * extent / times
    algebraic /= np.sqrt(ratio)
```

The same loop multiplies the four truncated one-dimensional theta factors:

```python
exponential *= (
    1.0
    + 2.0 * np.exp(-ratio / 8.0)
    + 2.0 * np.exp(-ratio / 2.0)
)
```

Finally, `algebraic + exponential - 1.0` returns Eqs. (8)--(9) in the
toolkit's $\Delta=C-1$ convention.

The rectangular product is mathematically natural for a separable free
momentum sum, and replacing $L^4$ by the four-volume is dimensionally
consistent. It is nonetheless an implementation extension: neither cited
paper explicitly derives the displayed rectangular formula. FVN evidence
appropriately records both
`"rectangular-torus product extension"` and
`"Jacobi theta series truncated at |n| <= 2"`.

### 2.3 Accuracy domain of the truncated theta series

For an isotropic box the first omitted one-dimensional contribution is

\[
2e^{-9/c^2}.
\]

At $c\le1/2$ this is at most $2e^{-36}$, so double precision cannot usually
distinguish the truncation from the full theta function. The implementation,
however, accepts every positive flow time. The approximation is not uniformly
accurate as $c$ grows. An audit calculation of Eq. (7) gave:

| $c$ | exact $C_{\rm FVN}$ | truncated $C_{\rm FVN}$ | relative difference |
|---:|---:|---:|---:|
| 0.50 | 0.9491588741 | 0.9491588741 | $2.1\times10^{-15}$ |
| 1.00 | 6.5838208002 | 6.5783177391 | $8.4\times10^{-4}$ |
| 2.00 | 105.2757803 | 65.0026642 | $3.8\times10^{-1}$ |

The values above are diagnostic calculations from the published theta
series, not a new authority. They show why the truncation should either have
an explicit applicability domain or an accuracy-based tail check. At present,
evidence names the truncation but does not prevent a low-quality correction.

### 2.4 Input and evidence behavior

`finite_volume_correction` requires a finite, positive, one-dimensional time
array and four positive extents extracted from the volume string. Its regular
expression is permissive rather than a full grammar: extra unrelated text can
surround four matching `l...`/`t...` components and still be accepted.

The result evidence correctly records:

- method `finite-volume-normalization`;
- DOI 10.1007/JHEP11(2012)007;
- the volume string;
- lattice-unit convention `t/a^2`;
- the rectangular extension and theta truncation.

Flow action, gauge action, and energy operator are `None`, which is appropriate
for this continuum tree-level normalization. They become relevant in TLN.

### 2.5 FVN audit verdict

- **Hypercubic formula:** correct relative to the 2012 source.
- **Correction direction:** correct.
- **Zero-mode factor:** correct for $L^4$; rectangular replacement is an
  implementation extension.
- **Theta term:** correct through $|n|=2$, not the paper's exact infinite
  series.
- **Evidence:** unusually good; it names both extensions.
- **Main gap:** no runtime domain or truncation-error control.

## 3. Finite-lattice tree-level normalization (`tln.py`)

### 3.1 Published master expression

The 2014 paper distinguishes three independent discretizations:

1. flow kernel $\mathcal S^f$;
2. dynamical gauge-action kernel $\mathcal S^g$;
3. energy-density kernel $\mathcal S^e$.

The paper's names are ordered Flow--Action--Observable. `W` denotes Wilson
coefficient zero, `S` denotes tree-level Symanzik coefficient $-1/12$, and
`C` denotes the separate clover energy kernel.

For periodic gauge fields, Eq. (6.2) gives

\[
\begin{aligned}
C(a^2/t,\sqrt{8t}/L)
={}&\frac{128\pi^2t^2}{3L^4}\\
&+\frac{64\pi^2t^2}{3L^4}
\sum_{p\ne0}\operatorname{Tr}\!\left[
e^{-t(\mathcal S^f+\mathcal G)}
(\mathcal S^g+\mathcal G)^{-1}
e^{-t(\mathcal S^f+\mathcal G)}
\mathcal S^e
\right].
\end{aligned}
\tag{10}
\]

The first term is the exact zero-mode contribution. The finite momentum sum
contains the lattice-spacing-dependent nonzero modes. The HTML preprint
renders this as sequential Eq. (44); the journal citation calls it Eq. (6.2).

The cited result is displayed with one common extent $L$, momenta
$p_\mu=2\pi n_\mu/L$, and four-volume $L^4$. It does not state the
$N_s^3N_t$ replacement. The implementation generalizes that formula to
$N_s^3N_t$ in lattice units $a=1$:

\[
C_{\rm impl}(t)
=\frac{128\pi^2t^2}{3N_s^3N_t}
+\frac{64\pi^2t^2}{3N_s^3N_t}\sum_{p\ne0}T(p,t).
\tag{11}
\]

It fixes Wilson flow, supports Wilson or Symanzik gauge action, and supports
plaquette, action-like Symanzik, or clover energy. Thus its supported triplets
are W--(W or S)--(W, S, or C). Rejecting non-Wilson flow is correct because
the numerical implementation hard-codes $c_f=0$.

### 3.2 Lattice momenta and action-like kernel

In lattice units, published Eq. (3.4) defines

\[
\widehat p_\mu=2\sin(p_\mu/2),
\qquad
\widetilde p_\mu=\sin(p_\mu),
\tag{12}
\]

with $p_\mu=2\pi n_\mu/N_\mu$. The code maps these literally:

```python
lattice_momentum = 2.0 * np.pi * indices / extents
half_momentum = 2.0 * np.sin(lattice_momentum / 2.0)
half_squared = half_momentum**2
half_norm = half_squared.sum(axis=1)
momentum = np.sin(lattice_momentum)
momentum_norm = (momentum**2).sum(axis=1)
cosine_half = np.cos(lattice_momentum / 2.0)
```

Published Eq. (3.5) is

\[
\begin{aligned}
S_{\mu\nu}(c)={}&
\delta_{\mu\nu}\left[
\widehat p^2
-c\sum_\rho\widehat p_\rho^4
-c\widehat p_\mu^2\widehat p^2
\right]\\
&-\widehat p_\mu\widehat p_\nu
\left[1-c(\widehat p_\mu^2+\widehat p_\nu^2)\right].
\end{aligned}
\tag{13}
\]

`_action_kernel` first builds the second line as a full outer-product matrix,
then adds the first line to the diagonal. The code is an exact vectorization
of Eq. (13). Coefficient zero selects Wilson; $-1/12$ selects tree-level
Symanzik.

### 3.3 Clover kernel and private dispatch sentinel

Published Eq. (3.6) is

\[
K_{\mu\nu}
=\left(\delta_{\mu\nu}\widetilde p^2
-\widetilde p_\mu\widetilde p_\nu\right)
\cos(p_\mu/2)\cos(p_\nu/2).
\tag{14}
\]

The implementation builds the parenthesized matrix and multiplies it by the
outer product of the cosine factors:

```python
matrix = -momentum[:, :, None] * momentum[:, None, :]
matrix[:, _DIRECTIONS, _DIRECTIONS] += momentum_norm[:, None]
return (
    matrix
    * cosine_half[:, :, None]
    * cosine_half[:, None, :]
)
```

The value `999.0` in `_ENERGY_KERNEL` is only a private branch sentinel. It is
not a physical Symanzik coefficient and must never appear as scientific
evidence.

### 3.4 Gauge fixing

Published Eq. (3.10) introduces

\[
\mathcal G_{\mu\nu}=\alpha^{-1}
\widehat p_\mu\widehat p_\nu.
\tag{15}
\]

The paper uses $\alpha=1$ for convenience and checks final gauge-parameter
independence. The code uses exactly that specialization:

```python
gauge_fixing = (
    half_momentum[:, :, None]
    * half_momentum[:, None, :]
)
flow_matrix = flow_kernel_matrix + gauge_fixing
gauge_matrix = gauge_kernel_matrix + gauge_fixing
```

Gauge fixing is not added to the transverse energy kernel. Removing the zero
mode before inversion makes `gauge_matrix` invertible for supported cases.

### 3.5 Momentum-orbit reduction

The paper states the full nonzero Fourier sum. `_momentum_orbits` reduces it
using exact symmetries of an isotropic spatial lattice:

- every component is reflected into $0\ldots\lfloor N_\mu/2\rfloor$;
- the three equal spatial components are sorted;
- factors 1, 3, or 6 restore spatial permutations;
- factors 1 or 2 restore reflected components, with zero and an even-lattice
  Nyquist component counted once.

The flattened first representative is the zero vector and is dropped. The
function checks

\[
\sum_{\rm orbits}w=N_s^3N_t-1.
\tag{16}
\]

This proves combinatorial closure of the orbit enumeration. It does not by
itself test the kernel values or numerical sum, but it ensures no momentum is
silently lost or duplicated under the assumed symmetries.

The reduction relies on the three spatial extents being equal. `_geometry`
rejects anisotropic spatial volumes but permits $N_t\ne N_s$. The latter is a
rectangular-torus extension beyond the displayed $L^4$ equation. Unlike FVN,
TLN does not record that extension in `implementation_notes`.

### 3.6 Spectral reduction of the two flow exponentials

For one nonzero momentum define

\[
F=\mathcal S^f+\mathcal G,
\qquad
P=\mathcal S^g+\mathcal G,
\qquad
O=\mathcal S^e.
\]

Since $F$ is real symmetric,

\[
F=V\operatorname{diag}(\lambda_i)V^{\mathsf T}.
\]

Let

\[
A=V^{\mathsf T}P^{-1}V,
\qquad
B=V^{\mathsf T}OV.
\]

Then the trace in Eq. (10) is

\[
\operatorname{Tr}(e^{-tF}P^{-1}e^{-tF}O)
=\sum_{i,j}A_{ij}B_{ji}
e^{-t(\lambda_i+\lambda_j)}.
\tag{17}
\]

The code calculates precisely these quantities:

```python
eigenvalues, eigenvectors = np.linalg.eigh(flow_matrix)
transpose = np.swapaxes(eigenvectors, 1, 2)
inverse_action = (
    transpose @ np.linalg.solve(gauge_matrix, eigenvectors)
)
measured_energy = transpose @ (energy_matrix @ eigenvectors)
coefficients.append(
    (
        inverse_action
        * np.swapaxes(measured_energy, 1, 2)
        * weights[:, None, None]
    ).ravel()
)
exponents.append(
    (
        eigenvalues[:, :, None]
        + eigenvalues[:, None, :]
    ).ravel()
)
```

The `solve` expression is $V^{\mathsf T}P^{-1}V$; the swapped energy matrix
changes $B_{ij}$ to $B_{ji}$; orbit multiplicities are folded into the scalar
coefficients. The two exponentials have not been incorrectly collapsed across
$P^{-1}$. Off-diagonal $A_{ij}$ terms retain the noncommuting case emphasized
after Eq. (3.15) of the paper.

Processing momentum representatives in blocks changes memory use only. It
does not change Eq. (17), apart from ordinary floating-point summation order.

### 3.7 Exponent merging

`_collapse_terms` stably sorts all exponents and merges adjacent terms when
their difference is no more than $10^{-11}$. Coefficients in each group are
summed, and the first exponent represents the group.

This is a numerical approximation, not a published step. It is exact for
mathematically degenerate eigenvalue sums up to rounding, but the grouping
rule is transitive through adjacent values: the full span of a group can
exceed $10^{-11}$ if many successive gaps are smaller than the tolerance.
No error estimate is attached to the merge and the tolerance is not recorded
in `CorrectionEvidence`.

### 3.8 Uniform-grid recurrence and zero mode

After reduction, the nonzero-mode sum has the form

\[
S(t)=\sum_k a_ke^{-t\mu_k}.
\]

On the uniform grid $t_j=j\epsilon$,

\[
a_ke^{-t_j\mu_k}
=a_k\left(e^{-\epsilon\mu_k}\right)^j.
\tag{18}
\]

`_sum_exponentials` evaluates one exponential per spectral term, then advances
Eq. (18) by multiplication. Chunk-local rows avoid concurrent writes. The
rows are summed afterward.

`_tree_level_grid` defines

```python
prefactor = (
    64.0
    * np.pi**2
    * flow_times**2
    / (3.0 * spatial_extent**3 * temporal_extent)
)
return flow_times, 2.0 * prefactor + prefactor * lattice_sum
```

Thus `2.0 * prefactor` is the first term of Eq. (11), and
`prefactor * lattice_sum` is the second. The code has the correct factor of
two and exactly two flowed kernels.

An explicit audit calculation enumerated all $4^3\times6-1$ nonzero momenta
without orbit reduction and evaluated the matrix trace directly. At grid
points it agreed with the optimized code to between $8\times10^{-16}$ and
$4\times10^{-15}$ relative error for WWP, WSS, and WSC cases. This supports
the orbit and spectral reorganizations, but it is not presently an automated
repository test.

### 3.9 Interpolation grid and domain boundary

The intended upper flow time is

\[
t_{\max}=\frac{N_s^2}{32},
\qquad
\frac{\sqrt{8t_{\max}}}{N_s}=\frac12.
\tag{19}
\]

The implementation chooses

```python
count = int(
    np.ceil(spatial_extent**2 / (32.0 * spacing)) + 1
)
flow_times = np.arange(count) * spacing
```

Therefore the last grid point is

\[
t_{\rm grid}=\epsilon
\left\lceil\frac{N_s^2}{32\epsilon}\right\rceil
\ge t_{\max},
\tag{20}
\]

with an overshoot smaller than one spacing. `tree_level_delta` rejects only
times greater than `grid[-1]`. Its docstring says requests beyond
$\sqrt{8t}/N_s=1/2$ reject, but Eq. (20) admits a narrow interval beyond that
boundary whenever $N_s^2/(32\epsilon)$ is not integral.

The public function evaluates a default not-a-knot `CubicSpline` on the grid
with extrapolation disabled. Grid spacing is recorded in evidence and is
correctly described as numerical interpolation spacing, not lattice spacing
$a$.

There is no interpolation error estimate and no lower bound relating $t$ to
$\epsilon$. An explicit full-sum comparison on a $4^3\times6$ lattice with
default $\epsilon=0.01$ found relative interpolation errors of approximately:

| $t$ | relative spline error |
|---:|---:|
| 0.0001 | 4.45 |
| 0.001 | 0.37 |
| 0.005 | 0.027 |
| 0.009 | 0.0018 |
| 0.025 | $3.0\times10^{-5}$ |

These values are audit diagnostics, not universal bounds; error depends on
geometry and discretization. They demonstrate that “positive flow time” is
not by itself a sufficient numerical accuracy contract. Typical analysis
times are much larger, but the public method does not enforce or evidence
that assumption.

### 3.10 Parallel summation, fast math, and cache identity

The number of accumulation chunks is `max(1, get_num_threads())`, and the
Numba kernel is compiled with `parallel=True, fastmath=True`. Floating-point
addition is not associative, so changing the process-global Numba thread
count can change the partition and final low bits.

The LRU cache key contains geometry, gauge coefficient, energy coefficient,
and interpolation spacing, but not thread count, Numba version, CPU target,
or fast-math settings. A value computed under one thread configuration can be
reused after the hidden environment changes. This is unlikely to produce a
large scientific shift, but it conflicts with strict bitwise determinism and
is not represented in evidence.

### 3.11 TLN output and evidence

`tree_level_delta` returns the spline value minus one, matching Eq. (4).
`tree_level_correction` then records:

- method `finite-lattice-tree-level-normalization`;
- DOI 10.1007/JHEP09(2014)018;
- volume;
- Wilson flow;
- Wilson or Symanzik gauge action;
- plaquette, Symanzik, or clover energy operator;
- units `t/a^2`;
- interpolation spacing.

This is enough to identify the scientific triplet and the main public
numerical parameter. It does not record the rectangular temporal extension,
exponent tolerance, spline boundary condition, fast-math policy, or summation
partition. Moreover, `ProcessingResult.__str__` prints only correction method
and source, not the triplet, even though the structured channel evidence
contains it.

### 3.12 TLN audit verdict

- **Paper kernels:** correctly implemented.
- **Gauge fixing:** correct $\alpha=1$ specialization.
- **Master factors and zero mode:** correct for $L^4$; $N_s^3N_t$ is an
  unrecorded rectangular extension.
- **Two exponentials/noncommutation:** correctly retained by the spectral
  formula.
- **Orbit and spectral optimization:** algebraically correct and independently
  spot-checked against the full sum.
- **Interpolation:** accurate at ordinary grid-resolved flow times, but has no
  guaranteed public error contract.
- **Determinism:** hidden thread-count and fast-math dependence remains.
- **Source field:** DOI 10.1007/JHEP09(2014)018 is appropriate for the
  scientific expression, not every numerical optimization.

## 4. Correct FVN--TLN convergence statement

Define

\[
x=\frac{a^2}{t},
\qquad
c=\frac{\sqrt{8t}}{L}.
\]

The 2014 paper's Eq. (6.1) defines the finite-lattice factor as $C(x,c)$;
Eq. (6.2) supplies its finite-volume momentum sum. The 2012 continuum
finite-volume result is $C(0,c)$. The sourced relationship is

\[
\lim_{x\to0\ \text{at fixed }c}
C_{\rm TLN}(x,c)=C_{\rm FVN}(c).
\tag{21}
\]

At a fixed nonzero $c$, the right-hand side is generally **not one**. For
example, the current $c=0.30$ oracle has
$C_{\rm FVN}=0.9734716362$. Unity requires a second limit, $c\to0$, after or
together with $x\to0$:

\[
\lim_{c\to0}\lim_{x\to0}C(x,c)=1.
\tag{22}
\]

In lattice units,

\[
\frac{t}{a^2}
=\frac{c^2}{8}\left(\frac{L}{a}\right)^2.
\]

Thus “large $t/a^2$” is a continuum statement only if $c$ is held fixed by
increasing $L/a$ simultaneously. Increasing $t/a^2$ on one fixed lattice also
changes $c$ and is not the same limit.

The current acceptance test follows the correct path: it fixes $c=0.30$ and
uses hypercubic $L/a=8,16,24,32$. The relative difference decreases
monotonically and is required to be below 1.2 percent at the largest lattice.
That is meaningful independent evidence for the complete correction path.

The test currently covers only WWP. Since every supported discretization must
share the same continuum finite-volume limit, WSP, WSS, WSC, and the remaining
supported Wilson/Symanzik gauge-action combinations would strengthen the
oracle. Different discretizations should not be expected to agree at finite
$a^2/t$.

### 4.1 Check against the current implementation manual

There is no disagreement on these two points with
`docs/tree-level-correction-implementation.md`. That manual explicitly calls
$V=N_s^3N_t$ a derived rectangular-time extension rather than a paper
formula, and its Eqs. (14)--(15) state the fixed-$c$ limit as
$C_{\rm TLN}(0,c)=C_{\rm FVN}(c)$. It does not claim this limit is unity.
`docs/tree-level-corrections.md` and `docs/tree-level-correction-sources.md`
also distinguish the three limits correctly. The documentation discrepancy
reported below concerns the number of flow exponentials, not either of these
attribution points.

## 5. Current test evidence and missing checks

### Existing evidence

`tests/test_processing_interface.py` currently provides:

- exact method/source reporting for FVN;
- a frozen end-to-end TLN regression on a $16^3\times32$ lattice;
- fixed-$c$ monotone convergence from WWP TLN to FVN;
- strict rejection of a non-Wilson flow.

The fixed-$c$ convergence test is the strongest independent scientific test.
The frozen values are useful regression evidence, but the test does not state
whether they came from the paper, QEX, the legacy implementation, or the
current implementation. Without a pinned authority/revision they are not an
independent oracle.

### High-value missing checks

The following checks would address distinct failure modes without duplicating
the production algorithm:

1. Compare FVN on a hypercube to an independently summed full Jacobi series at
   several $c$ values, including the edge $c=1/2$.
2. Add an explicit full-momentum TLN oracle on a tiny lattice for every kernel
   family, avoiding orbit reduction and exponent collapse.
3. Compare the default interpolation to direct evaluation at off-grid times,
   including a declared minimum supported $t/\epsilon$.
4. Verify convergence under halved interpolation spacing.
5. Test the exact $c=1/2$ boundary and reject the rounded-grid overshoot.
6. Test every supported Flow--Action--Observable code and every unsupported
   code.
7. Pin and identify any QEX/reference-table revision used for frozen values.
8. Run the same calculation under different Numba thread counts and declare
   an appropriate reproducibility tolerance or deterministic reduction.
9. Either source the $N_s^3N_t$ extension independently or restrict the
   scientific contract and tests to $L^4$.

## 6. Documentation details found during the audit

The existing correction documents and the implementation manual display the
two flow exponentials correctly: one on each side of the inverse action, as in
the paper and `tln.py`.

The `tree_level_delta` docstring promises rejection beyond exactly
$\sqrt{8t}/N_s=1/2$; the rounded-up grid check is slightly looser, as shown in
Eq. (20).

The current result report prints correction method and DOI, but not the
Flow--Action--Observable triplet or extension notes. The structured evidence
does preserve the triplet for TLN and the extension notes for FVN.

## 7. Source/evidence classification

| Behavior | Classification | Authority/evidence |
|---|---|---|
| Hypercubic FVN formula | Literature | Fodor et al. 2012, Eqs. (1.2), (1.3), (3.9) |
| Division by $C$ | Literature | Fodor et al. 2012 Eq. (4.1); Fodor et al. 2014 Sec. 7 |
| Rectangular FVN product | Toolkit/legacy extension | Named in FVN evidence; not explicit in cited papers |
| FVN theta truncation at $|n|\le2$ | Toolkit/legacy numerical policy | Named in FVN evidence |
| Action and clover kernels | Literature | Fodor et al. 2014, Eqs. (3.4)--(3.7) |
| Gauge fixing with $\alpha=1$ | Literature specialization | Fodor et al. 2014, Eqs. (3.10)--(3.15) |
| Finite-lattice master sum | Literature | Fodor et al. 2014, Eq. (6.2) |
| $N_s^3N_t$ TLN extension | Toolkit extension | Not currently named in TLN evidence |
| Wilson-flow-only support | Toolkit applicability restriction | Correctly rejected and recorded |
| Orbit reduction | Exact implementation optimization | Closure checked; not a paper method |
| Spectral trace reduction | Exact linear-algebra identity | Directly derivable from Eq. (6.2) |
| Exponent merge at $10^{-11}$ | Toolkit numerical policy | Not recorded in evidence |
| Uniform recurrence | Exact in real arithmetic | Floating implementation choice |
| Cubic spline and spacing | Toolkit numerical policy | Spacing recorded; spline/error policy not recorded |
| Numba fast math/thread chunks | Toolkit numerical policy | Not recorded in evidence or cache identity |
| Fixed-$c$ TLN-to-FVN limit | Literature relationship | Both canonical papers; automated WWP oracle |

## Primary references

- Z. Fodor, K. Holland, J. Kuti, D. Nogradi, and C. H. Wong,
  [The Yang--Mills Gradient Flow in Finite
  Volume](https://arxiv.org/html/1208.1051v2), JHEP 11 (2012) 007,
  DOI [10.1007/JHEP11(2012)007](https://doi.org/10.1007/JHEP11(2012)007).
- Z. Fodor, K. Holland, J. Kuti, S. Mondal, D. Nogradi, and C. H. Wong,
  [The Lattice Gradient Flow at Tree-Level and Its
  Improvement](https://arxiv.org/html/1406.0827v2), JHEP 09 (2014) 018,
  DOI [10.1007/JHEP09(2014)018](https://doi.org/10.1007/JHEP09(2014)018).
