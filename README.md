# Gradient flow renormalization group (RG) toolkit

A collection of tools for analyzing gradient flow running couplings

The statistical processing layer estimates aligned flow times and measurement
channels jointly. Its primary full-matrix estimator is the sourced,
positive-semidefinite Bartlett/Newey--West long-run covariance; batch means,
over-lugsail, and projected Wolff calculations are explicit validation or
systematic-variation paths.

Each statistical method owns its configuration, validation, evidence
construction, and numerical implementation in a separate module under
`stages/statistics`. The package-level statistics interface aggregates those
methods; higher-level setup and processing depend on that interface, while no
statistical method depends on `setup.py`.

Autocorrelation diagnostics record unresolved evidence without blocking an
exploratory result by default. Analyses that require a hard publication or CI
gate can opt into `UnresolvedAutocorrelationAction.Raise`.

Set `ProcessingConfiguration.verbosity=1` to stream deterministic
per-ensemble progress and exclusion reasons while processing. The default
`verbosity=0` remains quiet for library and test use.

See the
[compiled long-run covariance guide](docs/long-run-covariance-methods.pdf)
and its
[source document](docs/long-run-covariance-methods.md).

For a formula-to-code treatment of every implemented estimator, including
worked examples, normalization derivations, PSD/rank behavior, and evidence
semantics, see the
[compiled implementation manual](docs/covariance-estimator-implementation.pdf)
and its
[Markdown source](docs/covariance-estimator-implementation.md). The separate
[primary-source implementation audit](docs/covariance-implementation-source-audit.md)
records the literature cross-check and identified limitations.

The finite-volume and finite-lattice tree-level corrections are documented in
the [compiled correction guide](docs/tree-level-corrections.pdf) and its
[Markdown source](docs/tree-level-corrections.md).

Corrections implement a narrow `CorrectionMethod` interface and own their
configuration, validation, evidence, and numerical policy. FVN evaluates the
complete Jacobi-theta factor to a declared tolerance; TLN evaluates the
published finite momentum sum directly at requested flow times with a
deterministic inner reduction.

For the corresponding formula-to-code treatment of `fvn.py` and `tln.py`,
including momentum orbits, kernels, spectral reduction, direct requested-time
evaluation, worked values, and the fixed-scheme
convergence oracle, see the
[compiled correction implementation manual](docs/tree-level-correction-implementation.pdf)
and its
[Markdown source](docs/tree-level-correction-implementation.md). The separate
[primary-source correction audit](docs/tree-level-correction-implementation-source-audit.md)
records the independent literature cross-check and numerical-policy limits.
