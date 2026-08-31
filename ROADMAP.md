# GradientFlowRGToolkit roadmap

The roadmap advances through vertical scientific slices. A phase is complete
only when its behavior is observable through the public interface, its
scientific evidence is inspectable, its negative cases reject, documentation
matches reality, and the full repository check is green.

## Phase 0 — Contract and green baseline

Status: in progress.

Establish the repository discipline before scientific implementation grows:

- [x] record the thesis, invariants, vocabulary, and roadmap;
- [ ] choose and pin the supported Python/toolchain baseline;
- [ ] establish a conventional installable `src/gfrgtoolkit` package without the
  current placeholder import ambiguity;
- [ ] add a one-command `./tools/check.sh` covering formatting, linting, static type
  checks, tests, package build, and repository hygiene;
- [ ] add CI that invokes the same command rather than reimplementing it;
- [ ] define the test-data policy, including where generated, small frozen, and
  large external datasets belong;
- [ ] establish the protocol for freezing an acceptance workflow by recording
  its dataset manifest, analysis choices, expected outputs, tolerances, and
  scientific authority;
- [ ] document how scientific references and dataset revisions are pinned; and
- [ ] remove or replace placeholder source only when the package baseline tests pin
  the intended public import surface.

Completion criteria:

- a clean checkout can be installed and `import gfrgtoolkit` succeeds;
- `./tools/check.sh` passes locally from a clean environment;
- CI runs that same check;
- no scientific functionality is claimed by placeholders; and
- the thesis, invariants, and Phase 1 oracle have been reviewed.

## Phase 1 — Finite-ensemble processing tracer bullet

Make the toolkit useful for a complete finite-ensemble calculation. Implement
the smallest end-to-end route through the cohesive `RunningCoupling` workflow
while avoiding claims of generality beyond that route.

The processing-stage checkpoint is:

```text
trusted ensemble files
    -> Dataset selection
    -> raw-history observable combination
    -> joint Long-run covariance estimates
    -> finite-volume/tree-level normalization
    -> finite-ensemble coupling and beta estimates
    -> ProcessingResult
```

That checkpoint is the first part of the eventual Phase 1 vertical slice:

```text
finite-ensemble estimates
    -> optional Chiral limit
    -> Infinite-volume limit at fixed bare coupling
    -> interpolation in renormalized coupling
    -> Continuum limit at fixed renormalized coupling
```

- [x] execute the public processing path end to end on representative inputs;
- treat an in-flux external workflow as integration evidence, not as a source
  of scientific or numerical contracts: incomplete inputs may expose missing
  workflow support, but must not define estimator semantics, validity domains,
  tolerances, or acceptance thresholds;
- freeze dataset selections and reference tables/curves only after an explicit
  stable oracle revision is declared; until then, ground scientific kernels in
  published formulas plus analytic, synthetic, and tiny independent numerical
  oracles;
- [x] implement the public imports, `RunningCoupling`, and
  `ProcessingConfiguration` required by the tracer bullet;
- [x] replace the inherited rectangular covariance default with joint
  Bartlett/Newey--West, retain the legacy calculation only as an experimental
  comparator, and add batch-means, over-lugsail, projected-Wolff, and
  unresolved-tail evidence paths;
- introduce an `AnalysisPlan` only when the first post-processing stage gives
  it concrete behavior to own;
- support only the flow/operator/correction/model combinations the workflow
  actually uses;
- retain a staged exploratory interface and a one-shot result path over the
  same implementation;
- isolate legacy executable input behind an explicit trusted-input adapter if
  that is the format of the on-hand data;
- capture immutable typed Stage results even if the facade caches the current
  stage for notebook convenience; and
- produce the diagnostic plots/tables needed to judge agreement with the
  current trusted analysis.

Completion criteria:

- the processing checkpoint runs end to end on representative inputs before
  post-processing stages are added;
- an extended analysis eventually runs through the declared limit stages;
- when a stable oracle revision exists, selected intermediate and continuum
  outputs match its frozen expectations within independently justified
  tolerances;
- the workflow records its Dataset, plan, stage evidence, and exclusions;
- changing an upstream stage visibly invalidates dependent cached stages;
- a completed result remains stable after the facade is reused; and
- at least one compact automated acceptance path protects the notebook's
  essential behavior without requiring the full dataset.

## Phase 2 — Harden the finite-ensemble scientific kernel

Deepen the working Phase 1 implementation around explicit domain values and
independent numerical evidence:

- Gauge theory, Lattice geometry, Ensemble identity, Measurement channel, and
  Coupling scheme values;
- validated Flow-time grids and Monte Carlo histories;
- autocorrelation-aware correlated estimation;
- covariance-preserving energy-density combinations;
- coupling normalization and reviewed correction applicability;
- numerical differentiation on nonuniform grids;
- deterministic catalog discovery and typed Dataset queries; and
- malformed/empty Ensemble quarantine with visible reasons.

Add analytic and synthetic oracles whose derivative and covariance behavior
are known without repeating production code.

Completion criteria:

- the Phase 1 notebook remains unchanged and green;
- the analytic derivative is recovered within a justified tolerance;
- correlations survive averaging, combination, and differentiation;
- filesystem order cannot change the Dataset or result;
- malformed grids, mismatched histories, unsupported schemes, and insufficient
  support reject with useful diagnostics; and
- unsafe legacy input is never loaded implicitly.

## Phase 3 — Harden limit stages and evidence

Deepen the Phase 1 chiral, infinite-volume, interpolation, and continuum
implementation without widening its public interface unnecessarily.

- introduce only the Fit model and execution interfaces justified by real
  variation;
- require immutable Stage evidence with selected inputs, model, Fit result,
  domains, exclusions, and diagnostics;
- add a synthetic multi-coupling, multi-volume dataset with a known continuum
  curve and known finite-volume/lattice artifacts;
- make interpolation domains and Extrapolation explicit; and
- prevent fit-library-native containers from becoming public scientific state.

Completion criteria:

- the known synthetic continuum curve is recovered over its common Validity
  domain;
- the real-data oracle remains green;
- Continuum limits use a declared lattice-artifact coordinate at fixed
  renormalized coupling;
- underdetermined fits and incompatible domains reject; and
- evaluation beyond an interpolation domain is impossible unless explicit
  Extrapolation is enabled.

## Phase 4 — Alternative limit order and systematic evidence

Add the methodologically important variations without creating a second result
model:

- finite-volume interpolation followed by Infinite-volume extrapolation at
  fixed renormalized coupling;
- stable volume selection over the common interpolation domain;
- fit-window and interpolation-model scans;
- Measurement-channel comparisons;
- covariance policies and regularization as explicit choices;
- Bayesian model averaging with within- and between-model covariance; and
- systematic error summaries that retain every contributing variation.

Completion criteria:

- both Limit orders produce the same `AnalysisResult` shape and evidence model;
- the fixed-coupling route cannot switch volume subsets silently;
- model/window variations are reproducible and individually inspectable;
- model-average covariance passes an independent mixture calculation; and
- synthetic scenarios distinguish interpolation from unjustified
  Extrapolation.

## Phase 5 — Published-analysis reproduction

Make a published continuous-beta-function analysis the first real scientific
acceptance target. Start with the public twelve-flavor SU(3) analysis because
the legacy repository already records its workflow, public Zenodo source, fixed
point, and critical-exponent convention.

- pin the paper, public dataset revision, selected Ensembles, and analysis
  choices;
- reproduce the continuum curves for the declared Measurement channels;
- reproduce the fixed-point extraction without silently extending a fit domain;
- report the Flow and QFT beta-function conventions side by side;
- compare the fixed point and leading irrelevant exponent with documented
  statistical/systematic tolerances;
- preserve compact regression fixtures or derived reference tables so routine
  checks do not require the full external dataset; and
- independently review every legacy behavior promoted into the new toolkit.

Completion criteria:

- a pinned full-data command reproduces the selected published result within
  justified tolerances;
- a compact acceptance suite protects its essential scientific behavior;
- discrepancies are explained rather than tuned away; and
- the result contains enough Provenance and Stage evidence for another analyst
  to audit the route.

## Phase 6 — Research ergonomics and adapters

Add user-facing capabilities around the stable scientific core:

- deterministic, versioned Analysis-plan and Analysis-result persistence;
- plotting from immutable results rather than analysis internals;
- notebook helpers that call the same public interface as tests;
- command-line batch execution and machine-readable reports;
- explicit resource/memory diagnostics for large correlated analyses; and
- a second published workflow, such as the SU(2) adjoint analysis, to test
  generality rather than only adding options to the first case.

Completion criteria:

- adapters can be replaced without changing scientific modules;
- serialization round-trips preserve owned scientific meaning;
- plots and reports can be regenerated from a persisted Analysis result; and
- the second workflow requires domain extensions, not special-case branches in
  orchestration.

## Later candidates

These are directions, not commitments, and do not override the ordered phases:

- fixed-point and critical-exponent analysis as dedicated evidence-bearing
  results;
- perturbative matching and Lambda-parameter workflows;
- additional gradient flows, energy-density operators, gauge groups, and
  fermion representations;
- alternative statistical backends after a second real implementation
  justifies a public seam;
- scalable out-of-core data adapters; and
- publication bundles that combine plans, results, source citations, tables,
  and figures reproducibly.
