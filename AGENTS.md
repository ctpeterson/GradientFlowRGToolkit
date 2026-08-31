# GradientFlowRGToolkit development instructions

## Project thesis

GradientFlowRGToolkit makes continuous gradient-flow beta-function analyses
inspectable, reproducible, and comparable. The ensemble selection,
renormalization scheme, statistical transformations, limit ordering, fit
models, and resulting evidence are explicit scientific values, so a reported
curve cannot be separated from how it was obtained.

The central scientific interface is:

```text
Dataset + AnalysisPlan -> AnalysisResult
```

A notebook-facing `RunningCoupling` facade may provide a cohesive staged
interface over that transformation. The facade may cache the current Dataset,
plan, and stage results for exploration, provided that its lifecycle and
invalidation rules are explicit and previously returned results remain valid.

An `AnalysisResult` is an immutable scientific record, not a view of mutable
working state. It carries the continuum curve together with covariance,
provenance, validity domains, stage evidence, diagnostics, and systematic
variations.

The toolkit is object-oriented in the domain sense: objects represent
scientific concepts and own the behavior that preserves their invariants.
File formats, fit libraries, plotting systems, notebooks, and command-line
tools remain adapters around that model.

`../ContinuousBetaFunction` is a behavioral and research reference. Preserve
its validated scientific scenarios, published-analysis knowledge, useful
regression cases, and productive notebook workflow. Reassess rather than
blindly inherit its shared nested dictionaries, hidden stage dependencies,
compatibility surface, and incidental file conventions.

## Scientific invariants

1. **The beta-function convention is explicit.** The flow beta function is
   `beta_GF(g^2) = -t d g^2 / dt`. The QFT convention
   `mu d g^2 / d mu = 2 beta_GF(g^2)` is a distinct named quantity. Conversions
   are explicit; signs and factors of two are never inferred from context.
2. **A coupling value is inseparable from its scheme.** Gauge group, matter
   content, flow action, energy-density operator, normalization, correction,
   and scale convention accompany every gradient-flow coupling definition.
3. **Measurement coordinates are valid by construction.** Flow times and
   other fit coordinates are finite and strictly ordered, values are finite,
   shapes agree, and a numerical derivative is attempted only with enough
   support for its declared method.
4. **Ensemble identity is semantic.** Bare coupling, lattice geometry, fermion
   mass, flow definition, and other simulation parameters are typed identity
   fields, not information recovered repeatedly from filenames or compressed
   strings.
5. **Correlations are preserved.** Averaging, linear combinations,
   differentiation, interpolation, extrapolation, and model averaging retain
   the covariance they are mathematically entitled to. Diagonalization or
   covariance regularization is an explicit analysis choice and is recorded.
6. **Observable combinations occur before information is discarded.** Linear
   combinations of correlated Monte Carlo histories are formed before an
   averaging or reduction that would lose their cross-covariance.
7. **Limit order is part of the analysis.** Chiral, infinite-volume,
   interpolation, and continuum transformations do not silently commute. The
   chosen order and every omitted limit are recorded in the `AnalysisPlan` and
   result.
8. **Continuum claims require continuum evidence.** A finite-ensemble or
   finite-flow-time estimate is never labelled a continuum beta function.
   Continuum extrapolation is performed at fixed renormalized coupling with an
   explicit lattice-artifact coordinate and fit domain.
9. **Fixed-coupling volume extrapolation uses a stable ensemble set.** When the
   beta function is interpolated at finite volume before the infinite-volume
   limit, one declared volume set is used over its common interpolation domain;
   the set never changes silently with coupling.
10. **Interpolation is not evidence outside its domain.** Extrapolation is off
    by default. When enabled, its model, requested domain, justification, and
    diagnostics are explicit in the result.
11. **Corrections declare their applicability.** A finite-volume or tree-level
    correction identifies the flow action, operator, geometry, and assumptions
    it supports. An unsupported combination rejects instead of applying the
    nearest-looking formula.
12. **A fit is more than a parameter vector.** Inputs, model identity, priors,
    covariance policy, validity domain, convergence state, and quality
    diagnostics accompany every fit result.
13. **Scientific failure is explicit.** Non-finite arithmetic, singular or
    underdetermined fits, incompatible domains, insufficient ensembles, and
    failed convergence produce typed failures. They do not become empty curves,
    skipped points, or plausible-looking defaults.
14. **Missing and quarantined data remain visible.** A result records excluded,
    malformed, empty, and out-of-domain inputs together with the reason each was
    not used.
15. **Systematic variations are first-class evidence.** Alternative windows,
    models, discretizations, and orderings remain identifiable. Model averaging
    includes both within-model covariance and between-model variation.
16. **Scientific results are immutable.** A stateful exploratory facade may
    cache or replace its current working stages, but each completed Stage result
    and Analysis result is a stable value. Rerunning upstream work explicitly
    invalidates dependent cached stages and cannot mutate a previously returned
    result. Process-global numerical state is not part of the scientific core.
17. **Results are reproducible records.** A result identifies the complete
    plan, selected data, software/dependency versions, deterministic input
    digests where practical, and all choices that can change its scientific
    interpretation.
18. **Determinism is the default.** Identical declared inputs, versions, and
    configuration produce equivalent results independent of filesystem order,
    hash iteration order, wall-clock time, or hidden environment state.
19. **Independent evidence outranks self-consistency.** Analytic cases,
    synthetic data with known limits, published tables, and independent
    implementations are preferred as oracles over tests that merely repeat the
    production formulas.
20. **Reject rather than guess.** Silent production of a scientifically
    different analysis is more serious than an explicit failure.

## Oracle discipline

Notebook workflows under the repository's oracle directory are acceptance
surfaces that implementation work must preserve. A workflow becomes a frozen
oracle when its input dataset revision, analysis choices, expected scientific
outputs, tolerances, and source of authority are recorded.

- Drive the first vertical implementation from a concrete analysis workflow
  so the toolkit becomes useful immediately.
- While an oracle dataset is in flux, use it to discover integration and
  coverage gaps only. Derive scientific semantics, validity domains,
  tolerances, and acceptance thresholds from sources plus analytic,
  synthetic, or independent numerical oracles. Promote real-data values to
  regression expectations only after the dataset revision is explicitly
  frozen.
- Keep large datasets outside routine source control when necessary, but pin a
  manifest/digest and retain a compact fixture for ordinary checks.
- A notebook is an excellent end-to-end scenario; extract its essential
  expectations into automated acceptance tests so execution order and stale
  cell state cannot hide regressions.
- Once frozen, change an oracle only with explicit user approval and a recorded
  scientific reason. Implementation moves toward the oracle, not vice versa.
- Complement real-data oracles with analytic and synthetic cases. Agreement
  with one earlier implementation is regression evidence, not by itself an
  independent proof of correctness.

## Object-model discipline

- Prefer immutable domain results and composition where it clarifies ownership;
  inheritance is not prohibited.
- Preserve a cohesive, notebook-friendly `RunningCoupling` facade. A class is
  not a god object merely because it offers many related operations; it becomes
  problematic when callers must understand hidden shared state or unrelated
  responsibilities accumulate behind the same interface.
- Mixins are permitted as private implementation organization. They should have
  no independent lifecycle, should rely on a small documented host protocol,
  and should return or install typed Stage results through one centrally owned
  state/invalidation mechanism. A mixin that assumes dozens of undeclared host
  attributes is a hidden interface and should be deepened or replaced.
- Keep the scientific interface small. Callers may use the one-shot
  `Dataset + AnalysisPlan -> AnalysisResult` path or the staged facade, and both
  paths must exercise the same scientific implementation.
- Keep public collections typed and read-only. Nested `dict[str, Any]` values
  may exist inside an adapter but are not the scientific model.
- Put validation and invariant-preserving behavior with the value that owns the
  invariant. Do not duplicate the same scientific rule in orchestration,
  plotting, and persistence code.
- Design deep modules: substantial behavior behind a narrow interface. Avoid
  decorative wrappers, pass-through classes, and protocols introduced only for
  hypothetical future implementations.
- A seam becomes public when behavior actually varies across it. Internal
  numerical helpers remain implementation details until a second real adapter
  or caller justifies an interface.
- Accept dependencies at a seam; do not construct filesystem, plotting, random,
  or fit-engine dependencies inside domain objects.

## External-authority contract

The method papers, published analysis papers, public data records, and reviewed
correction formulas are scientific authorities. The legacy repository is an
implementation reference, not an authority that can override them.

- Record the exact paper, equation, dataset revision, or independent code
  revision behind a scientific formula or oracle.
- Distinguish a literature requirement from a toolkit policy and from an
  experimental option.
- Never promote a remembered formula, convention, or dataset shape into the
  core model without a reproducible source or fixture.
- Treat pickle and other executable serialization formats as trusted-input
  compatibility adapters only; they must never become the default safe data
  boundary.

## Architecture and ownership

Keep these responsibilities conceptually separate:

```text
domain values and validation
dataset catalog and ingestion adapters
Monte Carlo statistics and correlated estimates
gradient-flow coupling and correction definitions
numerical differentiation
fit models and fit execution
chiral, infinite-volume, interpolation, and continuum transformations
analysis orchestration and notebook-facing facade
evidence, diagnostics, provenance, and persistence
plotting, notebooks, and command-line presentation
```

Dependency direction points inward toward the domain model. Plotting and file
adapters may consume results; domain and analysis modules do not import them.
Fit-library-specific values do not escape into the public scientific model
without an explicit translation.

## Change discipline

For every coherent scientific change:

1. read `CONTEXT.md`, the active `ROADMAP.md` milestone, relevant ADRs,
   scientific references, implementation, and tests;
2. state the scientific or architectural intent;
3. identify the affected invariants;
4. add or strengthen an oracle, property, negative case, or regression test;
5. implement the smallest vertical slice through the public interface;
6. run focused checks and then the full repository check;
7. inspect result evidence and representative serialized/plot artifacts when
   those change;
8. inspect `git diff --check`, the full diff, and repository status;
9. update context, roadmap, references, and ADRs when their owned facts change;
10. finish only at a green, coherent checkpoint.

Do not mix a scientific-method change with broad mechanical refactoring. Do not
weaken an oracle, uncertainty check, domain restriction, or failure test merely
to continue.

## Testing strategy

Use complementary evidence:

- **Domain tests** for value construction, invariants, and typed failures.
- **Real-data workflow oracles** with pinned inputs, expected outputs, and
  justified tolerances.
- **Analytic/synthetic oracles** whose finite-volume and continuum answers are
  known independently of both the implementation and real-data reference.
- **Property tests** for covariance propagation, deterministic ordering,
  interpolation-domain intersection, serialization stability, and equivalent
  unit/convention conversions.
- **Negative tests** for malformed data, unsupported corrections,
  underdetermined fits, domain mismatch, and accidental extrapolation.
- **Differential tests** against selected legacy behavior only after deciding
  that behavior is scientifically intended.
- **Published-analysis acceptance tests** using pinned public inputs and
  documented tolerances.
- **Small golden tests** only where an exact report or serialized schema is part
  of the interface.

A tolerance must have a numerical or scientific rationale. Avoid assertions
chosen solely to make the current implementation pass.

## Determinism and reproducibility

Sort filesystem discoveries and serialized mappings. Seed randomized tests and
record any stochastic analysis seed. Keep timestamps and machine-specific paths
out of scientific identity. Persistence is versioned and deterministic, and
deserialization validates structure before constructing domain values.

## Documentation and decisions

- `AGENTS.md` owns the thesis, governing invariants, and engineering contract.
- `CONTEXT.md` owns canonical domain vocabulary and contains no implementation
  design.
- `ROADMAP.md` owns milestone order and completion criteria.
- `docs/adr/` records only hard-to-reverse, surprising decisions made after a
  real trade-off.
- Scientific reference notes should identify sources, equations, conventions,
  and the behavior independently reproduced by this project.

Clearly distinguish implemented behavior, planned behavior, and open research
questions. Do not document a desired capability as though it exists.

## Choosing the next milestone

When asked to continue, use the first incomplete milestone in `ROADMAP.md` whose
prerequisites are satisfied. State the milestone and its completion criterion
before editing. Do not choose work merely because the legacy repository already
contains a convenient implementation.

## Current objective

Complete Roadmap Phase 0, then complete the finite-ensemble processing tracer
bullet in Phase 1. Harden the scientific modules behind that working facade in
the following phases.

## Definition of engineering done

A change is complete only when:

1. its scientific semantics and affected invariants are clear;
2. the implementation is understandable and appropriately factored;
3. positive, negative, regression, and independent-oracle coverage exists as
   appropriate;
4. uncertainty and provenance remain intact;
5. behavior is deterministic;
6. diagnostics identify invalid inputs and violated constraints;
7. focused and full checks pass without unexplained warnings;
8. documentation and roadmap state agree with the code;
9. generated artifacts and caches are not accidentally committed; and
10. the final diff and repository status have been reviewed.

A curve is not done merely because it can be plotted once.

## Agent skills

### Issue tracker

Issues and specs are tracked in this repository’s GitHub Issues. See `docs/agents/issue-tracker.md`.

### Triage labels

Use the five default canonical triage labels. See `docs/agents/triage-labels.md`.

### Domain docs

This is a single-context repository. See `docs/agents/domain.md`.
