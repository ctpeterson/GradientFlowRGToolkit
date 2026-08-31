# Gradient-flow beta-function analysis

GradientFlowRGToolkit models the scientific concepts needed to infer a
continuous renormalization-group beta function from gradient-flow lattice data
and to retain the evidence behind that inference.

## Language

**Gauge theory**:
The continuum theory whose running coupling is being studied, including its
gauge group and matter content.
_Avoid_: Model, physics parameters

**Ensemble**:
A set of Monte Carlo configurations generated with one declared set of bare
parameters and one lattice geometry.
_Avoid_: File, run, volume

**Ensemble identity**:
The simulation parameters that distinguish one Ensemble from another, such as
bare coupling, fermion mass, and lattice geometry. Storage names may encode an
Ensemble identity but do not define it.
_Avoid_: Filename key, ensemble string

**Lattice geometry**:
The finite lattice extents and boundary information relevant to measurements
and finite-volume effects.
_Avoid_: Volume, when only a single scalar extent is meant

**Monte Carlo history**:
The ordered measurements of one quantity across configurations in an Ensemble.
It is the input to autocorrelation-aware statistical estimation.
_Avoid_: Samples, when their Markov-chain ordering matters

**Flow time**:
The gradient-flow evolution coordinate `t`. A Flow-time grid is an ordered set
of Flow times associated with a measured history.
_Avoid_: Energy scale

**Flow action**:
The discretized evolution equation used to generate flowed gauge fields.
_Avoid_: Gauge action, flow type

**Energy-density operator**:
The discretization used to measure the flowed energy density `E(t)`.
_Avoid_: Observable, when specifically distinguishing plaquette, Symanzik, or
clover energy-density definitions

**Measurement channel**:
A declared Flow action and Energy-density operator pair. Channels may differ at
finite lattice spacing while targeting the same continuum quantity.
_Avoid_: Flow, operator, or observable used alone for the pair

**Gradient-flow coupling**:
The renormalized coupling `g_GF^2(t)` obtained from a normalized flowed energy
density in a declared scheme.
_Avoid_: Coupling, when the scheme or scale would be ambiguous

**Coupling scheme**:
The conventions that give a Gradient-flow coupling its meaning, including its
normalization, Measurement channel, correction policy, and scale convention.
_Avoid_: Settings, options

**Flow beta function**:
The continuous beta function `beta_GF(g^2) = -t d g^2 / dt` in the toolkit's
flow-time convention.
_Avoid_: QFT beta function, beta function when the convention is ambiguous

**QFT beta function**:
The scale derivative `mu d g^2 / d mu`. For `mu = 1/sqrt(8t)`, it is twice the
Flow beta function.
_Avoid_: Flow beta function

**Correlated estimate**:
A numerical estimate together with the covariance relationships needed for
subsequent transformations. Its uncertainty is not an independent error bar
unless its covariance says so.
_Avoid_: Value with error

**Long-run covariance**:
The sum of all lagged cross-covariance matrices governing the asymptotic
covariance of a Monte Carlo history's mean.
_Avoid_: Error matrix, Gamma correction

**Bandwidth**:
The declared extent of lag information used by a Long-run covariance
estimator. It is a scientific bias--variance choice, not a plotting window.
_Avoid_: Window, when the lag-weight convention would be ambiguous

**Autocorrelation resolution**:
Evidence that reported uncertainty is stable over declared Bandwidth and
independent-estimator variations before the informative history ends.
_Avoid_: Convergence, decorrelation

**Unresolved autocorrelation**:
The failure state in which the observed Monte Carlo history does not support
a stable Long-run covariance over the declared diagnostics. It remains visible
as Stage evidence and does not prevent exploratory estimation unless the
Analysis plan explicitly requests strict rejection.
_Avoid_: Conservative error bar, inflated uncertainty

**Positive-leading-bias variation**:
A named estimator variation whose first asymptotic bias term is positive under
its declared persistence assumptions; it is not a finite-sample covariance
upper bound.
_Avoid_: Guaranteed conservative covariance, upper covariance

**Covariance projection**:
A declared transformation of an indefinite estimated covariance matrix into a
positive-semidefinite matrix suitable for correlated arithmetic. Its policy,
trigger, and adjustment magnitude are Stage evidence.
_Avoid_: Numerical cleanup, silent repair

**Correction**:
A declared modification of the finite-lattice coupling definition, with an
explicit applicability domain and assumptions.
_Avoid_: Fix, improvement used without naming its contract

**Fit model**:
A mathematical family and parameter policy used to infer values between or
beyond measured coordinates. A Fit model does not include the data fitted to
it or the resulting evidence.
_Avoid_: Fit, fit function

**Fit result**:
The parameters and evidence produced by applying a Fit model to declared data
over a declared domain.
_Avoid_: Coefficients

**Validity domain**:
The coordinate region supported by the inputs and assumptions of an estimate
or Fit result.
_Avoid_: Range, when it is only a plotting range

**Extrapolation**:
Evaluation beyond a Validity domain under an explicit model assumption.
_Avoid_: Interpolation

**Chiral limit**:
The zero-fermion-mass limit at fixed remaining analysis coordinates.
_Avoid_: Mass correction

**Infinite-volume limit**:
The limit in which finite spatial extent is removed at fixed declared analysis
coordinates.
_Avoid_: Volume correction

**Continuum limit**:
The zero-lattice-spacing limit at fixed renormalized coupling, after all other
limits required by the Analysis plan have been accounted for.
_Avoid_: Large-flow-time limit

**Limit order**:
The declared ordering of chiral, infinite-volume, interpolation, and continuum
transformations in an Analysis plan.
_Avoid_: Pipeline

**Dataset**:
A typed, queryable collection of Ensembles selected as the input to an
analysis. Exclusions remain part of its selection record.
_Avoid_: Data directory, nested dictionary

**Analysis plan**:
The complete immutable scientific choices for one analysis, including Dataset
selection, Coupling scheme, Limit order, Fit models, domains, and uncertainty
policies.
_Avoid_: Configuration dictionary, runtime options

**Stage evidence**:
The inputs, outputs, domains, diagnostics, exclusions, and model identity for
one scientific transformation within an analysis.
_Avoid_: Stage state, cache

**Systematic variation**:
A named alternative scientific choice evaluated to measure analysis
sensitivity, such as a different fit window, model, Measurement channel, or
Limit order.
_Avoid_: Retry, fallback

**Analysis result**:
An immutable scientific record produced by an Analysis plan, containing the
reported estimates and the evidence needed to understand and reproduce them.
_Avoid_: Plot, mutable analysis object

**Provenance**:
The identity of the inputs, scientific choices, source authorities, and
software environment that produced an Analysis result.
_Avoid_: Timestamp, metadata used without further qualification
