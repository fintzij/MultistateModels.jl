# ============================================================================
# Package-wide Constants
# ============================================================================
# Defines numerical tolerances and thresholds used throughout the package.
# Using named constants improves code readability and maintainability.
# ============================================================================

"""
Tolerance for treating Gompertz shape parameter as zero.

When |shape| < SHAPE_ZERO_TOL, the Gompertz hazard reduces to exponential.
This avoids numerical issues with division by near-zero shape values.
"""
const SHAPE_ZERO_TOL = 1e-10

"""
Tolerance for spline knot uniqueness.

Knots closer than this value are considered duplicates during spline basis construction.
"""
const KNOT_UNIQUENESS_TOL = 1e-10

"""
Minimum survival probability for numerical stability.

Used to clamp survival probabilities away from exactly 0 to prevent log(0).
"""
const SURVIVAL_PROB_EPS = eps(Float64)

"""
Tolerance for transition probability matrix row sums.

TPM rows should sum to 1; values within this tolerance are considered valid.
"""
const TPM_ROW_SUM_TOL = 1e-8

"""
Default maximum number of phases for phase-type distributions.
"""
const DEFAULT_MAX_PHASES = 10

"""
Threshold for Pareto-k diagnostic indicating unreliable importance weights.

When pareto_k > PARETO_K_THRESHOLD, weights may be unreliable and resampling 
or more samples may be needed.
"""
const PARETO_K_THRESHOLD = 0.7

# =============================================================================
# Observation Type Conventions
# =============================================================================

"""
    OBSTYPE_EXACT

Observation type code for exact (event time fully observed) observations.
"""
const OBSTYPE_EXACT = 1

"""
    OBSTYPE_PANEL

Observation type code for panel (state at discrete time) observations.
"""
const OBSTYPE_PANEL = 2

"""
    CENSORING_OBSTYPE_OFFSET

Offset for converting state indices to censoring obstypes and vice versa.

For censoring patterns, `obstype = CENSORING_OBSTYPE_OFFSET + statefrom` encodes
that the subject was known to be in state `statefrom` during the interval.
To recover the state: `statefrom = obstype - CENSORING_OBSTYPE_OFFSET`.

Convention:
- `obstype = 1`: Exact observation (transition time known)
- `obstype = 2`: Panel observation (state at discrete times)
- `obstype >= 3`: Censoring pattern where `statefrom = obstype - 2`
  - `obstype = 3`: Subject in state 1 during interval
  - `obstype = 4`: Subject in state 2 during interval
  - etc.

This convention allows the forward algorithm to constrain phase probabilities
to only those phases consistent with the observed sojourn state.
"""
const CENSORING_OBSTYPE_OFFSET = 2

# =============================================================================
# Simulation Defaults
# =============================================================================

"""
Default maximum number of jumps in a single path for workspace allocation.

Used for pre-allocating PathWorkspace buffers. Paths exceeding this will
trigger dynamic resizing.
"""
const DEFAULT_MAX_JUMPS = 1000

"""
Default maximum number of states for workspace pre-allocation.

Used for R matrix storage in ECCTMC sampling. Should cover typical models.
"""
const DEFAULT_MAX_WORKSPACE_STATES = 10

# =============================================================================
# MCEM Algorithm Defaults
# =============================================================================

"""
Default initial effective sample size target per subject.

This is a reasonable starting point that balances Monte Carlo variance
with computational cost.
"""
const DEFAULT_ESS_TARGET_INITIAL = 50

"""
Default maximum effective sample size before stopping.

If ESS reaches this without convergence, MCEM stops with a warning.
"""
const DEFAULT_MAX_ESS = 10000

"""
Default MCEM convergence tolerance for marginal log-likelihood change.
"""
const DEFAULT_MCEM_TOL = 1e-2

"""
Default maximum MCEM iterations.
"""
const DEFAULT_MCEM_MAXITER = 100

# =============================================================================
# Simulation Guards
# =============================================================================

"""
Maximum number of simulation iterations per subject before throwing an error.

This prevents infinite loops when all hazard rates from a state are zero
or when numerical issues cause the simulation to never terminate.
"""
const MAX_SIMULATION_ITERATIONS = 1_000_000

"""
Minimum total hazard rate required for simulation to proceed.

If all hazard rates from a state are below this threshold, simulation
throws an informative error rather than looping indefinitely.
"""
const MIN_TOTAL_HAZARD_RATE = 1e-300

# =============================================================================
# Spline Numerical Tolerances
# =============================================================================

"""
Relative tolerance for checking penalty matrix symmetry.

Used as `tol = SPLINE_SYMMETRY_RTOL * norm(S)` where S is the penalty matrix.
The penalty matrix S should satisfy norm(S - S') < tol.
"""
const SPLINE_SYMMETRY_RTOL = 1e-10

"""
Absolute tolerance fallback for symmetry check when matrix norm is zero.

When norm(S) == 0 (e.g., for zero penalty matrix), use this absolute tolerance.
"""
const SPLINE_SYMMETRY_ATOL = 1e-15

"""
Small positive time offset to avoid boundary issues in CDF grid evaluation.

When building CDF grids for quantile computation, we start at this offset
instead of exactly 0 to avoid potential singularities (e.g., Weibull shape < 1).
"""
const CDF_GRID_START = 1e-6

"""
Tolerance for detecting flat CDF regions during interpolation.

When |CDF(t_hi) - CDF(t_lo)| < this value, the CDF is considered flat
and we return t_lo directly instead of interpolating.
"""
const CDF_INTERPOLATION_TOL = 1e-12

"""
Tolerance for determining whether a parameter is at its bound.

A parameter θ is considered "at bound" if |θ - bound| < this tolerance.
Used in optimization post-processing to identify active constraints.
"""
const ACTIVE_CONSTRAINT_TOL = 1e-6

# =============================================================================
# Hazard Numerical Guards
# =============================================================================

"""
Large hazard value returned when Weibull hazard would be Inf.

For Weibull hazards with shape < 1, h(0) = ∞ mathematically.
We clamp to this large but finite value to prevent NaN propagation.
Likelihood optimization will naturally avoid such parameter combinations.
"""
const WEIBULL_MAX_HAZARD = 1e300

"""
Maximum exponent argument to prevent overflow in Gompertz hazard.

Gompertz hazard h(t) = rate * exp(shape * t) overflows when shape * t > ~709 (for Float64).
We clamp the exponent to prevent Inf and return the maximum representable hazard.
"""
const GOMPERTZ_MAX_EXPONENT = 700.0

# =============================================================================
# Spline Transform Conditioning
# =============================================================================

"""
Warning threshold for I-spline transformation matrix condition number.

When cond(L) exceeds this threshold, a warning is issued because the inverse
transformation (used in rectify_coefs! and coefficient recovery) may lose
significant precision. Common causes:
- Very closely spaced knots
- Very large number of spline basis functions
- Poorly chosen knot placement

Default: 1e10 (reasonable for Float64 precision ~1e-16)
"""
const ISPLINE_CONDITION_WARNING_THRESHOLD = 1e10

# =============================================================================
# Debug Mode Configuration
# =============================================================================

"""
Enable expensive debug assertions (invariant checks).

Set via environment variable `MSM_DEBUG_ASSERTIONS=true` before loading the package.
When enabled, additional runtime checks verify:
- Hazard non-negativity (M12_P2)
- TPM row sums (M11_P2)
- Other mathematical invariants

Disabled by default for performance. Enable during development/debugging:
```bash
export MSM_DEBUG_ASSERTIONS=true
julia -e 'using MultistateModels'
```
"""
const MSM_DEBUG_ASSERTIONS = get(ENV, "MSM_DEBUG_ASSERTIONS", "false") == "true"

# =============================================================================
# Optimization Constants (Ipopt and related)
# =============================================================================

"""
Default Ipopt convergence tolerance.

This is the primary convergence criterion for the interior point method.
Smaller values give more precise solutions but may require more iterations.
"""
const IPOPT_DEFAULT_TOL = 1e-7

"""
Default Ipopt acceptable tolerance.

When the optimization reaches this tolerance for `acceptable_iter` consecutive
iterations, Ipopt terminates with "acceptable" status. This provides a fallback
when strict convergence is difficult.
"""
const IPOPT_ACCEPTABLE_TOL = 1e-5

"""
Default number of acceptable iterations before termination.

After this many consecutive iterations at acceptable tolerance, Ipopt
terminates even if the strict tolerance has not been met.
"""
const IPOPT_ACCEPTABLE_ITER = 10

"""
Ipopt bound relaxation factor.

Controls how tightly bounds are enforced. Smaller values give tighter
bound enforcement. Default 1e-10 is tighter than Ipopt's default (1e-8)
to ensure parameters stay strictly within bounds for hazard positivity.
"""
const IPOPT_BOUND_RELAX_FACTOR = 1e-10

"""
Ipopt bound push parameter.

Controls how far the initial point is pushed away from bounds.
Smaller values allow starting closer to bounds.
"""
const IPOPT_BOUND_PUSH = 1e-4

"""
Ipopt bound fraction parameter.

Relative push from bounds as a fraction of the bound range.
"""
const IPOPT_BOUND_FRAC = 0.001

# =============================================================================
# Eigenvalue and Matrix Conditioning
# =============================================================================

"""
Threshold for treating eigenvalues as zero/near-zero.

Used in matrix operations (Cholesky downdates, eigendecompositions) to
determine when an eigenvalue should be treated as numerically zero.
"""
const EIGENVALUE_ZERO_TOL = 1e-10

"""
Small positive regularization for ill-conditioned matrices.

When matrices are near-singular, add this value to the diagonal to
improve numerical stability.
"""
const MATRIX_REGULARIZATION_EPS = 1e-6

# =============================================================================
# Importance Sampling
# =============================================================================

"""
Threshold for detecting degenerate importance weights.

When the range of log-importance weights is below this threshold,
the weights are considered degenerate (all equal) and resampling
may be needed.
"""
const IMPORTANCE_WEIGHT_RANGE_TOL = 1e-10

"""
Minimum lower bound for surrogate parameters.

Used when generating bounds for Markov surrogate parameters to ensure
all rates remain strictly positive.
"""
const SURROGATE_PARAM_MIN = 1e-6

# =============================================================================
# Smoothing Parameter Selection
# =============================================================================

"""
Default inner optimization tolerance for λ selection.

Used in Newton-type steps during smoothing parameter selection.
Slightly looser than main optimization to improve speed.
"""
const LAMBDA_SELECTION_INNER_TOL = 1e-6

"""
Cholesky downdate tolerance.

Used in rank-one Cholesky downdates to detect when the update would
make the matrix non-positive-definite.
"""
const CHOLESKY_DOWNDATE_TOL = 1e-10

# =============================================================================
# Constants Access Documentation
# =============================================================================
# These constants are not exported by default to avoid namespace pollution.
# Access them via qualified names:
#
#   MultistateModels.IPOPT_DEFAULT_TOL
#   MultistateModels.ACTIVE_CONSTRAINT_TOL
#   MultistateModels.TPM_ROW_SUM_TOL
#
# For custom tolerances in your code:
#   const MY_TOL = MultistateModels.IPOPT_DEFAULT_TOL
# =============================================================================
