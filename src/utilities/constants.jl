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
