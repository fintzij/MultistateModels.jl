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
