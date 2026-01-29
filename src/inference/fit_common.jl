# =============================================================================
# Model Fitting Entry Point and Dispatch
# =============================================================================
#
# This file contains:
# - Main fit() docstring and dispatch function
# - Common helpers (_is_ipopt_solver, _solve_optimization)
# - Constraint analysis helpers (identify_active_constraints, etc.)
# - The routing logic that selects _fit_exact, _fit_markov_panel, or _fit_mcem
#
# Related files:
# - fit_exact.jl: Exact (continuously observed) data fitting
# - fit_markov.jl: Markov panel data fitting (matrix exponential likelihood)
# - fit_mcem.jl: Semi-Markov panel fitting via Monte Carlo EM
#
# =============================================================================

"""
    fit(model::MultistateModel; kwargs...)

Fit a multistate model to data.

This is the unified entry point for model fitting. The appropriate fitting method is 
automatically selected based on the model's data type and hazard structure:

- **Exact data** (continuously observed): Direct MLE via Ipopt optimization
- **Panel data + Markov hazards**: Matrix exponential likelihood via Ipopt
- **Panel data + Semi-Markov hazards**: Monte Carlo EM (MCEM) algorithm

# Common Arguments
- `constraints`: parameter constraints (see Constraints documentation)
- `verbose::Bool=true`: print optimization/convergence messages
- `solver`: optimization solver (default: Ipopt)
- `adtype`: AD backend for gradients/Hessians (default: `:auto` selects based on solver).
  Use `Optimization.AutoForwardDiff()` for small models (<100 params),
  `Optimization.AutoReverseDiff()` or `Optimization.AutoZygote()` for large models.
- `penalty`: Penalty specification for spline hazards. Can be:
  - `:auto` (default): Apply `SplinePenalty()` if model has spline hazards, `nothing` otherwise
  - `:none`: Explicit opt-out for unpenalized fitting
  - `SplinePenalty()`: Curvature penalty on all spline hazards
  - `Vector{SplinePenalty}`: Multiple rules resolved by specificity
  - `nothing`: DEPRECATED - use `:none` instead
- `lambda_init::Float64=1.0`: Initial smoothing parameter value
- `vcov_type::Symbol=:ij`: Type of variance-covariance matrix to compute:
  - `:ij` (default): Infinitesimal jackknife / sandwich variance (robust, always computable)
  - `:model`: Model-based variance (inverse Fisher information, requires unconstrained)
  - `:jk`: Jackknife variance (leave-one-out, computationally expensive)
  - `:none`: Skip variance computation
- `vcov_threshold::Bool=true`: use adaptive threshold for pseudo-inverse

# Exact Data Arguments  
- `parallel::Bool=false`: enable parallel likelihood evaluation
- `nthreads::Union{Nothing,Int}=nothing`: number of threads for parallel execution

# Markov Panel Arguments
- `adbackend::ADBackend=ForwardDiffBackend()`: automatic differentiation backend

# MCEM Arguments (Semi-Markov Panel)
- `proposal::Union{Symbol,ProposalConfig}=:auto`: proposal distribution (`:markov` or `:phasetype`)
- `maxiter::Int=100`: maximum MCEM iterations
- `tol::Float64=0.01`: convergence tolerance
- `ess_target_initial::Int=50`: initial effective sample size target
- `max_ess::Int=10000`: maximum ESS
- See `_fit_mcem` for full argument list

# Returns
- `MultistateModelFitted`: fitted model with estimates and variance-covariance matrix

# Model Type Detection

The fitting method is selected via traits:
- `is_panel_data(model)`: true if constructed with panel/interval-censored observations
- `is_markov(model)`: true if all hazards are Markov (no sojourn time dependence)

# Variance Estimation

The `vcov_type` argument controls which variance estimator is computed:
- **IJ (sandwich)**: `H⁻¹ J H⁻¹` where H is Hessian, J is outer product of scores.
  Robust to model misspecification; always computable including with constraints.
  Parameters at active constraints get `NaN` variance with a warning.
- **Model-based**: `H⁻¹` (inverse Fisher information). Assumes correct model specification.
- **Jackknife**: Leave-one-out resampling variance. Computationally intensive.

For constrained models (including phase-type), IJ variance is recommended as it
handles active constraints by setting affected variances to `NaN`.

# Examples
```julia
# Exact data with default IJ variance (obstype=1 in data)
exact_model = multistatemodel(h12, h23; data=exact_data)
fitted = fit(exact_model)  # vcov_type=:ij by default
vcov = get_vcov(fitted)    # Returns IJ variance

# Model-based variance for unconstrained model
fitted = fit(model; vcov_type=:model)

# Skip variance computation for speed
fitted = fit(model; vcov_type=:none)

# Panel data with semi-Markov hazards
semimarkov = multistatemodel(h12_gom, h23_wei; data=panel_data)
fitted = fit(semimarkov; proposal=:markov, maxiter=50)
```

See also: [`get_vcov`](@ref), [`is_markov`](@ref), [`is_panel_data`](@ref), [`SplinePenalty`](@ref)
"""

# =============================================================================
# Internal helpers for solver-agnostic optimization
# =============================================================================

"""
    _is_ipopt_solver(solver)

Check if solver is Ipopt (supports Ipopt-specific options).
"""
_is_ipopt_solver(solver) = isnothing(solver) || solver isa IpoptOptimizer

"""
    DEFAULT_IPOPT_OPTIONS

Default Ipopt options for robust optimization with proper boundary handling.

These defaults are tuned for multistate model fitting:
- `print_level=0`: Suppress iteration output
- `honor_original_bounds="yes"`: Ensure solution exactly satisfies box constraints
- `bound_relax_factor=1e-10`: Tight bound relaxation (default 1e-8 can be too loose)
- `bound_push=1e-4`: Don't push initial point far from bounds
- `bound_frac=0.001`: Small relative push from bounds
- `mu_strategy="adaptive"`: More robust barrier parameter strategy
- `tol=1e-7`: Convergence tolerance
- `acceptable_tol=1e-5`: "Good enough" tolerance
- `acceptable_iter=10`: Iterations at acceptable before terminating

Note: The Ipopt banner is suppressed via `additional_options=Dict("sb"=>"yes")` 
in the IpoptOptimizer constructor.

Users can override any of these by passing keyword arguments to `_solve_optimization`.
"""
const DEFAULT_IPOPT_OPTIONS = (
    # Output control
    print_level = 0,
    
    # Boundary handling (critical for constrained parameters like β ≥ 0)
    honor_original_bounds = "yes",
    bound_relax_factor = IPOPT_BOUND_RELAX_FACTOR,
    bound_push = IPOPT_BOUND_PUSH,
    bound_frac = IPOPT_BOUND_FRAC,
    
    # Adaptive barrier (more robust for difficult problems)
    mu_strategy = "adaptive",
    
    # Convergence tolerances
    tol = IPOPT_DEFAULT_TOL,
    acceptable_tol = IPOPT_ACCEPTABLE_TOL,
    acceptable_iter = IPOPT_ACCEPTABLE_ITER,
)

"""
    _solve_optimization(prob, solver; ipopt_options...)

Solve optimization problem with solver-appropriate options.

For Ipopt, uses robust defaults from `DEFAULT_IPOPT_OPTIONS` which can be
overridden by passing keyword arguments:

```julia
_solve_optimization(prob, solver; tol=1e-8, mu_strategy="monotone")
```

See `DEFAULT_IPOPT_OPTIONS` for the full list of default options.
"""
function _solve_optimization(prob, solver; ipopt_options...)
    if _is_ipopt_solver(solver)
        # Create IpoptOptimizer with sb="yes" to suppress banner
        # This must be in additional_options, not solve() kwargs
        _solver = IpoptOptimizer(additional_options=Dict("sb" => "yes"))
        # Merge defaults with user overrides (user options take precedence)
        merged_options = merge(DEFAULT_IPOPT_OPTIONS, ipopt_options)
        return solve(prob, _solver; merged_options...)
    else
        # Non-Ipopt solver passed directly
        _solver = isnothing(solver) ? IpoptOptimizer(additional_options=Dict("sb" => "yes")) : solver
        return solve(prob, _solver)
    end
end

# =============================================================================
# Variance-Covariance Computation Helpers
# =============================================================================

"""
Tolerance for zeroing near-zero vcov entries.

After computing the pseudo-inverse of the Fisher information,
entries smaller than this tolerance (in absolute value) are set to zero.
This prevents numerical noise from appearing as spurious correlations.
"""
const VCOV_NEAR_ZERO_ATOL = sqrt(eps(Float64))

"""
Relative tolerance for zeroing near-zero vcov entries.

Used together with VCOV_NEAR_ZERO_ATOL for adaptive thresholding.
"""
const VCOV_NEAR_ZERO_RTOL = sqrt(eps(Float64))

"""
    _compute_vcov_tolerance(n_obs::Int, n_params::Int, use_adaptive::Bool)

Compute appropriate tolerance for pseudo-inverse of Fisher information matrix.

# Arguments
- `n_obs::Int`: Number of observations (subjects or sample paths)
- `n_params::Int`: Number of parameters
- `use_adaptive::Bool`: If true, use data-dependent adaptive tolerance;
                       if false, use conservative default

# Returns
- `Float64`: Tolerance value for `pinv(...; atol=...)`

# Mathematical Justification

The adaptive formula `(log(n) × p)^(-2)` is motivated by the following considerations:

1. **Fisher Information Scaling**: The Fisher information matrix I(θ) scales with 
   sample size as O(n). For well-specified models, eigenvalues of I(θ) are O(n).

2. **Inverse Scaling**: The variance-covariance matrix Var(θ̂) = I(θ)⁻¹ scales as O(1/n).
   Its eigenvalues are O(1/n).

3. **Numerical Stability**: Eigenvalues much smaller than O(1/n) typically indicate
   either numerical error or near-singularity (e.g., parameters at boundary, 
   collinearity). These should be truncated via pseudo-inverse.

4. **Tolerance Choice**: Setting tolerance ≈ (log(n) × p)⁻² ensures:
   - For n=100, p=10: tol ≈ 0.0005 (truncates eigenvalues < 0.05% of expected scale)
   - For n=1000, p=20: tol ≈ 0.00003 (more stringent for larger samples)
   - The log(n) factor provides conservative downward adjustment as n grows
   - The p factor accounts for higher-dimensional parameter spaces being more
     prone to numerical issues

5. **Connection to Condition Number**: For a well-conditioned Fisher information
   with condition number κ = O(1), eigenvalues span a range where the smallest
   is still O(n). The formula truncates only when eigenvalues fall orders of
   magnitude below this expected scale.

When `use_adaptive=false`, returns `sqrt(eps(Float64))` ≈ 1.5×10⁻⁸ which is a 
conservative machine-precision-based default. This works well for most cases but 
may fail to truncate problematic small eigenvalues in certain ill-conditioned scenarios.
"""
function _compute_vcov_tolerance(n_obs::Int, n_params::Int, use_adaptive::Bool)
    if use_adaptive
        return (log(n_obs) * n_params)^-2
    else
        return sqrt(eps(Float64))
    end
end

"""
    _clean_vcov_matrix!(vcov::AbstractMatrix)

Zero out near-zero entries in variance-covariance matrix.

Modifies `vcov` in-place to set entries that are approximately zero
(within `VCOV_NEAR_ZERO_ATOL` and `VCOV_NEAR_ZERO_RTOL`) to exactly zero.
This removes numerical noise from appearing as spurious correlations.

# Arguments
- `vcov::AbstractMatrix`: Variance-covariance matrix to clean

# Returns
- The modified matrix
"""
function _clean_vcov_matrix!(vcov::AbstractMatrix)
    vcov[isapprox.(vcov, 0.0; atol=VCOV_NEAR_ZERO_ATOL, rtol=VCOV_NEAR_ZERO_RTOL)] .= 0.0
    return vcov
end

# =============================================================================
# Penalty Resolution
# =============================================================================

"""
    _resolve_penalty(penalty, model::MultistateProcess) -> Union{Nothing, SplinePenalty, Vector{SplinePenalty}}

Resolve the penalty specification to a concrete penalty configuration.

# Arguments
- `penalty`: User specification - can be:
  - `:auto` (default): Apply `SplinePenalty()` if model has spline hazards, `nothing` otherwise
  - `:none`: Explicit opt-out, equivalent to no penalty
  - `nothing`: DEPRECATED - emits warning, treated as `:none`
  - `SplinePenalty`: Explicit penalty specification
  - `Vector{SplinePenalty}`: Multiple penalty rules
- `model::MultistateProcess`: The model to check for spline hazards

# Returns
- `nothing` if no penalty should be applied
- `SplinePenalty()` or the user-specified penalty otherwise

# Deprecation Warning
Using `penalty=nothing` is deprecated. Use `penalty=:none` for explicit unpenalized fitting.
The default behavior is now `penalty=:auto`, which automatically applies appropriate
penalization for spline hazards.

See also: [`SplinePenalty`](@ref), [`has_spline_hazards`](@ref)
"""
function _resolve_penalty(penalty, model::MultistateProcess)
    if penalty === :auto
        # Automatic: penalize if model has spline hazards
        if has_spline_hazards(model)
            return SplinePenalty()
        else
            return nothing
        end
    elseif penalty === :none
        # Explicit opt-out
        return nothing
    elseif isnothing(penalty)
        # Deprecated - warn and treat as :none
        @warn "penalty=nothing is deprecated. Use penalty=:none for explicit unpenalized fitting, " *
              "or penalty=:auto (default) for automatic penalization of spline hazards." maxlog=1
        return nothing
    else
        # User-specified penalty (SplinePenalty or Vector{SplinePenalty})
        return penalty
    end
end

"""
    _resolve_selector(select_lambda::Symbol, penalty::AbstractPenalty) -> AbstractHyperparameterSelector

Resolve the smoothing parameter selection method specification to a concrete selector type.

# Arguments
- `select_lambda::Symbol`: User specification:
  - `:none`: No selection (use fixed λ from lambda_init)
  - `:pijcv` (default): Newton-approximated LOO-CV (Wood 2024 NCV)
  - `:pijcv5`, `:pijcv10`, `:pijcv20`: k-fold Newton-approximated CV
  - `:loocv`: Exact leave-one-out cross-validation
  - `:cv5`, `:cv10`, `:cv20`: Exact k-fold cross-validation
  - `:efs`: REML/EFS criterion
  - `:perf`: PERF criterion (Marra & Radice 2020)
- `penalty::AbstractPenalty`: The resolved penalty configuration

# Returns
- `AbstractHyperparameterSelector`: Concrete selector type

# Notes
- If penalty is `NoPenalty`, returns `NoSelection()` regardless of select_lambda
- The `:pijlcv` alias is accepted as equivalent to `:pijcv`

See also: [`AbstractHyperparameterSelector`](@ref), [`PIJCVSelector`](@ref), [`ExactCVSelector`](@ref)
"""
function _resolve_selector(select_lambda::Symbol, penalty::AbstractPenalty)
    # No penalty means no selection needed
    penalty isa NoPenalty && return NoSelection()
    
    # Map symbol to selector type
    return if select_lambda == :none
        NoSelection()
    elseif select_lambda == :pijcv || select_lambda == :pijlcv
        PIJCVSelector(0)  # LOO (0 means leave-one-out)
    elseif select_lambda == :pijcv_fast || select_lambda == :pijcvq
        PIJCVSelector(0, true)  # LOO with fast quadratic approximation (V_q)
    elseif select_lambda == :pijcv5
        PIJCVSelector(5)
    elseif select_lambda == :pijcv10
        PIJCVSelector(10)
    elseif select_lambda == :pijcv20
        PIJCVSelector(20)
    elseif select_lambda == :loocv
        ExactCVSelector(0)  # 0 = n_subjects (leave-one-out)
    elseif select_lambda == :cv5
        ExactCVSelector(5)
    elseif select_lambda == :cv10
        ExactCVSelector(10)
    elseif select_lambda == :cv20
        ExactCVSelector(20)
    elseif select_lambda == :efs
        REMLSelector()
    elseif select_lambda == :perf
        PERFSelector()
    else
        throw(ArgumentError("Unknown select_lambda: :$select_lambda. " *
            "Valid options are: :none, :pijcv, :pijcv_fast, :pijcv5, :pijcv10, :pijcv20, " *
            ":loocv, :cv5, :cv10, :cv20, :efs, :perf"))
    end
end

"""
    _validate_vcov_type(vcov_type::Symbol) -> Symbol

Validate and return the vcov_type.

# Arguments
- `vcov_type::Symbol`: Type of variance (:ij, :model, :jk, :none)

# Returns
- `Symbol`: The validated vcov_type

# Throws
- `ArgumentError` if vcov_type is not valid
"""
function _validate_vcov_type(vcov_type::Symbol)
    if !(vcov_type in (:ij, :model, :jk, :none))
        throw(ArgumentError("vcov_type must be :ij, :model, :jk, or :none (got :$vcov_type)"))
    end
    return vcov_type
end

"""
    _resolve_adtype(adtype, solver)

Resolve the AD backend based on user specification and solver requirements.

# Arguments
- `adtype`: User specification - `:auto`, an ADType, or `nothing`
- `solver`: Optimization solver

# Returns
- ADType suitable for the solver

# Behavior
- `:auto` (default): Uses ForwardDiff with SecondOrder for Newton/Ipopt solvers
- Explicit ADType: Used as-is (user takes responsibility for Hessian support)
- For reverse-mode AD with Ipopt, consider using a gradient-only method or
  providing Hessian approximation via `Optimization.AutoForwardDiff()` for Hessian.

# Future: For large models (>100 params), use:
```julia
fit(model; adtype=Optimization.AutoReverseDiff())  # or AutoZygote()
```
"""
function _resolve_adtype(adtype, solver)
    if adtype === :auto
        # Default behavior: ForwardDiff, with SecondOrder for solvers that need Hessians
        base_ad = Optimization.AutoForwardDiff()
        if isnothing(solver) || solver isa IpoptOptimizer || 
           (isdefined(Optim, :Newton) && solver isa Optim.Newton) ||
           (isdefined(Optim, :NewtonTrustRegion) && solver isa Optim.NewtonTrustRegion)
            return DifferentiationInterface.SecondOrder(base_ad, base_ad)
        else
            return base_ad
        end
    else
        # User-specified ADType - use as-is
        return adtype
    end
end

# =============================================================================
# Constraint Analysis Helpers
# =============================================================================

"""
    identify_active_constraints(theta, constraints; tol=1e-6)

Identify which constraints are active (binding) at the parameter vector `theta`.

For inequality constraints `lcons ≤ c(θ) ≤ ucons`, a constraint is active if
`c(θ)` is within `tol` of either bound.

# Arguments
- `theta::AbstractVector`: Parameter vector at which to evaluate constraints
- `constraints`: Constraint specification with fields:
  - `cons_fn`: Function that evaluates constraint values c(θ)
  - `lcons`: Lower bounds on constraint values
  - `ucons`: Upper bounds on constraint values
- `tol::Float64=ACTIVE_CONSTRAINT_TOL`: Tolerance for determining if constraint is active

# Returns
- `BitVector`: Boolean vector where `true` indicates constraint is active (binding)

# Notes
- For equality constraints (lcons[i] == ucons[i]), the constraint is always active if satisfied
- For inequality constraints, active means c(θ) ≈ lcons or c(θ) ≈ ucons

# Example
```julia
active = identify_active_constraints(fitted_params, constraints)
n_active = sum(active)  # Number of binding constraints
```
"""
function identify_active_constraints(theta::AbstractVector, constraints; tol::Float64=ACTIVE_CONSTRAINT_TOL)
    # Evaluate constraints at theta
    c_vals = constraints.cons_fn(theta)
    
    # Check if each constraint is at its bound
    at_lower = abs.(c_vals .- constraints.lcons) .< tol
    at_upper = abs.(c_vals .- constraints.ucons) .< tol
    
    return at_lower .| at_upper
end

"""
    compute_constraint_jacobian(theta, constraints)

Compute the Jacobian of constraint functions at `theta` using automatic differentiation.

The Jacobian J has dimensions (p_c × p) where p_c is the number of constraints
and p is the number of parameters:
```math
J_{ij} = \\frac{\\partial c_i}{\\partial \\theta_j}
```

# Arguments
- `theta::AbstractVector`: Parameter vector at which to evaluate Jacobian
- `constraints`: Constraint specification with field `cons_fn`

# Returns
- `Matrix{Float64}`: p_c × p Jacobian matrix

# Example
```julia
J = compute_constraint_jacobian(fitted_params, constraints)
J_active = J[active_constraints, :]  # Jacobian of active constraints only
```
"""
function compute_constraint_jacobian(theta::AbstractVector, constraints)
    return ForwardDiff.jacobian(constraints.cons_fn, theta)
end

"""
    identify_bound_parameters(theta, lb, ub; tol=1e-6)

Identify parameters that are at (or very close to) their box bounds.

Parameters at bounds indicate that the optimization found a boundary solution,
which can affect variance estimates (the parameter cannot move in one direction).

# Arguments
- `theta::AbstractVector`: Parameter vector
- `lb::AbstractVector`: Lower bounds
- `ub::AbstractVector`: Upper bounds
- `tol::Float64=ACTIVE_CONSTRAINT_TOL`: Tolerance for determining if at bound

# Returns
- `BitVector`: Boolean vector indicating parameters at bounds

# Example
```julia
at_bounds = identify_bound_parameters(fitted_params, lb, ub)
if any(at_bounds)
    @warn "Parameters at bounds: \$(findall(at_bounds))"
end
```
"""
function identify_bound_parameters(theta::AbstractVector, lb::AbstractVector, ub::AbstractVector; tol::Float64=ACTIVE_CONSTRAINT_TOL)
    at_lower = abs.(theta .- lb) .< tol
    at_upper = abs.(theta .- ub) .< tol
    return at_lower .| at_upper
end

"""
    warn_bound_parameters(theta, lb, ub; tol=1e-6)

Check for parameters at bounds and issue a warning if any are found.

# Arguments
- `theta::AbstractVector`: Parameter vector  
- `lb::AbstractVector`: Lower bounds
- `ub::AbstractVector`: Upper bounds
- `tol::Float64=ACTIVE_CONSTRAINT_TOL`: Tolerance for bound detection

# Returns
- `BitVector`: Indicators of which parameters are at bounds

# Side Effects
- Issues `@warn` if any parameters are at their bounds
"""
function warn_bound_parameters(theta::AbstractVector, lb::AbstractVector, ub::AbstractVector; tol::Float64=ACTIVE_CONSTRAINT_TOL)
    at_bounds = identify_bound_parameters(theta, lb, ub; tol=tol)
    if any(at_bounds)
        bound_params = findall(at_bounds)
        @warn "Parameters $(bound_params) are at their box bounds. " *
              "Variance estimates for these parameters may be unreliable. " *
              "Consider widening bounds or using IJ (sandwich) variance."
    end
    return at_bounds
end

# =============================================================================

function fit(model::MultistateModel; kwargs...)
    # =========================================================================
    # THREAD SAFETY WARNING
    # =========================================================================
    # fit() is NOT thread-safe. Calling fit() concurrently on the same model
    # from multiple threads will cause data races and undefined behavior because:
    #   1. initialize_parameters!() may be called and modifies model.parameters
    #   2. The optimization process updates model state
    #   3. Surrogate fitting modifies model.surrogate
    #
    # Safe usage patterns:
    #   - Call fit() sequentially on the same model
    #   - Call fit() in parallel on DIFFERENT model instances (deepcopy first)
    #   - Use @threads with separate model copies per thread
    #
    # For parallel cross-validation or bootstrap, create independent model copies:
    #   models = [deepcopy(model) for _ in 1:nfolds]
    #   @threads for i in 1:nfolds
    #       fitted[i] = fit(models[i]; ...)
    #   end
    # =========================================================================
    
    # Validate model.bounds exists (required for optimization)
    # This can be missing for models deserialized from old versions or manually constructed
    if !hasproperty(model, :bounds) || isnothing(model.bounds)
        # Generate bounds on-demand if possible
        if hasproperty(model, :hazards) && !isempty(model.hazards)
            @warn "model.bounds is missing. This can happen with manually constructed models " *
                  "or models deserialized from older versions. Generating bounds on-demand. " *
                  "For full validation, consider rebuilding the model with multistatemodel()."
            # Generate bounds using the standard function
            model.bounds = build_parameter_bounds(model.hazards, model.parameters)
        else
            throw(ArgumentError(
                "model.bounds is required for optimization but is missing. " *
                "This can happen with:\n" *
                "  1. Models deserialized from older MultistateModels.jl versions\n" *
                "  2. Manually constructed MultistateModel objects\n" *
                "Please rebuild the model using multistatemodel() to generate bounds."
            ))
        end
    end
    
    # Dispatch based on observation type and model structure
    if is_panel_data(model)
        if is_markov(model)
            return _fit_markov_panel(model; kwargs...)
        else
            return _fit_mcem(model; kwargs...)
        end
    else
        return _fit_exact(model; kwargs...)
    end
end

"""
    _fit_exact(model::MultistateModel; kwargs...)

Internal implementation: Fit a multistate model to continuously observed (exact) data.

This is called by `fit()` when `is_panel_data(model) == false`.
"""
