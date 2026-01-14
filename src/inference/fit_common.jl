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
- `compute_vcov::Bool=true`: compute model-based variance-covariance matrix
- `vcov_threshold::Bool=true`: use adaptive threshold for pseudo-inverse
- `compute_ij_vcov::Bool=true`: compute infinitesimal jackknife (sandwich) variance
- `compute_jk_vcov::Bool=false`: compute jackknife variance
- `loo_method::Symbol=:direct`: method for LOO perturbations (`:direct` or `:cholesky`)

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
- `MultistateModelFitted`: fitted model with estimates and variance matrices

# Model Type Detection

The fitting method is selected via traits:
- `is_panel_data(model)`: true if constructed with panel/interval-censored observations
- `is_markov(model)`: true if all hazards are Markov (no sojourn time dependence)

# Examples
```julia
# Exact data (obstype=1 in data)
exact_model = multistatemodel(h12, h23; data=exact_data)
fitted = fit(exact_model)  # Automatic penalty for spline hazards

# Explicit unpenalized fitting
fitted_unpn = fit(exact_model; penalty=:none)

# Panel data with Markov hazards (obstype=2, exponential/Weibull hazards)
markov_panel = multistatemodel(h12_exp, h23_exp; data=panel_data)
fitted = fit(markov_panel)

# Panel data with semi-Markov hazards (Gompertz or non-Markov)
semimarkov = multistatemodel(h12_gom, h23_wei; data=panel_data)
fitted = fit(semimarkov; proposal=:markov, maxiter=50)
```

See also: [`get_vcov`](@ref), [`compare_variance_estimates`](@ref),
          [`is_markov`](@ref), [`is_panel_data`](@ref), [`SplinePenalty`](@ref)
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
- `print_level=0`: Suppress output
- `honor_original_bounds="yes"`: Ensure solution exactly satisfies box constraints
- `bound_relax_factor=1e-10`: Tight bound relaxation (default 1e-8 can be too loose)
- `bound_push=1e-4`: Don't push initial point far from bounds
- `bound_frac=0.001`: Small relative push from bounds
- `mu_strategy="adaptive"`: More robust barrier parameter strategy
- `tol=1e-7`: Convergence tolerance
- `acceptable_tol=1e-5`: "Good enough" tolerance
- `acceptable_iter=10`: Iterations at acceptable before terminating

Users can override any of these by passing keyword arguments to `_solve_optimization`.
"""
const DEFAULT_IPOPT_OPTIONS = (
    # Output control
    print_level = 0,
    
    # Boundary handling (critical for constrained parameters like β ≥ 0)
    honor_original_bounds = "yes",
    bound_relax_factor = 1e-10,
    bound_push = 1e-4,
    bound_frac = 0.001,
    
    # Adaptive barrier (more robust for difficult problems)
    mu_strategy = "adaptive",
    
    # Convergence tolerances
    tol = 1e-7,
    acceptable_tol = 1e-5,
    acceptable_iter = 10,
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
    _solver = isnothing(solver) ? IpoptOptimizer() : solver
    if _is_ipopt_solver(solver)
        # Merge defaults with user overrides (user options take precedence)
        merged_options = merge(DEFAULT_IPOPT_OPTIONS, ipopt_options)
        return solve(prob, _solver; merged_options...)
    else
        # Optim.jl and other solvers don't support Ipopt options
        return solve(prob, _solver)
    end
end

"""
    _clamp_to_bounds!(params::AbstractVector, lb::AbstractVector, ub::AbstractVector)

Clamp parameters to bounds in-place.

This function is needed when parameters may exceed bounds, e.g., after extrapolation
steps or when working with unbounded optimization algorithms.

Note: This should NOT be needed after Ipopt optimization when configured with
`honor_original_bounds="yes"`, which guarantees the final solution satisfies bounds.
If Ipopt returns out-of-bounds values, that indicates a configuration bug.

# Arguments
- `params`: Parameter vector to clamp (modified in-place)
- `lb`: Lower bounds
- `ub`: Upper bounds
- `eps`: Small buffer to keep parameters away from exact boundaries (default: 1e-8).
         This prevents infinite gradients when warm-starting optimization from
         parameters exactly on the boundary.
"""
function _clamp_to_bounds!(params::AbstractVector, lb::AbstractVector, ub::AbstractVector; eps::Float64=1e-8)
    for i in eachindex(params)
        # Clamp to [lb + eps, ub - eps] to avoid exact boundary values
        # which can cause infinite gradients in warm-started optimization
        lb_safe = lb[i] + eps
        ub_safe = ub[i] - eps
        # Only apply buffer if bounds are finite and have room for it
        if isfinite(lb[i]) && isfinite(ub[i]) && lb_safe < ub_safe
            params[i] = clamp(params[i], lb_safe, ub_safe)
        else
            # Fall back to exact bounds for infinite bounds or very tight constraints
            params[i] = clamp(params[i], lb[i], ub[i])
        end
    end
    return params
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
    elseif penalty === nothing
        # Deprecated - warn and treat as :none
        @warn "penalty=nothing is deprecated. Use penalty=:none for explicit unpenalized fitting, " *
              "or penalty=:auto (default) for automatic penalization of spline hazards."
        return nothing
    else
        # User-specified penalty (SplinePenalty or Vector{SplinePenalty})
        return penalty
    end
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
- `tol::Float64=1e-6`: Tolerance for determining if constraint is active

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
function identify_active_constraints(theta::AbstractVector, constraints; tol::Float64=1e-6)
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
- `tol::Float64=1e-6`: Tolerance for determining if at bound

# Returns
- `BitVector`: Boolean vector where `true` indicates parameter is at a bound

# Example
```julia
at_bounds = identify_bound_parameters(fitted_params, lb, ub)
if any(at_bounds)
    @warn "Parameters at bounds: \$(findall(at_bounds))"
end
```
"""
function identify_bound_parameters(theta::AbstractVector, lb::AbstractVector, ub::AbstractVector; tol::Float64=1e-6)
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
- `tol::Float64=1e-6`: Tolerance for bound detection

# Returns
- `BitVector`: Indicators of which parameters are at bounds

# Side Effects
- Issues `@warn` if any parameters are at their bounds
"""
function warn_bound_parameters(theta::AbstractVector, lb::AbstractVector, ub::AbstractVector; tol::Float64=1e-6)
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
