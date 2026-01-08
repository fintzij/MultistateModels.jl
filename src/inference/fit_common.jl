# =============================================================================
# Model Fitting Entry Point and Dispatch
# =============================================================================
#
# This file contains:
# - Main fit() docstring and dispatch function
# - Common helpers (_is_ipopt_solver, _solve_optimization)
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
