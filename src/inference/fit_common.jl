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
fitted = fit(exact_model)

# Panel data with Markov hazards (obstype=2, exponential/Weibull hazards)
markov_panel = multistatemodel(h12_exp, h23_exp; data=panel_data)
fitted = fit(markov_panel)

# Panel data with semi-Markov hazards (Gompertz or non-Markov)
semimarkov = multistatemodel(h12_gom, h23_wei; data=panel_data)
fitted = fit(semimarkov; proposal=:markov, maxiter=50)
```

See also: [`get_vcov`](@ref), [`compare_variance_estimates`](@ref),
          [`is_markov`](@ref), [`is_panel_data`](@ref)
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
