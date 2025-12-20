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

See also: [`get_vcov`](@ref), [`get_ij_vcov`](@ref), [`compare_variance_estimates`](@ref),
          [`is_markov`](@ref), [`is_panel_data`](@ref)
"""

# =============================================================================
# Internal helpers for solver-agnostic optimization
# =============================================================================

"""
    _is_ipopt_solver(solver)

Check if solver is Ipopt (supports print_level option).
"""
_is_ipopt_solver(solver) = isnothing(solver) || solver isa Ipopt.Optimizer

"""
    _solve_optimization(prob, solver)

Solve optimization problem with solver-appropriate options.
Ipopt supports print_level, but Optim.jl and others don't.
"""
function _solve_optimization(prob, solver)
    _solver = isnothing(solver) ? Ipopt.Optimizer() : solver
    if _is_ipopt_solver(solver)
        return solve(prob, _solver; print_level = 0)
    else
        # Optim.jl and other solvers don't support print_level
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
function _fit_exact(model::MultistateModel; constraints = nothing, verbose = true, solver = nothing, 
             parallel = false, nthreads = nothing,
             compute_vcov = true, vcov_threshold = true, compute_ij_vcov = true, 
             compute_jk_vcov = false, loo_method = :direct, kwargs...)

    # initialize array of sample paths
    samplepaths = extract_paths(model)

    # extract model parameters
    # Phase 3: Use ParameterHandling.jl flat parameters (log scale)
    parameters = get_parameters_flat(model)

    # Use model constraints if none provided and model has them
    if isnothing(constraints) && haskey(model.modelcall, :constraints) && !isnothing(model.modelcall.constraints)
        constraints = model.modelcall.constraints
    end

    # Configure threading
    n_paths = length(samplepaths)
    threading_config = ThreadingConfig(parallel=parallel, nthreads=nthreads)
    use_parallel = should_parallelize(threading_config, n_paths)
    
    if use_parallel && verbose
        println("Using $(threading_config.nthreads) threads for likelihood evaluation ($(n_paths) paths)")
    end

    # parse constraints, or not, and solve
    if isnothing(constraints) 
        # Create likelihood function - parallel or sequential
        # Note: ForwardDiff uses the function passed to OptimizationFunction for both
        # objective and gradient. We pass the sequential version for AD correctness,
        # but create a parallel version for objective-only evaluation.
        if use_parallel
            # Create wrapper that uses parallel evaluation
            # ForwardDiff will still use sequential for gradient computation
            loglik_fn = (params, data) -> loglik_exact(params, data; neg=true, parallel=true)
        else
            loglik_fn = loglik_exact
        end
        
        # get estimates - use Ipopt for unconstrained
        # Use SecondOrder AD if solver requires it (Newton, Ipopt) to avoid warnings
        adtype = Optimization.AutoForwardDiff()
        if isnothing(solver) || (solver isa Optim.Newton) || (solver isa Optim.NewtonTrustRegion) || (solver isa Ipopt.Optimizer)
            adtype = DifferentiationInterface.SecondOrder(Optimization.AutoForwardDiff(), Optimization.AutoForwardDiff())
        end

        optf = OptimizationFunction(loglik_fn, adtype)
        prob = OptimizationProblem(optf, parameters, ExactData(model, samplepaths))

        # solve with user-specified solver or default Ipopt
        sol = _solve_optimization(prob, solver)

        # rectify spline coefs
        if any(isa.(model.hazards, _SplineHazard))
            rectify_coefs!(sol.u, model)
        end
        
        # get vcov
        if compute_vcov && (sol.retcode == ReturnCode.Success) && !any(map(x -> (isa(x, _SplineHazard) && x.monotone != 0), model.hazards))
            # Compute subject-level gradients and Hessians to cache them for robust variance
            # This avoids redundant computation when computing robust variance estimates
            subject_grads_cache = compute_subject_gradients(sol.u, model, samplepaths)
            subject_hessians_cache = compute_subject_hessians(sol.u, model, samplepaths)
            
            # Aggregate Fisher information from subject Hessians
            nparams = length(sol.u)
            fishinf = zeros(Float64, nparams, nparams)
            for H_i in subject_hessians_cache
                fishinf .-= H_i
            end
            
            vcov = pinv(Symmetric(fishinf), atol = vcov_threshold ? (log(length(samplepaths)) * length(sol.u))^-2 : sqrt(eps(real(float(oneunit(eltype(fishinf)))))))
            vcov[isapprox.(vcov, 0.0; atol = sqrt(eps(Float64)), rtol = sqrt(eps(Float64)))] .= 0.0
            vcov = Symmetric(vcov)
        else
            subject_grads_cache = nothing
            subject_hessians_cache = nothing
            vcov = nothing
        end
    else
        # create constraint function and check that constraints are satisfied at the initial values
        _constraints = deepcopy(constraints)
        consfun_multistate = parse_constraints(_constraints.cons, model.hazards; consfun_name = :consfun_multistate)

        initcons = consfun_multistate(zeros(length(constraints.cons)), parameters, nothing)
        badcons = findall(initcons .< constraints.lcons .|| initcons .> constraints.ucons)
        if length(badcons) > 0
            error("Constraints $badcons are violated at the initial parameter values.")
        end

        # Create likelihood function - parallel or sequential
        if use_parallel
            loglik_fn = (params, data) -> loglik_exact(params, data; neg=true, parallel=true)
        else
            loglik_fn = loglik
        end

        # Use SecondOrder AD if solver requires it
        adtype = Optimization.AutoForwardDiff()
        if isnothing(solver) || (solver isa Optim.Newton) || (solver isa Optim.NewtonTrustRegion) || (solver isa Ipopt.Optimizer)
            adtype = DifferentiationInterface.SecondOrder(Optimization.AutoForwardDiff(), Optimization.AutoForwardDiff())
        end

        optf = OptimizationFunction(loglik_fn, adtype, cons = consfun_multistate)
        prob = OptimizationProblem(optf, parameters, ExactData(model, samplepaths), lcons = constraints.lcons, ucons = constraints.ucons)
        
        # solve with user-specified solver or default Ipopt
        sol = _solve_optimization(prob, solver)

        # rectify spline coefs
        if any(isa.(model.hazards, _SplineHazard))
            rectify_coefs!(sol.u, model)
        end

        # no hessian when there are constraints
        if compute_vcov == true
            @warn "No covariance matrix is returned when constraints are provided."
        end
        subject_grads_cache = nothing
        subject_hessians_cache = nothing
        vcov = nothing
    end

    # compute subject-level likelihood at the estimate
    ll_subj = loglik_exact(sol.u, ExactData(model, samplepaths); return_ll_subj = true)

    # compute robust variance estimates if requested
    ij_variance = nothing
    jk_variance = nothing
    subject_grads = nothing
    
    if (compute_ij_vcov || compute_jk_vcov) && !isnothing(vcov) && isnothing(constraints)
        if verbose
            println("Computing robust variance estimates...")
        end
        robust_result = compute_robust_vcov(sol.u, model, samplepaths;
                                           compute_ij = compute_ij_vcov,
                                           compute_jk = compute_jk_vcov,
                                           loo_method = loo_method,
                                           vcov_threshold = vcov_threshold,
                                           subject_grads = subject_grads_cache,
                                           subject_hessians = subject_hessians_cache)
        ij_variance = robust_result.ij_vcov
        jk_variance = robust_result.jk_vcov
        subject_grads = robust_result.subject_gradients
    end

    # Build parameters structure for fitted parameters
    # Use the unflatten function from the model to convert flat params back to nested
    params_nested = unflatten(model.parameters.reconstructor, sol.u)
    
    # Compute natural scale parameters (family-aware transformation)
    params_natural_pairs = [
        hazname => extract_natural_vector(params_nested[hazname], model.hazards[idx].family)
        for (hazname, idx) in sort(collect(model.hazkeys), by = x -> x[2])
    ]
    params_natural = NamedTuple(params_natural_pairs)
    
    parameters_fitted = (
        flat = Vector{Float64}(sol.u),
        nested = params_nested,
        natural = params_natural,
        reconstructor = model.parameters.reconstructor  # Keep same reconstructor
    )

    # Split sol.u into per-hazard log-scale vectors for spline remaking
    natural_vals = values(model.parameters.natural)
    block_sizes = [length(v) for v in natural_vals]
    log_scale_params = Vector{Vector{Float64}}(undef, length(block_sizes))
    offset = 0
    for i in eachindex(block_sizes)
        log_scale_params[i] = sol.u[(offset+1):(offset+block_sizes[i])]
        offset += block_sizes[i]
    end

    # wrap results
    model_fitted = MultistateModelFitted(
        model.data,
        parameters_fitted,
        (loglik = -sol.objective, subj_lml = ll_subj),
        vcov,
        isnothing(ij_variance) ? nothing : Matrix(ij_variance),
        isnothing(jk_variance) ? nothing : Matrix(jk_variance),
        subject_grads,
        model.hazards,
        model.totalhazards,
        model.tmat,
        model.emat,
        model.hazkeys,
        model.subjectindices,
        model.SubjectWeights,
        model.ObservationWeights,
        model.CensoringPatterns,
        model.markovsurrogate,
        (solution = sol,), # ConvergenceRecords::Union{Nothing, NamedTuple}
        nothing, # ProposedPaths::Union{Nothing, NamedTuple}
        model.modelcall,
        model.phasetype_expansion)

    # remake splines and calculate risk periods using log-scale parameters
    for i in eachindex(model_fitted.hazards)
        if isa(model_fitted.hazards[i], _SplineHazard)
            remake_splines!(model_fitted.hazards[i], log_scale_params[i])
            set_riskperiod!(model_fitted.hazards[i])
        end
    end

    # return fitted object
    return model_fitted;
end

# NOTE: fit(::PhaseTypeModel) has been removed as part of package streamlining.
# Phase-type hazards are now handled internally via MultistateModel with
# phasetype_expansion metadata. The standard fit() method handles expansion.

"""
    _fit_markov_panel(model::MultistateModel; kwargs...)

Internal implementation: Fit a Markov model to interval-censored/panel data.

This is called by `fit()` when `is_panel_data(model) && is_markov(model)`.

# Arguments
- `model::MultistateModel`: Markov model with panel observations
- `constraints`: parameter constraints (see Constraints documentation)
- `verbose::Bool=true`: print optimization messages
- `solver`: optimization solver (default: Ipopt)
- `adbackend::ADBackend=ForwardDiffBackend()`: automatic differentiation backend

# Notes
- For Markov models, the likelihood involves matrix exponentials of the intensity matrix
- Censored state observations are handled via marginalization over possible states
"""
function _fit_markov_panel(model::MultistateModel; constraints = nothing, verbose = true, solver = nothing, adbackend::ADBackend = ForwardDiffBackend(), compute_vcov = true, vcov_threshold = true, compute_ij_vcov = true, compute_jk_vcov = false, loo_method = :direct, kwargs...)

    # containers for bookkeeping TPMs
    books = build_tpm_mapping(model.data)

    # extract model parameters
    # Phase 3: Use ParameterHandling.jl flat parameters (log scale)
    parameters = get_parameters_flat(model)

    # number of subjects
    nsubj = length(model.subjectindices)

    # Use model constraints if none provided and model has them
    if isnothing(constraints) && haskey(model.modelcall, :constraints) && !isnothing(model.modelcall.constraints)
        constraints = model.modelcall.constraints
    end

    # Warn if using reverse-mode AD for Markov models (matrix exponential issue)
    if adbackend isa MooncakeBackend
        @warn "MooncakeBackend may fail for Markov models due to LAPACK calls in matrix exponential. " *
              "ChainRules.jl's exp rule also uses LAPACK internally. Use ForwardDiffBackend() if you encounter errors."
    end

    # Create closure that dispatches to correct implementation based on backend
    loglik_fn = (p, d) -> loglik_markov(p, d; backend=adbackend)

    # parse constraints, or not, and solve
    if isnothing(constraints)
        # get estimates - use Ipopt for unconstrained
        optf = OptimizationFunction(loglik_fn, get_optimization_ad(adbackend))
        prob = OptimizationProblem(optf, parameters, MPanelData(model, books))
        
        # solve with user-specified solver or default Ipopt
        sol = _solve_optimization(prob, solver)

        # get vcov
        if compute_vcov && (sol.retcode == ReturnCode.Success)
            # Compute subject-level gradients and Hessians to cache them for robust variance
            # This avoids redundant computation when computing robust variance estimates
            subject_grads_cache = compute_subject_gradients(sol.u, model, books)
            subject_hessians_cache = compute_subject_hessians(sol.u, model, books)
            
            # Aggregate Fisher information from subject Hessians
            nparams = length(sol.u)
            fishinf = zeros(Float64, nparams, nparams)
            for H_i in subject_hessians_cache
                fishinf .-= H_i
            end
            
            vcov = pinv(Symmetric(fishinf), atol = vcov_threshold ? (log(nsubj) * length(sol.u))^-2 : sqrt(eps(real(float(oneunit(eltype(fishinf)))))))
            vcov[isapprox.(vcov, 0.0; atol = eps(Float64))] .= 0.0
            vcov = Symmetric(vcov)
        else
            subject_grads_cache = nothing
            subject_hessians_cache = nothing
            vcov = nothing
        end
    else
        # create constraint function and check that constraints are satisfied at the initial values
        _constraints = deepcopy(constraints)
        consfun_markov = parse_constraints(_constraints.cons, model.hazards; consfun_name = :consfun_markov)

        initcons = consfun_markov(zeros(length(constraints.cons)), parameters, nothing)
        badcons = findall(initcons .< constraints.lcons .|| initcons .> constraints.ucons)
        if length(badcons) > 0
            error("Constraints $badcons are violated at the initial parameter values.")
        end

        optf = OptimizationFunction(loglik_fn, get_optimization_ad(adbackend), cons = consfun_markov)
        prob = OptimizationProblem(optf, parameters, MPanelData(model, books), lcons = constraints.lcons, ucons = constraints.ucons)
        
        # solve with user-specified solver or default Ipopt
        sol = _solve_optimization(prob, solver)

        # no hessian when there are constraints
        if compute_vcov == true
            @warn "No covariance matrix is returned when constraints are provided."
        end
        subject_grads_cache = nothing
        subject_hessians_cache = nothing
        vcov = nothing
    end

    # compute loglikelihood at the estimate
    logliks = (loglik = -sol.objective, subj_lml = loglik_markov(sol.u, MPanelData(model, books); return_ll_subj = true))

    # compute robust variance estimates if requested
    ij_variance = nothing
    jk_variance = nothing
    subject_grads = nothing
    
    if (compute_ij_vcov || compute_jk_vcov) && !isnothing(vcov) && isnothing(constraints)
        if verbose
            println("Computing robust variance estimates...")
        end
        robust_result = compute_robust_vcov(sol.u, model, books;
                                           compute_ij = compute_ij_vcov,
                                           compute_jk = compute_jk_vcov,
                                           loo_method = loo_method,
                                           vcov_threshold = vcov_threshold,
                                           subject_grads = subject_grads_cache,
                                           subject_hessians = subject_hessians_cache)
        ij_variance = robust_result.ij_vcov
        jk_variance = robust_result.jk_vcov
        subject_grads = robust_result.subject_gradients
    end

    # Build parameters structure for fitted parameters
    # Use the unflatten function from the model to convert flat params back to nested
    params_nested = unflatten(model.parameters.reconstructor, sol.u)
    
    # Compute natural scale parameters (family-aware transformation)
    params_natural_pairs = [
        hazname => extract_natural_vector(params_nested[hazname], model.hazards[idx].family)
        for (hazname, idx) in sort(collect(model.hazkeys), by = x -> x[2])
    ]
    params_natural = NamedTuple(params_natural_pairs)
    
    parameters_fitted = (
        flat = Vector{Float64}(sol.u),
        nested = params_nested,
        natural = params_natural,
        reconstructor = model.parameters.reconstructor  # Keep same reconstructor
    )

    # wrap results
    return MultistateModelFitted(
        model.data,
        parameters_fitted,
        logliks,
        vcov,
        isnothing(ij_variance) ? nothing : Matrix(ij_variance),
        isnothing(jk_variance) ? nothing : Matrix(jk_variance),
        subject_grads,
        model.hazards,
        model.totalhazards,
        model.tmat,
        model.emat,
        model.hazkeys,
        model.subjectindices,
        model.SubjectWeights,
        model.ObservationWeights,
        model.CensoringPatterns,
        model.markovsurrogate,
        (solution = sol,), # ConvergenceRecords::Union{Nothing, NamedTuple}
        nothing, # ProposedPaths::Union{Nothing, NamedTuple}
        model.modelcall,
        model.phasetype_expansion);
end


"""
    _fit_mcem(model::MultistateModel; kwargs...)

Internal implementation: Fit a semi-Markov model to panel data via Monte Carlo EM (MCEM).

This is called by `fit()` when `is_panel_data(model) && !is_markov(model)`.

Uses Ipopt optimization for both constrained and unconstrained M-steps by default.

!!! note "Surrogate Required"
    MCEM requires a Markov surrogate for importance sampling proposals. 
    You must call `set_surrogate!(model)` or use `surrogate=:markov` in 
    `multistatemodel()` before fitting.

# Algorithm

The MCEM algorithm iterates between:
1. **E-step**: Sample latent paths via importance sampling from a Markov surrogate
2. **M-step**: Maximize the expected complete-data log-likelihood

Convergence is assessed using the ascent-based stopping rule from Caffo et al. (2005),
with Pareto-smoothed importance sampling (PSIS) for stable weight estimation.

# References

- Morsomme, R., Liang, C. J., Mateja, A., Follmann, D. A., O'Brien, M. P., Wang, C.,
  & Fintzi, J. (2025). Assessing treatment efficacy for interval-censored endpoints
  using multistate semi-Markov models fit to multiple data streams. Biostatistics,
  26(1), kxaf038. https://doi.org/10.1093/biostatistics/kxaf038
- Wei, G. C., & Tanner, M. A. (1990). A Monte Carlo implementation of the 
  EM algorithm and the poor man's data augmentation algorithms. JASA, 85(411), 699-704.
- Caffo, B. S., Jank, W., & Jones, G. L. (2005). Ascent-based Monte Carlo 
  expectation-maximization. JRSS-B, 67(2), 235-251.
- Vehtari, A., Simpson, D., Gelman, A., Yao, Y., & Gabry, J. (2024). 
  Pareto Smoothed Importance Sampling. JMLR, 25(72), 1-58.
- Varadhan, R., & Roland, C. (2008). Simple and globally convergent methods 
  for accelerating the convergence of any EM algorithm. Scand. J. Stat., 35(2), 335-353.

# Arguments

**Model and constraints:**
- `model::MultistateSemiMarkovProcess`: semi-Markov model
- `constraints`: parameter constraints tuple

**Optimization:**
- `solver`: optimization solver for M-step (default: Ipopt for both constrained and unconstrained).
  See [Optimization Solvers](@ref) for available options.

**MCEM algorithm control:**
- `maxiter::Int=100`: maximum MCEM iterations
- `tol::Float64=1e-2`: tolerance for MLL change in stopping rule
- `ascent_threshold::Float64=0.1`: standard normal quantile for ascent lower bound
- `stopping_threshold::Float64=0.1`: standard normal quantile for stopping criterion
- `ess_growth_factor::Float64=sqrt(2.0)`: multiplicative factor for ESS increase when more paths needed
- `ess_increase_method::Symbol=:fixed`: method for increasing ESS. Options:
  - `:fixed` (default): multiply ESS by `ess_growth_factor` each time ascent is not confirmed
  - `:adaptive`: power-based calculation from Caffo et al. (2005) Equation 15 at iteration start
    to adaptively set ESS based on Monte Carlo variance; uses `ess_growth_factor` for mid-iteration
    increases when `ascent_lb < 0`
- `ascent_alpha::Float64=0.25`: type I error rate for adaptive (Caffo) power calculation
- `ascent_beta::Float64=0.25`: type II error rate for adaptive (Caffo) power calculation
- `ess_target_initial::Int=50`: initial effective sample size target per subject
- `max_ess::Int=10000`: maximum ESS before stopping for non-convergence
- `max_sampling_effort::Int=20`: maximum factor of ESS for additional path sampling
- `npaths_additional::Int=10`: increment for additional paths when augmenting
- `block_hessian_speedup::Float64=2.0`: minimum theoretical speedup required to use block-diagonal
  Hessian optimization. The Hessian is block-diagonal because each hazard's parameters only appear
  in its own cumulative hazard term (∂²L/∂θₐ∂θᵦ = 0 when θₐ, θᵦ belong to different hazards).
  For models with k hazards of sizes b₁,...,bₖ, block-diagonal inversion is O(Σbᵢ³) vs O(n³) for
  dense, where n = Σbᵢ. Set higher to prefer dense computation; set to 1.0 to always use blocks
- `acceleration::Symbol=:none`: acceleration method for MCEM. Options:
  - `:none` (default): standard MCEM without acceleration
  - `:squarem`: SQUAREM acceleration (Varadhan & Roland, 2008), applies quasi-Newton 
    extrapolation every 2 iterations to speed up convergence

**Sampling Importance Resampling (SIR):**
- `sir::Symbol=:adaptive_lhs`: SIR resampling method. Options:
  - `:none`: standard importance sampling without resampling
  - `:sir`: multinomial resampling from importance weights (from iteration 1)
  - `:lhs`: Latin Hypercube Sampling on the CDF, variance-reduced (from iteration 1)
  - `:adaptive_sir`: start with IS, switch to SIR when cost-effective
  - `:adaptive_lhs` (default): start with IS, switch to LHS when cost-effective
- `sir_pool_constant::Float64=2.0`: pool size multiplier (pool = c × m × log(m))
- `sir_max_pool::Int=8192`: maximum pool size cap
- `sir_resample::Symbol=:always`: when to resample. Options:
  - `:always`: resample every iteration
  - `:degeneracy`: resample only when Pareto-k exceeds threshold
- `sir_degeneracy_threshold::Float64=0.7`: Pareto-k threshold for `:degeneracy` mode
- `sir_adaptive_threshold::Float64=2.0`: for adaptive modes, minimum ratio of 
  (mean paths / ESS target) weighted by optimizer iterations before switching
- `sir_adaptive_min_iters::Int=3`: for adaptive modes, minimum iterations before 
  considering switch (allows stable ratio estimates)

**Output control:**
- `verbose::Bool=true`: print progress messages
- `return_convergence_records::Bool=true`: save iteration history
- `return_proposed_paths::Bool=false`: save latent paths and importance weights

**Variance estimation:**
- `compute_vcov::Bool=true`: compute model-based variance via Louis's identity (for diagnostics)
- `vcov_threshold::Bool=true`: use adaptive threshold for pseudo-inverse
- `compute_ij_vcov::Bool=true`: compute IJ/sandwich variance (robust, **recommended for inference**)
- `compute_jk_vcov::Bool=false`: compute jackknife variance
- `loo_method::Symbol=:direct`: method for LOO perturbations Δᵢ (jackknife only, not IJ):
  - `:direct`: Δᵢ = H⁻¹gᵢ (faster)
  - `:cholesky`: exact H₋ᵢ⁻¹ via Cholesky downdates (stable)

# Returns
- `MultistateModelFitted`: fitted model with MCEM solution and variance estimates

# Variance Estimation (Default Behavior)

By default, both model-based and IJ (sandwich) variance are computed:
- **IJ variance** (`compute_ij_vcov=true`): Primary for inference (robust to misspecification)
- **Model-based** (`compute_vcov=true`): For diagnostics via Louis's identity

For semi-Markov models, the observed Fisher information is computed using Louis's identity,
which accounts for the missing data (unobserved paths):
```
I_obs = E[I_comp | Y] - Var[S_comp | Y]
```

# Example
```julia
# Default fit: robust and model-based variance computed
fitted = fit(semimarkov_model; ess_target_initial=100, verbose=true)

# Use robust SEs for inference
robust_se = sqrt.(diag(get_ij_vcov(fitted)))

# Diagnose model specification
result = compare_variance_estimates(fitted)

# Check convergence
records = get_convergence_records(fitted)

# Use a custom solver for the M-step
using Optim
fitted = fit(semimarkov_model; solver=Optim.BFGS())

# Use SQUAREM acceleration for faster convergence
fitted = fit(semimarkov_model; acceleration=:squarem)
```

See also: [`fit`](@ref), [`compare_variance_estimates`](@ref)
"""
function _fit_mcem(model::MultistateModel; proposal::Union{Symbol, ProposalConfig} = :auto, constraints = nothing, solver = nothing, maxiter = 100, tol = 1e-2, ascent_threshold = 0.1, stopping_threshold = 0.1, ess_growth_factor = sqrt(2.0), ess_increase_method::Symbol = :fixed, ascent_alpha::Float64 = 0.25, ascent_beta::Float64 = 0.25, ess_target_initial = 50, max_ess = 10000, max_sampling_effort = 20, npaths_additional = 10, block_hessian_speedup = 2.0, acceleration::Symbol = :squarem, sir::Symbol = :adaptive_lhs, sir_pool_constant::Float64 = 2.0, sir_max_pool::Int = 8192, sir_resample::Symbol = :always, sir_degeneracy_threshold::Float64 = 0.7, sir_adaptive_threshold::Float64 = 2.0, sir_adaptive_min_iters::Int = 3, verbose = true, return_convergence_records = true, return_proposed_paths = false, compute_vcov = true, vcov_threshold = true, compute_ij_vcov = true, compute_jk_vcov = false, loo_method = :direct, kwargs...)

    # Validate acceleration parameter
    if acceleration ∉ (:none, :squarem)
        error("acceleration must be :none or :squarem, got :$acceleration")
    end
    use_squarem = acceleration === :squarem
    
    if verbose && use_squarem
        println("Using SQUAREM acceleration for MCEM.\n")
    end

    # Validate SIR parameters
    if sir ∉ (:none, :sir, :lhs, :adaptive_sir, :adaptive_lhs)
        error("sir must be :none, :sir, :lhs, :adaptive_sir, or :adaptive_lhs, got :$sir")
    end
    if sir_resample ∉ (:always, :degeneracy)
        error("sir_resample must be :always or :degeneracy, got :$sir_resample")
    end
    if sir_pool_constant <= 0
        error("sir_pool_constant must be positive, got $sir_pool_constant")
    end
    if sir_max_pool <= 0
        error("sir_max_pool must be positive, got $sir_max_pool")
    end
    if !(0 < sir_degeneracy_threshold < 1)
        error("sir_degeneracy_threshold must be in (0,1), got $sir_degeneracy_threshold")
    end
    if sir_adaptive_threshold <= 0
        error("sir_adaptive_threshold must be positive, got $sir_adaptive_threshold")
    end
    if sir_adaptive_min_iters < 1
        error("sir_adaptive_min_iters must be at least 1, got $sir_adaptive_min_iters")
    end
    
    # Derive SIR mode flags from sir parameter
    # use_sir: whether SIR is currently active (starts false for adaptive modes)
    # sir_adaptive: whether adaptive switching is enabled
    # sir_target_method: the SIR method to use (:sir or :lhs)
    sir_adaptive = sir in (:adaptive_sir, :adaptive_lhs)
    sir_target_method = sir in (:sir, :adaptive_sir) ? :sir : :lhs
    use_sir = sir in (:sir, :lhs)  # Immediate SIR, not adaptive
    
    if verbose && use_sir
        println("Using SIR ($sir) with pool constant $sir_pool_constant, max pool $sir_max_pool.\n")
    elseif verbose && sir_adaptive
        println("Using adaptive SIR (will switch to $sir_target_method when cost-effective).\n")
    end

    # Resolve proposal configuration
    proposal_config = resolve_proposal_config(proposal, model)
    use_phasetype = proposal_config.type === :phasetype
    
    if verbose && use_phasetype
        println("Using phase-type proposal for MCEM importance sampling.\n")
    end

    # copy of data
    data_original = deepcopy(model.data)

    # Use model constraints if none provided and model has them
    if isnothing(constraints) && haskey(model.modelcall, :constraints) && !isnothing(model.modelcall.constraints)
        constraints = model.modelcall.constraints
    end

    # check that constraints for the initial values are satisfied
    if !isnothing(constraints)
        # create constraint function and check that constraints are satisfied at the initial values
        _constraints = deepcopy(constraints)
        consfun_semimarkov = parse_constraints(_constraints.cons, model.hazards; consfun_name = :consfun_semimarkov)

        # Phase 3: Use ParameterHandling.jl flat parameters for constraint check
        initcons = consfun_semimarkov(zeros(length(constraints.cons)), get_parameters_flat(model), nothing)
        badcons = findall(initcons .< constraints.lcons .|| initcons .> constraints.ucons)
        if length(badcons) > 0
            error("Constraints $badcons are violated at the initial parameter values.")
        end
    end

    # check that max_sampling_effort is greater than 1
    if max_sampling_effort <= 1
        error("max_sampling_effort must be greater than 1.")
    end

    # check that ess_growth_factor is greater than 1
    if ess_growth_factor <= 1
        error("ess_growth_factor must be greater than 1.")
    end

    # Validate ess_increase_method parameter
    if ess_increase_method ∉ (:fixed, :adaptive)
        error("ess_increase_method must be :fixed or :adaptive, got :$ess_increase_method")
    end
    if ascent_alpha <= 0 || ascent_alpha >= 1
        error("ascent_alpha must be in (0,1), got $ascent_alpha")
    end
    if ascent_beta <= 0 || ascent_beta >= 1
        error("ascent_beta must be in (0,1), got $ascent_beta")
    end
    use_adaptive_ess = ess_increase_method === :adaptive
    
    if verbose && use_adaptive_ess
        println("Using adaptive (Caffo) power-based ESS increase (α=$ascent_alpha, β=$ascent_beta).\n")
    end

    # throw a warning if trying to fit a spline model where the degree is 0 for all splines
    if all(map(x -> (isa(x, _MarkovHazard) | (isa(x, _SplineHazard) && (x.degree == 0) && (length(x.knots) == 2))), model.hazards))
        error("Attempting to fit a time-homogeneous Markov model via MCEM. Recode as exponential hazards and refit.")
    end

    # MCEM initialization
    keep_going = true
    iter = 0
    is_converged = false

    # number of subjects
    nsubj = length(model.subjectindices)

    # identify absorbing states
    absorbingstates = findall(map(x -> all(x .== 0), eachrow(model.tmat)))

    # extract and initialize model parameters
    # Phase 3: Use ParameterHandling.jl flat parameters (log scale)
    params_cur = get_parameters_flat(model)

    # initialize ess target
    ess_target = ess_target_initial
    
    # Adaptive ESS method: track m_start for each iteration (m_{t,start} in Caffo et al. 2005)
    # At iteration t+1: m_{t+1,start} = max(m_{t,start}, m_t)
    adaptive_m_start = Float64(ess_target_initial)
    
    # Pre-compute z-quantiles for adaptive (Caffo) power calculation
    adaptive_z_sum_sq = use_adaptive_ess ? (quantile(Normal(), 1-ascent_alpha) + quantile(Normal(), 1-ascent_beta))^2 : 0.0

    # containers for latent sample paths, proposal and target log likelihoods, importance sampling weights
    ess_cur = zeros(nsubj)
    psis_pareto_k = zeros(nsubj)

    # initialize containers
    samplepaths     = [sizehint!(Vector{SamplePath}(), ess_target_initial * max_sampling_effort * 20) for i in 1:nsubj]

    # surrogate log likelihood
    loglik_surrog   = [sizehint!(Vector{Float64}(undef, 0), 
        ess_target_initial * max_sampling_effort * 2) for i in 1:nsubj]

    # target log likelihood - current parameters
    loglik_target_cur  = [sizehint!(Vector{Float64}(undef, 0), 
        ess_target_initial * max_sampling_effort * 2) for i in 1:nsubj]

    # target log likelihood - proposed parameters
    loglik_target_prop = [sizehint!(Vector{Float64}(undef, 0), 
        ess_target_initial * max_sampling_effort * 2) for i in 1:nsubj]

    # Log (unnormalize) importance weights
    _logImportanceWeights  = [sizehint!(Vector{Float64}(undef, 0), 
        ess_target_initial * max_sampling_effort * 2) for i in 1:nsubj]

    # exponentiated and normalized importance weights
    ImportanceWeights  = [sizehint!(Vector{Float64}(undef, 0), 
        ess_target_initial * max_sampling_effort * 2) for i in 1:nsubj]

    # SIR infrastructure
    sir_subsample_indices = [Vector{Int}() for _ in 1:nsubj]  # Indices into pool for each subject
    sir_pool_cap_exceeded = false  # Flag for convergence records warning
    sir_pool_target = use_sir ? sir_pool_size(ess_target, sir_pool_constant, sir_max_pool) : 0
    
    # Adaptive SIR state: track path evaluation ratios to decide when to switch
    # The key metric is: (mean paths per subject) / ess_target weighted by optimizer iterations
    # When this ratio exceeds threshold, switching to SIR reduces M-step cost
    adaptive_sir_switched = false           # Has switch occurred?
    adaptive_sir_switch_iter = 0            # Iteration when switch occurred
    adaptive_sir_cumul_path_ratio = 0.0     # Cumulative path ratio (weighted by optim iters)
    adaptive_sir_cumul_optim_iters = 0      # Cumulative optimizer iterations
    
    # make fbmats if necessary
    if any(model.data.obstype .> 2)
        fbmats = build_fbmats(model)
    else
        fbmats = nothing
    end

    # containers for traces
    mll_trace = Vector{Float64}() # marginal loglikelihood
    ess_trace = ElasticArray{Float64, 2}(undef, nsubj, 0) # effective sample size (one per subject)
    path_count_trace = ElasticArray{Int, 2}(undef, nsubj, 0) # actual path counts (one per subject)
    # Phase 3: Use ParameterHandling.jl flat parameter length
    parameters_trace = ElasticArray{Float64, 2}(undef, length(get_parameters_flat(model)), 0) # parameter estimates

    # Require a pre-built Markov surrogate for MCEM
    # Users should call set_surrogate!(model) or use surrogate=:markov in multistatemodel() beforehand
    if isnothing(model.markovsurrogate)
        error("MCEM requires a Markov surrogate. Call `set_surrogate!(model)` or use `surrogate=:markov` in `multistatemodel()` before fitting.")
    end
    
    # Check if surrogate needs to be fitted (not yet fitted)
    # This happens when surrogate=:markov is used with fit_surrogate=false
    if !model.markovsurrogate.fitted
        if verbose
            println("Markov surrogate not yet fitted. Fitting via MLE...")
        end
        # Fit the surrogate via set_surrogate! with MLE method
        set_surrogate!(model; type=:markov, method=:mle, verbose=verbose)
    end
    
    markov_surrogate = model.markovsurrogate
    if verbose
        println("Using model's Markov surrogate for MCEM.\n")
    end

    # Build phase-type surrogate if requested
    phasetype_surrogate = nothing
    tpm_book_ph = nothing
    hazmat_book_ph = nothing
    fbmats_ph = nothing
    emat_ph = nothing
    
    if use_phasetype
        phasetype_surrogate = fit_phasetype_surrogate(model, markov_surrogate; 
                                                       config=proposal_config, verbose=verbose)
    end
    
    # Use Markov surrogate for compatibility (phase-type uses it for sampling infrastructure)
    surrogate = markov_surrogate

     # containers for bookkeeping TPMs
     books = build_tpm_mapping(model.data)    

     # transition probability objects for Markov surrogate
     hazmat_book_surrogate = build_hazmat_book(Float64, model.tmat, books[1])
     tpm_book_surrogate = build_tpm_book(Float64, model.tmat, books[1])
 
     # allocate memory for matrix exponential
     cache = ExponentialUtilities.alloc_mem(similar(hazmat_book_surrogate[1]), ExpMethodGeneric())

    # Solve Kolmogorov equations for TPMs
    # Get natural-scale surrogate parameters for hazard evaluation (family-aware)
    surrogate_pars = get_hazard_params(surrogate.parameters, surrogate.hazards)
    for t in eachindex(books[1])
        # compute the transition intensity matrix
        compute_hazmat!(hazmat_book_surrogate[t], surrogate_pars, surrogate.hazards, books[1][t], model.data)
        # compute transition probability matrices
        compute_tmat!(tpm_book_surrogate[t], hazmat_book_surrogate[t], books[1][t], cache)
    end

    # Build phase-type infrastructure if using phase-type proposals
    # For exact observations, we need to expand the data to properly express
    # phase uncertainty during sojourn times
    expanded_ph_data = nothing
    ph_censoring_patterns = nothing
    ph_original_row_map = nothing
    ph_subjectindices = nothing
    expanded_ph_tpm_map = nothing  # tpm_map for expanded data
    
    if use_phasetype
        # Check if data needs expansion (has exact observations)
        if needs_data_expansion_for_phasetype(model.data)
            n_states = size(model.tmat, 1)
            expansion_result = expand_data_for_phasetype(model.data, n_states)
            expanded_ph_data = expansion_result.expanded_data
            ph_censoring_patterns = expansion_result.censoring_patterns
            ph_original_row_map = expansion_result.original_row_map
            ph_subjectindices = compute_expanded_subject_indices(expanded_ph_data)
            
            if verbose
                n_orig = nrow(model.data)
                n_exp = nrow(expanded_ph_data)
                println("  Expanded data for phase-type: $n_orig → $n_exp rows")
            end
        end
        
        # Build TPM book using original or expanded data
        data_for_ph = isnothing(expanded_ph_data) ? model.data : expanded_ph_data
        
        # Rebuild books for expanded data if needed
        if !isnothing(expanded_ph_data)
            books_ph = build_tpm_mapping(expanded_ph_data)
            expanded_ph_tpm_map = books_ph[2]  # Save tpm_map for sampling
            tpm_book_ph, hazmat_book_ph = build_phasetype_tpm_book(phasetype_surrogate, books_ph, expanded_ph_data)
        else
            tpm_book_ph, hazmat_book_ph = build_phasetype_tpm_book(phasetype_surrogate, books, model.data)
        end
        
        # Build fbmats with correct sizes (using expanded subject indices if available)
        subj_inds_for_ph = isnothing(ph_subjectindices) ? model.subjectindices : ph_subjectindices
        fbmats_ph = build_fbmats_phasetype_with_indices(subj_inds_for_ph, phasetype_surrogate)
        
        emat_ph = build_phasetype_emat_expanded(model, phasetype_surrogate;
                                                 expanded_data = expanded_ph_data,
                                                 censoring_patterns = ph_censoring_patterns)
    end

    # compute normalizing constant of proposal distribution
    # For Markov proposal: this is the log-likelihood under the Markov surrogate
    # For phase-type proposal: compute marginal likelihood via forward algorithm
    #   This is r(Y|θ') in the importance sampling formula:
    #   log f̂(Y|θ) = log r(Y|θ') + Σᵢ log(mean(νᵢ))
    if use_phasetype
        NormConstantProposal = compute_phasetype_marginal_loglik(
            model, phasetype_surrogate, emat_ph;
            expanded_data = expanded_ph_data,
            expanded_subjectindices = ph_subjectindices)
    else
        NormConstantProposal = compute_markov_marginal_loglik(model, markov_surrogate)
    end

    # draw sample paths until the target ess is reached 
    if verbose  println("Initializing sample paths ...\n") end
    
    # For SIR: sample pool_target paths instead of ess_target
    # After sampling, we'll resample ess_target indices from the pool
    sampling_target = use_sir ? sir_pool_target : ess_target
    
    DrawSamplePaths!(model; 
        ess_target = sampling_target, 
        ess_cur = ess_cur, 
        max_sampling_effort = max_sampling_effort,
        samplepaths = samplepaths, 
        loglik_surrog = loglik_surrog, 
        loglik_target_prop = loglik_target_prop, 
        loglik_target_cur = loglik_target_cur, 
        _logImportanceWeights = _logImportanceWeights, 
        ImportanceWeights = ImportanceWeights, 
        tpm_book_surrogate = tpm_book_surrogate, 
        hazmat_book_surrogate = hazmat_book_surrogate, 
        books = books, 
        npaths_additional = npaths_additional, 
        params_cur = params_cur, 
        surrogate = surrogate, 
        psis_pareto_k = psis_pareto_k,
        fbmats = fbmats,
        absorbingstates = absorbingstates,
        # Phase-type infrastructure (nothing if not using)
        phasetype_surrogate = phasetype_surrogate,
        tpm_book_ph = tpm_book_ph,
        hazmat_book_ph = hazmat_book_ph,
        fbmats_ph = fbmats_ph,
        emat_ph = emat_ph,
        # Expanded data infrastructure (nothing if not using/not needed)
        expanded_ph_data = expanded_ph_data,
        expanded_ph_subjectindices = ph_subjectindices,
        expanded_ph_tpm_map = expanded_ph_tpm_map,
        ph_original_row_map = ph_original_row_map)
    
    # Apply SIR: resample indices from pool and compute uniform weights
    if use_sir
        for i in 1:nsubj
            # Skip subjects with deterministic paths (single path or all equal logliks)
            if length(samplepaths[i]) <= 1 || allequal(loglik_surrog[i])
                sir_subsample_indices[i] = collect(1:length(samplepaths[i]))
            else
                # Resample ess_target indices from the pool using importance weights
                sir_subsample_indices[i] = get_sir_subsample_indices(
                    ImportanceWeights[i], ess_target, sir)
            end
        end
        # For SIR, ESS = subsample size (deterministic)
        fill!(ess_cur, Float64(ess_target))
    end
    
    # get current estimate of marginal log likelihood
    # For SIR: use subsampled paths with uniform weights
    if use_sir
        mll_cur = mcem_mll_sir(loglik_target_cur, sir_subsample_indices, model.SubjectWeights)
    else
        mll_cur = mcem_mll(loglik_target_cur, ImportanceWeights, model.SubjectWeights)
    end

    # generate optimization problem
    if isnothing(constraints)
        optf = OptimizationFunction(loglik, Optimization.AutoForwardDiff())
        prob = OptimizationProblem(optf, params_cur, SMPanelData(model, samplepaths, ImportanceWeights))
    else
        optf = OptimizationFunction(loglik, Optimization.AutoForwardDiff(), cons = consfun_semimarkov)
        prob = OptimizationProblem(optf, params_cur, SMPanelData(model, samplepaths, ImportanceWeights), lcons = constraints.lcons, ucons = constraints.ucons)
    end

    # print output
    if verbose
        println("Initial target ESS: $(round(ess_target;digits=2)) per-subject")
        println("Initial range of the number of sample paths per-subject: [$(ceil(ess_target)), $(maximum(length.(samplepaths)))]")
        println("Initial estimate of the marginal log-likelihood, Q: $(round(mll_cur;digits=3))")
        println("Initial estimate of the log marginal likelihood: $(round(compute_loglik(model, loglik_surrog, loglik_target_cur, NormConstantProposal).loglik;digits=3))\n")
        println("Starting Monte Carlo EM...\n")
    end
    
    # initialize MCEM
    mll_prop = mll_cur
    mll_change = 0.0
    ase = 0.0
    ascent_lb = 0.0
    ascent_ub = 0.0

    # Initialize SQUAREM state if using acceleration
    squarem_state = use_squarem ? SquaremState(length(params_cur)) : nothing

    # start algorithm
    while keep_going

        # increment the iteration
        iter += 1
        
        # =====================================================================
        # Adaptive ESS (Caffo et al. 2005): Set target ESS at START of iteration using Equation 15
        # m_k = ceil((z_α + z_β)² * σ̂²_k / Δ²)
        # where σ̂²_k = ase² is the estimated MC variance and Δ = tol
        # 
        # Key insight: Only increase ESS at start of iteration if:
        # 1. The previous iteration showed positive ascent (ascent_lb > 0)
        # 2. The ASE is small enough that we're near convergence (ASE < 5*tol)
        # This prevents aggressive ESS increases early when parameters are volatile.
        # =====================================================================
        if use_adaptive_ess && iter > 1 && ase > 0 && tol > 0 && ascent_lb > 0 && ase < 5 * tol
            adaptive_m_k = ceil(adaptive_z_sum_sq * ase^2 / tol^2)
            # Cap per-iteration increase to 2x to prevent degenerate behavior
            adaptive_m_k_capped = min(adaptive_m_k, 2.0 * ess_target)
            # m_{t+1,start} = max(m_{t,start}, m_t) - ESS never decreases
            adaptive_m_start = max(adaptive_m_start, ess_target)
            # New target is max of capped formula and m_start
            ess_target_new = max(adaptive_m_start, adaptive_m_k_capped)
            if ess_target_new > ess_target
                if verbose && adaptive_m_k > adaptive_m_k_capped
                    println("Adaptive start-of-iteration: m_k=$(Int(adaptive_m_k)) capped to $(Int(adaptive_m_k_capped)) (2x limit).")
                end
                ess_target = ess_target_new
                if verbose
                    println("Adaptive start-of-iteration: Target ESS set to $(Int(ess_target)) (m_k=$(Int(min(adaptive_m_k, adaptive_m_k_capped))), m_start=$(Int(adaptive_m_start))).\n")
                end
                # Update SIR pool target if using SIR
                if use_sir
                    new_pool_target = sir_pool_size(ess_target, sir_pool_constant, sir_max_pool)
                    if new_pool_target == sir_max_pool && !sir_pool_cap_exceeded
                        sir_pool_cap_exceeded = true
                        @warn "SIR pool size capped at $sir_max_pool; further ESS increases will reduce SIR effectiveness."
                    end
                    sir_pool_target = new_pool_target
                    
                    # Resample SIR indices with new ESS target
                    for i in 1:nsubj
                        if length(samplepaths[i]) <= 1 || allequal(loglik_surrog[i])
                            sir_subsample_indices[i] = collect(1:length(samplepaths[i]))
                        else
                            sir_subsample_indices[i] = get_sir_subsample_indices(
                                ImportanceWeights[i], Int(ess_target), sir)
                        end
                    end
                    fill!(ess_cur, Float64(ess_target))
                end
            end
        end
        
        # =====================================================================
        # SQUAREM: Save θ₀ at start of cycle (every 2 iterations)
        # =====================================================================
        if use_squarem && squarem_state.step == 0
            squarem_state.θ0 .= params_cur
            squarem_state.step = 1
        end
        
        # solve M-step: use user-specified solver or default Ipopt
        # For SIR: use subsampled paths with uniform weights
        if use_sir
            sir_data = create_sir_subsampled_data(samplepaths, sir_subsample_indices)
            mstep_data = SMPanelData(model, sir_data.paths, sir_data.weights)
        else
            mstep_data = SMPanelData(model, samplepaths, ImportanceWeights)
        end
        
        # Note: constraints handling happens in prob definition earlier
        # This if-else currently does the same thing - kept for future differentiation
        params_prop_optim = _solve_optimization(
            remake(prob, u0 = Vector(params_cur), p = mstep_data), 
            solver
        )
        params_prop = params_prop_optim.u
        
        # =====================================================================
        # Adaptive SIR: Track M-step path evaluation ratio
        # The key metric is (mean paths per subject) / ess_target, weighted by optimizer iterations
        # Each optimizer iteration evaluates the objective on all paths, so total evals ∝ n_iters × n_paths
        # With SIR, we'd evaluate on ess_target paths per subject instead
        # =====================================================================
        if sir_adaptive && !use_sir && !adaptive_sir_switched
            n_optim_iters = max(1, params_prop_optim.stats.iterations)  # At least 1
            mean_paths = mean(length.(samplepaths))
            # Ratio of M-step evaluations: current vs. what it would be with SIR
            # Current: n_optim_iters × total_paths
            # With SIR: n_optim_iters × (nsubj × ess_target)
            # Ratio per subject = mean_paths / ess_target
            path_ratio = mean_paths / ess_target
            
            # Accumulate weighted by optimizer iterations (more iterations = more cost savings)
            adaptive_sir_cumul_path_ratio += path_ratio * n_optim_iters
            adaptive_sir_cumul_optim_iters += n_optim_iters
        end

        # calculate the log likelihoods for the proposed parameters on FULL pool
        # (needed for importance weight recalculation)
        loglik!(params_prop, loglik_target_prop, SMPanelData(model, samplepaths, ImportanceWeights))
        
        # calculate the marginal log likelihood 
        # For SIR: use simple averages on subsampled paths
        if use_sir
            mll_cur  = mcem_mll_sir(loglik_target_cur, sir_subsample_indices, model.SubjectWeights)
            mll_prop = mcem_mll_sir(loglik_target_prop, sir_subsample_indices, model.SubjectWeights)
        else
            mll_cur  = mcem_mll(loglik_target_cur , ImportanceWeights, model.SubjectWeights)
            mll_prop = mcem_mll(loglik_target_prop, ImportanceWeights, model.SubjectWeights)
        end

        # compute the ALB and AUB
        if params_prop != params_cur

            # change in mll
            mll_change = mll_prop - mll_cur
    
            # calculate the ASE for ΔQ
            if use_sir
                ase = mcem_ase_sir(loglik_target_prop, loglik_target_cur, sir_subsample_indices, model.SubjectWeights)
            else
                ase = mcem_ase(loglik_target_prop, loglik_target_cur, ImportanceWeights, model.SubjectWeights)
            end
    
            # calculate the lower bound for ΔQ
            ascent_lb = quantile(Normal(mll_change, ase), ascent_threshold)
            ascent_ub = quantile(Normal(mll_change, ase), 1-stopping_threshold)
        else
            loglik_target_prop = loglik_target_cur
            mll_prop = mll_cur
            mll_change = 0
            ase = 0
            ascent_lb = 0
            ascent_ub = 0
            is_converged = true
        end

        # =====================================================================
        # SQUAREM: Apply acceleration every 2 iterations
        # =====================================================================
        if use_squarem && !is_converged
            if squarem_state.step == 1
                # After first EM step: save θ₁, continue to second step
                squarem_state.θ1 .= params_prop
                squarem_state.step = 2
                
                # Standard update for this iteration
                params_cur        = deepcopy(params_prop)
                mll_cur           = deepcopy(mll_prop)
                loglik_target_cur = deepcopy(loglik_target_prop)
                
            elseif squarem_state.step == 2
                # After second EM step: θ₂ = params_prop, now compute acceleration
                squarem_state.θ2 .= params_prop
                
                # Compute step length and acceleration vectors
                α, r, v = squarem_step_length(squarem_state.θ0, squarem_state.θ1, squarem_state.θ2)
                squarem_state.α = α
                
                if α == -1.0
                    # No acceleration possible (v too small), use standard EM update
                    params_cur        = deepcopy(params_prop)
                    mll_cur           = deepcopy(mll_prop)
                    loglik_target_cur = deepcopy(loglik_target_prop)
                    squarem_state.n_fallbacks += 1
                    if verbose
                        println("  [SQUAREM: no acceleration (Δ too small), using standard EM]\n")
                    end
                else
                    # Compute accelerated parameters
                    params_acc = squarem_accelerate(squarem_state.θ0, r, v, α)
                    
                    # Evaluate likelihood at accelerated point
                    loglik!(params_acc, loglik_target_prop, SMPanelData(model, samplepaths, ImportanceWeights))
                    if use_sir
                        mll_acc = mcem_mll_sir(loglik_target_prop, sir_subsample_indices, model.SubjectWeights)
                        mll_θ0 = mcem_mll_sir(loglik_target_cur, sir_subsample_indices, model.SubjectWeights)
                    else
                        mll_acc = mcem_mll(loglik_target_prop, ImportanceWeights, model.SubjectWeights)
                        mll_θ0 = mcem_mll(loglik_target_cur, ImportanceWeights, model.SubjectWeights)
                    end
                    
                    if squarem_should_accept(mll_acc, mll_prop, mll_θ0)
                        # Accept accelerated step
                        params_cur = params_acc
                        mll_cur = mll_acc
                        # loglik_target_cur already holds logliks at params_acc from loglik! call above
                        loglik_target_cur = deepcopy(loglik_target_prop)
                        squarem_state.n_accelerations += 1
                        
                        # Update mll_change for reporting
                        mll_change = mll_acc - mll_θ0
                        
                        if verbose
                            println("  [SQUAREM: acceleration accepted, α=$(round(α;sigdigits=3))]\n")
                        end
                    else
                        # Fallback to standard EM update (θ₂)
                        loglik!(squarem_state.θ2, loglik_target_prop, SMPanelData(model, samplepaths, ImportanceWeights))
                        params_cur        = deepcopy(squarem_state.θ2)
                        mll_cur           = deepcopy(mll_prop)
                        loglik_target_cur = deepcopy(loglik_target_prop)
                        squarem_state.n_fallbacks += 1
                        if verbose
                            println("  [SQUAREM: acceleration rejected (mll decreased), using θ₂]\n")
                        end
                    end
                end
                
                # Reset SQUAREM cycle
                squarem_state.step = 0
            end
        else
            # Standard MCEM (no SQUAREM) or converged
            # increment the current values of the parameter, marginal log likelihood, target loglik
            params_cur        = deepcopy(params_prop)
            mll_cur           = deepcopy(mll_prop)
            loglik_target_cur = deepcopy(loglik_target_prop)
        end

        # increment log importance weights, importance weights, effective sample size and pareto shape parameter
        ComputeImportanceWeightsESS!(loglik_target_cur, loglik_surrog, _logImportanceWeights, ImportanceWeights, ess_cur, ess_target, psis_pareto_k)

        # Note: psis_pareto_k > 0.7 indicates unreliable importance weights (Vehtari et al., 2024)
        # The values are stored in the returned convergence_records for diagnostic purposes

        # SIR: Check if resampling should occur based on mode
        if use_sir
            n_resampled = 0
            for i in 1:nsubj
                # Skip subjects with deterministic paths
                if length(samplepaths[i]) <= 1 || allequal(loglik_surrog[i])
                    continue
                end
                
                # Check if resampling is needed based on mode
                if should_resample(sir_resample, psis_pareto_k[i], sir_degeneracy_threshold)
                    sir_subsample_indices[i] = get_sir_subsample_indices(
                        ImportanceWeights[i], ess_target, sir)
                    n_resampled += 1
                end
            end
            
            # For SIR, ESS = subsample size (deterministic)
            fill!(ess_cur, Float64(ess_target))
            
            if verbose && n_resampled > 0
                mean_pareto_k = mean(psis_pareto_k)
                println("SIR: $n_resampled subjects resampled, mean Pareto-k = $(round(mean_pareto_k; sigdigits=3))")
            end
        end

        # print update
        if verbose
            println("Iteration: $iter")
            println("Target ESS: $(round(ess_target;digits=2)) per-subject")
            if use_sir
                println("Pool sizes per-subject: [$(minimum(length.(samplepaths))), $(maximum(length.(samplepaths)))]")
            else
                println("Range of the number of sampled paths per-subject: [$(ceil(ess_target)), $(max(length.(samplepaths)...))]")
            end
            println("Estimate of the marginal log-likelihood, Q: $(round(mll_cur;digits=3))")
            println("Gain in marginal log-likelihood, ΔQ: $(round(mll_change;sigdigits=3))")
            println("MCEM asymptotic standard error: $(round(ase;sigdigits=3))")
            println("Ascent lower and upper bound: [$(round(ascent_lb; sigdigits=3)), $(round(ascent_ub; sigdigits=3))]")
            println("Estimate of the log marginal likelihood, l(θ): $(round(compute_loglik(model, loglik_surrog, loglik_target_cur, NormConstantProposal).loglik;digits=3))")
            if use_squarem
                println("SQUAREM: $(squarem_state.n_accelerations) accelerations, $(squarem_state.n_fallbacks) fallbacks\n")
            else
                println()
            end
        end

        # save marginal log likelihood, parameters and effective sample size
        append!(parameters_trace, params_cur)
        push!(mll_trace, mll_cur)
        append!(ess_trace, ess_cur)
        append!(path_count_trace, length.(samplepaths))
        
        # =====================================================================
        # Adaptive SIR: Check if we should switch to LHS/SIR
        # Decision based on average ratio of path evaluations (current vs SIR)
        # weighted by optimizer iterations across all iterations so far
        # =====================================================================
        if sir_adaptive && !use_sir && !adaptive_sir_switched && iter >= sir_adaptive_min_iters
            # Compute average path ratio weighted by optimizer iterations
            avg_path_ratio = adaptive_sir_cumul_path_ratio / adaptive_sir_cumul_optim_iters
            
            if avg_path_ratio > sir_adaptive_threshold
                # Switch to LHS (or SIR if specified)
                use_sir = true
                sir = sir_target_method
                adaptive_sir_switched = true
                adaptive_sir_switch_iter = iter
                
                # Initialize SIR pool target
                sir_pool_target = sir_pool_size(ess_target, sir_pool_constant, sir_max_pool)
                
                # Initialize subsample indices from existing paths
                for i in 1:nsubj
                    if length(samplepaths[i]) <= 1 || allequal(loglik_surrog[i])
                        sir_subsample_indices[i] = collect(1:length(samplepaths[i]))
                    else
                        sir_subsample_indices[i] = get_sir_subsample_indices(
                            ImportanceWeights[i], Int(ess_target), sir)
                    end
                end
                fill!(ess_cur, Float64(ess_target))
                
                if verbose
                    println("\n[Adaptive SIR] Switching to $(uppercase(string(sir))) at iteration $iter")
                    println("  Average M-step path ratio: $(round(avg_path_ratio; digits=2)) (threshold: $sir_adaptive_threshold)")
                    println("  Mean paths/subject: $(round(mean(length.(samplepaths)); digits=1)), ESS target: $(Int(ess_target))")
                    println("  SIR pool target: $sir_pool_target\n")
                end
            end
        end

        # check for convergence
        if ascent_ub < tol
            is_converged = true
            if verbose  println("The MCEM algorithm has converged.\n") end
            break
        end

        # check if limits are reached
        if ess_target > max_ess  
            @warn "The maximum target ESS ($ess_target) has been reached.\n"; break 
        end
        
        if iter >= maxiter  
            @warn "The maximum number of iterations ($maxiter) has been reached.\n"; break 
        end

        # increase ESS if necessary
        if ascent_lb < 0
            # Multiplicative ESS increase (used by both :fixed and :adaptive methods)
            ess_target = ceil(ess_growth_factor * ess_target)
            if verbose
                println("Target ESS increased to $ess_target (growth factor $(round(ess_growth_factor, digits=3))).\n")
            end

            # Note: No need to clear arrays - DrawSamplePaths! will append only if ess_cur[i] < ess_target
            # The bug was in ComputeImportanceWeightsESS! incorrectly setting ess_cur[i] = ess_target
            # instead of the actual path count for uniform weights. This has been fixed.
            
            # SIR: Update pool target and check for cap
            if use_sir
                new_pool_target = sir_pool_size(ess_target, sir_pool_constant, sir_max_pool)
                if new_pool_target == sir_max_pool && !sir_pool_cap_exceeded
                    sir_pool_cap_exceeded = true
                    @warn "SIR pool size capped at $sir_max_pool; further ESS increases will reduce SIR effectiveness."
                end
                sir_pool_target = new_pool_target
                
                # Resample SIR indices with new ESS target
                for i in 1:nsubj
                    if length(samplepaths[i]) <= 1 || allequal(loglik_surrog[i])
                        sir_subsample_indices[i] = collect(1:length(samplepaths[i]))
                    else
                        sir_subsample_indices[i] = get_sir_subsample_indices(
                            ImportanceWeights[i], Int(ess_target), sir)
                    end
                end
                fill!(ess_cur, Float64(ess_target))
            end
        end

        # ensure that ess per person is sufficient
        # For SIR: sample to pool_target instead of ess_target
        sampling_target = use_sir ? sir_pool_target : ess_target
        
        DrawSamplePaths!(model; 
            ess_target = sampling_target, 
            ess_cur = ess_cur, 
            max_sampling_effort = max_sampling_effort,
            samplepaths = samplepaths, 
            loglik_surrog = loglik_surrog, 
            loglik_target_prop = loglik_target_prop, 
            loglik_target_cur = loglik_target_cur, 
            _logImportanceWeights = _logImportanceWeights, 
            ImportanceWeights = ImportanceWeights,   
            tpm_book_surrogate = tpm_book_surrogate,   
            hazmat_book_surrogate = hazmat_book_surrogate, 
            books = books, 
            npaths_additional = npaths_additional, 
            params_cur = params_cur, 
            surrogate = surrogate,
            psis_pareto_k = psis_pareto_k,
            fbmats = fbmats,
            absorbingstates = absorbingstates,
            # Phase-type infrastructure (nothing if not using)
            phasetype_surrogate = phasetype_surrogate,
            tpm_book_ph = tpm_book_ph,
            hazmat_book_ph = hazmat_book_ph,
            fbmats_ph = fbmats_ph,
            emat_ph = emat_ph,
            # Expanded data infrastructure (nothing if not using/not needed)
            expanded_ph_data = expanded_ph_data,
            expanded_ph_subjectindices = ph_subjectindices,
            expanded_ph_tpm_map = expanded_ph_tpm_map,
            ph_original_row_map = ph_original_row_map)
        
        # SIR: Resample after pool expansion
        if use_sir
            for i in 1:nsubj
                if length(samplepaths[i]) <= 1 || allequal(loglik_surrog[i])
                    sir_subsample_indices[i] = collect(1:length(samplepaths[i]))
                else
                    sir_subsample_indices[i] = get_sir_subsample_indices(
                        ImportanceWeights[i], Int(ess_target), sir)
                end
            end
            fill!(ess_cur, Float64(ess_target))
        end
    end # end-while

    if !is_converged
        @warn "MCEM did not converge."
    end

    # rectify spline coefs
    if any(isa.(model.hazards, _SplineHazard))
        rectify_coefs!(params_cur, model)
    end

    # hessian
    if !isnothing(constraints) || any(map(x -> (isa(x, _SplineHazard) && x.monotone != 0), model.hazards))

        # no hessian when there are constraints
        if compute_vcov == true
            @warn "No covariance matrix is returned when constraints are provided or when using monotone splines."
        end
        vcov = nothing
    
    elseif is_converged && compute_vcov
        if verbose
            println("Computing variance-covariance matrix at final estimates.")
        end

        # set up containers for path and sampling weight
        path = Array{SamplePath}(undef, 1)
        samplingweight = Vector{Float64}(undef, 1)
        
        # Get hazard parameter blocks from parameters.natural
        # Each hazard has a vector of parameters; compute block indices
        natural_pars = model.parameters.natural
        nhaz = length(natural_pars)
        block_sizes = [length(natural_pars[k]) for k in 1:nhaz]
        
        # Build index ranges for each hazard block
        elem_ptr = cumsum([1; block_sizes])
        blocks = [elem_ptr[k]:(elem_ptr[k+1]-1) for k in 1:nhaz]
        
        # initialize Fisher information matrix containers
        nparams = length(params_cur)
        fishinf = zeros(Float64, nparams, nparams)
        
        # The Hessian of the log-likelihood is BLOCK-DIAGONAL with blocks corresponding to 
        # each hazard's parameters. This is because:
        #   log L = Σᵢ [-Λ₁(t) - Λ₂(t) - ... + log hₖ(t)]  (for path with transition via hazard k)
        # Each hazard's parameters only appear in its own cumulative hazard Λₖ and log-hazard,
        # so ∂²L/∂θₐ∂θᵦ = 0 when θₐ and θᵦ belong to different hazards.
        #
        # Note: This block-diagonal structure holds because we skip vcov computation when 
        # there are constraints (constraints could couple parameters across hazards).
        # For constrained problems, one would need to compute the bordered Hessian or use
        # SparseDiffTools.jl with automatic sparsity detection.
        #
        # We exploit this structure by computing each block's Hessian separately, which is
        # O(Σᵢ bᵢ²) instead of O(n²) for a dense Hessian (where bᵢ = size of block i, n = total params).
        
        # Decide whether to use block-diagonal or dense Hessian computation
        # Due to function call overhead, we only use block-diagonal when theoretical speedup
        # exceeds the threshold (default 2.0× speedup required)
        sum_block_sq = sum(bs^2 for bs in block_sizes)
        theoretical_speedup = nparams^2 / sum_block_sq
        use_block_diagonal = theoretical_speedup > block_hessian_speedup && nhaz > 1
        
        # Full likelihood function
        ll_full = pars -> (loglik_AD(pars, ExactDataAD(path, samplingweight, model.hazards, model); neg=false))

        if use_block_diagonal
            # Block-diagonal Hessian computation (faster for models with many hazards)
            
            # Pre-allocate containers for each block
            diffres_blocks = [DiffResults.HessianResult(zeros(bs)) for bs in block_sizes]
            fish_i1_blocks = [zeros(Float64, bs, bs) for bs in block_sizes]
            
            # Create block likelihood functions (one per hazard)
            # These compute likelihood as a function of only that block's parameters
            params_base = copy(params_cur)
            ll_blocks = Vector{Function}(undef, nhaz)
            for k in 1:nhaz
                block = blocks[k]
                ll_blocks[k] = function(θ_block)
                    T = eltype(θ_block)
                    θ_full = T.(params_base)
                    θ_full[block] = θ_block
                    return ll_full(θ_full)
                end
            end
            
            # accumulate Fisher information
            for i in 1:nsubj

                # set importance weight
                samplingweight[1] = model.SubjectWeights[i]

                # number of paths
                npaths = length(samplepaths[i])

                # for accumulating gradients (full parameter vector)
                grads = zeros(Float64, nparams, npaths)
                
                # Process each block independently (exploiting block-diagonal Hessian structure)
                for (k, block) in enumerate(blocks)
                    bs = block_sizes[k]
                    fill!(fish_i1_blocks[k], 0.0)
                    grads_block = zeros(Float64, bs, npaths)
                    
                    # calculate gradient and hessian for paths (block k only)
                    for j in 1:npaths
                        path[1] = samplepaths[i][j]
                        diffres_blocks[k] = ForwardDiff.hessian!(diffres_blocks[k], ll_blocks[k], params_cur[block])

                        # grab block hessian and gradient
                        g_block = DiffResults.gradient(diffres_blocks[k])
                        H_block = DiffResults.hessian(diffres_blocks[k])
                        
                        grads_block[:, j] = g_block
                        grads[block, j] = g_block

                        # just to be safe wrt nans or infs
                        if !all(isfinite, H_block)
                            fill!(H_block, 0.0)
                        end

                        if !all(isfinite, g_block)
                            fill!(g_block, 0.0)
                            grads_block[:, j] .= 0.0
                            grads[block, j] .= 0.0
                        end

                        fish_i1_blocks[k] .+= ImportanceWeights[i][j] * (-H_block - g_block * transpose(g_block))
                    end
                    
                    # Block contribution to fish_i2: (Σⱼ wⱼgⱼ)(Σⱼ wⱼgⱼ)ᵀ for this block
                    g_weighted_block = grads_block * ImportanceWeights[i]
                    fish_i2_block = g_weighted_block * transpose(g_weighted_block)
                    
                    fishinf[block, block] .+= fish_i1_blocks[k] .+ fish_i2_block
                end
            end
        else
            # Dense Hessian computation (faster for models with few hazards/parameters)
            fish_i1 = zeros(Float64, nparams, nparams)
            diffres = DiffResults.HessianResult(params_cur)
            
            # accumulate Fisher information
            for i in 1:nsubj

                # set importance weight
                samplingweight[1] = model.SubjectWeights[i]

                # number of paths
                npaths = length(samplepaths[i])

                # for accumulating gradients and hessians
                grads = Array{Float64}(undef, nparams, npaths)

                # reset matrices for accumulating Fisher info contributions
                fill!(fish_i1, 0.0)

                # calculate gradient and hessian for paths
                for j in 1:npaths
                    path[1] = samplepaths[i][j]
                    diffres = ForwardDiff.hessian!(diffres, ll_full, params_cur)

                    # grab hessian and gradient
                    grads[:,j] = DiffResults.gradient(diffres)

                    # just to be safe wrt nans or infs
                    if !all(isfinite, DiffResults.hessian(diffres))
                        fill!(DiffResults.hessian(diffres), 0.0)
                    end

                    if !all(isfinite, DiffResults.gradient(diffres))
                        fill!(DiffResults.gradient(diffres), 0.0)
                    end

                    fish_i1 .+= ImportanceWeights[i][j] * (-DiffResults.hessian(diffres) - DiffResults.gradient(diffres) * transpose(DiffResults.gradient(diffres)))
                end

                # Optimized: Σⱼ Σₖ wⱼwₖ gⱼgₖᵀ = (Σⱼ wⱼgⱼ)(Σₖ wₖgₖ)ᵀ = g_weighted * g_weighted'
                g_weighted = grads * ImportanceWeights[i]
                fish_i2 = g_weighted * transpose(g_weighted)

                fishinf .+= fish_i1 .+ fish_i2
            end
        end

        # get the variance-covariance matrix
        vcov = pinv(Symmetric(fishinf), atol = vcov_threshold ? (log(nsubj) * length(params_cur))^-2 : sqrt(eps(real(float(oneunit(eltype(fishinf)))))))
        vcov[findall(isapprox.(vcov, 0.0; atol = sqrt(eps(Float64))))] .= 0.0
        vcov = Symmetric(vcov)
    else
        vcov = nothing
    end

    # subject marginal likelihood
    logliks = compute_loglik(model, loglik_surrog, loglik_target_cur, NormConstantProposal)

    # compute robust variance estimates if requested
    ij_variance = nothing
    jk_variance = nothing
    subject_grads = nothing
    
    if (compute_ij_vcov || compute_jk_vcov) && !isnothing(vcov) && isnothing(constraints) && is_converged
        if verbose
            println("Computing robust variance estimates...")
        end
        robust_result = compute_robust_vcov(params_cur, model, samplepaths, ImportanceWeights;
                                           compute_ij = compute_ij_vcov,
                                           compute_jk = compute_jk_vcov,
                                           loo_method = loo_method,
                                           vcov_threshold = vcov_threshold)
        ij_variance = robust_result.ij_vcov
        jk_variance = robust_result.jk_vcov
        subject_grads = robust_result.subject_gradients
    end

    # return convergence records
    if return_convergence_records
        # Base convergence records
        base_records = (
            mll_trace = mll_trace, 
            ess_trace = ess_trace, 
            path_count_trace = path_count_trace, 
            parameters_trace = parameters_trace, 
            psis_pareto_k = psis_pareto_k
        )
        
        # Add SIR-specific records if SIR was used (either from start or via adaptive switching)
        if use_sir
            sir_records = (
                sir_method = sir,
                sir_resample_mode = sir_resample,
                sir_pool_cap_exceeded = sir_pool_cap_exceeded
            )
            base_records = merge(base_records, sir_records)
        end
        
        # Add adaptive SIR records if adaptive switching was enabled
        if sir_adaptive || adaptive_sir_switched
            adaptive_records = (
                adaptive_sir_switched = adaptive_sir_switched,
                adaptive_sir_switch_iter = adaptive_sir_switched ? adaptive_sir_switch_iter : nothing,
                adaptive_sir_final_path_ratio = adaptive_sir_cumul_optim_iters > 0 ? 
                    adaptive_sir_cumul_path_ratio / adaptive_sir_cumul_optim_iters : nothing
            )
            base_records = merge(base_records, adaptive_records)
        end
        
        ConvergenceRecords = base_records
    else
        ConvergenceRecords = nothing
    end

    # return sampled paths and importance weights
    ProposedPaths = return_proposed_paths ? (paths=samplepaths, weights=ImportanceWeights) : nothing

    # Build ParameterHandling structure for fitted parameters
    # params_cur contains log-scale parameters; we need to split into per-hazard vectors
    # Compute block sizes from model.parameters.natural structure
    natural_vals = values(model.parameters.natural)
    block_sizes = [length(v) for v in natural_vals]
    
    # Split params_cur into per-hazard log-scale vectors
    log_scale_params = Vector{Vector{Float64}}(undef, length(block_sizes))
    offset = 0
    for i in eachindex(block_sizes)
        log_scale_params[i] = params_cur[(offset+1):(offset+block_sizes[i])]
        offset += block_sizes[i]
    end
    
    # Rebuild parameters with proper constraints using model's hazard info
    parameters_fitted = rebuild_parameters(log_scale_params, model)

    # wrap results
    model_fitted = MultistateModelFitted(
        data_original,
        parameters_fitted,
        logliks,
        vcov,
        isnothing(ij_variance) ? nothing : Matrix(ij_variance),
        isnothing(jk_variance) ? nothing : Matrix(jk_variance),
        subject_grads,
        model.hazards,
        model.totalhazards,
        model.tmat,
        model.emat,
        model.hazkeys,
        model.subjectindices,
        model.SubjectWeights,
        model.ObservationWeights,
        model.CensoringPatterns,
        surrogate,
        ConvergenceRecords,
        ProposedPaths,
        model.modelcall,
        model.phasetype_expansion)

    # remake splines and calculate risk periods using log-scale parameters
    for i in eachindex(model_fitted.hazards)
        if isa(model_fitted.hazards[i], _SplineHazard)
            remake_splines!(model_fitted.hazards[i], log_scale_params[i])
            set_riskperiod!(model_fitted.hazards[i])
        end
    end

    # return fitted object
    return model_fitted;
end
