"""
    fit(model::MultistateModel; constraints = nothing, verbose = true, compute_vcov = true, kwargs...)

Fit a multistate model to continuously observed (exact) data.

Uses L-BFGS optimization for unconstrained problems (5-6× faster than interior-point methods)
and Ipopt for constrained problems by default.

# Arguments
- `model::MultistateModel`: multistate model object with exact observation times
- `constraints`: parameter constraints (see Constraints documentation)
- `verbose::Bool=true`: print optimization messages
- `solver`: optimization solver (default: L-BFGS for unconstrained, Ipopt for constrained).
  See [Optimization Solvers](@ref) for available options.
- `compute_vcov::Bool=true`: compute model-based variance-covariance matrix (H⁻¹).
  Useful for diagnostics by comparing to robust variance.
- `vcov_threshold::Bool=true`: if true, uses adaptive threshold `1/√(log(n)·p)` for pseudo-inverse 
  of Fisher information; otherwise uses `√eps()`. Helps with near-singular Hessians.
- `compute_ij_vcov::Bool=true`: compute infinitesimal jackknife (sandwich/robust) variance H⁻¹KH⁻¹.
  **This is the recommended variance for inference** as it remains valid under model misspecification.
- `compute_jk_vcov::Bool=false`: compute jackknife variance ((n-1)/n)·Σᵢ ΔᵢΔᵢᵀ
- `loo_method::Symbol=:direct`: method for computing leave-one-out (LOO) perturbations Δᵢ.
  **Only affects jackknife variance** (not IJ), since IJ uses Var_IJ = H⁻¹KH⁻¹ directly
  without computing individual LOO estimates. Options:
  - `:direct`: approximate H₋ᵢ⁻¹ ≈ H⁻¹, so Δᵢ = H⁻¹gᵢ. O(p²n), faster when n >> p.
  - `:cholesky`: compute H₋ᵢ⁻¹ exactly via Cholesky rank-k downdates of H = LLᵀ.
    O(np³), more numerically stable for ill-conditioned problems.

# Returns
- `MultistateModelFitted`: fitted model object with estimates, standard errors, and diagnostics

# Variance Estimation (Default Behavior)

By default, both model-based and IJ (sandwich) variance are computed:
- **IJ/Sandwich variance** (`compute_ij_vcov=true`): **Primary for inference**. Robust variance
  H⁻¹KH⁻¹ that remains valid even under model misspecification.
- **Model-based variance** (`compute_vcov=true`): **For diagnostics**. Standard MLE variance H⁻¹,
  valid only under correct model specification. Compare to IJ variance to assess model adequacy.

Use `compare_variance_estimates(fitted)` to diagnose potential model misspecification.

# Example
```julia
# Default fit: computes both model-based and robust (IJ) variance
fitted = fit(model)

# Get robust standard errors for inference
robust_se = sqrt.(diag(get_ij_vcov(fitted)))

# Compare model-based and robust SEs to check model specification
result = compare_variance_estimates(fitted)
# If SE_robust >> SE_model, model may be misspecified

# Fit with model-based variance only (faster, but less robust)
fitted = fit(model; compute_ij_vcov=false)

# Use a custom solver (e.g., BFGS instead of L-BFGS)
using Optim
fitted = fit(model; solver=Optim.BFGS())
```

See also: [`get_vcov`](@ref), [`get_ij_vcov`](@ref), [`compare_variance_estimates`](@ref)
"""
function fit(model::MultistateModel; constraints = nothing, verbose = true, solver = nothing, compute_vcov = true, vcov_threshold = true, compute_ij_vcov = true, compute_jk_vcov = false, loo_method = :direct, kwargs...)

    # initialize array of sample paths
    samplepaths = extract_paths(model)

    # extract and initialize model parameters
    # Phase 3: Use ParameterHandling.jl flat parameters (log scale)
    parameters = get_parameters_flat(model)

    # parse constraints, or not, and solve
    if isnothing(constraints) 
        # get estimates - use L-BFGS for unconstrained (5-6x faster than Ipopt)
        optf = OptimizationFunction(loglik_exact, Optimization.AutoForwardDiff())
        prob = OptimizationProblem(optf, parameters, ExactData(model, samplepaths))

        # solve with user-specified solver or default L-BFGS
        _solver = isnothing(solver) ? Optim.LBFGS() : solver
        sol  = solve(prob, _solver)

        # rectify spline coefs
        if any(isa.(model.hazards, _SplineHazard))
            rectify_coefs!(sol.u, model)
        end
        
        # get vcov
        if compute_vcov && (sol.retcode == ReturnCode.Success) && !any(map(x -> (isa(x, _SplineHazard) && x.monotone != 0), model.hazards))
            # preallocate the hessian matrix
            diffres = DiffResults.HessianResult(sol.u)

            # single argument function for log-likelihood
            ll = pars -> loglik_exact(pars, ExactData(model, samplepaths); neg=false)

            # compute gradient and hessian
            diffres = ForwardDiff.hessian!(diffres, ll, sol.u)

            # grab results
            gradient = DiffResults.gradient(diffres)
            fishinf = -DiffResults.hessian(diffres)
            vcov = pinv(Symmetric(fishinf), atol = vcov_threshold ? (log(length(samplepaths)) * length(sol.u))^-2 : sqrt(eps(real(float(oneunit(eltype(fishinf)))))))
            vcov[isapprox.(vcov, 0.0; atol = sqrt(eps(Float64)), rtol = sqrt(eps(Float64)))] .= 0.0
            vcov = Symmetric(vcov)
        else
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

        optf = OptimizationFunction(loglik, Optimization.AutoForwardDiff(), cons = consfun_multistate)
        prob = OptimizationProblem(optf, parameters, ExactData(model, samplepaths), lcons = constraints.lcons, ucons = constraints.ucons)
        
        # solve with user-specified solver or default Ipopt
        _solver = isnothing(solver) ? Ipopt.Optimizer() : solver
        sol  = solve(prob, _solver; print_level = 0)

        # rectify spline coefs
        if any(isa.(model.hazards, _SplineHazard))
            rectify_coefs!(sol.u, model)
        end

        # no hessian when there are constraints
        if compute_vcov == true
            @warn "No covariance matrix is returned when constraints are provided."
        end
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
                                           vcov_threshold = vcov_threshold)
        ij_variance = robust_result.ij_vcov
        jk_variance = robust_result.jk_vcov
        subject_grads = robust_result.subject_gradients
    end

    # Build ParameterHandling structure for fitted parameters
    # Use the unflatten function from the model to convert flat params back to transformed
    params_transformed = model.parameters.unflatten(sol.u)
    params_natural = ParameterHandling.value(params_transformed)
    parameters_fitted = (
        flat = Vector{Float64}(sol.u),
        transformed = params_transformed,
        natural = params_natural,
        unflatten = model.parameters.unflatten
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
        model.modelcall)

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


"""
    fit(model::MultistateMarkovModel; constraints = nothing, verbose = true, compute_vcov = true, kwargs...)

Fit a Markov multistate model to interval-censored or panel data.

Uses L-BFGS optimization for unconstrained problems (5-6× faster than interior-point methods)
and Ipopt for constrained problems by default.

# Arguments
- `model::Union{MultistateMarkovModel, MultistateMarkovModelCensored}`: Markov model with panel observations
- `constraints`: parameter constraints (see Constraints documentation)
- `verbose::Bool=true`: print optimization messages
- `solver`: optimization solver (default: L-BFGS for unconstrained, Ipopt for constrained).
  See [Optimization Solvers](@ref) for available options.
- `compute_vcov::Bool=true`: compute model-based variance-covariance matrix (for diagnostics)
- `vcov_threshold::Bool=true`: use adaptive threshold for pseudo-inverse of Fisher information
- `compute_ij_vcov::Bool=true`: compute infinitesimal jackknife (sandwich/robust) variance.
  **This is the recommended variance for inference.**
- `compute_jk_vcov::Bool=false`: compute jackknife variance
- `loo_method::Symbol=:direct`: method for LOO perturbations Δᵢ = θ̂₋ᵢ - θ̂ (jackknife only, not IJ):
  - `:direct`: Δᵢ = H⁻¹gᵢ (faster, O(p²n))
  - `:cholesky`: exact H₋ᵢ⁻¹ via Cholesky downdates (stable, O(np³))

# Returns
- `MultistateModelFitted`: fitted model with estimates and variance matrices

# Variance Estimation (Default Behavior)

By default, both model-based and IJ (sandwich) variance are computed:
- **IJ variance** (`compute_ij_vcov=true`): Primary for inference (robust to misspecification)
- **Model-based** (`compute_vcov=true`): For diagnostics (compare SE ratios)

# Notes
- For Markov models, the likelihood involves matrix exponentials of the intensity matrix
- Censored state observations are handled via marginalization over possible states
- IJ/JK variance estimation works for both censored and uncensored Markov models

# Example
```julia
# Default fit: robust and model-based variance computed
fitted = fit(markov_model)

# Use robust SEs for inference
robust_se = sqrt.(diag(get_ij_vcov(fitted)))

# Diagnose model specification
result = compare_variance_estimates(fitted)

# Use a custom solver
using Optim
fitted = fit(markov_model; solver=Optim.BFGS())
```

See also: [`fit(::MultistateModel)`](@ref), [`compare_variance_estimates`](@ref)
"""
function fit(model::Union{MultistateMarkovModel,MultistateMarkovModelCensored}; constraints = nothing, verbose = true, solver = nothing, compute_vcov = true, vcov_threshold = true, compute_ij_vcov = true, compute_jk_vcov = false, loo_method = :direct, kwargs...)

    # containers for bookkeeping TPMs
    books = build_tpm_mapping(model.data)

    # extract and initialize model parameters
    # Phase 3: Use ParameterHandling.jl flat parameters (log scale)
    parameters = get_parameters_flat(model)

    # number of subjects
    nsubj = length(model.subjectindices)

    # parse constraints, or not, and solve
    if isnothing(constraints)
        # get estimates - use L-BFGS for unconstrained (5-6x faster than Ipopt)
        optf = OptimizationFunction(loglik_markov, Optimization.AutoForwardDiff())
        prob = OptimizationProblem(optf, parameters, MPanelData(model, books))
        
        # solve with user-specified solver or default L-BFGS
        _solver = isnothing(solver) ? Optim.LBFGS() : solver
        sol  = solve(prob, _solver)

        # get vcov
        if compute_vcov && (sol.retcode == ReturnCode.Success)
            # preallocate the hessian matrix
            diffres = DiffResults.HessianResult(sol.u)

            # single argument function for log-likelihood            
            ll = pars -> loglik_markov(pars, MPanelData(model, books); neg=false)

            # compute gradient and hessian
            diffres = ForwardDiff.hessian!(diffres, ll, sol.u)

            # grab results
            gradient = DiffResults.gradient(diffres)
            fishinf = Symmetric(.-DiffResults.hessian(diffres))
            vcov = pinv(Symmetric(fishinf), atol = vcov_threshold ? (log(nsubj) * length(sol.u))^-2 : sqrt(eps(real(float(oneunit(eltype(fishinf)))))))
            vcov[isapprox.(vcov, 0.0; atol = eps(Float64))] .= 0.0
            vcov = Symmetric(vcov)
        else
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

        optf = OptimizationFunction(loglik_markov, Optimization.AutoForwardDiff(), cons = consfun_markov)
        prob = OptimizationProblem(optf, parameters, MPanelData(model, books), lcons = constraints.lcons, ucons = constraints.ucons)
        
        # solve with user-specified solver or default Ipopt
        _solver = isnothing(solver) ? Ipopt.Optimizer() : solver
        sol  = solve(prob, _solver; print_level = 0)

        # no hessian when there are constraints
        if compute_vcov == true
            @warn "No covariance matrix is returned when constraints are provided."
        end
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
                                           vcov_threshold = vcov_threshold)
        ij_variance = robust_result.ij_vcov
        jk_variance = robust_result.jk_vcov
        subject_grads = robust_result.subject_gradients
    end

    # Build ParameterHandling structure for fitted parameters
    # Use the unflatten function from the model to convert flat params back to transformed
    params_transformed = model.parameters.unflatten(sol.u)
    params_natural = ParameterHandling.value(params_transformed)
    parameters_fitted = (
        flat = Vector{Float64}(sol.u),
        transformed = params_transformed,
        natural = params_natural,
        unflatten = model.parameters.unflatten
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
        model.modelcall);
end


"""
    fit(model::Union{MultistateSemiMarkovModel, MultistateSemiMarkovModelCensored}; kwargs...)

Fit a semi-Markov model to panel data via Monte Carlo EM (MCEM).

Uses L-BFGS optimization for unconstrained M-steps (5-6× faster than interior-point methods)
and Ipopt for constrained M-steps by default.

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
- `model::Union{MultistateSemiMarkovModel, MultistateSemiMarkovModelCensored}`: semi-Markov model
- `constraints`: parameter constraints tuple

**Optimization:**
- `solver`: optimization solver for M-step (default: L-BFGS for unconstrained, Ipopt for constrained).
  See [Optimization Solvers](@ref) for available options.

**MCEM algorithm control:**
- `maxiter::Int=100`: maximum MCEM iterations
- `tol::Float64=1e-2`: tolerance for MLL change in stopping rule
- `ascent_threshold::Float64=0.1`: standard normal quantile for ascent lower bound
- `stopping_threshold::Float64=0.1`: standard normal quantile for stopping criterion
- `ess_increase::Float64=2.0`: ESS inflation factor when more paths needed
- `ess_target_initial::Int=50`: initial effective sample size target per subject
- `max_ess::Int=10000`: maximum ESS before stopping for non-convergence
- `max_sampling_effort::Int=20`: maximum factor of ESS for additional path sampling
- `npaths_additional::Int=10`: increment for additional paths when augmenting
- `viterbi_init::Bool=true`: initialize MCEM with Viterbi MAP path per subject for warm start
- `block_hessian_speedup::Float64=2.0`: minimum speedup factor to use block-diagonal Hessian
- `acceleration::Symbol=:none`: acceleration method for MCEM. Options:
  - `:none` (default): standard MCEM without acceleration
  - `:squarem`: SQUAREM acceleration (Varadhan & Roland, 2008), applies quasi-Newton 
    extrapolation every 2 iterations to speed up convergence

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

See also: [`fit(::MultistateModel)`](@ref), [`compare_variance_estimates`](@ref)
"""
function fit(model::Union{MultistateSemiMarkovModel, MultistateSemiMarkovModelCensored}; proposal::Union{Symbol, ProposalConfig} = :auto, constraints = nothing, solver = nothing, maxiter = 100, tol = 1e-2, ascent_threshold = 0.1, stopping_threshold = 0.1, ess_increase = 2.0, ess_target_initial = 50, max_ess = 10000, max_sampling_effort = 20, npaths_additional = 10, viterbi_init = true, block_hessian_speedup = 2.0, acceleration::Symbol = :none, verbose = true, return_convergence_records = true, return_proposed_paths = false, compute_vcov = true, vcov_threshold = true, compute_ij_vcov = true, compute_jk_vcov = false, loo_method = :direct, kwargs...)

    # Validate acceleration parameter
    if acceleration ∉ (:none, :squarem)
        error("acceleration must be :none or :squarem, got :$acceleration")
    end
    use_squarem = acceleration === :squarem
    
    if verbose && use_squarem
        println("Using SQUAREM acceleration for MCEM.\n")
    end

    # Resolve proposal configuration
    proposal_config = resolve_proposal_config(proposal, model)
    use_phasetype = proposal_config.type === :phasetype
    
    if verbose && use_phasetype
        println("Using phase-type proposal for MCEM importance sampling.\n")
    end

    # copy of data
    data_original = deepcopy(model.data)

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

    # check that ess_increase is greater than 1
    if ess_increase <= 1
        error("ess_increase must be greater than 1.")
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

    # make fbmats if necessary
    if any(model.data.obstype .> 2)
        fbmats = build_fbmats(model)
    else
        fbmats = nothing
    end

    # containers for traces
    mll_trace = Vector{Float64}() # marginal loglikelihood
    ess_trace = ElasticArray{Float64, 2}(undef, nsubj, 0) # effective sample size (one per subject)
    # Phase 3: Use ParameterHandling.jl flat parameter length
    parameters_trace = ElasticArray{Float64, 2}(undef, length(get_parameters_flat(model)), 0) # parameter estimates

    # Require a pre-built Markov surrogate for MCEM
    # Users should call set_surrogate!(model) or use surrogate=:markov in multistatemodel() beforehand
    if isnothing(model.markovsurrogate)
        error("MCEM requires a Markov surrogate. Call `set_surrogate!(model)` or use `surrogate=:markov` in `multistatemodel()` before fitting.")
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
    # Get log-scale surrogate parameters for hazard evaluation
    surrogate_pars = get_log_scale_params(surrogate.parameters)
    for t in eachindex(books[1])
        # compute the transition intensity matrix
        compute_hazmat!(hazmat_book_surrogate[t], surrogate_pars, surrogate.hazards, books[1][t], model.data)
        # compute transition probability matrices
        compute_tmat!(tpm_book_surrogate[t], hazmat_book_surrogate[t], books[1][t], cache)
    end

    # Build phase-type TPM book if using phase-type proposals
    if use_phasetype
        tpm_book_ph, hazmat_book_ph = build_phasetype_tpm_book(phasetype_surrogate, books, model.data)
        fbmats_ph = build_fbmats_phasetype(model, phasetype_surrogate)
        emat_ph = build_phasetype_emat_expanded(model, phasetype_surrogate)
    end

    # compute normalizing constant of proposal distribution
    # For Markov proposal: this is the log-likelihood under the Markov surrogate
    # For phase-type proposal: compute marginal likelihood via forward algorithm
    #   This is r(Y|θ') in the importance sampling formula:
    #   log f̂(Y|θ) = log r(Y|θ') + Σᵢ log(mean(νᵢ))
    if use_phasetype
        NormConstantProposal = compute_phasetype_marginal_loglik(model, phasetype_surrogate, emat_ph)
    else
        NormConstantProposal = surrogate_fitted.loglik.loglik
    end

    # Viterbi MAP warm start initialization
    #
    # By initializing with the marginal posterior mode (Viterbi MAP) path for each
    # subject, the first M-step can take a large step toward the mode of the
    # likelihood surface, accelerating MCEM convergence.
    #
    # Note: Skip Viterbi initialization when using phase-type proposals because
    # Viterbi paths are in observed space, but phase-type importance weights
    # require expanded space paths. Paths will be drawn in expanded space directly.
    if viterbi_init && !use_phasetype
        if verbose println("Initializing with Viterbi MAP paths...\n") end
        
        for i in 1:nsubj
            # Get subject data indices
            subj_inds = model.subjectindices[i]
            # Skip subjects with only exact observations (obstype == 1)
            # For these subjects, the path is fully determined by the data
            if all(model.data.obstype[subj_inds] .== 1)
                continue
            end
            
            # compute Viterbi MAP path for subject i
            map_path = viterbi_map_path(i, model, tpm_book_surrogate, hazmat_book_surrogate, books[2], fbmats, absorbingstates)
            
            # initialize containers with single MAP path
            push!(samplepaths[i], map_path)
            
            # compute log-likelihoods using log-scale parameters
            surrogate_pars = get_log_scale_params(surrogate.parameters)
            ll_surrog = loglik(surrogate_pars, map_path, surrogate.hazards, model)
            target_pars = nest_params(params_cur, model.parameters)
            ll_target = loglik(target_pars, map_path, model.hazards, model)
            
            push!(loglik_surrog[i], ll_surrog)
            push!(loglik_target_cur[i], ll_target)
            push!(loglik_target_prop[i], 0.0)  # will be updated in M-step
            push!(_logImportanceWeights[i], ll_target - ll_surrog)
            push!(ImportanceWeights[i], 1.0)  # single path gets weight 1
            
            # set initial ESS to 1 (will be augmented by DrawSamplePaths!)
            ess_cur[i] = 1.0
        end
        
        if verbose println("Viterbi MAP initialization complete. Drawing additional paths...\n") end
    end

    # draw sample paths until the target ess is reached 
    if verbose  println("Initializing sample paths ...\n") end
    DrawSamplePaths!(model; 
        ess_target = ess_target, 
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
        emat_ph = emat_ph)
    
    # get current estimate of marginal log likelihood
    mll_cur = mcem_mll(loglik_target_cur, ImportanceWeights, model.SubjectWeights)

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
        # SQUAREM: Save θ₀ at start of cycle (every 2 iterations)
        # =====================================================================
        if use_squarem && squarem_state.step == 0
            squarem_state.θ0 .= params_cur
            squarem_state.step = 1
        end
        
        # solve M-step: use user-specified solver or default (L-BFGS for unconstrained, Ipopt for constrained)
        if isnothing(constraints)
            _solver = isnothing(solver) ? Optim.LBFGS() : solver
            params_prop_optim = solve(remake(prob, u0 = Vector(params_cur), p = SMPanelData(model, samplepaths, ImportanceWeights)), _solver)
        else
            _solver = isnothing(solver) ? Ipopt.Optimizer() : solver
            params_prop_optim = solve(remake(prob, u0 = Vector(params_cur), p = SMPanelData(model, samplepaths, ImportanceWeights)), _solver; print_level = 0)
        end
        params_prop = params_prop_optim.u

        # calculate the log likelihoods for the proposed parameters
        loglik!(params_prop, loglik_target_prop, SMPanelData(model, samplepaths, ImportanceWeights))
        
        # calculate the marginal log likelihood 
        mll_cur  = mcem_mll(loglik_target_cur , ImportanceWeights, model.SubjectWeights)
        mll_prop = mcem_mll(loglik_target_prop, ImportanceWeights, model.SubjectWeights)

        # compute the ALB and AUB
        if params_prop != params_cur

            # change in mll
            mll_change = mll_prop - mll_cur
    
            # calculate the ASE for ΔQ
            ase = mcem_ase(loglik_target_prop, loglik_target_cur, ImportanceWeights, model.SubjectWeights)
    
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
                    mll_acc = mcem_mll(loglik_target_prop, ImportanceWeights, model.SubjectWeights)
                    
                    # Decide whether to accept acceleration
                    # Use mll at θ₀ as reference (start of SQUAREM cycle)
                    mll_θ0 = mcem_mll(loglik_target_cur, ImportanceWeights, model.SubjectWeights)
                    
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

        # print update
        if verbose
            println("Iteration: $iter")
            println("Target ESS: $(round(ess_target;digits=2)) per-subject")
            println("Range of the number of sampled paths per-subject: [$(ceil(ess_target)), $(max(length.(samplepaths)...))]")
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

        # increase ess is necessary
        if ascent_lb < 0
            # increase the target ess for the factor ess_increase
            ess_target = ceil(ess_increase*ess_target) # TODO: alternatively, use Caffo's power calculation to determine the new ESS target
            if verbose  println("Target ESS is increased to $ess_target, because ascent lower bound < 0.\n") end

            # no need to sample paths for subjects with a single possible path
            ess_cur[findall(length.(ImportanceWeights) .== 1)] .= ess_target
        end

        # ensure that ess per person is sufficient
        DrawSamplePaths!(model; 
            ess_target = ess_target, 
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
            emat_ph = emat_ph)
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
    ConvergenceRecords = return_convergence_records ? (mll_trace=mll_trace, ess_trace=ess_trace, parameters_trace=parameters_trace, psis_pareto_k = psis_pareto_k) : nothing

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
    
    # Build transformed NamedTuple: exp to get natural scale, wrap with positive()
    haznames = sort(collect(model.hazkeys), by = x -> x[2])
    params_transformed_pairs = [
        haznames[i].first => safe_positive(exp.(log_scale_params[i]))
        for i in eachindex(haznames)
    ]
    params_transformed = NamedTuple(params_transformed_pairs)
    params_flat, unflatten_fn = ParameterHandling.flatten(params_transformed)
    params_natural = ParameterHandling.value(params_transformed)
    parameters_fitted = (
        flat = params_flat,
        transformed = params_transformed,
        natural = params_natural,
        unflatten = unflatten_fn
    )

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
        model.modelcall)

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
