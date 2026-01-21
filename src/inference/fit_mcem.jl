# =============================================================================
# Semi-Markov MCEM Fitting (_fit_mcem)
# =============================================================================
#
# Monte Carlo EM fitting for semi-Markov models with panel observations.
# Called by fit() when is_panel_data(model) && !is_markov(model).
#
# Features:
# - Importance sampling from Markov or phase-type surrogates
# - Pareto-smoothed importance sampling (PSIS) for stable weights
# - Adaptive ESS growth based on Caffo et al. (2005)
# - Sampling Importance Resampling (SIR/LHS) for weight degeneracy
# - Louis's identity for observed Fisher information
# - Block-diagonal Hessian optimization for multi-hazard models
#
# References:
# - Morsomme et al. (2025) Biostatistics - multistate semi-Markov models
# - Wei & Tanner (1990) JASA - Monte Carlo EM
# - Caffo et al. (2005) JRSS-B - ascent-based MCEM
# - Vehtari et al. (2024) JMLR - Pareto Smoothed Importance Sampling
#
# =============================================================================

# =============================================================================
# MCEM Configuration Struct (L1_P1)
# =============================================================================

"""
    MCEMConfig

Configuration struct for MCEM algorithm parameters.

This struct consolidates the many keyword arguments of `_fit_mcem` into a
single configuration object, improving readability and maintainability.

# Fields

## Algorithm Control
- `maxiter::Int`: Maximum MCEM iterations (default: 100)
- `tol::Float64`: Tolerance for MLL change in stopping rule (default: 1e-2)
- `ascent_threshold::Float64`: Standard normal quantile for ascent lower bound (default: 0.1)
- `stopping_threshold::Float64`: Standard normal quantile for stopping criterion (default: 0.1)

## ESS Control
- `ess_target_initial::Int`: Initial effective sample size target per subject (default: 50)
- `max_ess::Int`: Maximum ESS before stopping for non-convergence (default: 10000)
- `ess_growth_factor::Float64`: Multiplicative factor for ESS increase (default: √2)
- `ess_increase_method::Symbol`: Method for increasing ESS, `:fixed` or `:adaptive` (default: :fixed)
- `ascent_alpha::Float64`: Type I error rate for adaptive power calculation (default: 0.25)
- `ascent_beta::Float64`: Type II error rate for adaptive power calculation (default: 0.25)

## Sampling Control
- `max_sampling_effort::Int`: Maximum factor of ESS for additional path sampling (default: 20)
- `npaths_additional::Int`: Increment for additional paths when augmenting (default: 10)
- `block_hessian_speedup::Float64`: Minimum speedup to use block-diagonal Hessian (default: 2.0)

## SIR Configuration
- `sir::Symbol`: SIR resampling method (default: :adaptive_lhs)
- `sir_pool_constant::Float64`: Pool size multiplier (default: 2.0)
- `sir_max_pool::Int`: Maximum pool size cap (default: 8192)
- `sir_resample::Symbol`: When to resample, `:always` or `:degeneracy` (default: :always)
- `sir_degeneracy_threshold::Float64`: Pareto-k threshold for `:degeneracy` mode (default: 0.7)
- `sir_adaptive_threshold::Float64`: Ratio threshold for adaptive switching (default: 2.0)
- `sir_adaptive_min_iters::Int`: Minimum iterations before adaptive switch (default: 3)

## Output Control
- `verbose::Bool`: Print progress messages (default: true)
- `return_convergence_records::Bool`: Save iteration history (default: true)
- `return_proposed_paths::Bool`: Save latent paths and importance weights (default: false)

## Variance Estimation
- `compute_vcov::Bool`: Compute model-based variance (default: true)
- `vcov_threshold::Bool`: Use adaptive threshold for pseudo-inverse (default: true)
- `compute_ij_vcov::Bool`: Compute IJ/sandwich variance (default: true)
- `compute_jk_vcov::Bool`: Compute jackknife variance (default: false)
- `loo_method::Symbol`: Method for LOO perturbations (default: :direct)

## Penalty Configuration
- `penalty`: Penalty specification (default: :auto)
- `lambda_init::Float64`: Initial smoothing parameter (default: 1.0)

# Example
```julia
config = MCEMConfig(
    maxiter = 200,
    ess_target_initial = 100,
    sir = :lhs,
    verbose = true
)
fitted = fit(model; mcem_config = config)  # (future API)
```
"""
Base.@kwdef struct MCEMConfig
    # Algorithm control
    maxiter::Int = DEFAULT_MCEM_MAXITER
    tol::Float64 = DEFAULT_MCEM_TOL
    ascent_threshold::Float64 = 0.1
    stopping_threshold::Float64 = 0.1
    
    # ESS control
    ess_target_initial::Int = DEFAULT_ESS_TARGET_INITIAL
    max_ess::Int = DEFAULT_MAX_ESS
    ess_growth_factor::Float64 = sqrt(2.0)
    ess_increase_method::Symbol = :fixed
    ascent_alpha::Float64 = 0.25
    ascent_beta::Float64 = 0.25
    
    # Sampling control
    max_sampling_effort::Int = 20
    npaths_additional::Int = 10
    block_hessian_speedup::Float64 = 2.0
    
    # SIR configuration
    sir::Symbol = :adaptive_lhs
    sir_pool_constant::Float64 = 2.0
    sir_max_pool::Int = 8192
    sir_resample::Symbol = :always
    sir_degeneracy_threshold::Float64 = 0.7
    sir_adaptive_threshold::Float64 = 2.0
    sir_adaptive_min_iters::Int = 3
    
    # Output control
    verbose::Bool = true
    return_convergence_records::Bool = true
    return_proposed_paths::Bool = false
    
    # Variance estimation
    compute_vcov::Bool = true
    vcov_threshold::Bool = true
    compute_ij_vcov::Bool = true
    compute_jk_vcov::Bool = false
    loo_method::Symbol = :direct
    
    # Penalty configuration
    penalty::Any = :auto
    lambda_init::Float64 = 1.0
end

"""
    validate(config::MCEMConfig)

Validate MCEM configuration parameters.

# Returns
- `true` if all parameters are valid

# Throws
- `ArgumentError` if any parameter is invalid
"""
function validate(config::MCEMConfig)
    # SIR validation
    config.sir ∈ (:none, :sir, :lhs, :adaptive_sir, :adaptive_lhs) ||
        throw(ArgumentError("sir must be :none, :sir, :lhs, :adaptive_sir, or :adaptive_lhs, got :$(config.sir)"))
    config.sir_resample ∈ (:always, :degeneracy) ||
        throw(ArgumentError("sir_resample must be :always or :degeneracy, got :$(config.sir_resample)"))
    config.sir_pool_constant > 0 ||
        throw(ArgumentError("sir_pool_constant must be positive, got $(config.sir_pool_constant)"))
    config.sir_max_pool > 0 ||
        throw(ArgumentError("sir_max_pool must be positive, got $(config.sir_max_pool)"))
    0 < config.sir_degeneracy_threshold < 1 ||
        throw(ArgumentError("sir_degeneracy_threshold must be in (0,1), got $(config.sir_degeneracy_threshold)"))
    config.sir_adaptive_threshold > 0 ||
        throw(ArgumentError("sir_adaptive_threshold must be positive, got $(config.sir_adaptive_threshold)"))
    config.sir_adaptive_min_iters >= 1 ||
        throw(ArgumentError("sir_adaptive_min_iters must be at least 1, got $(config.sir_adaptive_min_iters)"))
    
    # Algorithm parameters
    config.max_sampling_effort > 1 ||
        throw(ArgumentError("max_sampling_effort must be greater than 1, got $(config.max_sampling_effort)"))
    config.ess_growth_factor > 1 ||
        throw(ArgumentError("ess_growth_factor must be greater than 1, got $(config.ess_growth_factor)"))
    config.ess_increase_method ∈ (:fixed, :adaptive) ||
        throw(ArgumentError("ess_increase_method must be :fixed or :adaptive, got :$(config.ess_increase_method)"))
    0 < config.ascent_alpha < 1 ||
        throw(ArgumentError("ascent_alpha must be in (0,1), got $(config.ascent_alpha)"))
    0 < config.ascent_beta < 1 ||
        throw(ArgumentError("ascent_beta must be in (0,1), got $(config.ascent_beta)"))
    
    return true
end

"""
    _fit_mcem(model::MultistateModel; kwargs...)

Internal implementation: Fit a semi-Markov model to panel data via Monte Carlo EM (MCEM).

This is called by `fit()` when `is_panel_data(model) && !is_markov(model)`.

Uses Ipopt optimization for both constrained and unconstrained M-steps by default.

!!! note "Surrogate Required"
    MCEM requires a Markov surrogate for importance sampling proposals. 
    You must call `initialize_surrogate!(model)` or use `surrogate=:markov` in 
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

# Arguments

**Model and constraints:**
- `model::MultistateProcess`: semi-Markov model
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
- `penalty`: Penalty specification for spline hazards. Can be:
  - `:auto` (default): Apply `SplinePenalty()` if model has spline hazards, `nothing` otherwise
  - `:none`: Explicit opt-out for unpenalized fitting
  - `SplinePenalty()`: Curvature penalty on all spline hazards
  - `Vector{SplinePenalty}`: Multiple rules resolved by specificity
  - `nothing`: DEPRECATED - use `:none` instead
- `lambda_init::Float64=1.0`: initial smoothing parameter value (used when penalty is specified)

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
robust_se = sqrt.(diag(get_vcov(fitted; type=:ij)))

# Diagnose model specification
result = compare_variance_estimates(fitted)

# Check convergence
records = get_convergence_records(fitted)

# Use a custom solver for the M-step
using Optim
fitted = fit(semimarkov_model; solver=Optim.BFGS())

```

See also: [`fit`](@ref), [`compare_variance_estimates`](@ref)
"""
function _fit_mcem(model::MultistateModel; proposal::Union{Symbol, ProposalConfig} = :auto, constraints = nothing, solver = nothing, maxiter = DEFAULT_MCEM_MAXITER, tol = DEFAULT_MCEM_TOL, ascent_threshold = 0.1, stopping_threshold = 0.1, ess_growth_factor = sqrt(2.0), ess_increase_method::Symbol = :fixed, ascent_alpha::Float64 = 0.25, ascent_beta::Float64 = 0.25, ess_target_initial = DEFAULT_ESS_TARGET_INITIAL, max_ess = DEFAULT_MAX_ESS, max_sampling_effort = 20, npaths_additional = 10, block_hessian_speedup = 2.0, sir::Symbol = :adaptive_lhs, sir_pool_constant::Float64 = 2.0, sir_max_pool::Int = 8192, sir_resample::Symbol = :always, sir_degeneracy_threshold::Float64 = 0.7, sir_adaptive_threshold::Float64 = 2.0, sir_adaptive_min_iters::Int = 3, verbose = true, return_convergence_records = true, return_proposed_paths = false, compute_vcov = true, vcov_threshold = true, compute_ij_vcov = true, compute_jk_vcov = false, loo_method = :direct, penalty = :auto, lambda_init = 1.0, kwargs...)

    # Resolve penalty specification (handles :auto, :none, deprecation warning)
    resolved_penalty = _resolve_penalty(penalty, model)

    # Validate SIR parameters
    if sir ∉ (:none, :sir, :lhs, :adaptive_sir, :adaptive_lhs)
        throw(ArgumentError("sir must be :none, :sir, :lhs, :adaptive_sir, or :adaptive_lhs, got :$sir"))
    end
    if sir_resample ∉ (:always, :degeneracy)
        throw(ArgumentError("sir_resample must be :always or :degeneracy, got :$sir_resample"))
    end
    if sir_pool_constant <= 0
        throw(ArgumentError("sir_pool_constant must be positive, got $sir_pool_constant"))
    end
    if sir_max_pool <= 0
        throw(ArgumentError("sir_max_pool must be positive, got $sir_max_pool"))
    end
    if !(0 < sir_degeneracy_threshold < 1)
        throw(ArgumentError("sir_degeneracy_threshold must be in (0,1), got $sir_degeneracy_threshold"))
    end
    if sir_adaptive_threshold <= 0
        throw(ArgumentError("sir_adaptive_threshold must be positive, got $sir_adaptive_threshold"))
    end
    if sir_adaptive_min_iters < 1
        throw(ArgumentError("sir_adaptive_min_iters must be at least 1, got $sir_adaptive_min_iters"))
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

        initcons = consfun_semimarkov(zeros(length(constraints.cons)), get_parameters_flat(model), nothing)
        badcons = findall(initcons .< constraints.lcons .|| initcons .> constraints.ucons)
        if length(badcons) > 0
            throw(ArgumentError("Constraints $badcons are violated at the initial parameter values."))
        end
    end

    # check that max_sampling_effort is greater than 1
    if max_sampling_effort <= 1
        throw(ArgumentError("max_sampling_effort must be greater than 1, got $max_sampling_effort"))
    end

    # check that ess_growth_factor is greater than 1
    if ess_growth_factor <= 1
        throw(ArgumentError("ess_growth_factor must be greater than 1, got $ess_growth_factor"))
    end

    # Validate ess_increase_method parameter
    if ess_increase_method ∉ (:fixed, :adaptive)
        throw(ArgumentError("ess_increase_method must be :fixed or :adaptive, got :$ess_increase_method"))
    end
    if ascent_alpha <= 0 || ascent_alpha >= 1
        throw(ArgumentError("ascent_alpha must be in (0,1), got $ascent_alpha"))
    end
    if ascent_beta <= 0 || ascent_beta >= 1
        throw(ArgumentError("ascent_beta must be in (0,1), got $ascent_beta"))
    end
    use_adaptive_ess = ess_increase_method === :adaptive
    
    if verbose && use_adaptive_ess
        println("Using adaptive (Caffo) power-based ESS increase (α=$ascent_alpha, β=$ascent_beta).\n")
    end

    # throw a warning if trying to fit a spline model where the degree is 0 for all splines
    if all(map(x -> (isa(x, _MarkovHazard) | (isa(x, _SplineHazard) && (x.degree == 0) && (length(x.knots) == 2))), model.hazards))
        throw(ArgumentError("Attempting to fit a time-homogeneous Markov model via MCEM. Recode as exponential hazards and refit."))
    end

    # MCEM initialization
    keep_going = true
    iter = 0
    is_converged = false

    # number of subjects
    nsubj = length(model.subjectindices)

    # extract and initialize model parameters
    # Phase 3: Use ParameterHandling.jl flat parameters (natural scale since v0.3.0)
    params_cur = get_parameters_flat(model)

    # Build penalty configuration from resolved penalty
    penalty_config = if !isnothing(resolved_penalty)
        penalties = resolved_penalty isa SplinePenalty ? [resolved_penalty] : resolved_penalty
        build_penalty_config(model, penalties; lambda_init=lambda_init)
    else
        PenaltyConfig()  # Empty config - no penalty
    end
    use_penalty = has_penalties(penalty_config)
    
    if verbose && use_penalty
        println("Using penalized likelihood with $(penalty_config.n_lambda) smoothing parameter(s).\n")
    end

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

    # containers for traces
    mll_trace = Vector{Float64}() # marginal loglikelihood
    ess_trace = ElasticArray{Float64, 2}(undef, nsubj, 0) # effective sample size (one per subject)
    path_count_trace = ElasticArray{Int, 2}(undef, nsubj, 0) # actual path counts (one per subject)
    # Phase 3: Use ParameterHandling.jl flat parameter length
    parameters_trace = ElasticArray{Float64, 2}(undef, length(get_parameters_flat(model)), 0) # parameter estimates

    # ==========================================================================
    # Surrogate Validation
    # ==========================================================================
    # MCEM requires a fitted surrogate. The surrogate should be created and fitted
    # either at model construction time (surrogate=:markov/:phasetype in multistatemodel())
    # or explicitly via initialize_surrogate!(model) before calling fit().
    #
    # fit() does NOT create or fit surrogates - this is by design to ensure clear
    # ownership and avoid unexpected computation during fitting.
    # ==========================================================================
    
    if isnothing(model.surrogate)
        throw(ArgumentError(
            "MCEM requires a surrogate for importance sampling.\n\n" *
            "Solutions:\n" *
            "  1. Create model with surrogate: multistatemodel(...; surrogate=:markov)\n" *
            "  2. Initialize surrogate explicitly: initialize_surrogate!(model)\n\n" *
            "See ?initialize_surrogate! for options (type=:markov/:phasetype, method=:mle/:heuristic)."))
    end
    
    if !model.surrogate.fitted
        throw(ArgumentError(
            "Model has a surrogate but it is not fitted.\n\n" *
            "Solution: Call initialize_surrogate!(model; method=:mle) to fit the surrogate.\n\n" *
            "Note: If you used fit_surrogate=false in multistatemodel(), you must fit " *
            "the surrogate explicitly before calling fit()."))
    end
    
    # Get surrogate from model - it's already fitted at this point
    # Surrogates are self-contained: MarkovSurrogate or PhaseTypeSurrogate
    surrogate = model.surrogate
    
    if verbose
        surrogate_name = surrogate isa PhaseTypeSurrogate ? "phase-type" : "Markov"
        println("Using $surrogate_name surrogate for MCEM importance sampling.\n")
    end

    # ==========================================================================
    # Build MCEM Infrastructure (Phase 4 refactor)
    # ==========================================================================
    # All surrogate-specific infrastructure is now built via dispatch on surrogate type.
    # This eliminates ~100 lines of conditional branching and duplicate variables.
    # See infrastructure.jl for MCEMInfrastructure struct and builders.
    infra = build_mcem_infrastructure(model, surrogate; verbose=verbose)
    
    # Create containers NamedTuple for DrawSamplePaths!
    containers = (
        samplepaths = samplepaths,
        loglik_surrog = loglik_surrog,
        loglik_target_prop = loglik_target_prop,
        loglik_target_cur = loglik_target_cur,
        _logImportanceWeights = _logImportanceWeights,
        ImportanceWeights = ImportanceWeights,
        ess_cur = ess_cur,
        psis_pareto_k = psis_pareto_k
    )

    # Compute normalizing constant of proposal distribution via dispatch
    # For Markov: log-likelihood under Markov surrogate
    # For PhaseType: marginal likelihood via forward algorithm on expanded space
    NormConstantProposal = compute_normalizing_constant(model, infra)

    # draw sample paths until the target ess is reached 
    if verbose  println("Initializing sample paths ...\n") end
    
    # For SIR: sample pool_target paths instead of ess_target
    # After sampling, we'll resample ess_target indices from the pool
    sampling_target = use_sir ? sir_pool_target : ess_target
    
    # Draw sample paths using infrastructure-based dispatch (Phase 4 refactor)
    DrawSamplePaths!(model, infra, containers;
        ess_target = sampling_target,
        max_sampling_effort = max_sampling_effort,
        npaths_additional = npaths_additional,
        params_cur = params_cur)
    
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

    # Use parameter bounds from model (generated at construction time)
    # This ensures box constraints are enforced during M-step optimization
    lb, ub = model.bounds.lb, model.bounds.ub

    # generate optimization problem
    # If using penalty, wrap objective to include penalty term
    if use_penalty
        # Create penalized objective that captures penalty_config
        penalized_loglik = (params, data) -> loglik(params, data, penalty_config)
        if isnothing(constraints)
            optf = OptimizationFunction(penalized_loglik, Optimization.AutoForwardDiff())
            prob = OptimizationProblem(optf, params_cur, SMPanelData(model, samplepaths, ImportanceWeights); lb=lb, ub=ub)
        else
            optf = OptimizationFunction(penalized_loglik, Optimization.AutoForwardDiff(), cons = consfun_semimarkov)
            prob = OptimizationProblem(optf, params_cur, SMPanelData(model, samplepaths, ImportanceWeights); lb=lb, ub=ub, lcons = constraints.lcons, ucons = constraints.ucons)
        end
    else
        if isnothing(constraints)
            optf = OptimizationFunction(loglik, Optimization.AutoForwardDiff())
            prob = OptimizationProblem(optf, params_cur, SMPanelData(model, samplepaths, ImportanceWeights); lb=lb, ub=ub)
        else
            optf = OptimizationFunction(loglik, Optimization.AutoForwardDiff(), cons = consfun_semimarkov)
            prob = OptimizationProblem(optf, params_cur, SMPanelData(model, samplepaths, ImportanceWeights); lb=lb, ub=ub, lcons = constraints.lcons, ucons = constraints.ucons)
        end
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
                        @warn "SIR pool size capped at $sir_max_pool; further ESS increases will reduce SIR effectiveness." maxlog=1
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
    
            # Guard against NaN or negative ASE (can occur with degenerate importance weights)
            if !isfinite(ase) || ase < 0
                ase = 0.0
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

        # Update parameters, marginal log likelihood, target loglik
        params_cur        = deepcopy(params_prop)
        mll_cur           = deepcopy(mll_prop)
        # Copy proposed log-likelihoods into current (preserves container reference)
        # Note: Array sizes should match since both were updated by DrawSamplePaths!
        for i in eachindex(loglik_target_cur)
            copyto!(loglik_target_cur[i], loglik_target_prop[i])
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
            println("Estimate of the log marginal likelihood, l(θ): $(round(compute_loglik(model, loglik_surrog, loglik_target_cur, NormConstantProposal).loglik;digits=3))\n")
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
                    @warn "SIR pool size capped at $sir_max_pool; further ESS increases will reduce SIR effectiveness." maxlog=1
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
        
        # Draw sample paths using infrastructure-based dispatch (Phase 4 refactor)
        DrawSamplePaths!(model, infra, containers;
            ess_target = sampling_target,
            max_sampling_effort = max_sampling_effort,
            npaths_additional = npaths_additional,
            params_cur = params_cur)
        
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

        # Model-based variance is complex with constraints in MCEM due to Louis's identity.
        # For now, skip model-based variance but still compute IJ/JK (which work regardless).
        if compute_vcov == true
            @warn "Model-based variance (vcov) is not available for constrained MCEM or monotone splines. " *
                  "IJ (sandwich) variance will be computed if compute_ij_vcov=true (default)."
        end
        vcov = nothing
    
    elseif is_converged && compute_vcov
        if verbose
            println("Computing variance-covariance matrix at final estimates.")
        end

        # set up containers for path and sampling weight
        path = Array{SamplePath}(undef, 1)
        samplingweight = Vector{Float64}(undef, 1)
        
        # Get hazard parameter block sizes from hazard objects
        nhaz = length(model.hazards)
        block_sizes = [model.hazards[k].npar_total for k in 1:nhaz]
        
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

        # Compute vcov using shared tolerance computation (H2_P1 fix)
        nparams = length(params_cur)
        pinv_tol = _compute_vcov_tolerance(nsubj, nparams, vcov_threshold)
        vcov = pinv(Symmetric(fishinf), atol = pinv_tol)
        _clean_vcov_matrix!(vcov)
        vcov = Symmetric(vcov)
    else
        vcov = nothing
    end

    # subject marginal likelihood
    logliks = compute_loglik(model, loglik_surrog, loglik_target_cur, NormConstantProposal)

    # Compute robust variance estimates if requested
    # IJ/JK variance can be computed with or without constraints - they use subject-level
    # gradients which are computed at the MLE regardless of how it was found.
    ij_variance = nothing
    jk_variance = nothing
    subject_grads = nothing
    
    if (compute_ij_vcov || compute_jk_vcov) && is_converged
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
    # params_cur contains parameters; we need to split into per-hazard vectors
    # Compute block sizes from hazard objects
    block_sizes = [model.hazards[i].npar_total for i in 1:length(model.hazards)]
    
    # Split params_cur into per-hazard vectors (natural scale since v0.3.0)
    fitted_params = Vector{Vector{Float64}}(undef, length(block_sizes))
    offset = 0
    for i in eachindex(block_sizes)
        fitted_params[i] = params_cur[(offset+1):(offset+block_sizes[i])]
        offset += block_sizes[i]
    end
    
    # Rebuild parameters with proper constraints using model's hazard info
    # Skip validation since optimizer should have respected bounds
    parameters_fitted = rebuild_parameters(fitted_params, model; validate_bounds=false)

    # The surrogate is already set from model.surrogate at MCEM start
    # (can be MarkovSurrogate or PhaseTypeSurrogate)

    # wrap results
    model_fitted = MultistateModelFitted(
        data_original,
        parameters_fitted,
        model.bounds,  # Pass bounds from unfitted model
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
        surrogate,  # Single unified surrogate field (from model.surrogate)
        ConvergenceRecords,
        ProposedPaths,
        model.modelcall,
        model.phasetype_expansion,
        nothing,  # smoothing_parameters (not yet implemented for MCEM)
        nothing)  # edf

    # return fitted object
    return model_fitted;
end
