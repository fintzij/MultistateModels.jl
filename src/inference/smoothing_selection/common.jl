# =============================================================================
# End of Phase 3 New Functions
# =============================================================================

"""
    SmoothingSelectionState

Internal state for smoothing parameter selection via PIJCV/CV, storing cached matrices and intermediate results.

Note: This is separate from `PIJCVState` in variance.jl which is the lower-level
state for computing PIJCV criterion from matrices.

# Fields
- `beta_hat::Vector{Float64}`: Current coefficient estimate
- `H_unpenalized::Matrix{Float64}`: Unpenalized Hessian (sum of subject Hessians)
- `subject_grads::Matrix{Float64}`: Subject gradients (p × n)
- `subject_hessians::Vector{Matrix{Float64}}`: Subject Hessians
- `penalty_config::PenaltyConfig`: Penalty configuration
- `n_subjects::Int`: Number of subjects
- `n_params::Int`: Number of parameters
- `model::MultistateProcess`: Model for likelihood evaluation
- `data::ExactData`: Data container
- `pijcv_eval_cache::Union{Nothing, PIJCVEvaluationCache}`: Pre-built cache for efficient LOO evaluation
"""
mutable struct SmoothingSelectionState
    beta_hat::Vector{Float64}
    H_unpenalized::Matrix{Float64}
    subject_grads::Matrix{Float64}
    subject_hessians::Vector{Matrix{Float64}}
    penalty_config::PenaltyConfig
    n_subjects::Int
    n_params::Int
    model::MultistateProcess
    data::ExactData
    pijcv_eval_cache::Union{Nothing, PIJCVEvaluationCache}  # PIJCV optimization: lazily built
end

# Constructor for backward compatibility (without cache)
function SmoothingSelectionState(beta_hat, H_unpenalized, subject_grads, subject_hessians,
                                  penalty_config, n_subjects, n_params, model, data)
    SmoothingSelectionState(beta_hat, H_unpenalized, subject_grads, subject_hessians,
                            penalty_config, n_subjects, n_params, model, data, nothing)
end

# =============================================================================
# Helper Functions for Performance Iteration
# =============================================================================

"""
    compute_penalty_from_lambda(beta::AbstractVector{T}, lambda::AbstractVector, 
                                 config::PenaltyConfig) where T

Compute penalty term Σⱼ λⱼ βⱼᵀ Sⱼ βⱼ / 2 using explicit lambda values.

Unlike `compute_penalty` which uses lambdas stored in the config, this function
takes explicit lambda values as an argument. Used during optimization when
lambda is being varied.

# Arguments
- `beta`: Coefficient vector (natural scale)
- `lambda`: Vector of smoothing parameters
- `config`: Penalty configuration containing S matrices and index mappings

# Returns
Scalar penalty value (half the quadratic form)

# Notes
- Parameters are on natural scale with box constraints (β ≥ 0)
- Penalty is quadratic: P(β) = (λ/2) βᵀSβ
- This must match the behavior of `compute_penalty` in penalties.jl
"""
function compute_penalty_from_lambda(beta::AbstractVector{T}, lambda::AbstractVector, 
                                      config::PenaltyConfig) where T
    penalty = zero(T)
    lambda_idx = 1
    
    # Baseline hazard penalties - parameters on natural scale
    for term in config.terms
        β_j = @view beta[term.hazard_indices]
        penalty += lambda[lambda_idx] * dot(β_j, term.S * β_j)
        lambda_idx += 1
    end
    
    # Total hazard penalties - sum natural-scale coefficients
    for term in config.total_hazard_terms
        K = size(term.S, 1)
        β_total = zeros(T, K)
        for idx_range in term.hazard_indices
            β_k = @view beta[idx_range]
            β_total .+= β_k  # Parameters already on natural scale
        end
        penalty += lambda[lambda_idx] * dot(β_total, term.S * β_total)
        lambda_idx += 1
    end
    
    # Smooth covariate penalties - no transformation (linear predictor scale)
    if !isempty(config.shared_smooth_groups)
        # Build term -> lambda mapping
        term_to_lambda = Dict{Int, Int}()
        for (group_idx, group) in enumerate(config.shared_smooth_groups)
            for term_idx in group
                term_to_lambda[term_idx] = lambda_idx
            end
            lambda_idx += 1
        end
        # Handle ungrouped terms
        for term_idx in 1:length(config.smooth_covariate_terms)
            if !haskey(term_to_lambda, term_idx)
                term_to_lambda[term_idx] = lambda_idx
                lambda_idx += 1
            end
        end
        # Apply penalties
        for (term_idx, term) in enumerate(config.smooth_covariate_terms)
            β_k = beta[term.param_indices]
            penalty += lambda[term_to_lambda[term_idx]] * dot(β_k, term.S * β_k)
        end
    else
        # No sharing - each term gets its own lambda
        for term in config.smooth_covariate_terms
            β_k = beta[term.param_indices]
            penalty += lambda[lambda_idx] * dot(β_k, term.S * β_k)
            lambda_idx += 1
        end
    end
    
    return penalty / 2
end

"""
    fit_penalized_beta(model::MultistateProcess, data::ExactData, 
                       lambda::Vector{Float64}, penalty_config::PenaltyConfig,
                       beta_init::Vector{Float64};
                       lb=nothing, ub=nothing,
                       maxiters::Int=100, use_polyalgorithm::Bool=false,
                       verbose::Bool=false) -> Vector{Float64}

Fit coefficients β given fixed smoothing parameters λ.

Minimizes the penalized negative log-likelihood:
    f(β) = -ℓ(β) + (1/2) Σⱼ λⱼ βⱼᵀ Sⱼ βⱼ

# Arguments
- `model`: MultistateProcess model
- `data`: ExactData container
- `lambda`: Vector of smoothing parameters (natural scale)
- `penalty_config`: Penalty configuration with S matrices
- `beta_init`: Warm start for optimization
- `lb`, `ub`: Parameter bounds (optional; extracted from model if not provided)
- `maxiters`: Maximum optimizer iterations
- `use_polyalgorithm`: If true, use LBFGS→Ipopt; if false, pure Ipopt
- `verbose`: Print optimization progress

# Returns
Fitted coefficient vector β
"""
function fit_penalized_beta(model::MultistateProcess, data::ExactData,
                            lambda::Vector{Float64}, penalty_config::PenaltyConfig,
                            beta_init::Vector{Float64};
                            lb::Union{Nothing, Vector{Float64}}=nothing,
                            ub::Union{Nothing, Vector{Float64}}=nothing,
                            maxiters::Int=100, use_polyalgorithm::Bool=false,
                            verbose::Bool=false, ipopt_options...)
    
    # Extract bounds from model if not provided
    param_lb = isnothing(lb) ? model.bounds.lb : lb
    param_ub = isnothing(ub) ? model.bounds.ub : ub
    
    # Define penalized negative log-likelihood objective
    function penalized_nll(β, p)
        # Unpenalized negative log-likelihood
        nll = loglik_exact(β, data; neg=true)
        # Add penalty
        pen = compute_penalty_from_lambda(β, lambda, penalty_config)
        return nll + pen
    end
    
    # Set up optimization with automatic differentiation
    adtype = DifferentiationInterface.SecondOrder(
        Optimization.AutoForwardDiff(), 
        Optimization.AutoForwardDiff()
    )
    optf = OptimizationFunction(penalized_nll, adtype)
    prob = OptimizationProblem(optf, beta_init, nothing; lb=param_lb, ub=param_ub)
    
    # Merge default options with user overrides
    # Convert kwargs to NamedTuple to enable merging
    ipopt_options_nt = (;ipopt_options...)
    merged_options = merge(DEFAULT_IPOPT_OPTIONS, (maxiters=maxiters, tol=LAMBDA_SELECTION_INNER_TOL), ipopt_options_nt)
    
    if use_polyalgorithm
        # Phase 1: LBFGS warm-start with loose tolerance
        if verbose
            println("    β fit: LBFGS warm-start...")
        end
        sol_warmstart = solve(prob, LBFGS();
                              maxiters=min(maxiters, 50),
                              abstol=1e-2,
                              reltol=1e-2,
                              show_trace=false)
        
        # Phase 2: Ipopt refinement
        if verbose
            println("    β fit: Ipopt refinement...")
        end
        prob_refined = remake(prob, u0=sol_warmstart.u)
        sol = solve(prob_refined, IpoptOptimizer(); merged_options...)
    else
        # Pure Ipopt from warm start
        if verbose
            println("    β fit: Ipopt from warm start...")
        end
        sol = solve(prob, IpoptOptimizer(); merged_options...)
    end
    
    return sol.u
end

"""
    extract_lambda_vector(config::PenaltyConfig) -> Vector{Float64}

Extract the current λ values from a penalty configuration as a flat vector.
"""
function extract_lambda_vector(config::PenaltyConfig)
    lambdas = Float64[]
    
    # Baseline hazard terms
    for term in config.terms
        push!(lambdas, term.lambda)
    end
    
    # Total hazard terms
    for term in config.total_hazard_terms
        push!(lambdas, term.lambda_H)
    end
    
    # Smooth covariate terms (handle shared groups)
    if !isempty(config.shared_smooth_groups)
        for group in config.shared_smooth_groups
            # Use first term's lambda as representative
            term = config.smooth_covariate_terms[group[1]]
            push!(lambdas, term.lambda)
        end
        # Handle ungrouped terms
        grouped_indices = Set(vcat(config.shared_smooth_groups...))
        for (idx, term) in enumerate(config.smooth_covariate_terms)
            if idx ∉ grouped_indices
                push!(lambdas, term.lambda)
            end
        end
    else
        for term in config.smooth_covariate_terms
            push!(lambdas, term.lambda)
        end
    end
    
    return lambdas
end

