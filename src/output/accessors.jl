
# =============================================================================
# Phase-Type Fitted Model Detection and Accessors
# =============================================================================

"""
    is_phasetype_fitted(model::MultistateModelFitted) -> Bool

Check if a fitted model was produced by fitting a model with phase-type hazards.

Phase-type fitted models store additional information in `modelcall`:
- `is_phasetype`: Flag indicating phase-type origin
- `mappings`: `PhaseTypeMappings` for state space translation
- `original_parameters`: User-facing phase-type parameters (λ, μ)
- `original_tmat`: Original transition matrix (on observed states)
- `original_data`: Original data (on observed states)

# Example
```julia
# Fit a phase-type model
h12 = Hazard(:pt, 1, 2)
model = multistatemodel(h12; data=data, n_phases=Dict(1=>3))
fitted = fit(model)

# Check if phase-type
is_phasetype_fitted(fitted)  # true

# Access phase-type parameters (convenience function)
params = get_phasetype_parameters(fitted)
```

See also: [`get_phasetype_parameters`](@ref), [`get_mappings`](@ref), [`PhaseTypeExpansion`](@ref)
"""
function is_phasetype_fitted(model::MultistateModelFitted)
    haskey(model.modelcall, :is_phasetype) && model.modelcall.is_phasetype === true
end

# Fallback for other model types
is_phasetype_fitted(::MultistateProcess) = false

"""
    get_phasetype_parameters(model::MultistateModelFitted; scale::Symbol=:natural)

Get phase-type parameters from a fitted phase-type model.

Returns the user-facing phase-type parameterization (λ progression rates, μ exit rates)
computed from the fitted expanded parameters.

Throws an error if the model is not a fitted phase-type model.

# Arguments
- `model::MultistateModelFitted`: A fitted model (must have phase-type hazards)
- `scale::Symbol=:natural`: Parameter scale
  - `:natural` - Human-readable scale (rates as positive values)
  - `:flat` or `:estimation` or `:log` - Flat vector on log scale
  - `:nested` - Nested NamedTuple structure

# Returns
- `NamedTuple`: Phase-type parameters per original hazard

# Example
```julia
fitted = fit(phasetype_model)
params = get_phasetype_parameters(fitted)
# (h12 = [λ₁, μ₁, μ₂], h23 = [rate])
```

See also: [`is_phasetype_fitted`](@ref), [`get_expanded_parameters`](@ref)
"""
function get_phasetype_parameters(model::MultistateModelFitted; scale::Symbol=:natural)
    if !is_phasetype_fitted(model)
        throw(ArgumentError("Model is not a fitted phase-type model. " *
                           "Use `get_parameters(model)` for standard models."))
    end
    
    # Get fitted expanded parameters
    exp_params = model.parameters.natural  # Expanded params on natural scale
    
    # Get original hazard specifications to rebuild user-facing structure
    original_hazards = model.phasetype_expansion.original_hazards
    mappings = model.phasetype_expansion.mappings
    
    # Build user-facing parameters from fitted expanded parameters
    result_pairs = Vector{Pair{Symbol, Vector{Float64}}}()
    
    for orig_haz in original_hazards
        orig_name = Symbol("h$(orig_haz.statefrom)$(orig_haz.stateto)")
        
        if orig_haz isa MultistateModels.PhaseTypeHazard
            # Phase-type hazard: collect λ and μ rates from expanded hazards
            n = orig_haz.n_phases
            user_params = Float64[]
            
            # Collect λ rates (progression: λ₁, λ₂, ..., λₙ₋₁)
            for i in 1:(n-1)
                prog_name = Symbol("h$(orig_haz.statefrom)_$(Char('a' + i - 1))$(Char('a' + i))")
                if haskey(exp_params, prog_name)
                    push!(user_params, exp_params[prog_name][1])
                end
            end
            
            # Collect μ rates (exits: μ₁, μ₂, ..., μₙ)
            for i in 1:n
                exit_name = Symbol("h$(orig_haz.statefrom)$(orig_haz.stateto)_$(Char('a' + i - 1))")
                if haskey(exp_params, exit_name)
                    push!(user_params, exp_params[exit_name][1])
                end
            end
            
            push!(result_pairs, orig_name => user_params)
        else
            # Non-phase-type hazard: use expanded params directly
            if haskey(exp_params, orig_name)
                push!(result_pairs, orig_name => exp_params[orig_name])
            end
        end
    end
    
    params_natural = NamedTuple(result_pairs)
    
    if scale == :natural
        return params_natural
    elseif scale == :flat || scale == :estimation || scale == :log
        # v0.3.0+: All parameters on natural scale, flatten directly (no log transform)
        return reduce(vcat, [v for v in values(params_natural)])
    elseif scale == :nested
        # v0.3.0+: Convert to nested format on natural scale (no log transform)
        nested_pairs = [
            name => (baseline = NamedTuple{Tuple([Symbol("p$i") for i in 1:length(v)])}(v),)
            for (name, v) in result_pairs
        ]
        return NamedTuple(nested_pairs)
    else
        throw(ArgumentError("scale must be :natural, :flat, :estimation, :log, or :nested (got :$scale)"))
    end
end

"""
    get_mappings(model::MultistateModelFitted)
    get_mappings(model::MultistateModel)

Get the `PhaseTypeMappings` from a phase-type model.

Throws an error if the model does not have phase-type expansion.

# Returns
- `PhaseTypeMappings`: State space translation information

See also: [`is_phasetype_fitted`](@ref), [`has_phasetype_expansion`](@ref), [`PhaseTypeMappings`](@ref)
"""
function get_mappings(model::MultistateModelFitted)
    if !is_phasetype_fitted(model)
        throw(ArgumentError("Model is not a fitted phase-type model."))
    end
    return model.phasetype_expansion.mappings
end

function get_mappings(model::MultistateModel)
    if !has_phasetype_expansion(model)
        throw(ArgumentError("Model does not have phase-type expansion."))
    end
    return model.phasetype_expansion.mappings
end

"""
    get_original_data(model::MultistateModelFitted)
    get_original_data(model::MultistateModel)

Get the original (non-expanded) data from a phase-type model.

Throws an error if the model does not have phase-type expansion.

# Returns
- `DataFrame`: Original data on observed state space

See also: [`is_phasetype_fitted`](@ref), [`has_phasetype_expansion`](@ref)
"""
function get_original_data(model::MultistateModelFitted)
    if !is_phasetype_fitted(model)
        throw(ArgumentError("Model is not a fitted phase-type model."))
    end
    return model.phasetype_expansion.original_data
end

function get_original_data(model::MultistateModel)
    if !has_phasetype_expansion(model)
        throw(ArgumentError("Model does not have phase-type expansion."))
    end
    return model.phasetype_expansion.original_data
end

"""
    get_original_tmat(model::MultistateModelFitted)
    get_original_tmat(model::MultistateModel)

Get the original transition matrix from a phase-type model.

Throws an error if the model does not have phase-type expansion.

# Returns
- `Matrix{Int64}`: Original transition matrix on observed state space

See also: [`is_phasetype_fitted`](@ref), [`has_phasetype_expansion`](@ref)
"""
function get_original_tmat(model::MultistateModelFitted)
    if !is_phasetype_fitted(model)
        throw(ArgumentError("Model is not a fitted phase-type model."))
    end
    return model.phasetype_expansion.original_tmat
end

function get_original_tmat(model::MultistateModel)
    if !has_phasetype_expansion(model)
        throw(ArgumentError("Model does not have phase-type expansion."))
    end
    return model.phasetype_expansion.original_tmat
end

"""
    get_convergence(model::MultistateModelFitted)

Check if the model fitting converged.

Checks the convergence records from optimization.

# Returns
- `Bool`: Whether the model fitting converged
"""
function get_convergence(model::MultistateModelFitted)
    # Check ConvergenceRecords for all model types
    if !isnothing(model.ConvergenceRecords)
        if haskey(model.ConvergenceRecords, :retcode)
            return model.ConvergenceRecords.retcode == :Success
        elseif haskey(model.ConvergenceRecords, :solution)
            sol = model.ConvergenceRecords.solution
            if hasfield(typeof(sol), :retcode)
                # Works with both Symbol and ReturnCode types
                return sol.retcode == ReturnCode.Success || sol.retcode == :Success
            end
        end
    end
    return true  # Assume converged if no info
end

# Deprecated: PhaseTypeFittedModel is no longer used
# Fitted phase-type models are now returned as MultistateModelFitted with
# phase-type specific info stored in modelcall. Use is_phasetype_fitted() to check.

"""
    get_loglik(model::MultistateModelFitted; type::Symbol=:loglik, ll::Union{Nothing,String}=nothing) 

Return the log likelihood at the maximum likelihood estimates. 

# Arguments 
- `model`: fitted model
- `type::Symbol=:loglik`: one of:
  - `:loglik` (default) - observed data log likelihood
  - `:subj_lml` - log marginal likelihood at the subject level
- `ll::String` (deprecated): use `type` instead

# Examples
```julia
get_loglik(fitted)                  # Total log-likelihood
get_loglik(fitted; type=:subj_lml)  # Subject-level marginal log-likelihoods
```
"""
function get_loglik(model::MultistateModelFitted; type::Symbol=:loglik, ll::Union{Nothing,String}=nothing)
    # Handle deprecated string argument
    if !isnothing(ll)
        Base.depwarn("String argument `ll` is deprecated, use `type::Symbol` instead (e.g., `type=:loglik`)", :get_loglik)
        type = Symbol(ll)
    end
    
    if type === :loglik
        return model.loglik.loglik
    elseif type === :subj_lml
        return model.loglik.subj_lml
    else
        throw(ArgumentError("Unknown type :$type. Use :loglik or :subj_lml"))
    end
end

"""
    get_parameters(model::MultistateProcess; scale::Symbol=:natural, expanded::Bool=false)

Return model parameters on the specified scale.

For phase-type models (unfitted), `expanded=false` (default) returns the user-facing 
phase-type parameters (λ, μ) from the original hazard specifications.
Set `expanded=true` to get the internal expanded Markov parameters.

For non-phase-type models, the `expanded` argument is ignored.

# Arguments 
- `model`: A MultistateProcess (unfitted)
- `scale::Symbol=:natural`: Parameter scale
  - `:natural` - Human-readable scale (exp applied to baseline params)
  - `:estimation` or `:log` - Flat vector on log scale (for optimization)
  - `:nested` - Nested NamedTuple with ParameterHandling structure
- `expanded::Bool=false`: For phase-type models, whether to return expanded parameters

# Returns
- For `:natural`: `NamedTuple` with parameters per hazard on natural scale
- For `:estimation`/`:log`: `Vector{Float64}` flat parameter vector
- For `:nested`: `NamedTuple` with full ParameterHandling structure

# Example
```julia
# Get human-readable parameters
params = get_parameters(model)  # Same as scale=:natural

# For phase-type models, get user-facing params
params = get_parameters(model)  # (h12 = [λ, μ₁, μ₂], h23 = [rate])

# Get expanded internal parameters
params = get_parameters(model; expanded=true)  # (h1_ab = [...], h12_a = [...], ...)
```

# See also
- [`set_parameters!`](@ref) - Set model parameters
- [`get_parnames`](@ref) - Get parameter names
- [`get_expanded_parameters`](@ref) - Get expanded parameters
"""
function get_parameters(model::MultistateProcess; scale::Symbol=:natural, expanded::Bool=false)
    # For phase-type models with expanded=false, return user-facing parameters
    if !expanded && has_phasetype_expansion(model)
        orig_params = model.phasetype_expansion.original_parameters
        if scale == :natural
            return orig_params.natural
        elseif scale == :estimation || scale == :log || scale == :flat
            return orig_params.flat
        elseif scale == :nested
            return orig_params.nested
        else
            throw(ArgumentError("scale must be :natural, :estimation, :log, :flat, or :nested (got :$scale)"))
        end
    end
    
    # Otherwise, return expanded/internal parameters
    if scale == :natural
        return model.parameters.natural
    elseif scale == :estimation || scale == :log || scale == :flat
        return model.parameters.flat
    elseif scale == :nested
        return model.parameters.nested
    else
        throw(ArgumentError("scale must be :natural, :estimation, :log, :flat, or :nested (got :$scale)"))
    end
end

"""
    get_parameters(model::MultistateModelFitted; scale::Symbol=:natural, expanded::Bool=false)

Return model parameters on the specified scale.

For fitted phase-type models, `expanded=false` (default) returns the user-facing 
phase-type parameters (λ, μ). Set `expanded=true` to get the internal expanded
Markov parameters.

For non-phase-type fitted models, the `expanded` argument is ignored.

# Arguments 
- `model`: A fitted model (MultistateModelFitted)
- `scale::Symbol=:natural`: Parameter scale
  - `:natural` - Human-readable scale (rates as positive values)
  - `:estimation` or `:log` or `:flat` - Flat vector on log scale
  - `:nested` - Nested NamedTuple structure
- `expanded::Bool=false`: For phase-type models, whether to return expanded parameters

# Returns
- For `:natural`: `NamedTuple` with parameters per hazard on natural scale
- For `:estimation`/`:log`/`:flat`: `Vector{Float64}` flat parameter vector
- For `:nested`: `NamedTuple` with full structure

# Example
```julia
# For phase-type fitted models:
fitted = fit(phasetype_model)

# Get user-facing phase-type parameters (default)
params = get_parameters(fitted)
# (h12 = [λ₁, μ₁, μ₂], h23 = [rate])

# Get expanded internal parameters  
exp_params = get_parameters(fitted; expanded=true)
# (h1_prog1 = [λ₁], h12_exit1 = [μ₁], ...)
```

# See also
- [`get_phasetype_parameters`](@ref) - Get phase-type params explicitly
- [`get_expanded_parameters`](@ref) - Get expanded params explicitly
- [`is_phasetype_fitted`](@ref) - Check if model is fitted phase-type
"""
function get_parameters(model::MultistateModelFitted; scale::Symbol=:natural, expanded::Bool=false)
    # For phase-type models with expanded=false, return user-facing parameters
    if is_phasetype_fitted(model) && !expanded
        return get_phasetype_parameters(model; scale=scale)
    end
    
    # Otherwise, return expanded/internal parameters
    if scale == :natural
        return model.parameters.natural
    elseif scale == :estimation || scale == :log || scale == :flat
        return model.parameters.flat
    elseif scale == :nested
        return model.parameters.nested
    else
        throw(ArgumentError("scale must be :natural, :estimation, :log, :flat, or :nested (got :$scale)"))
    end
end

"""
    get_expanded_parameters(model::MultistateModelFitted; scale::Symbol=:natural)
    get_expanded_parameters(model::MultistateModel; scale::Symbol=:natural)

Get the expanded (internal) parameters from a model.

For phase-type models, this returns the internal Markov parameters 
(progression λ and exit μ rates as separate hazards).

For non-phase-type models, this is equivalent to `get_parameters`.

# Example
```julia
fitted = fit(phasetype_model)
exp_params = get_expanded_parameters(fitted)
# (h1_prog1 = [λ₁], h1_prog2 = [λ₂], h12_exit1 = [μ₁], ...)
```

See also: [`get_parameters`](@ref), [`get_phasetype_parameters`](@ref)
"""
function get_expanded_parameters(model::MultistateModelFitted; scale::Symbol=:natural)
    return get_parameters(model; scale=scale, expanded=true)
end

function get_expanded_parameters(model::MultistateModel; scale::Symbol=:natural)
    # For unfitted models, expanded parameters are just the model's parameters
    # (which are already on the expanded space for phase-type models)
    if scale == :natural
        return model.parameters.natural
    elseif scale == :estimation || scale == :log || scale == :flat
        return model.parameters.flat
    elseif scale == :nested
        return model.parameters.nested
    else
        throw(ArgumentError("scale must be :natural, :estimation, :log, :flat, or :nested (got :$scale)"))
    end
end

"""
    get_parnames(model; flatten=false)    

Return the parameter names.

# Arguments
- `model`: A MultistateProcess or MarkovSurrogate
- `flatten::Bool = false`: If true, return a single vector of all parameter names.
  If false (default), return a vector of vectors grouped by hazard.

# Returns
- If `flatten=false`: Vector of parameter name vectors, one per hazard
- If `flatten=true`: Single vector of all parameter names (useful for constraints)

# Example
```julia
# Per-hazard grouping (default)
get_parnames(model)  # [[:log_λ_h12], [:log_λ_h21, :log_shape_h21]]

# Flattened for constraint specification
get_parnames(model; flatten=true)  # [:log_λ_h12, :log_λ_h21, :log_shape_h21]
```
"""
function get_parnames(model::MultistateProcess; flatten::Bool = false)
    names_per_hazard = [x.parnames for x in model.hazards]
    if flatten
        return reduce(vcat, names_per_hazard)
    else
        return names_per_hazard
    end
end

"""
    get_parnames(surrogate::MarkovSurrogate; flatten=false)

Return the parameter names for a Markov surrogate.

# Arguments
- `surrogate::MarkovSurrogate`: A Markov surrogate
- `flatten::Bool = false`: If true, return a single vector of all parameter names.

# Returns
- Vector of parameter name vectors, one per hazard (or flattened if requested)
"""
function get_parnames(surrogate::MarkovSurrogate; flatten::Bool = false)
    names_per_hazard = [x.parnames for x in surrogate.hazards]
    if flatten
        return reduce(vcat, names_per_hazard)
    else
        return names_per_hazard
    end
end

"""
    get_vcov(model::MultistateModelFitted; type::Symbol=:model) 

Return the variance-covariance matrix at the maximum likelihood estimate.

# Arguments 
- `model::MultistateModelFitted`: fitted model
- `type::Symbol=:model`: Type of variance estimator
  - `:model` - Model-based variance (inverse Hessian, H⁻¹)
  - `:ij` - Infinitesimal jackknife / sandwich / robust variance (H⁻¹ K H⁻¹)
  - `:jk` - Jackknife variance ((n-1)/n times IJ variance)

# Returns
- `Symmetric{Float64, Matrix{Float64}}`: p × p variance-covariance matrix, or `nothing` if not computed

# Details
- `:model` - Valid under correct model specification. Requires `compute_vcov=true` (default).
- `:ij` - Also known as robust/Huber-White variance. Valid under model misspecification.
  Requires `compute_ij_vcov=true` during fitting.
- `:jk` - Jackknife variance with finite-sample correction. 
  Related to IJ by: Var_JK = ((n-1)/n) × Var_IJ.
  Requires `compute_jk_vcov=true` during fitting.

# Example
```julia
# Fit model with all variance types
fitted = fit(model; compute_ij_vcov=true, compute_jk_vcov=true)

# Compare standard errors
model_se = sqrt.(diag(get_vcov(fitted)))                    # Model-based
robust_se = sqrt.(diag(get_vcov(fitted; type=:ij)))         # Robust/sandwich
jk_se = sqrt.(diag(get_vcov(fitted; type=:jk)))             # Jackknife

# Ratio of robust to model-based SE (>1 suggests misspecification)
println("SE ratio (robust/model): ", robust_se ./ model_se)
```

# See also
- [`get_subject_gradients`](@ref) - Get subject-level score vectors
- [`get_pseudovalues`](@ref) - Get jackknife or IJ pseudo-values
"""
function get_vcov(model::MultistateModelFitted; type::Symbol=:model)
    if type == :model
        if isnothing(model.vcov)
            @warn "Model-based variance-covariance matrix was not computed for this model."
        end
        return model.vcov
    elseif type == :ij
        if isnothing(model.ij_vcov)
            @warn "IJ variance-covariance matrix was not computed. Refit with compute_ij_vcov=true."
        end
        return model.ij_vcov
    elseif type == :jk
        if isnothing(model.jk_vcov)
            @warn "Jackknife variance-covariance matrix was not computed. Refit with compute_jk_vcov=true."
        end
        return model.jk_vcov
    else
        throw(ArgumentError("type must be :model, :ij, or :jk (got :$type)"))
    end
end

# Internal helpers - prefer get_vcov(model; type=:ij/:jk) for new code
function get_ij_vcov(model::MultistateModelFitted)
    return get_vcov(model; type=:ij)
end

function get_jk_vcov(model::MultistateModelFitted)
    return get_vcov(model; type=:jk)
end

"""
    get_subject_gradients(model::MultistateModelFitted)

Return the subject-level score vectors (gradients of log-likelihood).

# Description
The score vector for subject i is:
```math
g_i = ∇_θ ℓ_i(θ̂) = ∂ log L_i(θ̂) / ∂θ
```

These represent each subject's contribution to the gradient of the total log-likelihood.
At the MLE, the sum of scores is approximately zero (Σᵢ gᵢ ≈ 0), but individual
scores are typically non-zero.

# Arguments
- `model::MultistateModelFitted`: fitted model (must have been fit with `compute_ij_vcov=true` or `compute_jk_vcov=true`)

# Returns
- `Matrix{Float64}`: p × n matrix where column i contains the score vector gᵢ for subject i,
  or `nothing` if not computed

# Uses
- Building blocks for IJ/JK variance: K = Σᵢ gᵢgᵢᵀ
- Influence diagnostics: large ||gᵢ|| indicates influential observations
- Model checking: patterns in scores may indicate misspecification

# Example
```julia
fitted = fit(model, data; compute_ij_vcov=true)
grads = get_subject_gradients(fitted)

# Check that scores sum to ~0 at MLE
println("Sum of scores: ", sum(grads, dims=2))

# Find most influential subjects
grad_norms = [norm(grads[:, i]) for i in 1:size(grads, 2)]
top_influential = sortperm(grad_norms, rev=true)[1:5]
```

See also: [`get_influence_functions`](@ref), [`get_loo_perturbations`](@ref)
"""
function get_subject_gradients(model::MultistateModelFitted)
    if isnothing(model.subject_gradients)
        println("Subject gradients were not computed for this model.")
    end
    model.subject_gradients
end

"""
    get_loo_perturbations(model::MultistateModelFitted; method=:direct)

Return the leave-one-out (LOO) parameter perturbations.

# Formula
The LOO perturbation for subject i approximates how much the estimates would change
if that subject were removed:
```math
Δᵢ = θ̂_{-i} - θ̂ ≈ H⁻¹ gᵢ
```
where:
- θ̂₋ᵢ is the estimate with subject i removed
- H is the Hessian (observed Fisher information)
- gᵢ is the score vector for subject i

The full LOO estimate is: `θ̂₋ᵢ ≈ θ̂ + Δᵢ`

# Arguments
- `model::MultistateModelFitted`: fitted model (requires `compute_ij_vcov=true` or `compute_jk_vcov=true`)
- `method::Symbol=:direct`: computation method (currently only `:direct` is supported for accessor)

# Returns
- `Matrix{Float64}`: p × n matrix where column i is the perturbation Δᵢ

# Example
```julia
fitted = fit(model, data; compute_ij_vcov=true)
deltas = get_loo_perturbations(fitted)
theta_hat = get_parameters_flat(fitted)

# Approximate LOO estimate for subject 5
theta_loo_5 = theta_hat .+ deltas[:, 5]

# Which parameters are most affected by removing subject 5?
most_affected = sortperm(abs.(deltas[:, 5]), rev=true)[1:3]
```

# Note
This uses the one-step Newton approximation, which is very accurate for smooth
likelihood functions and avoids refitting the model n times.

See also: [`get_influence_functions`](@ref), [`get_jk_pseudovalues`](@ref)
"""
function get_loo_perturbations(model::MultistateModelFitted; method::Symbol=:direct)
    if isnothing(model.subject_gradients)
        throw(ArgumentError("Subject gradients were not computed. Refit with compute_ij_vcov=true or compute_jk_vcov=true."))
    end
    if isnothing(model.vcov)
        throw(ArgumentError("Variance-covariance matrix was not computed. Refit with compute_vcov=true."))
    end
    
    return loo_perturbations_direct(model.vcov, model.subject_gradients)
end

"""
    get_pseudovalues(model::MultistateModelFitted; type::Symbol=:jk)

Return pseudo-values for each subject.

# Arguments
- `model::MultistateModelFitted`: fitted model (requires `compute_ij_vcov=true` or `compute_jk_vcov=true`)
- `type::Symbol=:jk`: Type of pseudo-values
  - `:jk` - Jackknife pseudo-values: θ̃ᵢ = θ̂ - (n-1)·Δᵢ
  - `:ij` - IJ pseudo-values (LOO estimates): θ̃ᵢ = θ̂ + Δᵢ

# Returns
- `Matrix{Float64}`: p × n matrix where column i is the pseudo-value for subject i

# Details
**Jackknife pseudo-values** (`:jk`):
- Formula: θ̃ᵢ = n·θ̂ - (n-1)·θ̂₋ᵢ = θ̂ - (n-1)·Δᵢ
- Property: mean(θ̃) = θ̂ (unbiased)
- Use: Can be treated as approximately IID for variance estimation

**IJ pseudo-values** (`:ij`):
- Formula: θ̃ᵢ = θ̂ + Δᵢ ≈ θ̂₋ᵢ (leave-one-out estimate)
- Property: These ARE the approximate LOO estimates
- Use: Influence diagnostics, model checking

# Example
```julia
fitted = fit(model; compute_ij_vcov=true)

# Get jackknife pseudo-values
jk_pseudo = get_pseudovalues(fitted)  # default :jk
# Mean equals the MLE
mean(jk_pseudo, dims=2) ≈ get_parameters(fitted; scale=:flat)

# Get IJ pseudo-values (LOO estimates)
ij_pseudo = get_pseudovalues(fitted; type=:ij)
# Column i is approximately θ̂₋ᵢ
```

# References
- Efron, B. (1982). The Jackknife, the Bootstrap and Other Resampling Plans. SIAM.
- Miller, R. G. (1974). The jackknife - a review. Biometrika, 61(1), 1-15.

# See also
- [`get_vcov`](@ref) - Get variance-covariance matrices
- [`get_loo_perturbations`](@ref) - Get raw LOO perturbations Δᵢ
- [`get_influence_functions`](@ref) - Alias for LOO perturbations
"""
function get_pseudovalues(model::MultistateModelFitted; type::Symbol=:jk)
    deltas = get_loo_perturbations(model)
    theta_hat = get_parameters(model; scale=:flat)
    
    if type == :jk
        # θ̃ᵢ = θ̂ - (n-1)·Δᵢ
        n = size(deltas, 2)
        return theta_hat .- (n - 1) .* deltas
    elseif type == :ij
        # θ̃ᵢ = θ̂ + Δᵢ (the LOO estimate itself)
        return theta_hat .+ deltas
    else
        throw(ArgumentError("type must be :jk or :ij (got :$type)"))
    end
end

# Internal helpers - prefer get_pseudovalues(model; type=:jk/:ij) for new code
function get_jk_pseudovalues(model::MultistateModelFitted)
    return get_pseudovalues(model; type=:jk)
end

function get_ij_pseudovalues(model::MultistateModelFitted)
    return get_pseudovalues(model; type=:ij)
end

"""
    get_influence_functions(model::MultistateModelFitted)

Return the empirical influence function values for each subject.

# Formula
The influence function for subject i is:
```math
IF_i = H⁻¹ gᵢ = Δᵢ
```

This is identical to the LOO perturbation and measures how much each subject
"pulls" the estimate away from where it would be without that subject.

# Interpretation
- Large ||IFᵢ|| indicates subject i has high influence on the estimates
- Direction of IFᵢ shows which parameters are most affected
- Sum of squared influence functions relates to IJ variance

# Arguments
- `model::MultistateModelFitted`: fitted model (requires `compute_ij_vcov=true` or `compute_jk_vcov=true`)

# Returns
- `Matrix{Float64}`: p × n matrix where column i is the influence function IFᵢ

# Diagnostic Use
```julia
fitted = fit(model, data; compute_ij_vcov=true)
IF = get_influence_functions(fitted)

# Identify highly influential subjects
IF_norms = [norm(IF[:, i]) for i in 1:size(IF, 2)]
high_influence = findall(IF_norms .> 2 * mean(IF_norms))
println("Subjects with high influence: ", high_influence)

# Plot influence for first parameter
using Plots
scatter(1:size(IF, 2), IF[1, :], xlabel="Subject", ylabel="Influence on θ₁")
```

# Note
This function is an alias for `get_loo_perturbations(model)` provided for
conceptual clarity when the focus is on influence diagnostics.

See also: [`get_loo_perturbations`](@ref), [`get_subject_gradients`](@ref)
"""
function get_influence_functions(model::MultistateModelFitted)
    return get_loo_perturbations(model)
end


"""
    get_convergence_records(model::MultistateModelFitted) 

Return the convergence records for the fit. 

# Arguments 
- model: fitted model
"""
function get_convergence_records(model::MultistateModelFitted) 
    model.ConvergenceRecords
end

"""
    summary(model::MultistateModelFitted; compute_se=true, confidence_level=0.95, 
            estimate_likelihood=false, min_ess=100)

Generate a summary of a fitted multistate model with parameter estimates,
standard errors, and confidence intervals.

# Arguments
- `model::MultistateModelFitted`: A fitted multistate model
- `compute_se::Bool=true`: Whether to compute standard errors and confidence intervals
- `confidence_level::Float64=0.95`: Confidence level for intervals (default 95%)
- `estimate_likelihood::Bool=false`: Whether to estimate log-likelihood via importance sampling
- `min_ess::Int=100`: Minimum effective sample size for IS likelihood estimation

# Returns
A NamedTuple containing:
- `summary`: NamedTuple of DataFrames with parameter estimates for each hazard
- `loglik`: Log-likelihood value
- `AIC`: Akaike Information Criterion
- `BIC`: Bayesian Information Criterion  
- `MCSE_loglik`: Monte Carlo standard error of log-likelihood (if estimated)

# Example
```julia
fitted = fit(model)
s = summary(fitted)
s.summary.h12  # DataFrame for hazard h12
s.loglik       # Log-likelihood
s.AIC          # AIC
```
"""
function summary(model::MultistateModelFitted; compute_se::Bool=true, confidence_level::Float64=0.95, 
                 estimate_likelihood::Bool=false, min_ess::Int=100) 
    
    # Get parameters on natural scale
    mle = get_parameters(model; scale=:natural)
    
    # Get flat parameters for SE computation (vcov is on estimation scale)
    flat_pars = model.parameters.flat
    
    # Container for summary tables
    summary_tables = Dict{Symbol, DataFrame}()
    
    # Process each hazard
    sorted_hazkeys = sort(collect(model.hazkeys), by = x -> x[2])
    par_offset = 0  # Track position in flat parameter vector
    
    for (hazname, idx) in sorted_hazkeys
        haz = model.hazards[idx]
        npar = haz.npar_total
        parnames = haz.parnames
        estimates = collect(mle[hazname])
        
        # Clean up parameter names for display
        clean_names = String[]
        for pname in parnames
            pname_str = string(pname)
            # Remove hazard prefix if present
            prefix = string(hazname) * "_"
            if startswith(pname_str, prefix)
                push!(clean_names, pname_str[length(prefix)+1:end])
            else
                push!(clean_names, pname_str)
            end
        end
        
        if isnothing(model.vcov) || !compute_se
            # No standard errors available
            df = DataFrame(
                parameter = clean_names,
                estimate = estimates
            )
        else
            # Extract SEs for this hazard from the diagonal of vcov
            varcov = get_vcov(model)
            se_flat = sqrt.(diag(varcov))
            se = se_flat[par_offset+1:par_offset+npar]
            
            # Critical value for CI
            z_crit = quantile(Normal(0.0, 1.0), 1 - (1 - confidence_level) / 2)
            
            df = DataFrame(
                parameter = clean_names,
                estimate = estimates,
                se = se,
                lower = estimates .- z_crit .* se,
                upper = estimates .+ z_crit .* se
            )
        end
        
        summary_tables[hazname] = df
        par_offset += npar
    end
    
    # Convert to NamedTuple
    summary_nt = NamedTuple(summary_tables)
    
    # Print warning if no vcov
    if isnothing(model.vcov) && compute_se
        @warn "Variance-covariance matrix not computed. Standard errors and confidence intervals unavailable."
    end
    
    # Log likelihood
    if estimate_likelihood
        ll_result = estimate_loglik(model; min_ess=min_ess)
        ll = ll_result.loglik
        mcse = ll_result.mcse_loglik
    else
        ll = get_loglik(model; ll="loglik")
        mcse = nothing
    end
    
    # Information criteria
    AIC = MultistateModels.aic(model; loglik=ll)
    BIC = MultistateModels.bic(model; loglik=ll)

    return (summary=summary_nt, loglik=ll, AIC=AIC, BIC=BIC, MCSE_loglik=mcse)
end

"""
    estimate_loglik(model::MultistateProcess; min_ess = 100)
    
Estimate the log marginal likelihood for a fitted multistate model. Require that the minimum effective sample size per subject is greater than min_ess.  

# Arguments
- min_ess: minimum effective sample size, defaults to 100.
"""
function estimate_loglik(model::MultistateProcess; min_ess = 100)

    # sample paths and grab logliks
    result = draw_paths(model; min_ess = min_ess, paretosmooth = false, return_logliks = true)
    
    # For exact Markov data, draw_paths returns (loglik, subj_lml) directly
    if haskey(result, :loglik) && haskey(result, :subj_lml) && !haskey(result, :samplepaths)
        return (loglik = result.loglik, loglik_subj = result.subj_lml, 
                mcse_loglik = 0.0, mcse_loglik_subj = zeros(length(result.subj_lml)))
    end
    
    # For importance sampling case, unpack the full result
    samplepaths, loglik_target, subj_ess, loglik_surrog, ImportanceWeightsNormalized, ImportanceWeights = result

    # calculate the log marginal likelihood
    subj_ml = map(w -> mean(w), ImportanceWeights) # need to use the *un-normalized* weights
    subj_lml = log.(subj_ml)
    observed_lml = sum(subj_lml .* model.SubjectWeights)

    # calculate MCSEs
    subj_ml_var = map(w -> length(w) == 1 ? 0.0 : var(w) / length(w), ImportanceWeights)
    subj_lml_var = subj_ml_var ./ (subj_ml.^2) # delta method

    # sum and include subject weights
    observed_lml_var = sum(subj_lml_var .* model.SubjectWeights.^2)

    # return log likelihoods
    return (loglik = observed_lml, loglik_subj = subj_lml, mcse_loglik = sqrt(observed_lml_var), mcse_loglik_subj = sqrt.(subj_lml_var))
end

"""
    compute_loglik(model::MultistateProcess, loglik_surrog)
    
Compute the log marginal likelihood from sample paths.  

# Arguments
- min_ess: minimum effective sample size, defaults to 100.
"""
function compute_loglik(model::MultistateProcess, loglik_surrog, loglik_target, NormConstantProposal)
    ImportanceWeightsUnnormalized = [exp.(loglik_target[i] .- loglik_surrog[i]) for i in eachindex(loglik_surrog)]

    # calculate the log marginal likelihood
    ml_subj = map(w -> mean(w), ImportanceWeightsUnnormalized) # need to use the *un-normalized* weights
    lml_subj = log.(ml_subj)
    lml = sum(lml_subj .* model.SubjectWeights) + NormConstantProposal

    # # calculate MCSEs
    # var_lml_subj = map(w -> length(w) == 1 ? 0.0 : var(w) / length(w), ImportanceWeightsUnnormalized)
    # var_lml_subj = var_lml_subj ./ (lml_subj.^2) # delta method

    # # sum and include subject weights
    # var_lml = sum(var_lml_subj .* model.SubjectWeights.^2)

    # return log likelihoods
    return (loglik = lml, loglik_subj = lml_subj) #, mcse = sqrt(var_lml), mcse_loglik_subj = sqrt.(var_lml_subj))

end


"""
    aic(model::MultistateModelFitted; estimate_likelihood = false, min_ess = 100)

Akaike's Information Criterion, defined as -2*log(L) + 2*k, where L is the likelihood and k is the number of consumed degrees of freedom. 

# Arguments
- loglik: value of the loglikelihood to use, if provided.
- estimate_likelihood: logical; whether to estimate the loglikelihood, defaults to true.  
- min_ess: minimum effective sample size per subject, defaults to 100.
"""
function aic(model::MultistateModelFitted; loglik = nothing, estimate_likelihood = true, min_ess = 100)

    if !estimate_likelihood & all(isa.(model.hazards, _MarkovHazard))
        @warn "The log-likelihood for a Markov model includes a normalizing constant that is intractable for other families. Setting estimate_likelihood=true is strongly recommended when comparing different families of models."
    end

    # number of parameters
    # Phase 3: Use ParameterHandling.jl flat parameter length
    p = length(get_parameters_flat(model))

    # loglik
    ll = if !isnothing(loglik)
        loglik
    elseif !estimate_likelihood
        get_loglik(model; ll = "loglik")
    else
        estimate_loglik(model; min_ess = min_ess).loglik
    end

    # AIC
    AIC = - 2 * ll + 2 * p

    return AIC
end

"""
    bic(model::MultistateModelFitted; estimate_likelihood = false, min_ess = 100, paretosmooth = true)

Bayesian Information Criterion, defined as -2*log(L) + k*log(n), where L is the likelihood and k is the number of consumed degrees of freedom.

# Arguments
- loglik: value of the loglikelihood to use, if provided.
- estimate_likelihood: logical; whether to estimate the loglikelihood, defaults to true.  
- min_ess: minimum effective sample size per subject, defaults to 100.
"""
function bic(model::MultistateModelFitted; loglik = nothing, estimate_likelihood = true, min_ess = 100)

    if !estimate_likelihood & all(isa.(model.hazards, _MarkovHazard))
        @warn "The log-likelihood for a Markov model includes a normalizing constant that is intractable for other families. Setting estimate_likelihood=true is strongly recommended when comparing different families of models."
    end

    # number of parameters
    # Phase 3: Use ParameterHandling.jl flat parameter length
    p = length(get_parameters_flat(model))

    # number of individuals
    n = sum(model.SubjectWeights)

    # loglik
    ll = if !isnothing(loglik)
        loglik
    elseif !estimate_likelihood
        get_loglik(model; ll = "loglik")
    else
        estimate_loglik(model; min_ess = min_ess).loglik
    end

    # BIC
    BIC = - 2 * ll + log(n) * p

    return BIC
end

# =============================================================================
# Pretty printing for fitted models
# =============================================================================

"""
    Base.show(io::IO, model::MultistateModelFitted)

Pretty print a fitted multistate model with parameter estimates and standard errors.
For phase-type fitted models, displays user-facing parameters.
"""
function Base.show(io::IO, model::MultistateModelFitted)
    # Check if this is a phase-type fitted model
    is_pt = is_phasetype_fitted(model)
    
    # Header
    if is_pt
        println(io, "MultistateModelFitted (Phase-Type)")
    else
        println(io, "MultistateModelFitted")
    end
    println(io, "─" ^ 60)
    
    # Basic info
    n_subj = length(model.subjectindices)
    n_states = size(model.tmat, 1)
    n_hazards = length(model.hazards)
    
    if is_pt
        # Show both original and expanded state counts
        orig_tmat = model.modelcall.original_tmat
        n_orig_states = size(orig_tmat, 1)
        println(io, "  Subjects: $n_subj")
        println(io, "  States: $n_orig_states (observed), $n_states (expanded)")
        println(io, "  Hazards: $n_hazards (expanded)")
    else
        println(io, "  Subjects: $n_subj")
        println(io, "  States: $n_states")
        println(io, "  Hazards: $n_hazards")
    end
    
    # Log-likelihood and information criteria (compute directly to avoid warnings)
    ll = model.loglik.loglik
    npar = length(get_parameters_flat(model))
    nsubj = sum(model.SubjectWeights)
    aic_val = -2 * ll + 2 * npar
    bic_val = -2 * ll + npar * log(nsubj)
    
    println(io, "  Log-likelihood: $(round(ll, digits=4))")
    println(io, "  AIC: $(round(aic_val, digits=4))")
    println(io, "  BIC: $(round(bic_val, digits=4))")
    
    # Convergence status for phase-type
    if is_pt
        conv = get_convergence(model) ? "converged" : "not converged"
        println(io, "  Status: $conv")
    end
    
    # Get standard errors if vcov is available
    has_se = !isnothing(model.vcov)
    se_flat = nothing
    if has_se
        varcov = get_vcov(model)
        se_flat = sqrt.(diag(varcov))
    end
    
    # Parameters
    println(io)
    if is_pt
        println(io, "Phase-type parameter estimates (natural scale):")
    else
        println(io, "Parameter estimates (natural scale):")
    end
    println(io, "─" ^ 60)
    
    if is_pt
        # Display user-facing phase-type parameters
        pt_params = get_phasetype_parameters(model; scale=:natural)
        mappings = model.modelcall.mappings
        original_hazards = mappings.original_hazards
        
        for hazname in keys(pt_params)
            pars = pt_params[hazname]
            hazname_str = string(hazname)
            
            # Find matching original hazard to get phase-type info
            pt_haz = nothing
            for haz in original_hazards
                if Symbol("h$(haz.statefrom)$(haz.stateto)") == hazname
                    pt_haz = haz
                    break
                end
            end
            
            if !isnothing(pt_haz) && pt_haz isa PhaseTypeHazard
                n_phases = pt_haz.n_phases
                statefrom = pt_haz.statefrom
                stateto = pt_haz.stateto
                println(io, "  $hazname ($(statefrom)→$(stateto), pt, $n_phases phases):")
                
                # Display λ and μ parameters separately
                n_lambda = n_phases - 1
                n_mu = n_phases
                
                for i in 1:n_lambda
                    println(io, "    λ$i = $(round(pars[i], digits=4))")
                end
                for i in 1:n_mu
                    println(io, "    μ$i = $(round(pars[n_lambda + i], digits=4))")
                end
            elseif !isnothing(pt_haz)
                # Non-phase-type hazard (e.g., exp) in mixed model
                trans_str = "$(pt_haz.statefrom)→$(pt_haz.stateto)"
                family_str = pt_haz.family
                println(io, "  $hazname ($trans_str, $family_str):")
                for (i, pval) in enumerate(pars)
                    println(io, "    par$i = $(round(pval, digits=4))")
                end
            else
                # Fallback
                println(io, "  $hazname:")
                for (i, pval) in enumerate(pars)
                    println(io, "    par$i = $(round(pval, digits=4))")
                end
            end
        end
    else
        # Standard display for non-phase-type models with optional SE
        sorted_hazkeys = sort(collect(model.hazkeys), by = x -> x[2])
        par_offset = 0
        
        for (hazname, idx) in sorted_hazkeys
            haz = model.hazards[idx]
            parnames = haz.parnames
            npar = haz.npar_total
            natural_pars = model.parameters.natural[hazname]
            
            # Format transition
            trans_str = "$(haz.statefrom)→$(haz.stateto)"
            family_str = string(haz.family)
            println(io, "  $hazname ($trans_str, $family_str):")
            
            for (i, (pname, pval)) in enumerate(zip(parnames, natural_pars))
                # Clean up parameter name for display
                pname_str = string(pname)
                # Remove hazard prefix if present
                if startswith(pname_str, string(hazname) * "_")
                    pname_str = pname_str[length(string(hazname))+2:end]
                end
                
                # Format with or without SE
                if has_se
                    se = se_flat[par_offset + i]
                    println(io, "    $pname_str = $(round(pval, digits=4)) (SE: $(round(se, digits=4)))")
                else
                    println(io, "    $pname_str = $(round(pval, digits=4))")
                end
            end
            par_offset += npar
        end
    end
    
    # Variance-covariance status
    println(io)
    if has_se
        println(io, "  Variance-covariance: computed (robust sandwich)")
    else
        println(io, "  Variance-covariance: not computed")
    end
    println(io)
    println(io, "Use `summary(model)` for confidence intervals.")
end

"""
    Base.show(io::IO, ::MIME"text/plain", model::MultistateModelFitted)

Extended pretty print for REPL display.
"""
function Base.show(io::IO, ::MIME"text/plain", model::MultistateModelFitted)
    show(io, model)
end

# =============================================================================
# NOTE: PhaseTypeModel show methods removed (Package Streamlining)
# =============================================================================
# Phase-type hazards are now handled internally via MultistateModel.
# Standard show() methods handle models with phase-type expansion metadata.
# =============================================================================
