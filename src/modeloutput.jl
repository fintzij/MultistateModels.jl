
# =============================================================================
# Phase-Type Fitted Model Detection and Accessors
# =============================================================================

"""
    is_phasetype_fitted(model::MultistateModelFitted) -> Bool

Check if a fitted model was produced by fitting a `PhaseTypeModel`.

Phase-type fitted models store additional information in `modelcall`:
- `is_phasetype`: Flag indicating phase-type origin
- `mappings`: `PhaseTypeMappings` for state space translation
- `original_parameters`: User-facing phase-type parameters (λ, μ)
- `original_tmat`: Original transition matrix (on observed states)
- `original_data`: Original data (on observed states)

# Example
```julia
# Fit a phase-type model
h12 = Hazard(@formula(0 ~ 1), "pt", 1, 2; n_phases=3)
model = multistatemodel(h12; data=data)
fitted = fit(model)

# Check if phase-type
is_phasetype_fitted(fitted)  # true

# Access phase-type parameters (convenience function)
params = get_phasetype_parameters(fitted)
```

See also: [`get_phasetype_parameters`](@ref), [`get_mappings`](@ref), [`PhaseTypeModel`](@ref)
"""
function is_phasetype_fitted(model::MultistateModelFitted)
    haskey(model.modelcall, :is_phasetype) && model.modelcall.is_phasetype === true
end

# Fallback for other model types
is_phasetype_fitted(::MultistateProcess) = false

"""
    get_phasetype_parameters(model::MultistateModelFitted; scale::Symbol=:natural)

Get phase-type parameters from a fitted phase-type model.

Returns the user-facing phase-type parameterization (λ progression rates, μ exit rates).
Throws an error if the model is not a fitted phase-type model.

# Arguments
- `model::MultistateModelFitted`: A fitted model (must be from `PhaseTypeModel`)
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
# (h12 = (λ = [1.2, 0.8], μ = [0.5, 0.3, 0.4]), h23 = [0.6, 0.25])
```

See also: [`is_phasetype_fitted`](@ref), [`get_expanded_parameters`](@ref)
"""
function get_phasetype_parameters(model::MultistateModelFitted; scale::Symbol=:natural)
    if !is_phasetype_fitted(model)
        throw(ArgumentError("Model is not a fitted phase-type model. " *
                           "Use `get_parameters(model)` for standard models."))
    end
    
    original_params = model.modelcall.original_parameters
    if scale == :natural
        return original_params.natural
    elseif scale == :flat || scale == :estimation || scale == :log
        return original_params.flat
    elseif scale == :nested
        return original_params.nested
    else
        throw(ArgumentError("scale must be :natural, :flat, :estimation, :log, or :nested (got :$scale)"))
    end
end

"""
    get_mappings(model::MultistateModelFitted)

Get the `PhaseTypeMappings` from a fitted phase-type model.

Throws an error if the model is not a fitted phase-type model.

# Returns
- `PhaseTypeMappings`: State space translation information

See also: [`is_phasetype_fitted`](@ref), [`PhaseTypeMappings`](@ref)
"""
function get_mappings(model::MultistateModelFitted)
    if !is_phasetype_fitted(model)
        throw(ArgumentError("Model is not a fitted phase-type model."))
    end
    return model.modelcall.mappings
end

"""
    get_original_data(model::MultistateModelFitted)

Get the original (non-expanded) data from a fitted phase-type model.

Throws an error if the model is not a fitted phase-type model.

# Returns
- `DataFrame`: Original data on observed state space

See also: [`is_phasetype_fitted`](@ref)
"""
function get_original_data(model::MultistateModelFitted)
    if !is_phasetype_fitted(model)
        throw(ArgumentError("Model is not a fitted phase-type model."))
    end
    return model.modelcall.original_data
end

"""
    get_original_tmat(model::MultistateModelFitted)

Get the original transition matrix from a fitted phase-type model.

Throws an error if the model is not a fitted phase-type model.

# Returns
- `Matrix{Int64}`: Original transition matrix on observed state space

See also: [`is_phasetype_fitted`](@ref)
"""
function get_original_tmat(model::MultistateModelFitted)
    if !is_phasetype_fitted(model)
        throw(ArgumentError("Model is not a fitted phase-type model."))
    end
    return model.modelcall.original_tmat
end

"""
    get_convergence(model::MultistateModelFitted)

Check if the model fitting converged.

For phase-type models, returns the convergence status from optimization.
For other fitted models, checks the convergence records if available.

# Returns
- `Bool`: Whether the model fitting converged
"""
function get_convergence(model::MultistateModelFitted)
    if is_phasetype_fitted(model)
        return model.modelcall.convergence
    else
        # For non-phase-type models, check ConvergenceRecords
        if !isnothing(model.ConvergenceRecords)
            if haskey(model.ConvergenceRecords, :retcode)
                return model.ConvergenceRecords.retcode == :Success
            elseif haskey(model.ConvergenceRecords, :solution)
                sol = model.ConvergenceRecords.solution
                return hasfield(typeof(sol), :retcode) && sol.retcode == ReturnCode.Success
            end
        end
        return true  # Assume converged if no info
    end
end

# Deprecated: PhaseTypeFittedModel is no longer used
# Fitted phase-type models are now returned as MultistateModelFitted with
# phase-type specific info stored in modelcall. Use is_phasetype_fitted() to check.

"""
get_loglik(model::MultistateModelFitted) 

Return the log likelihood at the maximum likelihood estimates. 

# Arguments 
- model: fitted model
- `ll`: one of "loglik" (default) for the observed data log likelihood or "subj_lml" for log marginal likelihood at the subject level
"""
function get_loglik(model::MultistateModelFitted; ll = "loglik") 
    if ll == "loglik"
        model.loglik.loglik
    elseif ll == "subj_lml"
        model.loglik.subj_lml
    end
end

"""
    get_parameters(model::MultistateProcess; scale::Symbol=:natural)

Return model parameters on the specified scale.

# Arguments 
- `model`: A MultistateProcess (fitted or unfitted)
- `scale::Symbol=:natural`: Parameter scale
  - `:natural` - Human-readable scale (exp applied to baseline params)
  - `:estimation` or `:log` - Flat vector on log scale (for optimization)
  - `:nested` - Nested NamedTuple with ParameterHandling structure

# Returns
- For `:natural`: `NamedTuple` with parameters per hazard on natural scale
- For `:estimation`/`:log`: `Vector{Float64}` flat parameter vector
- For `:nested`: `NamedTuple` with full ParameterHandling structure

# Example
```julia
# Get human-readable parameters
params = get_parameters(fitted_model)  # Same as scale=:natural
# (h12 = [1.0, 0.3, 0.5], h21 = [1.2, 0.25, 0.3])

# Get flat vector for optimization
params_flat = get_parameters(fitted_model; scale=:estimation)
# [0.0, -1.2, 0.5, 0.18, -1.39, 0.3]
```

# See also
- [`set_parameters!`](@ref) - Set model parameters
- [`get_parnames`](@ref) - Get parameter names
"""
function get_parameters(model::MultistateProcess; scale::Symbol=:natural)
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

Get the expanded (internal) parameters from a fitted model.

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

"""
    get_parnames(model)    

Return the parameter names.

# Arguments
- `model`: A MultistateProcess or MarkovSurrogate
"""
function get_parnames(model::MultistateProcess)
    [x.parnames for x in model.hazards]
end

"""
    get_parnames(surrogate::MarkovSurrogate)

Return the parameter names for a Markov surrogate.

# Arguments
- `surrogate::MarkovSurrogate`: A Markov surrogate

# Returns
- Vector of parameter name vectors, one per hazard
"""
function get_parnames(surrogate::MarkovSurrogate)
    [x.parnames for x in surrogate.hazards]
end

"""
    get_vcov(model::MultistateModelFitted) 

Return the variance covariance matrix at the maximum likelihood estimate. 

# Arguments 
- model: fitted model
"""
function get_vcov(model::MultistateModelFitted)
    if isnothing(model.vcov)
        println("The variance-covariance matrix was not computed for this model.")
    end
    model.vcov
end

"""
    get_ij_vcov(model::MultistateModelFitted)

Return the infinitesimal jackknife (IJ) / sandwich variance-covariance matrix.

# Formula
The IJ variance is computed as:
```math
Var_{IJ}(θ̂) = H⁻¹ K H⁻¹
```
where:
- H is the observed Fisher information (negative Hessian at the MLE)
- K = Σᵢ gᵢgᵢᵀ is the sum of outer products of subject-level score vectors

# Interpretation
This is also known as the **robust** or **Huber-White** variance estimator.
Unlike the model-based variance (H⁻¹), the sandwich variance remains valid
even when the model is misspecified.

# Arguments
- `model::MultistateModelFitted`: fitted model (must have been fit with `compute_ij_vcov=true`)

# Returns
- `Symmetric{Float64, Matrix{Float64}}`: p × p variance-covariance matrix, or `nothing` if not computed

# Example
```julia
# Fit model with IJ variance
fitted = fit(model, data; compute_ij_vcov=true)

# Get robust standard errors
ij_vcov = get_ij_vcov(fitted)
if !isnothing(ij_vcov)
    robust_se = sqrt.(diag(ij_vcov))
    println("Robust SEs: ", robust_se)
end

# Compare to model-based SE
model_se = sqrt.(diag(get_vcov(fitted)))
println("Model-based SEs: ", model_se)
println("Ratio (robust/model): ", robust_se ./ model_se)
```

See also: [`get_jk_vcov`](@ref), [`get_vcov`](@ref), [`fit`](@ref)
"""
function get_ij_vcov(model::MultistateModelFitted)
    if isnothing(model.ij_vcov)
        println("The IJ variance-covariance matrix was not computed for this model.")
    end
    model.ij_vcov
end

"""
    get_jk_vcov(model::MultistateModelFitted)

Return the jackknife variance-covariance matrix.

# Formula
The jackknife variance is computed as:
```math
Var_{JK}(θ̂) = \frac{n-1}{n} Σᵢ ΔᵢΔᵢᵀ
```
where Δᵢ = H⁻¹gᵢ are the leave-one-out parameter perturbations.

# Relationship to IJ Variance
The jackknife and IJ variances are related by:
```math
Var_{JK}(θ̂) = \frac{n-1}{n} Var_{IJ}(θ̂)
```
The factor (n-1)/n is a finite-sample correction; for large n they are essentially equivalent.

# Arguments
- `model::MultistateModelFitted`: fitted model (must have been fit with `compute_jk_vcov=true`)

# Returns
- `Symmetric{Float64, Matrix{Float64}}`: p × p variance-covariance matrix, or `nothing` if not computed

# Example
```julia
# Fit model with jackknife variance
fitted = fit(model, data; compute_jk_vcov=true)

# Get jackknife standard errors  
jk_vcov = get_jk_vcov(fitted)
if !isnothing(jk_vcov)
    jk_se = sqrt.(diag(jk_vcov))
    println("Jackknife SEs: ", jk_se)
end
```

See also: [`get_ij_vcov`](@ref), [`get_vcov`](@ref), [`get_jk_pseudovalues`](@ref)
"""
function get_jk_vcov(model::MultistateModelFitted)
    if isnothing(model.jk_vcov)
        println("The jackknife variance-covariance matrix was not computed for this model.")
    end
    model.jk_vcov
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
        error("Subject gradients were not computed. Refit with compute_ij_vcov=true or compute_jk_vcov=true.")
    end
    if isnothing(model.vcov)
        error("Variance-covariance matrix was not computed. Refit with compute_vcov=true.")
    end
    
    return loo_perturbations_direct(model.vcov, model.subject_gradients)
end

"""
    get_jk_pseudovalues(model::MultistateModelFitted)

Return the jackknife pseudo-values for each subject.

# Formula
The jackknife pseudo-value for subject i is:
```math
θ̃ᵢ = n θ̂ - (n-1) θ̂_{-i} = θ̂ - (n-1) Δᵢ
```

# Statistical Properties
1. **Mean equals estimate**: `(1/n) Σᵢ θ̃ᵢ = θ̂`
2. **Variance estimates estimator variance**: `Var(θ̃) / n ≈ Var(θ̂)`
3. **Approximately IID**: Under regularity conditions, pseudo-values are approximately
   independent with mean θ (the true parameter)

# Arguments
- `model::MultistateModelFitted`: fitted model (requires `compute_ij_vcov=true` or `compute_jk_vcov=true`)

# Returns
- `Matrix{Float64}`: p × n matrix where column i is the pseudo-value θ̃ᵢ

# Uses
- **Jackknife variance**: `Var(θ̂) ≈ (1/(n(n-1))) Σᵢ (θ̃ᵢ - θ̂)(θ̃ᵢ - θ̂)ᵀ`
- **Jackknife confidence intervals**: Treat pseudo-values as IID sample, use t-interval
- **Regression on pseudo-values**: Standard regression methods with pseudo-values as outcomes

# Example
```julia
fitted = fit(model, data; compute_jk_vcov=true)
pseudo = get_jk_pseudovalues(fitted)

# Verify mean equals estimate
theta_hat = get_parameters_flat(fitted)
mean_pseudo = mean(pseudo, dims=2)
println("Difference from estimate: ", maximum(abs.(mean_pseudo .- theta_hat)))

# Manual jackknife variance calculation
n = size(pseudo, 2)
jk_var_manual = (1/(n*(n-1))) * sum((pseudo[:, i] .- theta_hat) * (pseudo[:, i] .- theta_hat)' 
                                    for i in 1:n)
```

# References
- Efron, B. (1982). The Jackknife, the Bootstrap and Other Resampling Plans. SIAM.
- Miller, R. G. (1974). The jackknife - a review. Biometrika, 61(1), 1-15.

See also: [`get_ij_pseudovalues`](@ref), [`get_jk_vcov`](@ref)
"""
function get_jk_pseudovalues(model::MultistateModelFitted)
    deltas = get_loo_perturbations(model)
    n = size(deltas, 2)
    theta_hat = get_parameters_flat(model)
    
    # θ̃ᵢ = θ̂ - (n-1)·Δᵢ
    pseudovalues = theta_hat .- (n - 1) .* deltas
    return pseudovalues
end

"""
    get_ij_pseudovalues(model::MultistateModelFitted)

Return the infinitesimal jackknife (IJ) pseudo-values for each subject.

# Formula
The IJ pseudo-value for subject i is:
```math
θ̃ᵢ^{IJ} = θ̂ + Δᵢ = θ̂ + H⁻¹ gᵢ ≈ θ̂_{-i}
```

This is simply the approximate LOO estimate for subject i. Unlike jackknife
pseudo-values, IJ pseudo-values are NOT scaled by (n-1).

# Relationship to Jackknife Pseudo-values
- JK pseudo-value: `θ̃ᵢ = θ̂ - (n-1)Δᵢ` (bias-corrected, mean = θ̂)
- IJ pseudo-value: `θ̃ᵢ^{IJ} = θ̂ + Δᵢ` (LOO estimate, mean ≠ θ̂)

# Arguments
- `model::MultistateModelFitted`: fitted model (requires `compute_ij_vcov=true` or `compute_jk_vcov=true`)

# Returns
- `Matrix{Float64}`: p × n matrix where column i is the IJ pseudo-value (LOO estimate)

# Variance Calculation
The IJ variance can be computed from pseudo-values as:
```math
Var_{IJ}(θ̂) = \frac{1}{n^2} Σᵢ (θ̃ᵢ^{IJ} - θ̂)(θ̃ᵢ^{IJ} - θ̂)ᵀ = \frac{1}{n^2} Σᵢ ΔᵢΔᵢᵀ
```

# Example
```julia
fitted = fit(model, data; compute_ij_vcov=true)
ij_pseudo = get_ij_pseudovalues(fitted)
theta_hat = get_parameters_flat(fitted)

# These are the approximate LOO estimates
println("LOO estimate for subject 1: ", ij_pseudo[:, 1])
println("Full estimate: ", theta_hat)
println("Difference: ", ij_pseudo[:, 1] .- theta_hat)  # Same as delta_1
```

See also: [`get_jk_pseudovalues`](@ref), [`get_loo_perturbations`](@ref)
"""
function get_ij_pseudovalues(model::MultistateModelFitted)
    deltas = get_loo_perturbations(model)
    theta_hat = get_parameters_flat(model)
    
    # θ̃ᵢ = θ̂ + Δᵢ (the LOO estimate itself)
    pseudovalues = theta_hat .+ deltas
    return pseudovalues
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
    summary(model::MultistateModelFitted) 

Summary of model output. 

# Arguments 
- `model::MultistateModelFitted`: fitted model
- `confidence_level::Float64`: confidence level of the confidence intervals
"""
function summary(model::MultistateModelFitted; compute_se = true, confidence_level::Float64 = 0.95, estimate_likelihood = false, min_ess = 100) 
    
    # maximum likelihood estimates
    mle = get_parameters(model)

    # container for summary table
    summary_table = Vector{DataFrame}(undef, length(model.hazards))

    if isnothing(model.vcov) | !compute_se

        if isnothing(model.vcov)
            println("Confidence intervals are not computed for models without a variance-covariance matrix.")
        end
        # populate summary tables for each hazard
        for s in eachindex(summary_table)
            # summary for hazard s (only the estimate when there is no vcov)
            summary_table[s] = DataFrame(estimate = reduce(vcat, mle[s]))
        end
    else    
        # standard error
        varcov = get_vcov(model)
        se = sqrt.(varcov[diagind(varcov)])
        se_nested = unflatten(model.parameters.reconstructor, se)
        # critical value
        z_critical = quantile(Normal(0.0, 1.0), 1-(1-confidence_level)/2)

        # populate summary tables for each hazard
        for s in eachindex(summary_table)
            # summary for hazard s
            summary_table[s] = DataFrame(
                estimate = reduce(vcat, mle[s]),
                se = reduce(vcat, se_nested[s]))
                summary_table[s].lower = summary_table[s].estimate .- z_critical .* summary_table[s].se
                summary_table[s].upper = summary_table[s].estimate .+ z_critical .* summary_table[s].se
        end
    end

    # add hazard names to the table
    haznames = map(x -> model.hazards[x].hazname, collect(1:length(model.hazards)))
    summary_table = (;zip(haznames, summary_table)...)
    
    # log likelihood  
    if estimate_likelihood
        ll_result = estimate_loglik(model; min_ess = min_ess)
        ll = ll_result.loglik
        mcse = ll_result.mcse_loglik
    else
        ll = get_loglik(model; ll = "loglik")
        mcse = nothing
    end

    # information criteria
    AIC = MultistateModels.aic(model; loglik = ll)

    BIC = MultistateModels.bic(model; loglik = ll)

    return (summary = summary_table, loglik = ll, AIC = AIC, BIC = BIC, MCSE_loglik = mcse)
end

"""
    estimate_loglik(model::MultistateProcess; min_ess = 100)
    
Estimate the log marginal likelihood for a fitted multistate model. Require that the minimum effective sample size per subject is greater than min_ess.  

# Arguments
- min_ess: minimum effective sample size, defaults to 100.
"""
function estimate_loglik(model::MultistateProcess; min_ess = 100)

    # sample paths and grab logliks
    samplepaths, loglik_target, subj_ess, loglik_surrog, ImportanceWeightsNormalized, ImportanceWeights = draw_paths(model; min_ess = min_ess, paretosmooth = false, return_logliks = true)

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

Pretty print a fitted multistate model with parameter estimates.
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
    println(io, "─" ^ 50)
    
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
    
    # Log-likelihood
    ll = model.loglik.loglik
    println(io, "  Log-likelihood: $(round(ll, digits=4))")
    
    # Convergence status for phase-type
    if is_pt
        conv = get_convergence(model) ? "converged" : "not converged"
        println(io, "  Status: $conv")
    end
    
    # Parameters
    println(io)
    if is_pt
        println(io, "Phase-type parameter estimates (natural scale):")
    else
        println(io, "Parameter estimates (natural scale):")
    end
    println(io, "─" ^ 50)
    
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
            
            if !isnothing(pt_haz) && pt_haz isa PhaseTypeHazardSpec
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
        # Standard display for non-phase-type models
        sorted_hazkeys = sort(collect(model.hazkeys), by = x -> x[2])
        
        for (hazname, idx) in sorted_hazkeys
            haz = model.hazards[idx]
            parnames = haz.parnames
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
                println(io, "    $pname_str = $(round(pval, digits=4))")
            end
        end
    end
    
    # Variance-covariance status
    println(io)
    if !isnothing(model.vcov)
        println(io, "  Variance-covariance: computed")
    else
        println(io, "  Variance-covariance: not computed")
    end
end

"""
    Base.show(io::IO, ::MIME"text/plain", model::MultistateModelFitted)

Extended pretty print for REPL display.
"""
function Base.show(io::IO, ::MIME"text/plain", model::MultistateModelFitted)
    show(io, model)
end

# =============================================================================
# Pretty printing for PhaseTypeModel (unfitted)
# =============================================================================

"""
    Base.show(io::IO, model::PhaseTypeModel)

Pretty print an unfitted phase-type model showing state space structure.
"""
function Base.show(io::IO, model::PhaseTypeModel)
    println(io, "PhaseTypeModel")
    println(io, "─" ^ 50)
    
    # Basic info
    n_subj = length(model.subjectindices)
    n_orig_states = size(model.original_tmat, 1)
    n_exp_states = size(model.tmat, 1)
    n_orig_hazards = length(model.hazards_spec)
    n_exp_hazards = length(model.hazards)
    
    println(io, "  Subjects: $n_subj")
    println(io, "  Observed states: $n_orig_states")
    println(io, "  Expanded states: $n_exp_states")
    println(io, "  Original hazards: $n_orig_hazards")
    println(io, "  Expanded hazards: $n_exp_hazards")
    
    # Show phase-type hazards
    println(io)
    println(io, "Hazard specifications:")
    println(io, "─" ^ 50)
    
    for (i, haz) in enumerate(model.hazards_spec)
        trans_str = "$(haz.statefrom)→$(haz.stateto)"
        
        if haz isa PhaseTypeHazardSpec
            structure_str = string(haz.structure)
            println(io, "  h$(haz.statefrom)$(haz.stateto) ($trans_str, pt):")
            println(io, "    n_phases: $(haz.n_phases)")
            println(io, "    structure: $structure_str")
            println(io, "    parameters: $(2*haz.n_phases - 1) baseline")
        else
            family_str = haz.family
            println(io, "  h$(haz.statefrom)$(haz.stateto) ($trans_str, $family_str)")
        end
    end
    
    # Show state mappings summary
    if !isnothing(model.mappings) && hasfield(typeof(model.mappings), :observed_to_expanded)
        println(io)
        println(io, "State expansion:")
        println(io, "─" ^ 50)
        
        obs_to_exp = model.mappings.observed_to_expanded
        for (obs_state, exp_states) in sort(collect(obs_to_exp), by = x -> x[1])
            if length(exp_states) == 1
                println(io, "  State $obs_state → state $(exp_states[1])")
            else
                println(io, "  State $obs_state → states $(first(exp_states))-$(last(exp_states)) ($(length(exp_states)) phases)")
            end
        end
    end
    
    # Parameters status
    println(io)
    params = model.original_parameters
    if haskey(params, :natural) && !isnothing(params.natural)
        println(io, "  Parameters: initialized")
    else
        println(io, "  Parameters: not initialized")
    end
end

"""
    Base.show(io::IO, ::MIME"text/plain", model::PhaseTypeModel)

Extended pretty print for REPL display.
"""
function Base.show(io::IO, ::MIME"text/plain", model::PhaseTypeModel)
    show(io, model)
end