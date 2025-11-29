
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
get_parameters(model::MultistateProcess) 

Return the maximum likelihood estimates. 

# Arguments 
- model: fitted model
"""
function get_parameters(model::MultistateProcess)
    model.parameters
end

"""
    get_parnames(model)    

Return the parameter names.

# Arguments
- `model`
"""
function get_parnames(model::MultistateProcess)
    [x.parnames for x in model.hazards]
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
        se_vv = VectorOfVectors(se, model.parameters.elem_ptr)
        # critical value
        z_critical = quantile(Normal(0.0, 1.0), 1-(1-confidence_level)/2)

        # populate summary tables for each hazard
        for s in eachindex(summary_table)
            # summary for hazard s
            summary_table[s] = DataFrame(
                estimate = reduce(vcat, mle[s]),
                se = reduce(vcat, se_vv[s]))
                summary_table[s].lower = summary_table[s].estimate .- z_critical .* summary_table[s].se
                summary_table[s].upper = summary_table[s].estimate .+ z_critical .* summary_table[s].se
        end
    end

    # add hazard names to the table
    haznames = map(x -> model.hazards[x].hazname, collect(1:length(model.hazards)))
    summary_table = (;zip(haznames, summary_table)...)
    
    # log likelihood
    ll = estimate_likelihood ? estimate_loglik(model; min_ess = min_ess) : get_loglik(model; ll = "loglik")

    # information criteria
    AIC = MultistateModels.aic(model; loglik = ll.loglik)

    BIC = MultistateModels.bic(model; loglik = ll.loglik)

    return (summary = summary_table, loglik = ll.loglik, AIC = AIC, BIC = BIC, MCSE_loglik = ll.mcse_loglik)
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