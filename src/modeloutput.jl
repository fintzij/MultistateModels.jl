
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
    get_ConvergenceRecords(model::MultistateModelFitted) 

Return the convergence records for the fit. 

# Arguments 
- model: fitted model
"""
function get_ConvergenceRecords(model::MultistateModelFitted) 
    model.ConvergenceRecords
end

"""
    summary(model::MultistateModelFitted) 

Summary of model output. 

# Arguments 
- `model::MultistateModelFitted`: fitted model
- `confidence_level::Float64`: confidence level of the confidence intervals
"""
function summary(model::MultistateModelFitted; confidence_level::Float64 = 0.95, estimate_likelihood = false, min_ess = 100) 
    
    # maximum likelihood estimates
    mle = get_parameters(model)

    # container for summary table
    summary_table = Vector{DataFrame}(undef, length(model.hazards))

    if isnothing(model.vcov)
        println("Confidence intervals are not computed for models without a variance-covariance matrix.")
        # populate summary tables for each hazard
        for s in eachindex(summary_table)
            # summary for hazard s (only the estimate when there is no vcov)
            summary_table[s] = DataFrame(estimate = reduce(vcat, mle[s]))
        end
    else    
        # standard error
        vcov = get_vcov(model)
        se = sqrt.(vcov[diagind(vcov)])
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
    observed_lml = sum(subj_lml .* model.SamplingWeights)

    # calculate MCSEs
    subj_ml_var = map(w -> length(w) == 1 ? 0.0 : var(w) / length(w), ImportanceWeights)
    subj_lml_var = subj_ml_var ./ (subj_ml.^2) # delta method

    # sum and include sampling weights
    observed_lml_var = sum(subj_lml_var .* model.SamplingWeights.^2)

    # return log likelihoods
    return (loglik = observed_lml, loglik_subj = subj_lml, mcse_loglik = sqrt(observed_lml_var), mcse_loglik_subj = sqrt.(subj_lml_var))
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
    p = length(flatview(model.parameters))

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
    p = length(flatview(model.parameters))

    # number of individuals
    n = sum(model.SamplingWeights)

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