
"""
get_loglik(model::MultistateModelFitted) 

Return the log likelihood at the maximum likelihood estimates. 

# Arguments 
- `model::MultistateModelFitted`: fitted model
"""
function get_loglik(model::MultistateModelFitted) 
    model.loglik
end

"""
GetParameters(model::MultistateModelFitted; transformed::Bool = true) 

Return the maximum likelihood estimates. 

# Arguments 
- `model::MultistateModelFitted`: fitted model
"""
function get_parameters(model::MultistateModelFitted)
    model.parameters
end

"""
    Return the parameter names.

# Arguments
- `model`
"""
function get_parnames(model)
    [x.parnames for x in model.hazards]
end

"""
GetVcov(model::MultistateModelFitted) 

Return the variance covariance matrix at the maximum likelihood estimate. 

# Arguments 
- `model::MultistateModelFitted`: fitted model
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
- `model::MultistateModelFitted`: fitted model
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
function summary(model::MultistateModelFitted; confidence_level::Float64 = 0.95) 
    
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
            summary_table[s].upper = summary_table[s].estimate .+ z_critical .* summary_table[s].se
            summary_table[s].lower = summary_table[s].estimate .- z_critical .* summary_table[s].se
        end
    end

    # add hazard names to the table
    haznames = map(x -> model.hazards[x].hazname, collect(1:length(model.hazards)))
    summary_table = (;zip(haznames, summary_table)...)
    
    # log likelihood
    ll = get_loglik(model)

    # information criteria
    p = length(reduce(vcat, mle))
    n = nrow(model.data)
    AIC = - 2 * ll + 2      * p
    BIC = - 2 * ll + log(n) * p

    return summary_table, ll, AIC, BIC
end