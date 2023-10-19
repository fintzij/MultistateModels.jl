
"""
GetLoglik(model::MultistateModelFitted) 

Return the maximum likelihood estimates. 

# Arguments 
- `model::MultistateModelFitted`: fitted model
"""
function GetLoglik(model::MultistateModelFitted) 
    model.loglik
end

"""
GetParameters(model::MultistateModelFitted; transformed::Bool = true) 

Return the maximum likelihood estimates. 

# Arguments 
- `model::MultistateModelFitted`: fitted model
"""
function GetParameters(model::MultistateModelFitted)
    model.parameters
#    reduce(vcat, model.parameters)
end


"""
GetVcov(model::MultistateModelFitted) 

Return the variance covariance matrix at the maximum likelihood estimate. 

# Arguments 
- `model::MultistateModelFitted`: fitted model
"""
function GetVcov(model::MultistateModelFitted) 
    model.vcov
end


"""
    optim(model::MultistateModelFitted) 

Return the variance covariance matrix at the maximum likelihood estimate. 

# Arguments 
- `model::MultistateModelFitted`: fitted model
"""
function GetConvergenceRecords(model::MultistateModelFitted) 
    model.ConvergenceRecords
end

"""
    summary(model::MultistateModelFitted) 

Return the variance covariance matrix at the maximum likelihood estimate. 

# Arguments 
- `model::MultistateModelFitted`: fitted model
- `confidence_level::Float64`: confidence level of the confidence intervals
"""
function summary(model::MultistateModelFitted; confidence_level::Float64 = 0.95) 
    
    # maximum likelihood estimates
    mle=GetParameters(model)
    
    # standard error
    vcov = GetVcov(model)
    se = sqrt.(vcov[diagind(vcov)])
    se_vv = VectorOfVectors(se, model.parameters.elem_ptr)

    # name of hazards
    haznames = map(x -> model.hazards[x].hazname, collect(1:length(model.hazards)))

    # summary tables for each hazard
    summary_table = Vector{DataFrame}(undef, length(haznames))
    z_critical = quantile(Normal(0.0, 1.0), 1-(1-confidence_level)/2)
    for s in eachindex(summary_table)
        # summary for hazard s
        summary_table[s] = DataFrame(
            estimate = reduce(vcat, mle[s]),
            se = reduce(vcat, se_vv[s]))
        summary_table[s].upper = summary_table[s].estimate .+ z_critical .* summary_table[s].se
        summary_table[s].lower = summary_table[s].estimate .- z_critical .* summary_table[s].se
    end

    # add hazard names to the table
    summary_table = (;zip(haznames, summary_table)...)
    
    # log likelihood
    ll = MultistateModels.GetLoglik(model)

    # information criteria
    p = length(reduce(vcat, mle))
    n = nrow(model.data)
    AIC = -2*ll + 2     *p
    BIC = -2*ll + log(n)*p

    return summary_table, ll, AIC, BIC
end