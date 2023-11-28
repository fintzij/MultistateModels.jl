"""
    mcem_mll(logliks, ImportanceWeights, SamplingWeights)

Compute the marginal log likelihood for MCEM.
"""
function mcem_mll(logliks, ImportanceWeights, SamplingWeights)

    obj = 0.0
    
    for i in eachindex(logliks)
        for j in eachindex(logliks[i])
            obj += logliks[i][j] * ImportanceWeights[i][j] * SamplingWeights[i]
        end
    end

    return obj
end


"""
    var_ris(l, w)

Helper to compute the variance of two means estimated via importance sampling. 

# Arguments:

- l: For MCEM, the difference in log-likelihoods
- w: Normalized importance weights
"""
function var_ris(l, w)

    vris = (sum(l .* w)) ^ 2 * ((sum((w .* l).^2)) / sum(w .* l) ^2 - 2 * (sum(w.^2 .* l) / (sum(w .* l))) + sum(w .^ 2))

    vris < eps(Float64) ? 0.0 : vris
end


"""
    mcem_ase(loglik_target_prop, loglik_target_cur, ImportanceWeights, SamplingWeights)

Asymptotic standard error of the change in the MCEM objective function.
"""
function mcem_ase(loglik_target_prop, loglik_target_cur, ImportanceWeights, SamplingWeights)

    VarRis = 0.0
    for i in eachindex(SamplingWeights)
        VarRis += var_ris(loglik_target_prop[i] - loglik_target_cur[i], ImportanceWeights[i]) / length(ImportanceWeights[i]) * SamplingWeights[i]
    end

    # return the asymptotic standard error
    sqrt(VarRis)
end