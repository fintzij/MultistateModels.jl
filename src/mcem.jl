"""
    mcem_mll(logliks, ImportanceWeights, SamplingWeights)

Compute the marginal log likelihood for MCEM.
"""
function mcem_mll(logliks, ImportanceWeights, SamplingWeights)

    obj = 0.0
    
    for i in eachindex(logliks)
        obj += dot(logliks[i], ImportanceWeights[i]) * SamplingWeights[i]
    end

    return obj
end

"""
    mcem_mll(logliks, ImportanceWeights, SamplingWeights)

Compute the log marginal likelihood for MCEM.
"""
function mcem_lml(logliks, ImportanceWeights, SamplingWeights)

    obj = 0.0
    
    for i in eachindex(logliks)
        obj += log(dot(exp.(logliks[i]), ImportanceWeights[i])) * SamplingWeights[i]
    end

    return obj
end

"""
    mcem_lml_subj(logliks, ImportanceWeights)

Compute the log marginal likelihood of each subject for MCEM.
"""
function mcem_lml_subj(logliks, ImportanceWeights)

    subj_lml = zeros(length(logliks))
    
    for i in eachindex(logliks)
        subj_lml[i] = log(sum(exp.(logliks[i]) .* ImportanceWeights[i])) 
    end

    return subj_lml
end

"""
    var_ris(l, w)

Helper to compute the variance of the ratio of two means estimated via importance sampling. 

# Arguments:

- l: For MCEM, the difference in log-likelihoods
- w: Normalized importance weights
"""
function var_ris(l, w)

    vris = (sum(l .* w)) ^ 2 * ((sum((w .* l).^2)) / sum(w .* l) ^2 - 2 * (sum(w.^2 .* l) / (sum(w .* l))) + sum(w .^ 2))

    vris < eps() ? eps() : vris
end

"""
    mcem_ase(loglik_target_prop, loglik_target_cur, ImportanceWeights, SamplingWeights)

Asymptotic standard error of the change in the MCEM objective function.
"""
function mcem_ase(loglik_target_prop, loglik_target_cur, ImportanceWeights, SamplingWeights)

    VarRis = 0.0
    for i in eachindex(SamplingWeights)
        if length(ImportanceWeights[i]) != 1
            VarRis += var_ris(loglik_target_prop[i] - loglik_target_cur[i], ImportanceWeights[i]) / length(ImportanceWeights[i]) * SamplingWeights[i]^2
        end
    end

    # return the asymptotic standard error
    sqrt(VarRis)
end