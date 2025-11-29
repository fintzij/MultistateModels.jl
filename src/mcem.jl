"""
    mcem_mll(logliks, ImportanceWeights, SubjectWeights)

Compute the marginal log likelihood for MCEM.
"""
function mcem_mll(logliks, ImportanceWeights, SubjectWeights)

    obj = 0.0
    
    for i in eachindex(logliks)
        obj += dot(logliks[i], ImportanceWeights[i]) * SubjectWeights[i]
    end

    return obj
end

"""
    mcem_lml(logliks, ImportanceWeights, SubjectWeights)

Compute the log marginal likelihood for MCEM.
"""
function mcem_lml(logliks, ImportanceWeights, SubjectWeights)

    obj = 0.0
    
    for i in eachindex(logliks)
        obj += log(dot(exp.(logliks[i]), ImportanceWeights[i])) * SubjectWeights[i]
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

    all(isapprox.(l, 0.0)) ? 0.0 : (sum(l .* w)) ^ 2 * ((sum((w .* l).^2)) / sum(w .* l) ^2 - 2 * (sum(w.^2 .* l) / (sum(w .* l))) + sum(w .^ 2)) 
    
end

"""
    mcem_ase(loglik_target_prop, loglik_target_cur, ImportanceWeights, SubjectWeights)

Asymptotic standard error of the change in the MCEM objective function.
"""
function mcem_ase(loglik_target_prop, loglik_target_cur, ImportanceWeights, SubjectWeights)

    VarRis = 0.0
    for i in eachindex(SubjectWeights)
        if length(ImportanceWeights[i]) != 1
            VarRis += var_ris(loglik_target_prop[i] - loglik_target_cur[i], ImportanceWeights[i]) * SubjectWeights[i]^2
        end
    end

    # return the asymptotic standard error
    sqrt(VarRis)
end