"""
    mcem_mll(logliks, ImportanceWeights, TotImportanceWeights)

Compute the marginal log likelihood for MCEM.
"""
function mcem_mll(logliks, ImportanceWeights, TotImportanceWeights)

    obj = 0.0
    
    for i in eachindex(logliks)
        for j in eachindex(logliks[i])
            obj += logliks[i][j] * ImportanceWeights[i][j] / TotImportanceWeights[i]
        end
    end

    return obj
end


"""
    var_ris(l, w, t)

Helper to compute the variance of two means estimated via importance sampling. 

# Arguments:

- l: For MCEM, the difference in log-likelihoods
- w: Importance weights
- t: Total importance weight
"""
function var_ris(l, w, t)

    vris = (sum(l .* w) / t) ^ 2 * ((sum((w .* l).^2)) / sum(w .* l) ^2 - 2 * (sum(w.^2 .* l) / (sum(w .* l) * t)) + sum(w .^ 2) / t^2)

    isapprox(vris, 0.0; atol = sqrt(eps(Float64))) ? 0.0 : vris
end


"""
    mcem_ase(loglik_target_prop, loglik_target_cur, ImportanceWeights, TotImportanceWeights)

Asymptotic standard error of the change in the MCEM objective function.
"""
function mcem_ase(loglik_target_prop, loglik_target_cur, ImportanceWeights, TotImportanceWeights)

    VarRis = 0.0
    for i in eachindex(TotImportanceWeights)
        VarRis += var_ris(loglik_target_prop[i] - loglik_target_cur[i], ImportanceWeights[i], TotImportanceWeights[i]) / length(ImportanceWeights[i])
    end

    # return the asymptotic standard error
    sqrt(VarRis)
end