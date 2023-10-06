"""
    mcem_mll(logliks, ImportanceWeights, TotImportanceWeights)

Compute the marginal log likelihood for MCEM.
"""
function mcem_mll(logliks, ImportanceWeights, TotImportanceWeights)

    obj = 0.0
    #for j in 1:size(logliks, 2)
        #for i in 1:size(logliks, 1)
            #obj += logliks[i,j] * ImportanceWeights[i,j] / TotImportanceWeights[i]
    for i in 1:length(logliks)
        for j in 1:length(logliks[i])
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

    (sum(l .* w) / t) ^ 2 * ((sum((w .* l).^2)) / sum(w .* l) ^2 - 2 * (sum(w.^2 .* l) / (sum(w .* l) * t)) + sum(w .^ 2) / t^2)

end


"""
    mcem_ase(delta_ll, ImportanceWeights, TotImportanceWeights)

Asymptotic standard error of the change in the MCEM objective function.
"""
function mcem_ase(delta_ll, ImportanceWeights, TotImportanceWeights)

    #sqrt.(sum(map(i -> var_ris(delta_ll[i,:], ImportanceWeights[i,:], TotImportanceWeights[i]), collect(1:length(TotImportanceWeights))))) 
    VarRis = zeros(length(TotImportanceWeights))
    for i in 1:length(TotImportanceWeights)
        VarRis[i]=var_ris(delta_ll[i], ImportanceWeights[i], TotImportanceWeights[i])
    end
    sqrt(sum(VarRis))

end