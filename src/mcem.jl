"""
    mcem_objective(logliks, weights, totweights)

Compute the marginal log likelihood for MCEM.
"""
function mcem_mll(logliks, weights, totweights) 

    obj = 0.0
    for j in 1:size(logliks, 2)
        for i in 1:size(logliks, 1)
            obj += logliks[i,j] * weights[i,j] / totweights[i]
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
    mcem_ase(delta_ll, weights, totweights)

Asymptotic standard error of the change in the MCEM objective function.
"""
function mcem_ase(delta_ll, weights, totweights)

    sqrt.(sum(map(i -> var_ris(delta_ll[i,:], weights[i,:], totweights[i]), collect(1:length(totweights))))) 

end



