# =============================================================================
# MCEM Helper Functions
# =============================================================================
#
# This module provides helper functions for Monte Carlo EM (MCEM) estimation
# of semi-Markov multistate models. The MCEM algorithm iterates between:
#
#   E-step: Estimate E[Q(θ|θ') | Y] via importance sampling from Markov surrogate
#   M-step: Maximize Q(θ|θ') = Σᵢ Σⱼ wᵢⱼ ℓᵢⱼ(θ) w.r.t. θ
#
# where wᵢⱼ are normalized importance weights and ℓᵢⱼ is the complete-data
# log-likelihood for path j of subject i.
#
# # References
#
# - Morsomme, R., Liang, C. J., Mateja, A., Follmann, D. A., O'Brien, M. P., Wang, C.,
#   & Fintzi, J. (2025). Assessing treatment efficacy for interval-censored endpoints
#   using multistate semi-Markov models fit to multiple data streams. Biostatistics,
#   26(1), kxaf038. https://doi.org/10.1093/biostatistics/kxaf038
# - Wei, G. C., & Tanner, M. A. (1990). A Monte Carlo implementation of the 
#   EM algorithm and the poor man's data augmentation algorithms. JASA, 85(411), 699-704.
# - Caffo, B. S., Jank, W., & Jones, G. L. (2005). Ascent-based Monte Carlo 
#   expectation-maximization. JRSS-B, 67(2), 235-251.
#
# =============================================================================

"""
    mcem_mll(logliks, ImportanceWeights, SubjectWeights)

Compute the marginal log likelihood Q(θ|θ') for MCEM.

This is the importance-weighted expected complete-data log-likelihood:
```math
Q(θ|θ') = Σᵢ SubjectWeights[i] × Σⱼ ImportanceWeights[i][j] × logliks[i][j]
```

See also: [`mcem_ase`](@ref), [`mcem_lml`](@ref)
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