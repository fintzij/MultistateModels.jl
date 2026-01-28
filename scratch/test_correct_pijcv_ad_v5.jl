# =============================================================================
# PIJCV Gradient via Correct Formula - WITH THIRD DERIVATIVES
# =============================================================================
#
# Key insight from v4: Subject 1 has ratio = -0.35 (wrong sign!)
# This means the simplified chain rule (ignoring ∂Hᵢ/∂β) is inadequate
# 
# The full derivative requires:
#   dΔ⁻ⁱ/dρ = H_loo⁻¹ [dgᵢ/dρ - dH_loo/dρ · Δᵢ]
# where:
#   dH_loo/dρ = dH_λ/dρ - dHᵢ/dρ
#   dH_λ/dρ = λS + Σⱼ (∂Hⱼ/∂β)·dβ̂/dρ  [third derivatives!]
#   dHᵢ/dρ = (∂Hᵢ/∂β)·dβ̂/dρ
# =============================================================================

using LinearAlgebra, Random, DataFrames, Distributions, Printf
using ForwardDiff, ADTypes
using ImplicitDifferentiation
using ImplicitDifferentiation: MatrixRepresentation, DirectLinearSolver
using MultistateModels
using StatsModels: @formula

import MultistateModels: multistatemodel, Hazard,
                          ExactData, extract_paths, get_parameters_flat,
                          build_penalty_config, SplinePenalty, QuadraticPenalty,
                          compute_penalty, n_hyperparameters, set_hyperparameters,
                          _fit_inner_coefficients, loglik_subject

# =============================================================================
# Test Fixture Setup (Same as v4)
# =============================================================================
Random.seed!(12345)
id = Int[]; tstart = Float64[]; tstop = Float64[]
statefrom = Int[]; stateto = Int[]; obstype = Int[]
for i in 1:50
    push!(id, i); push!(tstart, 0.0)
    t_event = min(rand(Exponential(2.0)), 5.0)
    push!(tstop, t_event); push!(statefrom, 1)
    push!(stateto, t_event < 5.0 ? 2 : 1)
    push!(obstype, t_event < 5.0 ? 1 : 2)
end
dat = DataFrame(id=id, tstart=tstart, tstop=tstop, statefrom=statefrom, stateto=stateto, obstype=obstype)
h12 = Hazard(@formula(0 ~ 1), :sp, 1, 2)
model = multistatemodel(h12; data=dat)
samplepaths = extract_paths(model)
data = ExactData(model, samplepaths)
beta_init = get_parameters_flat(model)
penalty = build_penalty_config(model, SplinePenalty())
lb, ub = model.bounds.lb, model.bounds.ub
n_subj = length(samplepaths)
n_beta = length(beta_init)

println("="^70)
println("THIRD DERIVATIVE GRADIENT TEST")
println("="^70)
println("n_subj = $n_subj, n_beta = $n_beta")

# Forward solve (same as v4)
function forward_solve(log_lambda::AbstractVector)
    λ_float = [ForwardDiff.value(x) for x in log_lambda]
    λ = exp.(λ_float)
    penalty_current = set_hyperparameters(penalty, λ)
    β_opt = _fit_inner_coefficients(model, data, penalty_current, beta_init; 
                                     lb=lb, ub=ub, maxiter=50)
    return β_opt, (lambda=λ,)
end

function optimality_conditions(log_lambda::AbstractVector, β::AbstractVector, z)
    λ = exp.(log_lambda)
    grad_ll = ForwardDiff.gradient(b -> MultistateModels.loglik_exact(b, data; neg=false), collect(β))
    n_β = length(β)
    T = promote_type(eltype(β), eltype(λ))
    grad_penalty = zeros(T, n_β)
    for term in penalty.terms
        β_j = β[term.hazard_indices]
        grad_j = λ[1] * (term.S * β_j)
        grad_penalty[term.hazard_indices] .+= grad_j
    end
    return grad_ll - grad_penalty
end

implicit_beta = ImplicitFunction(forward_solve, optimality_conditions;
    representation=MatrixRepresentation(),
    linear_solver=DirectLinearSolver()
)

# Build S_full matrix
S_full = zeros(n_beta, n_beta)
for term in penalty.terms
    idx = term.hazard_indices
    S_full[idx, idx] .= Matrix(term.S)
end

# Helper function for H_λ
function compute_H_λ_at(β, λ_val)
    H = zeros(n_beta, n_beta)
    for i in 1:n_subj
        H_i = -ForwardDiff.hessian(b -> loglik_subject(b, data, i), β)
        H .+= H_i
    end
    for term in penalty.terms
        idx = term.hazard_indices
        H[idx, idx] .+= λ_val * Matrix(term.S)
    end
    return H
end

# =============================================================================
# Main Test at ρ = 2.0
# =============================================================================
ρ = 2.0
β_opt, _ = forward_solve([ρ])
λ = exp(ρ)

println("\nAt ρ = $ρ, λ = $λ")
println("β̂ = ", β_opt)

# dβ̂/dρ via ImplicitDifferentiation.jl
dbeta_drho = ForwardDiff.jacobian(ρ_vec -> implicit_beta(ρ_vec)[1], [ρ])[:, 1]
println("dβ̂/dρ = ", dbeta_drho)

# Verify via FD
ε_fd = 1e-6
β_p, _ = forward_solve([ρ + ε_fd])
β_m, _ = forward_solve([ρ - ε_fd])
dbeta_drho_fd = (β_p - β_m) / (2ε_fd)
println("dβ̂/dρ (FD) = ", dbeta_drho_fd)
println("Ratio = ", dbeta_drho ./ dbeta_drho_fd)

# Compute per-subject quantities
println("\nComputing subject-level Hessians and third derivatives...")
subject_grads = Vector{Float64}[]
subject_hessians = Matrix{Float64}[]
subject_dH_dbeta = []  # Third derivative tensors

for i in 1:n_subj
    g_i = -ForwardDiff.gradient(b -> loglik_subject(b, data, i), β_opt)
    H_i = -ForwardDiff.hessian(b -> loglik_subject(b, data, i), β_opt)
    
    # Third derivative: ∂Hᵢ/∂β
    H_flat_jac = ForwardDiff.jacobian(
        β -> vec(-ForwardDiff.hessian(b -> loglik_subject(b, data, i), β)),
        β_opt
    )
    dH_dbeta_i = reshape(H_flat_jac, n_beta, n_beta, n_beta)
    
    push!(subject_grads, g_i)
    push!(subject_hessians, H_i)
    push!(subject_dH_dbeta, dH_dbeta_i)
end

# Build H_λ
H_λ = sum(subject_hessians)
for term in penalty.terms
    idx = term.hazard_indices
    H_λ[idx, idx] .+= λ * Matrix(term.S)
end

# Compute dH_λ/dρ WITH third derivatives
# dH_λ/dρ = λS + Σⱼ (∂Hⱼ/∂β)·dβ̂/dρ
dH_λ_drho = λ * S_full
for i in 1:n_subj
    for l in 1:n_beta
        dH_λ_drho .+= subject_dH_dbeta[i][:,:,l] * dbeta_drho[l]
    end
end

# Verify dH_λ/dρ via FD
H_λ_p = compute_H_λ_at(β_p, exp(ρ + ε_fd))
H_λ_m = compute_H_λ_at(β_m, exp(ρ - ε_fd))
dH_λ_drho_fd = (H_λ_p - H_λ_m) / (2ε_fd)

println("\ndH_λ/dρ comparison:")
println("  Analytical (with 3rd deriv) Frobenius norm: ", norm(dH_λ_drho))
println("  FD Frobenius norm:                          ", norm(dH_λ_drho_fd))
println("  Relative error:                             ", norm(dH_λ_drho - dH_λ_drho_fd) / norm(dH_λ_drho_fd))

# =============================================================================
# Per-subject gradient with FULL chain rule
# =============================================================================
println("\n" * "="^70)
println("Per-subject contribution analysis (with third derivatives)")
println("="^70)

V_total = 0.0
dV_drho_total = 0.0
dV_drho_total_fd = 0.0

for i in 1:n_subj
    global V_total, dV_drho_total, dV_drho_total_fd
    
    gᵢ = subject_grads[i]
    Hᵢ = subject_hessians[i]
    dH_dbeta_i = subject_dH_dbeta[i]
    
    H_loo = H_λ - Hᵢ
    Δᵢ = H_loo \ gᵢ
    β_tilde_i = β_opt - Δᵢ
    
    # V contribution
    ll_at_pseudo = loglik_subject(β_tilde_i, data, i)
    V_i = -ll_at_pseudo
    V_total += V_i
    
    # Gradient at pseudo-estimate
    grad_ll_at_pseudo = ForwardDiff.gradient(b -> loglik_subject(b, data, i), β_tilde_i)
    
    # FULL chain rule including third derivatives
    # gᵢ = -∇ℓᵢ(β̂), so dgᵢ/dρ = -∇²ℓᵢ(β̂)·dβ̂/dρ = Hᵢ·dβ̂/dρ
    dgᵢ_drho = Hᵢ * dbeta_drho  # FIXED: was -Hᵢ
    
    # dHᵢ/dρ = (∂Hᵢ/∂β)·dβ̂/dρ
    dHᵢ_drho = zeros(n_beta, n_beta)
    for l in 1:n_beta
        dHᵢ_drho .+= dH_dbeta_i[:,:,l] * dbeta_drho[l]
    end
    
    # dH_loo/dρ = dH_λ/dρ - dHᵢ/dρ
    dH_loo_drho = dH_λ_drho - dHᵢ_drho
    
    dDelta_drho = H_loo \ (dgᵢ_drho - dH_loo_drho * Δᵢ)
    dbeta_tilde_drho = dbeta_drho - dDelta_drho
    
    dV_i_drho = -dot(grad_ll_at_pseudo, dbeta_tilde_drho)
    dV_drho_total += dV_i_drho
    
    # FD gradient for comparison
    function V_i_func(ρ_val)
        β_tmp, _ = forward_solve([ρ_val])
        λ_tmp = exp(ρ_val)
        
        g_i_tmp = -ForwardDiff.gradient(b -> loglik_subject(b, data, i), β_tmp)
        H_i_tmp = -ForwardDiff.hessian(b -> loglik_subject(b, data, i), β_tmp)
        
        H_λ_tmp = compute_H_λ_at(β_tmp, λ_tmp)
        H_loo_tmp = H_λ_tmp - H_i_tmp
        Δ_i_tmp = H_loo_tmp \ g_i_tmp
        β_tilde_tmp = β_tmp - Δ_i_tmp
        
        return -loglik_subject(β_tilde_tmp, data, i)
    end
    
    dV_i_drho_fd = (V_i_func(ρ + ε_fd) - V_i_func(ρ - ε_fd)) / (2ε_fd)
    dV_drho_total_fd += dV_i_drho_fd
    
    if i <= 5
        ratio = dV_i_drho / dV_i_drho_fd
        println("\nSubject $i:")
        println("  V_i = $V_i")
        println("  dV_i/dρ (full chain) = $dV_i_drho")
        println("  dV_i/dρ (FD)         = $dV_i_drho_fd")
        println("  Ratio = $ratio")
    end
end

println("\n" * "="^70)
println("SUMMARY at ρ = $ρ")
println("="^70)
println("V_total = $V_total")
println("dV/dρ (full chain rule with 3rd deriv) = $dV_drho_total")
println("dV/dρ (FD)                             = $dV_drho_total_fd")
println("Ratio = $(dV_drho_total / dV_drho_total_fd)")

# =============================================================================
# Test at multiple ρ values
# =============================================================================
println("\n" * "="^70)
println("Test at multiple ρ values")
println("="^70)

function compute_gradient_full(ρ_val)
    β_loc, _ = forward_solve([ρ_val])
    λ_loc = exp(ρ_val)
    
    dbeta_loc = ForwardDiff.jacobian(ρ_vec -> implicit_beta(ρ_vec)[1], [ρ_val])[:, 1]
    
    # Subject quantities
    local_grads = [-ForwardDiff.gradient(b -> loglik_subject(b, data, i), β_loc) for i in 1:n_subj]
    local_hess = [-ForwardDiff.hessian(b -> loglik_subject(b, data, i), β_loc) for i in 1:n_subj]
    local_dH = [reshape(ForwardDiff.jacobian(β -> vec(-ForwardDiff.hessian(b -> loglik_subject(b, data, i), β)), β_loc), n_beta, n_beta, n_beta) for i in 1:n_subj]
    
    H_λ_loc = compute_H_λ_at(β_loc, λ_loc)
    
    # dH_λ/dρ with third derivatives
    dH_λ_drho_loc = λ_loc * S_full
    for i in 1:n_subj
        for l in 1:n_beta
            dH_λ_drho_loc .+= local_dH[i][:,:,l] * dbeta_loc[l]
        end
    end
    
    dV = 0.0
    for i in 1:n_subj
        H_loo_loc = H_λ_loc - local_hess[i]
        Δ_loc = H_loo_loc \ local_grads[i]
        β_tilde_loc = β_loc - Δ_loc
        
        grad_ll_pseudo = ForwardDiff.gradient(b -> loglik_subject(b, data, i), β_tilde_loc)
        
        # dgᵢ/dρ = Hᵢ·dβ̂/dρ (not -Hᵢ)
        dg_drho = local_hess[i] * dbeta_loc  # FIXED: was -local_hess[i]
        
        dH_i_drho = zeros(n_beta, n_beta)
        for l in 1:n_beta
            dH_i_drho .+= local_dH[i][:,:,l] * dbeta_loc[l]
        end
        
        dH_loo_drho_loc = dH_λ_drho_loc - dH_i_drho
        dDelta = H_loo_loc \ (dg_drho - dH_loo_drho_loc * Δ_loc)
        dbeta_tilde = dbeta_loc - dDelta
        
        dV += -dot(grad_ll_pseudo, dbeta_tilde)
    end
    return dV
end

function compute_gradient_fd(ρ_val)
    function V_func(ρ_v)
        β_tmp, _ = forward_solve([ρ_v])
        λ_tmp = exp(ρ_v)
        
        V = 0.0
        H_λ_tmp = compute_H_λ_at(β_tmp, λ_tmp)
        for i in 1:n_subj
            g_i_tmp = -ForwardDiff.gradient(b -> loglik_subject(b, data, i), β_tmp)
            H_i_tmp = -ForwardDiff.hessian(b -> loglik_subject(b, data, i), β_tmp)
            H_loo_tmp = H_λ_tmp - H_i_tmp
            Δ_i_tmp = H_loo_tmp \ g_i_tmp
            β_tilde_tmp = β_tmp - Δ_i_tmp
            V += -loglik_subject(β_tilde_tmp, data, i)
        end
        return V
    end
    return (V_func(ρ_val + ε_fd) - V_func(ρ_val - ε_fd)) / (2ε_fd)
end

println("\nρ      dV/dρ_full    dV/dρ_FD      ratio")
println("-"^50)
for ρ_test in [0.0, 1.0, 2.0, 3.0, 4.0]
    dv_full = compute_gradient_full(ρ_test)
    dv_fd = compute_gradient_fd(ρ_test)
    ratio = dv_full / dv_fd
    println(@sprintf("%5.1f  %12.5f  %12.5f  %7.4f", ρ_test, dv_full, dv_fd, ratio))
end

println("\n" * "="^70)
println("Done")
println("="^70)
