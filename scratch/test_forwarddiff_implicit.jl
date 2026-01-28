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

# Create test fixture
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

println("Setup complete: $(n_beta) β params, 1 λ param")

# Forward solve (same as before)
function forward_solve(log_lambda::AbstractVector)
    λ_float = [ForwardDiff.value(x) for x in log_lambda]
    λ = exp.(λ_float)
    penalty_current = set_hyperparameters(penalty, λ)
    β_opt = _fit_inner_coefficients(model, data, penalty_current, beta_init; lb=lb, ub=ub, maxiter=50)
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

# Create ImplicitFunction with ForwardDiff backends
implicit_beta = ImplicitFunction(forward_solve, optimality_conditions;
    representation=MatrixRepresentation(),
    linear_solver=DirectLinearSolver(),
    backends=(x=AutoForwardDiff(), y=AutoForwardDiff()))

println("\nTest 1: AD-compatible PIJCV computation")
println("-"^50)

# AD-compatible PIJCV computation that takes Dual-number ρ
function pijcv_full_ad(ρ::AbstractVector{T}) where T
    # 1. Get β̂(ρ) - this returns Dual numbers when ρ are Duals
    β_opt, _ = implicit_beta(ρ)
    
    # 2. Compute λ from ρ (AD-compatible)
    λ = exp.(ρ)
    
    # 3. Extract Float64 β for computing gᵢ, Hᵢ
    β_float = [ForwardDiff.value(x) for x in β_opt]
    
    # 4. Compute per-subject gradients and Hessians via ForwardDiff
    function subject_loglik(b, i)
        return loglik_subject(b, data, i)
    end
    
    subject_lls = Float64[]
    subject_grads = Vector{Float64}[]
    subject_hessians = Matrix{Float64}[]
    
    for i in 1:n_subj
        ll_i = subject_loglik(β_float, i)
        grad_i = ForwardDiff.gradient(b -> subject_loglik(b, i), β_float)
        hess_i = ForwardDiff.hessian(b -> subject_loglik(b, i), β_float)
        push!(subject_lls, ll_i)
        push!(subject_grads, -grad_i)  # gᵢ = -∇ℓᵢ
        push!(subject_hessians, -hess_i)  # Hᵢ = -∇²ℓᵢ
    end
    
    # 5. Build penalized Hessian H_λ (with Dual λ for proper AD)
    H_unpenalized = sum(subject_hessians)
    
    # Construct H_λ with proper element type for AD
    H_λ = zeros(T, n_beta, n_beta)
    H_λ .= H_unpenalized
    for term in penalty.terms
        idx = term.hazard_indices
        H_λ[idx, idx] .+= λ[1] * Matrix(term.S)
    end
    
    # 6. Compute V criterion
    V = zero(T)
    for i in 1:n_subj
        gᵢ = subject_grads[i]
        Hᵢ = subject_hessians[i]
        ℓᵢ = subject_lls[i]
        
        H_loo = H_λ - Hᵢ
        Δᵢ = H_loo \ gᵢ
        
        V += -ℓᵢ + dot(gᵢ, Δᵢ) + T(0.5) * dot(Δᵢ, Hᵢ * Δᵢ)
    end
    
    return V
end

# Test at ρ = 2.0
ρ_test = [2.0]
V_ad = pijcv_full_ad(ρ_test)
println("V([2.0]) via AD-compatible function = ", V_ad)

# Now let's use chain rule: dV/dρ = (∂V/∂β)·(dβ/dρ) + (∂V/∂λ)·(dλ/dρ)
println("\n" * "="^60)
println("Chain Rule Gradient Computation")
println("="^60)

# Compute dβ/dρ at a point using ImplicitDifferentiation
function compute_dbeta_drho(ρ_val::Float64)
    J = ForwardDiff.jacobian(ρ -> implicit_beta(ρ)[1], [ρ_val])
    return J[:, 1]
end

# Compute V(β, λ) at fixed β and λ (for partial derivative computation)
function compute_V_fixed(β::AbstractVector{T1}, λ::T2) where {T1, T2}
    T = promote_type(T1, T2, Float64)
    
    # Use Float64 values for the loglik_subject calls (it doesn't support Duals)
    β_float = [ForwardDiff.value(x) for x in β]
    
    subject_lls = Float64[]
    subject_grads = Vector{Float64}[]
    subject_hessians = Matrix{Float64}[]
    
    for i in 1:n_subj
        ll_i = loglik_subject(β_float, data, i)
        grad_i = ForwardDiff.gradient(b -> loglik_subject(b, data, i), β_float)
        hess_i = ForwardDiff.hessian(b -> loglik_subject(b, data, i), β_float)
        push!(subject_lls, ll_i)
        push!(subject_grads, -grad_i)
        push!(subject_hessians, -hess_i)
    end
    
    H_unpen = sum(subject_hessians)
    H_λ = Matrix{T}(undef, n_beta, n_beta)
    H_λ .= H_unpen
    for term in penalty.terms
        idx = term.hazard_indices
        H_λ[idx, idx] .+= λ * Matrix(term.S)
    end
    
    V = zero(T)
    for i in 1:n_subj
        gᵢ = subject_grads[i]
        Hᵢ = subject_hessians[i]
        ℓᵢ = subject_lls[i]
        H_loo = H_λ - Hᵢ
        Δᵢ = H_loo \ gᵢ
        V += -ℓᵢ + dot(gᵢ, Δᵢ) + T(0.5) * dot(Δᵢ, Hᵢ * Δᵢ)
    end
    return V
end

# Full chain rule gradient
function compute_gradient_chain_rule(ρ_val::Float64)
    β_opt, _ = forward_solve([ρ_val])
    λ = exp(ρ_val)
    
    dbeta_drho = compute_dbeta_drho(ρ_val)
    dV_dbeta = ForwardDiff.gradient(b -> compute_V_fixed(b, λ), β_opt)
    dV_dlambda = ForwardDiff.derivative(l -> compute_V_fixed(β_opt, l), λ)
    dlambda_drho = λ
    
    dV_drho = dot(dV_dbeta, dbeta_drho) + dV_dlambda * dlambda_drho
    return dV_drho
end

# Finite difference
ε = 1e-5

println("\nDiagnostic breakdown:")
println("-"^60)

ρ_test = 2.0
β_opt, _ = forward_solve([ρ_test])
λ_test = exp(ρ_test)

# Components
dbeta_drho = compute_dbeta_drho(ρ_test)
println("dβ/dρ = ", dbeta_drho)

# ∂V/∂λ at fixed β
dV_dlambda = ForwardDiff.derivative(l -> compute_V_fixed(β_opt, l), λ_test)
dlambda_drho = λ_test
println("\n∂V/∂λ = ", dV_dlambda)
println("dλ/dρ = ", dlambda_drho)
println("∂V/∂λ · dλ/dρ = ", dV_dlambda * dlambda_drho)

# ∂V/∂β at fixed λ (THIS IS THE PROBLEM - gᵢ, Hᵢ change with β!)
dV_dbeta = ForwardDiff.gradient(b -> compute_V_fixed(b, λ_test), β_opt)
println("\n∂V/∂β = ", dV_dbeta)
println("∂V/∂β · dβ/dρ = ", dot(dV_dbeta, dbeta_drho))

println("\nTotal chain rule gradient = ", dot(dV_dbeta, dbeta_drho) + dV_dlambda * dlambda_drho)

# Now let's compute true FD gradient
V_p = compute_V_fixed(forward_solve([ρ_test+ε])[1], exp(ρ_test+ε))
V_m = compute_V_fixed(forward_solve([ρ_test-ε])[1], exp(ρ_test-ε))
grad_fd = (V_p - V_m) / (2ε)
println("FD gradient = ", grad_fd)

println("\n" * "="^60)
println("The problem: gᵢ and Hᵢ in compute_V_fixed use β_float, not β!")
println("So ∂V/∂β doesn't capture how gᵢ, Hᵢ change with β.")
println("="^60)

println("\nρ        V         grad_chain   grad_FD      ratio")
println("-"^60)
for ρ in [0.0, 1.0, 2.0, 3.0, 4.0]
    V = compute_V_fixed(forward_solve([ρ])[1], exp(ρ))
    V_p = compute_V_fixed(forward_solve([ρ+ε])[1], exp(ρ+ε))
    V_m = compute_V_fixed(forward_solve([ρ-ε])[1], exp(ρ-ε))
    grad_fd = (V_p - V_m) / (2ε)
    
    grad_chain = compute_gradient_chain_rule(ρ)
    ratio = grad_chain / grad_fd
    
    @printf("%.1f    %8.4f    %8.5f    %8.5f    %6.3f\n", ρ, V, grad_chain, grad_fd, ratio)
end

println("\n" * "="^60)
println("Test Complete")
println("="^60)
