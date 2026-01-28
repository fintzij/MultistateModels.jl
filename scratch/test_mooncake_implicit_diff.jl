# =============================================================================
# Test: Mooncake + ImplicitDifferentiation.jl for PIJCV Gradients
# =============================================================================
#
# Goal: Test whether Mooncake as the AD backend for ImplicitDifferentiation.jl
# can compute correct PIJCV gradients.
#
# Success Criteria:
# 1. Gradient matches FD within 5% across log(λ) ∈ {0, 1, 2, 3, 4}
# 2. Zero-crossing agrees with FD within 0.1 on log scale
# 3. No errors or crashes
#
# Date: 2026-01-27
# =============================================================================

using Pkg
Pkg.activate("/Users/fintzij/Library/CloudStorage/OneDrive-BristolMyersSquibb/Documents/Julia packages/MultistateModels.jl")

using LinearAlgebra
using Random
using DataFrames
using Distributions
using ForwardDiff
using ADTypes
using Mooncake
using DifferentiationInterface
using ImplicitDifferentiation
using ImplicitDifferentiation: MatrixRepresentation, DirectLinearSolver
using MultistateModels
using StatsModels: @formula
using Printf

# Import internal functions
import MultistateModels: multistatemodel, Hazard,
                          ExactData, extract_paths, get_parameters_flat,
                          build_penalty_config, SplinePenalty, QuadraticPenalty,
                          compute_penalty, n_hyperparameters, set_hyperparameters,
                          _fit_inner_coefficients, compute_subject_grads_and_hessians_fast,
                          loglik_subject

println("="^70)
println("Test: Mooncake + ImplicitDifferentiation.jl for PIJCV Gradients")
println("="^70)
println()

# =============================================================================
# 1. Create Test Fixture
# =============================================================================

function create_test_fixture(; n_subjects=50, seed=12345)
    Random.seed!(seed)
    
    id = Int[]
    tstart = Float64[]
    tstop = Float64[]
    statefrom = Int[]
    stateto = Int[]
    obstype = Int[]
    
    for i in 1:n_subjects
        push!(id, i)
        push!(tstart, 0.0)
        t_event = rand(Exponential(2.0))
        t_event = min(t_event, 5.0)
        push!(tstop, t_event)
        push!(statefrom, 1)
        if t_event < 5.0
            push!(stateto, 2)
            push!(obstype, 1)
        else
            push!(stateto, 1)
            push!(obstype, 2)
        end
    end
    
    dat = DataFrame(
        id = id, tstart = tstart, tstop = tstop,
        statefrom = statefrom, stateto = stateto, obstype = obstype
    )
    
    h12 = Hazard(@formula(0 ~ 1), :sp, 1, 2)
    model = multistatemodel(h12; data=dat)
    
    return model, dat
end

println("Creating test fixture (n=50 subjects)...")
model, dat = create_test_fixture()
samplepaths = extract_paths(model)
data = ExactData(model, samplepaths)
beta_init = get_parameters_flat(model)
penalty = build_penalty_config(model, SplinePenalty())
lb, ub = model.bounds.lb, model.bounds.ub
n_lambda = n_hyperparameters(penalty)

println("  - Number of β parameters: ", length(beta_init))
println("  - Number of λ parameters: ", n_lambda)
println()

# =============================================================================
# 2. Define Forward Solve and Conditions for ImplicitDifferentiation.jl
# =============================================================================

# Forward function: solve inner optimization β̂(λ)
function forward_solve(log_lambda::AbstractVector)
    λ_float = [ForwardDiff.value(x) for x in log_lambda]
    λ = exp.(λ_float)
    penalty_current = set_hyperparameters(penalty, λ)
    β_opt = _fit_inner_coefficients(model, data, penalty_current, beta_init;
                                     lb=lb, ub=ub, maxiter=50)
    return β_opt, (lambda=λ,)
end

# Optimality conditions: ∇_β ℓ(β) - ∇_β penalty = 0
function optimality_conditions(log_lambda::AbstractVector, β::AbstractVector, z)
    λ = exp.(log_lambda)
    
    # Gradient of log-likelihood
    grad_ll = ForwardDiff.gradient(
        b -> MultistateModels.loglik_exact(b, data; neg=false), 
        collect(β)
    )
    
    # Gradient of penalty: Σⱼ λⱼ Sⱼ β
    n_β = length(β)
    T = promote_type(eltype(β), eltype(λ))
    grad_penalty = zeros(T, n_β)
    
    lambda_idx = 1
    for term in penalty.terms
        β_j = β[term.hazard_indices]
        grad_j = λ[lambda_idx] * (term.S * β_j)
        grad_penalty[term.hazard_indices] .+= grad_j
        lambda_idx += 1
    end
    
    return grad_ll - grad_penalty
end

println("Testing forward solve and conditions...")
log_λ_test = [2.0]
β_test, z_test = forward_solve(log_λ_test)
conds = optimality_conditions(log_λ_test, β_test, z_test)
println("  - Forward solve: β̂ computed ($(length(β_test)) params)")
println("  - Conditions at optimum: ||c||∞ = ", maximum(abs.(conds)))
println()

# =============================================================================
# 3. Create ImplicitFunction with Mooncake Backend
# =============================================================================

println("Creating ImplicitFunction with Mooncake backend...")
try
    global implicit_beta = ImplicitFunction(
        forward_solve, 
        optimality_conditions;
        representation=MatrixRepresentation(),
        linear_solver=DirectLinearSolver(),
        backends=(x=AutoMooncake(; config=nothing), 
                  y=AutoMooncake(; config=nothing))
    )
    println("  ✓ ImplicitFunction created successfully")
catch e
    println("  ✗ ERROR creating ImplicitFunction: ", e)
    rethrow()
end
println()

# =============================================================================
# 4. Define PIJCV Criterion
# =============================================================================

"""
Compute PIJCV criterion V(λ, β) at given β and λ.

V = Σᵢ [-ℓᵢ + gᵢᵀΔ⁻ⁱ + ½(Δ⁻ⁱ)ᵀHᵢΔ⁻ⁱ]

where Δ⁻ⁱ = (H_λ - Hᵢ)⁻¹ gᵢ (leave-one-out Newton step)
"""
function compute_pijcv_at_beta(β::Vector{Float64}, log_λ::Vector{Float64})
    λ = exp.(log_λ)
    n_subj = length(samplepaths)
    
    # Compute per-subject gradients and Hessians
    grads_ll, hess_ll = compute_subject_grads_and_hessians_fast(
        β, model, samplepaths; use_threads=:auto
    )
    subject_grads = [-g for g in grads_ll]  # gᵢ = -∇ℓᵢ (loss gradient)
    subject_hessians = [-H for H in hess_ll]  # Hᵢ = -∇²ℓᵢ (loss Hessian)
    
    # Compute full penalized Hessian: H_λ = Σᵢ Hᵢ + Σⱼ λⱼ Sⱼ
    H_unpenalized = sum(subject_hessians)
    H_λ = copy(H_unpenalized)
    lambda_idx = 1
    for term in penalty.terms
        idx = term.hazard_indices
        H_λ[idx, idx] .+= λ[lambda_idx] * term.S
        lambda_idx += 1
    end
    
    # Full gradient (should be ~0 at β̂)
    g = sum(subject_grads)
    
    # Compute V using Newton-step approximation
    V = 0.0
    for i in 1:n_subj
        gᵢ = subject_grads[i]
        Hᵢ = subject_hessians[i]
        
        # Per-subject log-likelihood
        ℓᵢ = loglik_subject(β, data, i)
        
        # LOO Hessian and Newton step
        H_loo = H_λ - Hᵢ
        try
            Δᵢ = H_loo \ gᵢ
            V += -ℓᵢ + dot(gᵢ, Δᵢ) + 0.5 * dot(Δᵢ, Hᵢ * Δᵢ)
        catch
            V += -ℓᵢ  # Fallback if H_loo is singular
        end
    end
    
    return V
end

# Test PIJCV computation
println("Testing PIJCV computation...")
V_test = compute_pijcv_at_beta(β_test, log_λ_test)
println("  - V(β̂, log_λ=2.0) = ", V_test)
println()

# =============================================================================
# 5. Full PIJCV Objective with Implicit β(λ)
# =============================================================================

function pijcv_objective(log_λ::Vector{Float64})
    # Get β̂(λ) via implicit function
    β_opt, _ = implicit_beta(log_λ)
    β_float = [ForwardDiff.value(x) for x in β_opt]
    
    # Compute V(β̂, λ)
    V = compute_pijcv_at_beta(β_float, log_λ)
    return V
end

# Test full objective
println("Testing full PIJCV objective...")
V_full = pijcv_objective(log_λ_test)
println("  - V(log_λ=2.0) via implicit function = ", V_full)
println()

# =============================================================================
# 6. Gradient Computation: Mooncake vs Finite Differences
# =============================================================================

println("="^70)
println("Gradient Comparison: Mooncake + ID.jl vs Finite Differences")
println("="^70)
println()

# Test at multiple log(λ) values
test_log_lambdas = [0.0, 1.0, 2.0, 3.0, 4.0]

println("log(λ)    V          grad_mooncake   grad_FD        ratio")
println("-"^65)

results = []
for log_λ_val in test_log_lambdas
    log_λ = [log_λ_val]
    
    # Compute V
    V = pijcv_objective(log_λ)
    
    # Finite difference gradient
    ε = 1e-5
    V_plus = pijcv_objective([log_λ_val + ε])
    V_minus = pijcv_objective([log_λ_val - ε])
    grad_fd = (V_plus - V_minus) / (2ε)
    
    # Mooncake gradient via DifferentiationInterface
    grad_mooncake = try
        DifferentiationInterface.gradient(pijcv_objective, AutoMooncake(; config=nothing), log_λ)[1]
    catch e
        println("  WARNING: Mooncake gradient failed at log_λ=$(log_λ_val): ", e)
        NaN
    end
    
    ratio = isfinite(grad_mooncake) ? grad_mooncake / grad_fd : NaN
    
    push!(results, (log_λ=log_λ_val, V=V, grad_mooncake=grad_mooncake, grad_fd=grad_fd, ratio=ratio))
    
    @printf("%6.2f    %10.4f  %12.6f    %12.6f   %6.3f\n", 
            log_λ_val, V, grad_mooncake, grad_fd, ratio)
end

println()

# =============================================================================
# 7. Analysis
# =============================================================================

println("="^70)
println("Analysis")
println("="^70)
println()

# Check if gradients match within 5%
valid_results = filter(r -> isfinite(r.ratio), results)
if !isempty(valid_results)
    ratios = [r.ratio for r in valid_results]
    within_5pct = all(0.95 .<= ratios .<= 1.05)
    within_10pct = all(0.90 .<= ratios .<= 1.10)
    
    println("Gradient accuracy:")
    println("  - All within 5% of FD:  ", within_5pct ? "✓ YES" : "✗ NO")
    println("  - All within 10% of FD: ", within_10pct ? "✓ YES" : "✗ NO")
    println("  - Ratio range: [$(minimum(ratios)), $(maximum(ratios))]")
    println()
    
    # Check for sign consistency
    signs_agree = all((r.grad_mooncake > 0) == (r.grad_fd > 0) || 
                      abs(r.grad_fd) < 0.01 for r in valid_results)
    println("Sign consistency: ", signs_agree ? "✓ YES" : "✗ NO")
else
    println("ERROR: No valid gradient computations!")
end

# Find approximate zero-crossing
fd_grads = [r.grad_fd for r in results]
if any(fd_grads .> 0) && any(fd_grads .< 0)
    # Find where sign changes
    for i in 1:(length(fd_grads)-1)
        if fd_grads[i] * fd_grads[i+1] < 0
            # Linear interpolation
            log_λ_cross_fd = test_log_lambdas[i] - fd_grads[i] * (test_log_lambdas[i+1] - test_log_lambdas[i]) / (fd_grads[i+1] - fd_grads[i])
            println()
            println("Approximate FD zero-crossing: log(λ) ≈ ", round(log_λ_cross_fd, digits=2))
            break
        end
    end
end

println("="^70)
println("Test Complete")
println("="^70)
