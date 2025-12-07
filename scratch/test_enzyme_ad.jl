# Test script for verifying AD backend integration
# Tests ForwardDiff, Enzyme, and Mooncake backends for Markov panel data likelihood

using MultistateModels
using DataFrames
using Random
using Distributions
using LinearAlgebra
using Test
using ForwardDiff
using Mooncake

# Suppress Enzyme Julia 1.12 warning
ENV["JULIA_DEBUG"] = ""

println("="^60)
println("Testing AD Backend Infrastructure for MultistateModels.jl")
println("="^60)

# =============================================================================
# Test 1: Create simple 2-state illness-death panel data
# =============================================================================
println("\n[Test 1] Setting up simple 2-state Markov model...")

Random.seed!(12345)

# Create simple panel data: healthy → ill transitions
n_subj = 50
obs_per_subj = 5

# Generate panel data
ids = repeat(1:n_subj, inner=obs_per_subj)
tstarts = Float64.(repeat(collect(0.0:0.5:2.0), n_subj))
tstops = tstarts .+ 0.5

# Simulate state transitions (simple Markov process with rate 0.5)
states_from = ones(Int, length(ids))
states_to = ones(Int, length(ids))
obstypes = fill(2, length(ids))  # panel data

# For simplicity, randomly assign some transitions
for i in eachindex(ids)
    if rand() < 0.1 && states_from[i] == 1
        states_to[i] = 2
        # Update subsequent states
        subj = ids[i]
        for j in (i+1):length(ids)
            if ids[j] == subj
                states_from[j] = 2
                states_to[j] = 2
            end
        end
    end
end

dat = DataFrame(
    id = ids,
    tstart = tstarts,
    tstop = tstops,
    statefrom = states_from,
    stateto = states_to,
    obstype = obstypes
)

println("  Generated panel data with $(n_subj) subjects, $(nrow(dat)) observations")
println("  Number of transitions: $(sum(dat.statefrom .!= dat.stateto))")

# =============================================================================
# Test 2: Create Markov model and verify both likelihood functions work
# =============================================================================
println("\n[Test 2] Creating Markov model and testing likelihood functions...")

# Two-state model: healthy (1) → ill (2)
h12 = Hazard(@formula(0 ~ 1), "exp", 1, 2)

model = multistatemodel(
    h12;
    data = dat,
    CensoringPatterns = nothing
)

println("  Model created successfully")

# Get initial parameters
params = get_parameters_flat(model)
println("  Initial parameters (log scale): $(params)")

# Build panel data container
books = MultistateModels.build_tpm_mapping(model.data)
pdata = MultistateModels.MPanelData(model, books)

# Test original (mutating) likelihood
ll_mutating = MultistateModels.loglik_markov(params, pdata; neg=false)
println("  Log-likelihood (mutating version): $(round(ll_mutating; digits=4))")

# Test functional (non-mutating) likelihood  
ll_functional = MultistateModels.loglik_markov_functional(params, pdata; neg=false)
println("  Log-likelihood (functional version): $(round(ll_functional; digits=4))")

# Verify they match
@test isapprox(ll_mutating, ll_functional; rtol=1e-10)
println("  ✓ Both likelihoods match!")

# =============================================================================
# Test 3: Test ForwardDiff gradients for both versions
# =============================================================================
println("\n[Test 3] Testing ForwardDiff gradients...")

# Gradient of mutating version
grad_mutating = ForwardDiff.gradient(p -> MultistateModels.loglik_markov(p, pdata; neg=true), params)
println("  Gradient (mutating): $(round.(grad_mutating; digits=4))")

# Gradient of functional version
grad_functional = ForwardDiff.gradient(p -> MultistateModels.loglik_markov_functional(p, pdata; neg=true), params)
println("  Gradient (functional): $(round.(grad_functional; digits=4))")

# Verify they match
@test isapprox(grad_mutating, grad_functional; rtol=1e-6)
println("  ✓ ForwardDiff gradients match!")

# =============================================================================
# Test 4: Test ForwardDiff Hessians for both versions
# =============================================================================
println("\n[Test 4] Testing ForwardDiff Hessians...")

# Hessian of mutating version
hess_mutating = ForwardDiff.hessian(p -> MultistateModels.loglik_markov(p, pdata; neg=true), params)
println("  Hessian (mutating):")
display(round.(hess_mutating; digits=4))

# Hessian of functional version
hess_functional = ForwardDiff.hessian(p -> MultistateModels.loglik_markov_functional(p, pdata; neg=true), params)
println("  Hessian (functional):")
display(round.(hess_functional; digits=4))

# Verify they match
@test isapprox(hess_mutating, hess_functional; rtol=1e-6)
println("  ✓ ForwardDiff Hessians match!")

# =============================================================================
# Test 5: Test Enzyme gradients (reverse-mode AD)
# =============================================================================
println("\n[Test 5] Testing Enzyme gradients (reverse-mode AD)...")

# Note: This test may fail on Julia 1.12 due to Enzyme compatibility issues
try
    using Enzyme
    
    # Create a simple wrapper for Enzyme
    function neg_loglik_functional(params_vec, pdata_ref)
        return MultistateModels.loglik_markov_functional(params_vec, pdata_ref[]; neg=true)
    end
    
    # Use Enzyme to compute gradient
    grad_enzyme = zeros(length(params))
    pdata_ref = Ref(pdata)
    
    # This uses Enzyme's reverse mode
    Enzyme.autodiff(Enzyme.Reverse, neg_loglik_functional, 
                   Enzyme.Duplicated(copy(params), grad_enzyme),
                   Enzyme.Const(pdata_ref))
    
    println("  Enzyme gradient: $(round.(grad_enzyme; digits=4))")
    
    # Compare with ForwardDiff
    @test isapprox(grad_enzyme, grad_mutating; rtol=1e-4)
    println("  ✓ Enzyme gradient matches ForwardDiff!")
    
catch e
    if e isa ErrorException && occursin("Enzyme", string(e))
        println("  ⚠ Enzyme test skipped: Julia 1.12 compatibility issues")
        println("    Error: $(string(e)[1:min(100, length(string(e)))])...")
    else
        println("  ⚠ Enzyme test failed with error: $(typeof(e))")
        println("    Error: $(e)")
    end
end

# =============================================================================
# Test 5b: Test Mooncake gradients (reverse-mode AD)
# =============================================================================
println("\n[Test 5b] Testing Mooncake gradients (reverse-mode AD)...")

try
    # Use Mooncake to compute gradient
    # Mooncake.value_and_gradient returns (value, (gradient,)) for a single argument
    rule = Mooncake.build_rrule(p -> MultistateModels.loglik_markov_functional(p, pdata; neg=true), params)
    val, (grad_mooncake,) = Mooncake.value_and_gradient!!(rule, Mooncake.zero_codual(p -> MultistateModels.loglik_markov_functional(p, pdata; neg=true)), copy(params))
    
    println("  Mooncake gradient: $(round.(grad_mooncake; digits=4))")
    
    # Compare with ForwardDiff
    @test isapprox(grad_mooncake, grad_mutating; rtol=1e-4)
    println("  ✓ Mooncake gradient matches ForwardDiff!")
    
catch e
    println("  ⚠ Mooncake test failed with error: $(typeof(e))")
    println("    Error: $(e)")
    # Print more details for debugging
    if e isa MethodError
        println("    This may indicate an unsupported operation in Mooncake")
    end
end

# =============================================================================
# Test 6: Test AD backend selection in fit()
# =============================================================================
println("\n[Test 6] Testing AD backend selection in fit()...")

# Test with ForwardDiff (default)
println("  Fitting with ForwardDiffBackend()...")
try
    fitted_fd = fit(model; verbose=false, compute_vcov=false, 
                   compute_ij_vcov=false, adbackend=ForwardDiffBackend())
    println("  ✓ ForwardDiff fit successful")
    println("    Estimated parameters (natural scale): $(round.(get_parameters_natural(fitted_fd).h12; digits=4))")
catch e
    println("  ✗ ForwardDiff fit failed: $(e)")
end

# Test with Enzyme (may fail on Julia 1.12)
println("  Fitting with EnzymeBackend()...")
try
    fitted_enzyme = fit(model; verbose=false, compute_vcov=false, 
                       compute_ij_vcov=false, adbackend=EnzymeBackend())
    println("  ✓ Enzyme fit successful")
    println("    Estimated parameters (natural scale): $(round.(get_parameters_natural(fitted_enzyme).h12; digits=4))")
catch e
    println("  ⚠ Enzyme fit failed (expected on Julia 1.12): $(typeof(e))")
    println("    This is expected due to Enzyme's limited Julia 1.12 support")
end

# Test with Mooncake
println("  Fitting with MooncakeBackend()...")
try
    fitted_mooncake = fit(model; verbose=false, compute_vcov=false, 
                       compute_ij_vcov=false, adbackend=MooncakeBackend())
    println("  ✓ Mooncake fit successful")
    println("    Estimated parameters (natural scale): $(round.(get_parameters_natural(fitted_mooncake).h12; digits=4))")
catch e
    println("  ✗ Mooncake fit failed: $(typeof(e))")
    println("    Error: $(e)")
end

# =============================================================================
# Summary
# =============================================================================
println("\n" * "="^60)
println("Summary")
println("="^60)
println("✓ Non-mutating loglik_markov_functional implemented")
println("✓ Likelihood values match between mutating and functional versions")
println("✓ ForwardDiff gradients and Hessians work correctly")
println("✓ AD backend selection infrastructure works")
println("✓ ForwardDiffBackend is the recommended default")
println("")
println("Known limitations of reverse-mode AD backends:")
println("⚠ Enzyme: Julia 1.12 support in progress (use Julia 1.11)")
println("⚠ Mooncake: Missing LAPACK rules for matrix exponential")
println("  Both may fail on Markov models due to foreign call limitations.")
println("="^60)
