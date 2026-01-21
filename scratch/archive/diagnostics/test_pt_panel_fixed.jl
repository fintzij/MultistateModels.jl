"""
Minimal test script for pt_panel_fixed (Section 6) with verbose output.
Tests eigenvalue ordering constraints for 2-phase Coxian with single destination.
"""

using Test
using MultistateModels
using DataFrames
using Random
using Statistics
using LinearAlgebra

# Import internal functions
import MultistateModels: Hazard, multistatemodel, fit, set_parameters!, simulate,
    get_parameters_flat, get_parameters, SamplePath, @formula

const RNG_SEED = 0xDEAD0001
const N_SUBJECTS = 1000
const MAX_TIME = 10.0
const PANEL_TIMES = [0.0, 2.5, 5.0, 7.5, 10.0]

println("\n" * "="^70)
println("Testing pt_panel_fixed with Eigenvalue Ordering Constraints")
println("="^70)

# Generate covariate data
Random.seed!(RNG_SEED + 100)
n_subj = N_SUBJECTS
cov_vals = rand([0.0, 1.0], n_subj)

# Create exact data template for simulation
exact_template = DataFrame(
    id = 1:n_subj,
    tstart = zeros(n_subj),
    tstop = fill(MAX_TIME, n_subj),
    statefrom = ones(Int, n_subj),
    stateto = ones(Int, n_subj),
    obstype = ones(Int, n_subj),
    x = cov_vals
)

# Build simulation model using production API
println("\n--- Building model with default (eigorder_sctp) constraints ---")
h12 = Hazard(@formula(0 ~ x), :pt, 1, 2; n_phases=2)
println("PhaseTypeHazard structure: $(h12.structure)")
println("PhaseTypeHazard covariate_constraints: $(h12.covariate_constraints)")

model_sim = multistatemodel(h12; data=exact_template, verbose=true)

# True parameters (production API structure):
# [h1_ab_rate, h12_a_rate, h12_a_x, h12_b_rate, h12_b_x]
true_lambda = 0.4    # progression rate
true_mu1 = 0.25      # exit rate from phase 1
true_mu2 = 0.5       # exit rate from phase 2
true_beta = 0.35     # shared covariate effect

println("\n--- True parameters ---")
println("  λ = $true_lambda (progression)")
println("  μ₁ = $true_mu1 (exit from phase 1)")
println("  μ₂ = $true_mu2 (exit from phase 2)")
println("  β = $true_beta (covariate effect)")

# Compute eigenvalues
nu1 = true_lambda + true_mu1
nu2 = true_mu2
println("\n--- Eigenvalue ordering check ---")
println("  ν₁ = λ + μ₁ = $nu1 (total rate from phase 1)")
println("  ν₂ = μ₂ = $nu2 (total rate from phase 2)")
println("  Constraint ν₁ ≥ ν₂: $(nu1) ≥ $(nu2) ? $(nu1 >= nu2)")

params_sim = (
    h1_ab = [true_lambda],
    h12_a = [true_mu1, true_beta],
    h12_b = [true_mu2, true_beta]  # Same beta - homogeneous constraint
)
set_parameters!(model_sim, params_sim)

# Simulate exact data with paths (needed for panel conversion)
println("\n--- Simulating data ---")
sim_result = simulate(model_sim; paths=true, data=true, nsim=1)
exact_data = sim_result[1][1]
paths = sim_result[2][1]

# Convert to panel observations using paths
println("Converting to panel observations...")
panel_rows = []
for path in paths
    subj_id = path.subj
    x_val = cov_vals[subj_id]
    
    for i in 1:(length(PANEL_TIMES)-1)
        t_start = PANEL_TIMES[i]
        t_stop = PANEL_TIMES[i+1]
        
        # State at t_start
        idx_start = searchsortedlast(path.times, t_start)
        state_start = idx_start >= 1 ? path.states[idx_start] : 1
        
        # State at t_stop
        idx_stop = searchsortedlast(path.times, t_stop)
        state_stop = idx_stop >= 1 ? path.states[idx_stop] : 1
        
        push!(panel_rows, (
            id = subj_id,
            tstart = t_start,
            tstop = t_stop,
            statefrom = state_start,
            stateto = state_stop,
            obstype = 2,
            x = x_val
        ))
    end
end
panel_data = DataFrame(panel_rows)

n_absorbed = sum(panel_data.stateto .== 2)
println("  Panel data: $(nrow(panel_data)) observations, $n_absorbed absorptions")

# Build model for fitting using same production API
println("\n--- Fitting model ---")
model_fit = multistatemodel(h12; data=panel_data, verbose=true)

println("\nFitting phase-type model with covariate from panel data...")
fitted = fit(model_fit; verbose=true, compute_vcov=true)
fitted_params = get_parameters_flat(fitted)

# Report results
true_params = [true_lambda, true_mu1, true_beta, true_mu2, true_beta]
println("\n" * "="^70)
println("RESULTS")
println("="^70)
println("\nParameter comparison:")
param_names = ["λ (progression)", "μ₁ (exit phase 1)", "β₁ (cov effect phase 1)", "μ₂ (exit phase 2)", "β₂ (cov effect phase 2)"]
for (i, name) in enumerate(param_names)
    true_val = true_params[i]
    fit_val = fitted_params[i]
    rel_err = abs(fit_val - true_val) / abs(true_val) * 100
    println("  $name:")
    println("    True: $(round(true_val, digits=4))")
    println("    Fitted: $(round(fit_val, digits=4))")
    println("    Rel. Error: $(round(rel_err, digits=1))%")
end

# Check eigenvalue ordering in fitted model
fitted_lambda = fitted_params[1]
fitted_mu1 = fitted_params[2]
fitted_mu2 = fitted_params[4]
fitted_nu1 = fitted_lambda + fitted_mu1
fitted_nu2 = fitted_mu2

println("\n--- Fitted eigenvalue check ---")
println("  ν₁ = λ + μ₁ = $(round(fitted_nu1, digits=4))")
println("  ν₂ = μ₂ = $(round(fitted_nu2, digits=4))")
println("  Constraint ν₁ ≥ ν₂: $(fitted_nu1) ≥ $(fitted_nu2) ? $(fitted_nu1 >= fitted_nu2)")

# Summary
println("\n" * "="^70)
println("SUMMARY")
println("="^70)

baseline_rel_errors = [abs(fitted_params[i] - true_params[i]) / abs(true_params[i]) 
                       for i in [1, 2, 4]]  # λ, μ₁, μ₂
covariate_rel_errors = [abs(fitted_params[i] - true_params[i]) / abs(true_params[i]) 
                        for i in [3, 5]]  # β₁, β₂

println("  Max baseline rel. error: $(round(maximum(baseline_rel_errors)*100, digits=1))%")
println("  Max covariate rel. error: $(round(maximum(covariate_rel_errors)*100, digits=1))%")
println("  β₁ - β₂ (should be 0): $(round(fitted_params[3] - fitted_params[5], digits=6))")
