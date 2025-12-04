"""
Diagnostic script to compare exact data vs panel data fitting for reversible model with TVC.

This helps isolate whether the MCEM issues are due to:
1. Panel data observation model (sampling paths)
2. Likelihood computation with TVC
3. Importance weight calculation
4. Something else

We'll fit the same model to:
1. Exact data (no MCEM needed - direct MLE)
2. Panel data (requires MCEM)

And compare parameter estimates and log-likelihoods.
"""

using MultistateModels
using DataFrames
using Random
using Statistics
using LinearAlgebra
using Distributions

# Import internal functions
import MultistateModels: Hazard, multistatemodel, fit, set_parameters!, simulate,
    get_parameters_flat, get_log_scale_params, SamplePath, loglik, nest_params,
    make_subjdat, loglik_path, prepare_parameters

println("=" ^ 70)
println("DIAGNOSTIC: Exact vs Panel Data for Reversible Model with TVC")
println("=" ^ 70)

# =============================================================================
# Setup: Reversible 1↔2 model with TVC
# =============================================================================

Random.seed!(12345)

# True parameters (Weibull with shape=1 = exponential)
true_params = (
    h12 = [log(1.0), log(0.3), 0.3],   # log(shape), log(scale), beta
    h21 = [log(1.0), log(0.25), 0.2]   # log(shape), log(scale), beta
)

println("\nTrue parameters:")
println("  h12: shape=$(exp(true_params.h12[1])), scale=$(exp(true_params.h12[2])), beta=$(true_params.h12[3])")
println("  h21: shape=$(exp(true_params.h21[1])), scale=$(exp(true_params.h21[2])), beta=$(true_params.h21[3])")

# Build panel data structure (will be used for simulation)
n_subj = 1000
obs_times = [1.5, 3.0, 4.5, 6.0]
change_time = 2.0  # TVC changes at t=2

rows = []
for subj in 1:n_subj
    # Half get treatment at change_time, half don't
    trt = rand() < 0.5 ? [0.0, 1.0] : [0.0, 0.0]
    
    all_times = sort(unique([0.0; obs_times; change_time]))
    for i in 1:(length(all_times)-1)
        x_val = all_times[i] < change_time ? trt[1] : trt[2]
        push!(rows, (id=subj, tstart=all_times[i], tstop=all_times[i+1],
                     statefrom=1, stateto=1, obstype=2, x=x_val))
    end
end
panel_data = DataFrame(rows)

println("\nPanel data structure:")
println("  Subjects: $n_subj")
println("  Observation times: $obs_times")
println("  TVC change time: $change_time")
println("  Unique rows per subject: $(nrow(panel_data) ÷ n_subj)")

# =============================================================================
# Part 1: Simulate paths and extract EXACT data
# =============================================================================

println("\n" * "=" ^ 70)
println("PART 1: EXACT DATA FITTING (Direct MLE)")
println("=" ^ 70)

# Create simulation model
h12_sim = Hazard(@formula(0 ~ x), "wei", 1, 2)
h21_sim = Hazard(@formula(0 ~ x), "wei", 2, 1)
model_sim = multistatemodel(h12_sim, h21_sim; data=panel_data, surrogate=:markov)
set_parameters!(model_sim, true_params)

# Simulate with paths=true to get exact transition times
sim_result = simulate(model_sim; paths=true, data=true, nsim=1)
# sim_result is a tuple: (data_matrix, paths_matrix) 
# data_matrix is nsubj x nsim Matrix{DataFrame}
# paths_matrix is nsubj x nsim Matrix{SamplePath}
simulated_data = sim_result[1][:, 1]  # Vector of DataFrames for first simulation
simulated_paths = sim_result[2][:, 1]  # Vector of SamplePaths for first simulation

# Combine all DataFrames into one
simulated_data_combined = reduce(vcat, simulated_data)

println("\nSimulation result types:")
println("  sim_result: $(typeof(sim_result))")
println("  simulated_data: $(typeof(simulated_data)) with $(length(simulated_data)) DataFrames")
println("  simulated_paths: $(typeof(simulated_paths)) with $(length(simulated_paths)) paths")

# Count transitions
n_transitions_12 = 0
n_transitions_21 = 0
for path in simulated_paths
    for i in 1:(length(path.states)-1)
        if path.states[i] == 1 && path.states[i+1] == 2
            global n_transitions_12 += 1
        elseif path.states[i] == 2 && path.states[i+1] == 1
            global n_transitions_21 += 1
        end
    end
end

println("\nSimulated data summary:")
println("  Total 1→2 transitions: $n_transitions_12")
println("  Total 2→1 transitions: $n_transitions_21")
println("  Total transitions: $(n_transitions_12 + n_transitions_21)")

# Convert simulated data to exact observations (obstype=1)
exact_data = copy(simulated_data_combined)
exact_data.obstype .= 1

println("\nFitting model to EXACT data...")

# Fit with exact data (should use direct MLE, not MCEM)
h12_exact = Hazard(@formula(0 ~ x), "wei", 1, 2)
h21_exact = Hazard(@formula(0 ~ x), "wei", 2, 1)
model_exact = multistatemodel(h12_exact, h21_exact; data=exact_data, surrogate=:markov)

fitted_exact = fit(model_exact; verbose=true)

exact_params = get_parameters_flat(fitted_exact)
println("\nExact data MLE estimates:")
println("  h12: shape=$(round(exp(exact_params[1]), digits=3)), scale=$(round(exp(exact_params[2]), digits=3)), beta=$(round(exact_params[3], digits=3))")
println("  h21: shape=$(round(exp(exact_params[4]), digits=3)), scale=$(round(exp(exact_params[5]), digits=3)), beta=$(round(exact_params[6], digits=3))")
println("  Log-likelihood: $(round(fitted_exact.loglik.loglik, digits=3))")

# Compare to truth
println("\nParameter recovery (exact data):")
println("  h12 shape error: $(round(exp(exact_params[1]) - exp(true_params.h12[1]), digits=3))")
println("  h12 scale error: $(round(exp(exact_params[2]) - exp(true_params.h12[2]), digits=3))")
println("  h12 beta error: $(round(exact_params[3] - true_params.h12[3], digits=3))")
println("  h21 shape error: $(round(exp(exact_params[4]) - exp(true_params.h21[1]), digits=3))")
println("  h21 scale error: $(round(exp(exact_params[5]) - exp(true_params.h21[2]), digits=3))")
println("  h21 beta error: $(round(exact_params[6] - true_params.h21[3], digits=3))")

# =============================================================================
# Part 2: Fit PANEL data (requires MCEM)
# =============================================================================

println("\n" * "=" ^ 70)
println("PART 2: PANEL DATA FITTING (MCEM)")
println("=" ^ 70)

# Use the same simulated data but keep obstype=2 (panel)
panel_obs_data = copy(simulated_data_combined)
# obstype should already be 2 from simulation

println("\nFitting model to PANEL data (MCEM)...")

h12_panel = Hazard(@formula(0 ~ x), "wei", 1, 2)
h21_panel = Hazard(@formula(0 ~ x), "wei", 2, 1)
model_panel = multistatemodel(h12_panel, h21_panel; data=panel_obs_data, surrogate=:markov)

fitted_panel = fit(model_panel;
    verbose=true,
    maxiter=15,
    tol=0.05,
    ess_target_initial=30,
    max_ess=500,
    compute_vcov=false,
    return_convergence_records=true)

panel_params = get_parameters_flat(fitted_panel)
println("\nPanel data MCEM estimates:")
println("  h12: shape=$(round(exp(panel_params[1]), digits=3)), scale=$(round(exp(panel_params[2]), digits=3)), beta=$(round(panel_params[3], digits=3))")
println("  h21: shape=$(round(exp(panel_params[4]), digits=3)), scale=$(round(exp(panel_params[5]), digits=3)), beta=$(round(panel_params[6], digits=3))")
println("  Log-likelihood: $(round(fitted_panel.loglik.loglik, digits=3))")

# Compare to truth
println("\nParameter recovery (panel data):")
println("  h12 shape error: $(round(exp(panel_params[1]) - exp(true_params.h12[1]), digits=3))")
println("  h12 scale error: $(round(exp(panel_params[2]) - exp(true_params.h12[2]), digits=3))")
println("  h12 beta error: $(round(panel_params[3] - true_params.h12[3], digits=3))")
println("  h21 shape error: $(round(exp(panel_params[4]) - exp(true_params.h21[1]), digits=3))")
println("  h21 scale error: $(round(exp(panel_params[5]) - exp(true_params.h21[2]), digits=3))")
println("  h21 beta error: $(round(panel_params[6] - true_params.h21[3], digits=3))")

# =============================================================================
# Part 3: Compare exact vs panel estimates
# =============================================================================

println("\n" * "=" ^ 70)
println("COMPARISON: Exact vs Panel Data Estimates")
println("=" ^ 70)

println("\nParameter comparison:")
println("  Parameter     | True    | Exact   | Panel   | Exact-Panel")
println("  " * "-"^60)
println("  h12 shape     | $(round(exp(true_params.h12[1]), digits=3)) | $(round(exp(exact_params[1]), digits=3)) | $(round(exp(panel_params[1]), digits=3)) | $(round(exp(exact_params[1]) - exp(panel_params[1]), digits=3))")
println("  h12 scale     | $(round(exp(true_params.h12[2]), digits=3)) | $(round(exp(exact_params[2]), digits=3)) | $(round(exp(panel_params[2]), digits=3)) | $(round(exp(exact_params[2]) - exp(panel_params[2]), digits=3))")
println("  h12 beta      | $(round(true_params.h12[3], digits=3)) | $(round(exact_params[3], digits=3)) | $(round(panel_params[3], digits=3)) | $(round(exact_params[3] - panel_params[3], digits=3))")
println("  h21 shape     | $(round(exp(true_params.h21[1]), digits=3)) | $(round(exp(exact_params[4]), digits=3)) | $(round(exp(panel_params[4]), digits=3)) | $(round(exp(exact_params[4]) - exp(panel_params[4]), digits=3))")
println("  h21 scale     | $(round(exp(true_params.h21[2]), digits=3)) | $(round(exp(exact_params[5]), digits=3)) | $(round(exp(panel_params[5]), digits=3)) | $(round(exp(exact_params[5]) - exp(panel_params[5]), digits=3))")
println("  h21 beta      | $(round(true_params.h21[3], digits=3)) | $(round(exact_params[6], digits=3)) | $(round(panel_params[6], digits=3)) | $(round(exact_params[6] - panel_params[6], digits=3))")

# =============================================================================
# Part 4: Verify likelihood calculation manually
# =============================================================================

println("\n" * "=" ^ 70)
println("PART 4: MANUAL LIKELIHOOD VERIFICATION")
println("=" ^ 70)

# Pick a specific path and verify the likelihood calculation
test_path = simulated_paths[1]
println("\nTest path (subject 1):")
println("  Times: $(test_path.times)")
println("  States: $(test_path.states)")

# Compute likelihood at true parameters
subj_inds = model_exact.subjectindices[1]
subj_dat = view(model_exact.data, subj_inds, :)

println("\nSubject data for path:")
println(subj_dat)

# Make subjdat DataFrame
subjdat_df = make_subjdat(test_path, subj_dat)
println("\nConverted subjdat DataFrame:")
println(subjdat_df)

# Compute likelihood manually
true_pars_nested = (true_params.h12, true_params.h21)
ll_true = loglik(true_pars_nested, test_path, model_exact.hazards, model_exact)
println("\nLog-likelihood at TRUE parameters: $(round(ll_true, digits=4))")

# Compute at fitted parameters
exact_pars_nested = nest_params(exact_params, model_exact.parameters)
ll_exact = loglik(exact_pars_nested, test_path, model_exact.hazards, model_exact)
println("Log-likelihood at EXACT MLE: $(round(ll_exact, digits=4))")

panel_pars_nested = nest_params(panel_params, model_panel.parameters)
ll_panel = loglik(panel_pars_nested, test_path, model_panel.hazards, model_panel)
println("Log-likelihood at PANEL MLE: $(round(ll_panel, digits=4))")

# =============================================================================
# Part 5: Check sojourn time computation
# =============================================================================

println("\n" * "=" ^ 70)
println("PART 5: SOJOURN TIME VERIFICATION")
println("=" ^ 70)

println("\nVerifying sojourn times in subjdat_df:")
println("Expected: sojourn resets to 0 when state changes")
for i in 1:nrow(subjdat_df)
    row = subjdat_df[i, :]
    println("  Row $i: sojourn=$(round(row.sojourn, digits=3)), increment=$(round(row.increment, digits=3)), " *
            "state=$(row.statefrom)→$(row.stateto), x=$(row.x)")
    
    # Check sojourn reset
    if i > 1 && subjdat_df[i-1, :statefrom] != subjdat_df[i-1, :stateto]
        if row.sojourn != 0.0
            println("    ⚠️  WARNING: sojourn should be 0 after state change!")
        else
            println("    ✓ Sojourn correctly reset after state change")
        end
    end
end

println("\n" * "=" ^ 70)
println("DIAGNOSTIC COMPLETE")
println("=" ^ 70)
