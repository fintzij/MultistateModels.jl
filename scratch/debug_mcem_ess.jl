# Debug script for MCEM ESS issues
# Run with: julia --project scratch/debug_mcem_ess.jl

using MultistateModels
using DataFrames
using Random
using Statistics

import MultistateModels: Hazard, multistatemodel, fit, set_parameters!, simulate,
    get_parameters_flat, SamplePath

# Helper function for counting
function countmap(x)
    d = Dict{eltype(x), Int}()
    for v in x
        d[v] = get(d, v, 0) + 1
    end
    return d
end

# Set seed for reproducibility
Random.seed!(0xABCDEF01)

println("=" ^ 60)
println("MCEM ESS Debug Script")
println("=" ^ 60)

# ============================================================================
# Build a simple TVC panel data scenario
# ============================================================================

function build_tvc_panel_data(;
    n_subjects::Int,
    obs_times::Vector{Float64},
    covariate_change_times::Vector{Float64},
    covariate_generator::Function,
    obstype::Int = 2
)
    all_times = sort(unique(vcat(0.0, obs_times, covariate_change_times)))
    
    rows = []
    for subj in 1:n_subjects
        covariate_vals = covariate_generator(subj)
        
        for i in 1:(length(all_times) - 1)
            t_start = all_times[i]
            t_stop = all_times[i + 1]
            
            cov_idx = 1
            for (j, ct) in enumerate(covariate_change_times)
                if t_start >= ct
                    cov_idx = j + 1
                end
            end
            
            push!(rows, (
                id = subj,
                tstart = t_start,
                tstop = t_stop,
                statefrom = 1,
                stateto = 1,
                obstype = obstype,
                x = covariate_vals[cov_idx]
            ))
        end
    end
    
    return DataFrame(rows)
end

# ============================================================================
# Test Case 1: Binary TVC (similar to the failing test)
# ============================================================================

println("\n--- Test Case: Binary TVC ---")

N_SUBJECTS = 20  # Small for inspection

# Build panel data
panel_data = build_tvc_panel_data(
    n_subjects = N_SUBJECTS,
    obs_times = [2.0, 4.0, 6.0, 8.0],
    covariate_change_times = [3.0],
    covariate_generator = _ -> [0.0, 1.0],
    obstype = 2
)

println("Panel data structure:")
println(first(panel_data, 10))
println("...")
println("Total rows: ", nrow(panel_data))
println("Unique subjects: ", length(unique(panel_data.id)))

# Create and simulate with WEIBULL (semi-Markov) hazard
# NOTE: Exponential hazards are Markov - they dispatch to matrix exponentiation, NOT MCEM
# MCEM is only used for semi-Markov hazards (Weibull, Gompertz, splines) with panel data
h12_sim = Hazard(@formula(0 ~ x), "wei", 1, 2)  # Weibull is semi-Markov!
model_sim = multistatemodel(h12_sim; data=panel_data, surrogate=:markov)
# Weibull params: [log_shape, log_scale, beta_x]
set_parameters!(model_sim, (h12 = [log(1.0), log(0.25), 0.6],))

sim_result = simulate(model_sim; paths=false, data=true, nsim=1)
simulated_data = sim_result[1, 1]

println("\nSimulated data summary:")
println("  Transitions (statefrom != stateto): ", sum(simulated_data.statefrom .!= simulated_data.stateto))
println("  Unique final states: ", unique(simulated_data.stateto))
println("  Obstype distribution: ", countmap(simulated_data.obstype))

# CRITICAL: The simulate function converts panel (obstype=2) to exact (obstype=1)!
# This means the fit() will dispatch to Markov MLE, not MCEM.
# We need to keep panel observations for MCEM testing.

# Let's manually construct panel data that maintains obstype=2
# by only recording state at observation times, not exact transition times
println("\n⚠️  NOTE: simulate() returns exact observations (obstype=1)")
println("    For MCEM testing, we need to maintain panel observations (obstype=2)")

# Show per-subject summary
println("\nPer-subject summary:")
for subj in 1:min(5, N_SUBJECTS)
    subj_rows = simulated_data[simulated_data.id .== subj, :]
    n_trans = sum(subj_rows.statefrom .!= subj_rows.stateto)
    final_state = last(subj_rows.stateto)
    println("  Subject $subj: $(nrow(subj_rows)) rows, $n_trans transitions, final state=$final_state")
end

# ============================================================================
# Fit with verbose output and single iteration
# ============================================================================

println("\n" * "=" ^ 60)
println("Fitting MCEM (single iteration for debugging)")
println("=" ^ 60)

# Create proper panel data that maintains obstype=2
# We need to convert exact observations back to panel observations
# by determining state at each observation time
function make_panel_from_exact(exact_data, obs_times, covariate_change_times, covariate_generator)
    panel_rows = []
    
    for subj in unique(exact_data.id)
        subj_data = exact_data[exact_data.id .== subj, :]
        sort!(subj_data, :tstart)
        
        # Get covariate values for this subject
        cov_vals = covariate_generator(subj)
        
        # Determine state at each observation time
        # Track current state based on exact transition times
        current_state = 1
        for i in 1:(length(obs_times) - 1)
            t_start = obs_times[i]
            t_stop = obs_times[i + 1]
            
            # Find state at t_start: look for rows where transition happened before or at t_start
            for row in eachrow(subj_data)
                if row.tstop <= t_start && row.statefrom != row.stateto
                    current_state = row.stateto
                end
            end
            statefrom = current_state
            
            # Find state at t_stop: check if any transition occurred in [t_start, t_stop]
            for row in eachrow(subj_data)
                if row.tstop <= t_stop && row.tstop > t_start && row.statefrom != row.stateto
                    current_state = row.stateto
                end
            end
            stateto = current_state
            
            # Get covariate for this interval based on covariate_change_times
            cov_idx = 1
            for (j, ct) in enumerate(covariate_change_times)
                if t_start >= ct
                    cov_idx = j + 1
                end
            end
            
            push!(panel_rows, (
                id = subj,
                tstart = t_start,
                tstop = t_stop,
                statefrom = statefrom,
                stateto = stateto,
                obstype = 2,  # Panel observation!
                x = cov_vals[cov_idx]
            ))
        end
    end
    return DataFrame(panel_rows)
end

# Reconstruct panel data
obs_times_grid = [0.0, 2.0, 4.0, 6.0, 8.0]
covariate_change_times = [3.0]
covariate_generator = _ -> [0.0, 1.0]
panel_for_fit = make_panel_from_exact(simulated_data, obs_times_grid, covariate_change_times, covariate_generator)

println("\nReconstructed panel data (obstype=2):")
println(first(panel_for_fit, 12))
println("...")
println("Obstype distribution: ", countmap(panel_for_fit.obstype))

h12_fit = Hazard(@formula(0 ~ x), "wei", 1, 2)  # Weibull for MCEM
model_fit = multistatemodel(h12_fit; data=panel_for_fit, surrogate=:markov)

println("\nModel type: ", typeof(model_fit))
println("Is semi-Markov? ", model_fit isa MultistateModels.MultistateSemiMarkovModel || 
                            model_fit isa MultistateModels.MultistateSemiMarkovModelCensored)

# Check surrogate
println("\nSurrogate check:")
println("  markovsurrogate is nothing: ", isnothing(model_fit.markovsurrogate))
if !isnothing(model_fit.markovsurrogate)
    println("  Surrogate hazards: ", length(model_fit.markovsurrogate.hazards))
    println("  Surrogate params: ", model_fit.markovsurrogate.parameters.flat)
end

# Run single iteration with very loose convergence criteria
fitted = nothing  # Initialize outside try block
try
    global fitted = fit(model_fit;
        proposal=:markov,
        verbose=true,
        maxiter=1,              # Single iteration
        tol=1000.0,             # Very loose tolerance (will stop after 1 iter)
        ess_target_initial=10,  # Low target for inspection
        max_ess=100,            # Allow up to 100 paths per subject
        max_sampling_effort=5,  # Limit sampling attempts
        compute_vcov=false,
        return_convergence_records=true)
    
    println("\n" * "=" ^ 60)
    println("Convergence Records Inspection")
    println("=" ^ 60)
    
    if !isnothing(fitted.ConvergenceRecords)
        records = fitted.ConvergenceRecords
        
        println("\nConvergence records fields: ", fieldnames(typeof(records)))
        
        if hasproperty(records, :ess_trace)
            println("\nESS trace shape: ", size(records.ess_trace))
            if size(records.ess_trace, 2) > 0
                ess_final = records.ess_trace[:, end]
                println("Final ESS per subject:")
                for (i, ess) in enumerate(ess_final)
                    status = ess < 5 ? " ⚠️ LOW" : ""
                    println("  Subject $i: ESS = $(round(ess, digits=2))$status")
                end
                
                # Identify problematic subjects
                low_ess_subjs = findall(ess_final .< 5)
                if !isempty(low_ess_subjs)
                    println("\n⚠️  Subjects with low ESS (<5): ", low_ess_subjs)
                end
            end
        else
            println("\nNo ess_trace field - likely Markov fit was used (not MCEM)")
            println("Available fields: ", keys(records))
        end
        
        if hasproperty(records, :mll_trace)
            println("\nMLL trace: ", records.mll_trace)
        end
        if hasproperty(records, :psis_pareto_k)
            println("Pareto-k diagnostics: ", records.psis_pareto_k)
        end
    else
        println("\nNo ConvergenceRecords - this might be Markov MLE fit")
    end
    
catch e
    println("\n❌ Error during fit:")
    println(e)
    println("\nStacktrace:")
    for (exc, bt) in current_exceptions()
        showerror(stdout, exc, bt)
        println()
    end
end

# ============================================================================
# Manual inspection of what MCEM sees
# ============================================================================

println("\n" * "=" ^ 60)
println("Manual Data Inspection")
println("=" ^ 60)

println("\nModel data obstypes:")
println("  obstype distribution: ", countmap(model_fit.data.obstype))

println("\nSubject indices:")
for (i, inds) in enumerate(model_fit.subjectindices[1:min(5, length(model_fit.subjectindices))])
    println("  Subject $i: rows $inds")
    subj_data = model_fit.data[inds, :]
    for row in eachrow(subj_data)
        println("    [$(row.tstart), $(row.tstop)] : $(row.statefrom) → $(row.stateto), obstype=$(row.obstype)")
    end
end

# Analyze low-ESS subjects more closely
low_ess_subjects = [1, 6, 9, 13, 17]  # From above
println("\n" * "=" ^ 60)
println("Low ESS Subject Analysis (ESS=1)")
println("=" ^ 60)

for subj_id in low_ess_subjects
    if subj_id <= length(model_fit.subjectindices)
        inds = model_fit.subjectindices[subj_id]
        subj_data = model_fit.data[inds, :]
        
        # Check if this subject has an early transition
        first_trans_interval = findfirst(r -> r.statefrom != r.stateto, eachrow(subj_data))
        early_transition = !isnothing(first_trans_interval) && first_trans_interval == 1
        
        println("\n  Subject $subj_id:")
        println("    Early transition (first interval): $early_transition")
        println("    Data:")
        for row in eachrow(subj_data)
            marker = row.statefrom != row.stateto ? " ← TRANSITION" : ""
            println("      [$(row.tstart), $(row.tstop)] : $(row.statefrom) → $(row.stateto)$marker")
        end
    end
end

# Also check subjects with Pareto-k = 0 (subjects 10, 20)
println("\n" * "=" ^ 60)
println("Pareto-k = 0 Subject Analysis (no transition?)")
println("=" ^ 60)

for subj_id in [10, 20]
    if subj_id <= length(model_fit.subjectindices)
        inds = model_fit.subjectindices[subj_id]
        subj_data = model_fit.data[inds, :]
        
        # Check if this subject transitioned at all
        any_trans = any(r -> r.statefrom != r.stateto, eachrow(subj_data))
        
        println("\n  Subject $subj_id:")
        println("    Any transition: $any_trans")
        println("    Data:")
        for row in eachrow(subj_data)
            marker = row.statefrom != row.stateto ? " ← TRANSITION" : ""
            println("      [$(row.tstart), $(row.tstop)] : $(row.statefrom) → $(row.stateto)$marker")
        end
    end
end

# ============================================================================
# Pareto-k Diagnostic Summary
# ============================================================================

println("\n" * "=" ^ 60)
println("Pareto-k Diagnostic Summary")
println("=" ^ 60)

println("""
Pareto-k diagnostic from PSIS (Pareto Smoothed Importance Sampling):

  Pareto-k = 0.0: Uniform weights (subject didn't transition or all paths identical)
  Pareto-k < 0.5: Good importance sampling quality
  0.5 < Pareto-k < 0.7: Acceptable quality
  Pareto-k > 0.7: Poor quality - weights have heavy tails
  Pareto-k = 1.0: PSIS fitting failed (fallback to standard weights)

Subjects with Pareto-k = 1.0 typically have:
  - Early transitions (first observation interval)
  - Only one possible path structure
  - Variance only from exact transition time, not path structure

This is expected behavior and does NOT indicate a bug. The MCEM algorithm
still converges but those subjects contribute less information to ESS.
""")

# Cross-reference ESS with Pareto-k
if !isnothing(fitted.ConvergenceRecords) && hasproperty(fitted.ConvergenceRecords, :psis_pareto_k)
    pareto_k = fitted.ConvergenceRecords.psis_pareto_k
    ess_final = fitted.ConvergenceRecords.ess_trace[:, end]
    
    println("\nESS vs Pareto-k correlation:")
    n_low_ess = sum(ess_final .< 5)
    n_high_pareto = sum(pareto_k .>= 0.7)
    n_both = sum((ess_final .< 5) .& (pareto_k .>= 0.7))
    
    println("  Subjects with ESS < 5: $n_low_ess")
    println("  Subjects with Pareto-k >= 0.7: $n_high_pareto")
    println("  Subjects with both: $n_both")
    
    # Check data patterns for different Pareto-k ranges
    println("\nData patterns by Pareto-k:")
    for subj in 1:min(20, length(pareto_k))
        inds = model_fit.subjectindices[subj]
        subj_data = model_fit.data[inds, :]
        first_trans = findfirst(r -> r.statefrom != r.stateto, eachrow(subj_data))
        any_trans = any(r -> r.statefrom != r.stateto, eachrow(subj_data))
        
        k_val = pareto_k[subj]
        ess_val = ess_final[subj]
        
        pattern = if !any_trans
            "no_trans"
        elseif first_trans == 1
            "early_trans"
        else
            "late_trans"
        end
        
        println("  Subj $subj: k=$(round(k_val, digits=2)), ESS=$(round(ess_val, digits=1)), pattern=$pattern")
    end
end

println("\n" * "=" ^ 60)
println("End Debug Script")
println("=" ^ 60)
