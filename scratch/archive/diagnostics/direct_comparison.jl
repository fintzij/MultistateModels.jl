# Direct comparison: exact vs panel data spline fitting
# This is a minimal reproduction of the bug

using MultistateModels
using DataFrames
using Random
using BSplineKit

import MultistateModels: Hazard, @formula, multistatemodel, fit, set_parameters!, 
    simulate, get_parameters_flat, get_parameters, set_surrogate!

Random.seed!(12345)

println("=" ^ 60)
println("MINIMAL REPRODUCTION: Spline MCEM Panel vs Exact")
println("=" ^ 60)
println()

# Configuration
N_SUBJECTS = 500
MAX_TIME = 15.0

# Spline setup (matches longtest_spline_suite.jl)
DEGREE = 3
INTERIOR_KNOTS = [5.0, 10.0]  # 2 interior knots
BOUNDARY_KNOTS = [0.0, MAX_TIME]

# True coefficients (NATURAL SCALE, must be positive)
TRUE_COEFS_H12 = [0.08, 0.10, 0.14, 0.18]  # Increasing hazard

println("Configuration:")
println("  N_SUBJECTS = $N_SUBJECTS")
println("  MAX_TIME = $MAX_TIME")
println("  DEGREE = $DEGREE")
println("  INTERIOR_KNOTS = $INTERIOR_KNOTS")
println("  BOUNDARY_KNOTS = $BOUNDARY_KNOTS")
println("  TRUE_COEFS_H12 = $TRUE_COEFS_H12")
println()

# Create spline hazard specification
h12 = Hazard(@formula(0 ~ 1), "sp", 1, 2;
    degree=DEGREE,
    knots=INTERIOR_KNOTS,
    boundaryknots=BOUNDARY_KNOTS,
    extrapolation="constant")

# Create panel data template (simple 2-state model: 1 → 2)
panel_times = [0.0, 2.5, 5.0, 7.5, 10.0, 12.5, 15.0]
nobs = length(panel_times) - 1

panel_template = DataFrame(
    id = repeat(1:N_SUBJECTS, inner=nobs),
    tstart = repeat(panel_times[1:end-1], N_SUBJECTS),
    tstop = repeat(panel_times[2:end], N_SUBJECTS),
    statefrom = ones(Int, N_SUBJECTS * nobs),
    stateto = ones(Int, N_SUBJECTS * nobs),
    obstype = fill(2, N_SUBJECTS * nobs)
)

println("Step 1: Creating simulation model...")
model_sim = multistatemodel(h12; data=panel_template, initialize=false)
set_parameters!(model_sim, (h12 = TRUE_COEFS_H12,))

println("Step 2: Simulating data with exact observations...")
# Simulate with exact observation type
sim_result = simulate(model_sim; paths=true, data=true, nsim=1, autotmax=false)
exact_data = sim_result[1, 1]

# Count transitions
n_trans = sum(exact_data.obstype .== 1)
n_no_trans = sum((exact_data.statefrom .== exact_data.stateto) .& (exact_data.obstype .== 1))
println("  Simulated $(nrow(exact_data)) rows, $n_trans exact observations")
println()

# Create panel version of same data (discretize to panel times)
println("Step 3: Creating panel version of data...")
# For each subject, find state at each panel time
panel_data = DataFrame(
    id = Int[],
    tstart = Float64[],
    tstop = Float64[],
    statefrom = Int[],
    stateto = Int[],
    obstype = Int[]
)

for subj in 1:N_SUBJECTS
    subj_exact = filter(row -> row.id == subj, exact_data)
    
    for i in 1:(length(panel_times)-1)
        t0, t1 = panel_times[i], panel_times[i+1]
        
        # Find state at t0 and t1
        state_t0 = 1
        state_t1 = 1
        
        for row in eachrow(subj_exact)
            if row.tstart <= t0 < row.tstop
                state_t0 = row.statefrom
            end
            if row.tstart < t1 <= row.tstop
                state_t1 = row.statefrom
                if row.obstype == 1 && row.tstop == t1
                    state_t1 = row.stateto
                end
            end
        end
        
        # Use final state if we've passed the transition
        final_row = subj_exact[end, :]
        if final_row.obstype == 1 && final_row.tstop <= t1
            state_t1 = final_row.stateto
        end
        if final_row.obstype == 1 && final_row.tstop <= t0
            state_t0 = final_row.stateto
        end
        
        push!(panel_data, (id=subj, tstart=t0, tstop=t1, 
                          statefrom=state_t0, stateto=state_t1, obstype=2))
    end
end

println("  Created panel data with $(nrow(panel_data)) rows")

# Count state distribution at final time
final_states = [filter(row -> row.id == i && row.tstop == MAX_TIME, panel_data)[1, :stateto] for i in 1:N_SUBJECTS]
n_state1 = sum(final_states .== 1)
n_state2 = sum(final_states .== 2)
println("  Final state distribution: state 1 = $n_state1, state 2 = $n_state2")
println()

# ============================================================
# FIT 1: Exact data (should give correct estimates)
# ============================================================
println("Step 4: Fitting to EXACT data (MLE)...")
h12_exact = Hazard(@formula(0 ~ 1), "sp", 1, 2;
    degree=DEGREE,
    knots=INTERIOR_KNOTS,
    boundaryknots=BOUNDARY_KNOTS,
    extrapolation="constant")

model_exact = multistatemodel(h12_exact; data=exact_data, initialize=false)

# Initial guess: slightly perturbed from true
initial_guess = TRUE_COEFS_H12 .* 1.2
set_parameters!(model_exact, (h12 = initial_guess,))

println("  Initial parameters: $initial_guess")

# Fit via MLE (exact data)
try
    fitted_exact = fit(model_exact;
        verbose=true,
        compute_vcov=true)
    
    pars_exact = get_parameters(fitted_exact; scale=:natural)
    println("  Fitted parameters (exact): $(round.(pars_exact.h12, digits=4))")
catch e
    println("  ERROR fitting exact data: $e")
    pars_exact = nothing
end
println()

# ============================================================
# FIT 2: Panel data via MCEM (this is where the bug occurs)
# ============================================================
println("Step 5: Fitting to PANEL data (MCEM)...")
h12_panel = Hazard(@formula(0 ~ 1), "sp", 1, 2;
    degree=DEGREE,
    knots=INTERIOR_KNOTS,
    boundaryknots=BOUNDARY_KNOTS,
    extrapolation="constant")

model_panel = multistatemodel(h12_panel; data=panel_data, surrogate=:markov)

# Initial guess
set_parameters!(model_panel, (h12 = initial_guess,))
println("  Initial parameters: $initial_guess")

# Fit via MCEM
println("  Running MCEM (this may take a few minutes)...")
try
    fitted_panel = fit(model_panel;
        proposal=:markov,
        verbose=true,
        maxiter=15,
        tol=0.05,
        ess_target_initial=25,
        max_ess=200,
        compute_vcov=false)
    
    pars_panel = get_parameters(fitted_panel; scale=:natural)
    println("  Fitted parameters (panel): $(round.(pars_panel.h12, digits=4))")
catch e
    println("  ERROR fitting panel data: $e")
    pars_panel = nothing
end
println()

# ============================================================
# COMPARISON
# ============================================================
println("=" ^ 60)
println("COMPARISON")
println("=" ^ 60)
println()

println("| Param    | True  | Exact Est | Panel Est | Panel Rel Err |")
println("|----------|-------|-----------|-----------|---------------|")
for i in 1:4
    true_val = TRUE_COEFS_H12[i]
    exact_val = isnothing(pars_exact) ? NaN : pars_exact.h12[i]
    panel_val = isnothing(pars_panel) ? NaN : pars_panel.h12[i]
    panel_err = isnothing(pars_panel) ? NaN : abs(panel_val - true_val) / true_val * 100
    
    println("| coef_$i   | $(round(true_val, digits=2))  | $(round(exact_val, digits=3))     | $(round(panel_val, digits=3))     | $(round(panel_err, digits=0))%           |")
end
