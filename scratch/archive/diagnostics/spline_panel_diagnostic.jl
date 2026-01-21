# =============================================================================
# Diagnostic: Spline Panel Data Generation vs Fitting
# =============================================================================
# 
# Goal: Identify exactly where the discrepancy enters in panel spline tests.
# Focus on nocov case (simplest) to isolate the issue.
#
# Key things to check:
# 1. DGP model knots vs fitting model knots
# 2. True parameters vs auto-initialized parameters  
# 3. Data generation process
# 4. Model classification (is_markov, is_panel_data)
# =============================================================================

using MultistateModels
using DataFrames
using Random
using Statistics
using Printf

import MultistateModels: Hazard, @formula, multistatemodel, fit, set_parameters!, 
    simulate, get_parameters_flat, get_parameters, is_markov, is_panel_data

# =============================================================================
# Configuration (from longtest_spline_suite.jl)
# =============================================================================
const N_SUBJECTS = 1000
const SPLINE_DEGREE = 2
const N_INTERIOR_KNOTS = 1
const SPLINE_MAX_TIME = 5.0
const SPLINE_BOUNDARYKNOTS = [0.0, SPLINE_MAX_TIME]
const SPLINE_PANEL_TIMES = [0.0, 1.5, 3.0, 4.5]  # Original: 3 intervals

# True coefficients (natural scale)
const TRUE_SPLINE_COEFS_H12 = [0.15, 0.25]
const TRUE_SPLINE_COEFS_H23 = [0.10, 0.20]

println("="^70)
println("SPLINE PANEL DIAGNOSTIC")
println("="^70)

# =============================================================================
# Step 1: Create the spline specification
# =============================================================================
println("\n[STEP 1] Spline Specification")
println("-"^50)

function get_spline_knots(n_interior::Int, boundary_knots::Vector{Float64})
    t_min, t_max = boundary_knots
    return collect(range(t_min + (t_max - t_min) / (n_interior + 1), 
                         t_max - (t_max - t_min) / (n_interior + 1), 
                         length=n_interior))
end

knots = get_spline_knots(N_INTERIOR_KNOTS, SPLINE_BOUNDARYKNOTS)
println("  Degree: $SPLINE_DEGREE")
println("  Interior knots: $knots")
println("  Boundary knots: $SPLINE_BOUNDARYKNOTS")
println("  Expected n_coeffs: $(SPLINE_DEGREE + N_INTERIOR_KNOTS + 1 - 2)") # RecombinedBSplineBasis

# =============================================================================
# Step 2: Create DGP model and examine
# =============================================================================
println("\n[STEP 2] DGP Model Construction")
println("-"^50)

Random.seed!(12345)

# Create panel data template (nocov case)
nobs = length(SPLINE_PANEL_TIMES) - 1
panel_template = DataFrame(
    id = repeat(1:N_SUBJECTS, inner=nobs),
    tstart = repeat(SPLINE_PANEL_TIMES[1:end-1], N_SUBJECTS),
    tstop = repeat(SPLINE_PANEL_TIMES[2:end], N_SUBJECTS),
    statefrom = ones(Int, N_SUBJECTS * nobs),
    stateto = ones(Int, N_SUBJECTS * nobs),
    obstype = fill(2, N_SUBJECTS * nobs)
)

println("  Template rows: $(nrow(panel_template))")
println("  Unique subjects: $(length(unique(panel_template.id)))")
println("  Panel times: $SPLINE_PANEL_TIMES")
println("  Intervals per subject: $nobs")

# Create DGP hazards
h12_dgp = Hazard(@formula(0 ~ 1), "sp", 1, 2;
    degree=SPLINE_DEGREE,
    knots=knots,
    boundaryknots=SPLINE_BOUNDARYKNOTS,
    extrapolation="constant")

h23_dgp = Hazard(@formula(0 ~ 1), "sp", 2, 3;
    degree=SPLINE_DEGREE,
    knots=knots,
    boundaryknots=SPLINE_BOUNDARYKNOTS,
    extrapolation="constant")

# Build DGP model
model_dgp = multistatemodel(h12_dgp, h23_dgp; data=panel_template, initialize=false)

# Set true parameters
true_params = (h12 = TRUE_SPLINE_COEFS_H12, h23 = TRUE_SPLINE_COEFS_H23)
set_parameters!(model_dgp, true_params)

println("\n  DGP Model:")
println("    is_markov: $(is_markov(model_dgp))")
println("    is_panel_data: $(is_panel_data(model_dgp))")
println("    Number of hazards: $(length(model_dgp.hazards))")
println("    h12 type: $(typeof(model_dgp.hazards[1]))")
println("    h23 type: $(typeof(model_dgp.hazards[2]))")

# Print knots from internal hazard structs
println("\n  Internal knot configuration:")
for (i, haz) in enumerate(model_dgp.hazards)
    hazname = haz.hazname
    println("    $hazname:")
    if hasproperty(haz, :meta) && !isnothing(haz.meta)
        m = haz.meta
        println("      degree: $(hasproperty(m, :degree) ? m.degree : "N/A")")
        println("      interior_knots: $(hasproperty(m, :interior_knots) ? m.interior_knots : "N/A")")
        println("      boundary_knots: $(hasproperty(m, :boundary_knots) ? m.boundary_knots : "N/A")")
    else
        println("      (no meta)")
    end
end

println("\n  True parameters (set):")
println("    h12: $(TRUE_SPLINE_COEFS_H12)")
println("    h23: $(TRUE_SPLINE_COEFS_H23)")

# Verify parameters are set correctly
pars_dgp_h12 = get_parameters(model_dgp, 1, scale=:natural)
pars_dgp_h23 = get_parameters(model_dgp, 2, scale=:natural)
println("\n  Parameters from model (get_parameters, natural):")
println("    h12: $pars_dgp_h12")
println("    h23: $pars_dgp_h23")

# =============================================================================
# Step 3: Evaluate true hazard at test times
# =============================================================================
println("\n[STEP 3] True Hazard Values")
println("-"^50)

eval_times = [0.5, 1.5, 2.5, 3.5, 4.5]
println("  Evaluation times: $eval_times")

println("\n  True h12(t):")
h12_haz = model_dgp.hazards[1]
for t in eval_times
    h_val = h12_haz(t, pars_dgp_h12, NamedTuple())
    println("    h($t) = $(round(h_val, digits=4))")
end

println("\n  True h23(t):")
h23_haz = model_dgp.hazards[2]
for t in eval_times
    h_val = h23_haz(t, pars_dgp_h23, NamedTuple())
    println("    h($t) = $(round(h_val, digits=4))")
end

# =============================================================================
# Step 4: Simulate panel data
# =============================================================================
println("\n[STEP 4] Panel Data Simulation")
println("-"^50)

# The issue may be here: obstype_map Dict(1 => 2, 2 => 1) makes h12 panel, h23 exact
# For a 3-state model 1 → 2 → 3:
#   Transition 1: 1 → 2 (h12)
#   Transition 2: 2 → 3 (h23)
obstype_map = Dict(1 => 2, 2 => 1)
println("  obstype_map: $obstype_map")
println("    Transition 1 (1→2): obstype=$(obstype_map[1]) (panel)")
println("    Transition 2 (2→3): obstype=$(obstype_map[2]) (exact)")

sim_result = simulate(model_dgp; paths=false, data=true, nsim=1, autotmax=false,
                     obstype_by_transition=obstype_map)
panel_data = sim_result[1, 1]

println("\n  Simulated data summary:")
println("    Rows: $(nrow(panel_data))")
println("    Unique subjects: $(length(unique(panel_data.id)))")

# Count transitions
n_12_trans = sum((panel_data.statefrom .== 1) .& (panel_data.stateto .== 2))
n_23_trans = sum((panel_data.statefrom .== 2) .& (panel_data.stateto .== 3))
n_censor = sum(panel_data.statefrom .== panel_data.stateto)
println("    1→2 transitions: $n_12_trans")
println("    2→3 transitions: $n_23_trans")
println("    Censored intervals: $n_censor")

# Check obstype distribution
println("\n  Obstype distribution in simulated data:")
for ot in sort(unique(panel_data.obstype))
    cnt = sum(panel_data.obstype .== ot)
    println("    obstype=$ot: $cnt rows")
end

# Show first few rows
println("\n  First 20 rows of simulated data:")
show(first(panel_data, 20))
println()

# =============================================================================
# Step 5: Create fitting model (same spec) and check initialization
# =============================================================================
println("\n[STEP 5] Fitting Model Construction")
println("-"^50)

# Create fitting hazards with IDENTICAL specification
h12_fit = Hazard(@formula(0 ~ 1), "sp", 1, 2;
    degree=SPLINE_DEGREE,
    knots=knots,
    boundaryknots=SPLINE_BOUNDARYKNOTS,
    extrapolation="constant")

h23_fit = Hazard(@formula(0 ~ 1), "sp", 2, 3;
    degree=SPLINE_DEGREE,
    knots=knots,
    boundaryknots=SPLINE_BOUNDARYKNOTS,
    extrapolation="constant")

# Build fitting model (with automatic initialization)
model_fit = multistatemodel(h12_fit, h23_fit; data=panel_data, surrogate=:markov)

println("  Fitting Model:")
println("    is_markov: $(is_markov(model_fit))")
println("    is_panel_data: $(is_panel_data(model_fit))")
println("    Number of hazards: $(length(model_fit.hazards))")
println("    h12 type: $(typeof(model_fit.hazards[1]))")
println("    h23 type: $(typeof(model_fit.hazards[2]))")

# Get auto-initialized parameters
init_pars_h12 = get_parameters(model_fit, 1, scale=:natural)
init_pars_h23 = get_parameters(model_fit, 2, scale=:natural)
println("\n  Auto-initialized parameters:")
println("    h12: $init_pars_h12")
println("    h23: $init_pars_h23")

println("\n  True vs Initial comparison:")
println("    h12 true:  $(TRUE_SPLINE_COEFS_H12)")
println("    h12 init:  $init_pars_h12")
println("    h12 ratio: $(init_pars_h12 ./ TRUE_SPLINE_COEFS_H12)")
println()
println("    h23 true:  $(TRUE_SPLINE_COEFS_H23)")
println("    h23 init:  $init_pars_h23")
println("    h23 ratio: $(init_pars_h23 ./ TRUE_SPLINE_COEFS_H23)")

# =============================================================================
# Step 6: Compare knots between DGP and fitting models
# =============================================================================
println("\n[STEP 6] Knot Comparison (DGP vs Fitting)")
println("-"^50)

function extract_knots_from_hazard(haz)
    if hasproperty(haz, :meta) && !isnothing(haz.meta)
        m = haz.meta
        return (
            degree = hasproperty(m, :degree) ? m.degree : nothing,
            interior = hasproperty(m, :interior_knots) ? m.interior_knots : nothing,
            boundary = hasproperty(m, :boundary_knots) ? m.boundary_knots : nothing
        )
    end
    return nothing
end

println("  h12 comparison:")
dgp_k12 = extract_knots_from_hazard(model_dgp.hazards[1])
fit_k12 = extract_knots_from_hazard(model_fit.hazards[1])
println("    DGP: $dgp_k12")
println("    Fit: $fit_k12")
println("    Match: $(dgp_k12 == fit_k12 ? "YES ✓" : "NO ✗")")

println("\n  h23 comparison:")
dgp_k23 = extract_knots_from_hazard(model_dgp.hazards[2])
fit_k23 = extract_knots_from_hazard(model_fit.hazards[2])
println("    DGP: $dgp_k23")
println("    Fit: $fit_k23")
println("    Match: $(dgp_k23 == fit_k23 ? "YES ✓" : "NO ✗")")

# =============================================================================
# Step 7: Fit model and compare results
# =============================================================================
println("\n[STEP 7] Model Fitting")
println("-"^50)

# Fit with MCEM
println("  Starting MCEM fit...")
fitted = fit(model_fit;
    verbose=true,
    compute_vcov=false,
    method=:MCEM,
    penalty=:none,
    tol=0.05,
    ess_target_initial=30,
    max_ess=500,
    maxiter=25
)

# Get fitted parameters
fitted_pars_h12 = get_parameters(fitted, 1, scale=:natural)
fitted_pars_h23 = get_parameters(fitted, 2, scale=:natural)

println("\n  Fitted parameters:")
println("    h12: $fitted_pars_h12")
println("    h23: $fitted_pars_h23")

# =============================================================================
# Step 8: Compare hazard functions
# =============================================================================
println("\n[STEP 8] Hazard Function Comparison")
println("-"^50)

println("\n  h12(t) comparison:")
println("  Time      True        Fitted      Rel Diff")
println("  " * "-"^50)
for t in eval_times
    h_true = h12_haz(t, pars_dgp_h12, NamedTuple())
    h_fit = fitted.hazards[1](t, fitted_pars_h12, NamedTuple())
    rel_diff = abs(h_fit - h_true) / h_true * 100
    status = rel_diff <= 50 ? "✓" : "✗"
    @printf("  %-9.1f %-11.4f %-11.4f %6.1f%% %s\n", t, h_true, h_fit, rel_diff, status)
end

println("\n  h23(t) comparison:")
println("  Time      True        Fitted      Rel Diff")
println("  " * "-"^50)
for t in eval_times
    h_true = h23_haz(t, pars_dgp_h23, NamedTuple())
    h_fit = fitted.hazards[2](t, fitted_pars_h23, NamedTuple())
    rel_diff = abs(h_fit - h_true) / h_true * 100
    status = rel_diff <= 50 ? "✓" : "✗"
    @printf("  %-9.1f %-11.4f %-11.4f %6.1f%% %s\n", t, h_true, h_fit, rel_diff, status)
end

println("\n" * "="^70)
println("DIAGNOSTIC COMPLETE")
println("="^70)
