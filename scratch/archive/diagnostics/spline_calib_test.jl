# Test calibration approach for spline panel tests
# DIAGNOSTIC: Check spline configuration matches between DGP and fitting model

using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using MultistateModels
using DataFrames
using Random
import MultistateModels: Hazard, @formula, multistatemodel, fit, set_parameters!, 
    simulate, get_parameters_flat, get_parameters

println("="^70)
println("SPLINE CONFIGURATION DIAGNOSTIC")
println("="^70)

const N_SUBJECTS = 2000
const MAX_TIME = 5.0
const DEGREE = 2
const knots = [2.5]
const boundaryknots = [0.0, MAX_TIME]
const PANEL_TIMES = collect(0.0:0.5:5.0)
const eval_times = [0.5, 1.5, 2.5, 3.5, 4.5]

Random.seed!(12345)

# ============================================================================
# Step 1: Generate exact data from Weibull (2-state: 1→2)
# ============================================================================
println("\nSTEP 1: Generate exact data from Weibull (2-state model)")

template = DataFrame(
    id = 1:N_SUBJECTS,
    tstart = zeros(N_SUBJECTS),
    tstop = fill(MAX_TIME, N_SUBJECTS),
    statefrom = ones(Int, N_SUBJECTS),
    stateto = fill(2, N_SUBJECTS),
    obstype = ones(Int, N_SUBJECTS)
)

wei_h12 = Hazard(@formula(0 ~ 1), "wei", 1, 2)
wei_model = multistatemodel(wei_h12; data=template)
set_parameters!(wei_model, (h12 = [1.3, 0.15],))

exact_data = simulate(wei_model; data=true, paths=false, nsim=1)[1]
println("  Generated $(nrow(exact_data)) rows from Weibull DGP")

# ============================================================================
# Step 2: Fit spline to exact data → Calibrated coefficients
# ============================================================================
println("\nSTEP 2: Fit spline to exact Weibull data")

sp_h12 = Hazard(@formula(0 ~ 1), "sp", 1, 2;
    degree=DEGREE, knots=knots, boundaryknots=boundaryknots, extrapolation="constant")

calib_model = multistatemodel(sp_h12; data=exact_data)

# DIAGNOSTIC: Print spline configuration from calibration model
println("\n  === CALIBRATION MODEL SPLINE CONFIG ===")
calib_haz = calib_model.hazards[1]
println("  Type: $(typeof(calib_haz))")
println("  Degree: $(calib_haz.degree)")
println("  Knots (from hazard): $(calib_haz.knots)")
println("  Extrapolation: $(calib_haz.extrapolation)")
println("  Monotone constraint: $(calib_haz.monotone)")
println("  Natural spline: $(calib_haz.natural_spline)")
# Check all available fields
println("  All fields: $(fieldnames(typeof(calib_haz)))")

calib_fit = fit(calib_model; verbose=false, compute_vcov=false, penalty=:none)

calib_h12 = get_parameters(calib_fit, 1, scale=:natural)
println("\n  Calibrated h12 coefficients: $(round.(calib_h12, digits=4))")
println("  Number of coefficients: $(length(calib_h12))")

calib_params = (h12 = calib_h12,)

# ============================================================================
# Step 3: Generate panel data from calibrated spline DGP
# ============================================================================
println("\nSTEP 3: Generate panel data from calibrated spline DGP")

nobs = length(PANEL_TIMES) - 1
panel_template = DataFrame(
    id = repeat(1:N_SUBJECTS, inner=nobs),
    tstart = repeat(PANEL_TIMES[1:end-1], N_SUBJECTS),
    tstop = repeat(PANEL_TIMES[2:end], N_SUBJECTS),
    statefrom = ones(Int, N_SUBJECTS * nobs),
    stateto = ones(Int, N_SUBJECTS * nobs),
    obstype = fill(2, N_SUBJECTS * nobs)
)

dgp_h12 = Hazard(@formula(0 ~ 1), "sp", 1, 2;
    degree=DEGREE, knots=knots, boundaryknots=boundaryknots, extrapolation="constant")

dgp_model = multistatemodel(dgp_h12; data=panel_template, initialize=false)

# DIAGNOSTIC: Print spline configuration from DGP model  
println("\n  === DGP MODEL SPLINE CONFIG ===")
dgp_haz = dgp_model.hazards[1]
println("  Type: $(typeof(dgp_haz))")
println("  Degree: $(dgp_haz.degree)")
println("  Knots (from hazard): $(dgp_haz.knots)")
println("  Extrapolation: $(dgp_haz.extrapolation)")
println("  Monotone constraint: $(dgp_haz.monotone)")
println("  Natural spline: $(dgp_haz.natural_spline)")

set_parameters!(dgp_model, calib_params)

# Verify DGP hazard matches calibrated hazard
println("\n  Verify DGP hazard matches calibrated hazard:")
dgp_h12_params = get_parameters(dgp_model, 1, scale=:natural)
println("  DGP params: $(round.(dgp_h12_params, digits=4))")
println("  Calib params: $(round.(calib_h12, digits=4))")
println("  Match: $(all(dgp_h12_params .≈ calib_h12))")

panel_data = simulate(dgp_model; paths=false, data=true, nsim=1, autotmax=false,
                     obstype_by_transition=Dict(1 => 2))[1, 1]

n_panel = sum(panel_data.obstype .== 2)
println("  Generated $(nrow(panel_data)) rows: $n_panel panel")

# ============================================================================
# Step 4: Fit spline to panel data via MCEM
# ============================================================================
println("\nSTEP 4: Fit spline to panel data via MCEM")

fit_h12 = Hazard(@formula(0 ~ 1), "sp", 1, 2;
    degree=DEGREE, knots=knots, boundaryknots=boundaryknots, extrapolation="constant")

fit_model = multistatemodel(fit_h12; data=panel_data, surrogate=:markov, initialize=false)

# DIAGNOSTIC: Print spline configuration from fitting model  
println("\n  === FITTING MODEL SPLINE CONFIG ===")
fit_haz = fit_model.hazards[1]
println("  Type: $(typeof(fit_haz))")
println("  Degree: $(fit_haz.degree)")
println("  Knots (from hazard): $(fit_haz.knots)")
println("  Extrapolation: $(fit_haz.extrapolation)")
println("  Monotone constraint: $(fit_haz.monotone)")
println("  Natural spline: $(fit_haz.natural_spline)")

# DIAGNOSTIC: Compare all three spline configurations
println("\n  === CONFIGURATION COMPARISON ===")
println("  Degree match (calib==dgp==fit): $(calib_haz.degree == dgp_haz.degree == fit_haz.degree)")
println("  Knots match: $(calib_haz.knots == dgp_haz.knots == fit_haz.knots)")
println("    Calib knots: $(calib_haz.knots)")
println("    DGP knots: $(dgp_haz.knots)")
println("    Fit knots: $(fit_haz.knots)")
println("  Extrapolation match: $(calib_haz.extrapolation == dgp_haz.extrapolation == fit_haz.extrapolation)")
println("  Monotone match: $(calib_haz.monotone == dgp_haz.monotone == fit_haz.monotone)")
println("    Calib monotone: $(calib_haz.monotone)")
println("    DGP monotone: $(dgp_haz.monotone)")
println("    Fit monotone: $(fit_haz.monotone)")

# DIAGNOSTIC: Check parameter bounds
println("\n  === PARAMETER BOUNDS ===")
println("  Fitting model lb: $(fit_model.bounds.lb)")
println("  Fitting model ub: $(fit_model.bounds.ub)")

# Initialize close to truth with 10% perturbation
perturbed_h12 = calib_h12 .* (1 .+ 0.1 * randn(length(calib_h12)))
set_parameters!(fit_model, (h12 = perturbed_h12,))
println("\n  Initial h12 (perturbed): $(round.(perturbed_h12, digits=4))")

# DIAGNOSTIC: Verify hazard values before fitting
println("\n  === HAZARD VALUES BEFORE FITTING ===")
init_params = get_parameters(fit_model, 1, scale=:natural)
println("  Init params in model: $(round.(init_params, digits=4))")
for t in [1.0, 2.5, 4.0]
    h_calib = calib_fit.hazards[1](t, calib_h12, NamedTuple())
    h_dgp = dgp_model.hazards[1](t, dgp_h12_params, NamedTuple())
    h_init = fit_model.hazards[1](t, init_params, NamedTuple())
    println("  t=$t: calib=$(round(h_calib, digits=4)), dgp=$(round(h_dgp, digits=4)), init=$(round(h_init, digits=4))")
end

println("\n  Fitting with MCEM (surrogate=:markov)...")

# DIAGNOSTIC: Compare hazard functions directly before fitting
println("\n  === DIRECT HAZARD FUNCTION COMPARISON ===")
using BSplineKit
calib_basis = MultistateModels._rebuild_spline_basis(calib_haz)
dgp_basis = MultistateModels._rebuild_spline_basis(dgp_haz)
fit_basis = MultistateModels._rebuild_spline_basis(fit_haz)
println("  Calib nbasis: $(length(calib_basis)), DGP nbasis: $(length(dgp_basis)), Fit nbasis: $(length(fit_basis))")

# The key - verify that the three models produce the same hazard for the same params
# Use 3 params to match the nbasis=3
println("\n  Testing with SAME params [0.15, 0.25, 0.30]:")
test_params = [0.15, 0.25, 0.30]
test_pars_nt = (baseline=(h12_sp1=0.15, h12_sp2=0.25, h12_sp3=0.30),)
for t_test in [1.0, 2.5, 4.0]
    h_calib = calib_fit.hazards[1].hazard_fn(t_test, test_pars_nt, NamedTuple())
    h_dgp = dgp_model.hazards[1].hazard_fn(t_test, test_pars_nt, NamedTuple())
    h_fit = fit_model.hazards[1].hazard_fn(t_test, test_pars_nt, NamedTuple())
    println("  t=$t_test: calib=$(round(h_calib, digits=4)), dgp=$(round(h_dgp, digits=4)), fit=$(round(h_fit, digits=4)), match=$(h_calib ≈ h_dgp ≈ h_fit)")
end

fitted = fit(fit_model;
    verbose=true,
    compute_vcov=false,
    method=:MCEM,
    penalty=:none,
    tol=0.05,
    ess_target_initial=100,
    max_ess=500,
    maxiter=25,
    initialize=false
)

fit_h12_params = get_parameters(fitted, 1, scale=:natural)
println("\n  Fitted h12 coefficients: $(round.(fit_h12_params, digits=4))")

# ============================================================================
# Step 5: Compare fitted hazard to calibrated
# ============================================================================
println("\nSTEP 5: Compare fitted hazard to calibrated")

println("\n  h12 comparison (fitted vs calibrated):")
println("  Time    Calib       Fitted      RelErr    Status")
max_h12_err = 0.0
for t in eval_times
    hc = calib_fit.hazards[1](t, calib_h12, NamedTuple())
    hf = fitted.hazards[1](t, fit_h12_params, NamedTuple())
    re = abs(hf - hc) / hc
    global max_h12_err = max(max_h12_err, re)
    status = re <= 0.5 ? "✓" : "✗"
    println("  $(rpad(t, 6)) $(rpad(round(hc, digits=4), 11)) $(rpad(round(hf, digits=4), 11)) $(rpad(round(re*100, digits=1), 9))% $status")
end

h12_pass = max_h12_err <= 0.5

println("\n" * "="^70)
println("SUMMARY")
println("="^70)
println("  h12 max error: $(round(max_h12_err*100, digits=1))% (tolerance: 50%): $(h12_pass ? "PASS" : "FAIL")")
println("="^70)
