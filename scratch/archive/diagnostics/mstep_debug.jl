# M-step debugging script
# Test whether optimization actually finds the correct maximum

using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using MultistateModels
using DataFrames
using Random
using Statistics
using Optimization
using OptimizationOptimJL
import MultistateModels: Hazard, @formula, multistatemodel, fit, set_parameters!, 
    simulate, get_parameters, draw_paths, unflatten_parameters, SMPanelData, 
    loglik_semi_markov, get_parameters_flat

println("="^70)
println("M-STEP OPTIMIZATION DEBUG")
println("="^70)

const N_SUBJECTS = 500
const MAX_TIME = 5.0
const DEGREE = 2
const knots = [2.5]
const boundaryknots = [0.0, MAX_TIME]
const PANEL_TIMES = collect(0.0:1.0:5.0)

Random.seed!(99999)

# ============================================================================
# Step 1: Create data and model (same as before)
# ============================================================================
println("\n[1] Setting up model and data...")

# Create ground truth
template_exact = DataFrame(
    id = 1:N_SUBJECTS,
    tstart = zeros(N_SUBJECTS),
    tstop = fill(MAX_TIME, N_SUBJECTS),
    statefrom = ones(Int, N_SUBJECTS),
    stateto = fill(2, N_SUBJECTS),
    obstype = ones(Int, N_SUBJECTS)
)

wei_h12 = Hazard(@formula(0 ~ 1), "wei", 1, 2)
wei_model = multistatemodel(wei_h12; data=template_exact)
set_parameters!(wei_model, (h12 = [1.3, 0.15],))

exact_data = simulate(wei_model; data=true, paths=false, nsim=1)[1]

sp_h12_calib = Hazard(@formula(0 ~ 1), "sp", 1, 2;
    degree=DEGREE, knots=knots, boundaryknots=boundaryknots, extrapolation="constant")
calib_model = multistatemodel(sp_h12_calib; data=exact_data)
calib_fit = fit(calib_model; verbose=false, compute_vcov=false, penalty=:none)

true_params = get_parameters(calib_fit, 1, scale=:natural)
println("  True spline coefficients: $(round.(true_params, digits=4))")

# Generate panel data
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
set_parameters!(dgp_model, (h12 = true_params,))

panel_data = simulate(dgp_model; paths=false, data=true, nsim=1, autotmax=false,
                     obstype_by_transition=Dict(1 => 2))[1, 1]

fit_h12 = Hazard(@formula(0 ~ 1), "sp", 1, 2;
    degree=DEGREE, knots=knots, boundaryknots=boundaryknots, extrapolation="constant")
fit_model = multistatemodel(fit_h12; data=panel_data, surrogate=:markov, initialize=false)
set_parameters!(fit_model, (h12 = true_params,))

# ============================================================================
# Step 2: Draw paths and get weights
# ============================================================================
println("\n[2] Drawing paths with importance sampling...")

result = draw_paths(fit_model; npaths=300, return_logliks=true)
paths = result.samplepaths
weights = result.ImportanceWeightsNormalized

println("  Drew paths, ESS: min=$(round(minimum(result.subj_ess), digits=1)), median=$(round(median(result.subj_ess), digits=1))")

# ============================================================================
# Step 3: Define M-step objective
# ============================================================================
println("\n[3] Creating M-step optimization problem...")

mstep_data = SMPanelData(fit_model, paths, weights)

# Objective function: negative log-likelihood
function mstep_objective(params, data)
    return loglik_semi_markov(params, data; neg=true, use_sampling_weight=true)
end

# Test objective at true params
obj_true = mstep_objective(true_params, mstep_data)
println("  Objective at TRUE params: $(round(obj_true, digits=2))")

# Test objective at wrong params
wrong_params = [0.25, 0.05, 1.2]
obj_wrong = mstep_objective(wrong_params, mstep_data)
println("  Objective at WRONG params: $(round(obj_wrong, digits=2))")

# ============================================================================
# Step 4: Run optimization from different starting points
# ============================================================================
println("\n[4] Running optimization from different starting points...")

lb = [0.0, 0.0, 0.0]
ub = [Inf, Inf, Inf]

function optimize_from(x0, label)
    optf = OptimizationFunction(mstep_objective, Optimization.AutoForwardDiff())
    prob = OptimizationProblem(optf, x0, mstep_data; lb=lb, ub=ub)
    sol = solve(prob, BFGS())
    return sol.u
end

# From true params (slightly perturbed)
x0_near_true = true_params .* (1 .+ 0.05 * randn(3))
println("\n  Starting from near TRUE: $(round.(x0_near_true, digits=4))")
solution_from_true = optimize_from(x0_near_true, "near_true")
println("  Solution: $(round.(solution_from_true, digits=4))")
obj_solution_true = mstep_objective(solution_from_true, mstep_data)
println("  Objective at solution: $(round(obj_solution_true, digits=2))")

# From wrong params
x0_wrong = wrong_params .* (1 .+ 0.05 * randn(3))
println("\n  Starting from near WRONG: $(round.(x0_wrong, digits=4))")
solution_from_wrong = optimize_from(x0_wrong, "near_wrong")
println("  Solution: $(round.(solution_from_wrong, digits=4))")
obj_solution_wrong = mstep_objective(solution_from_wrong, mstep_data)
println("  Objective at solution: $(round(obj_solution_wrong, digits=2))")

# From default initialization (ones)
x0_default = [1.0, 1.0, 1.0]
println("\n  Starting from DEFAULT [1,1,1]:")
solution_from_default = optimize_from(x0_default, "default")
println("  Solution: $(round.(solution_from_default, digits=4))")
obj_solution_default = mstep_objective(solution_from_default, mstep_data)
println("  Objective at solution: $(round(obj_solution_default, digits=2))")

# ============================================================================
# Step 5: Check the landscape with gradient
# ============================================================================
println("\n[5] Checking gradient at various points...")

using ForwardDiff

function grad_at(x)
    ForwardDiff.gradient(p -> mstep_objective(p, mstep_data), x)
end

g_true = grad_at(true_params)
println("  Gradient at TRUE: $(round.(g_true, digits=4)), norm=$(round(norm(g_true), digits=6))")

g_wrong = grad_at(wrong_params)
println("  Gradient at WRONG: $(round.(g_wrong, digits=4)), norm=$(round(norm(g_wrong), digits=6))")

g_solution = grad_at(solution_from_default)
println("  Gradient at SOLUTION: $(round.(g_solution, digits=4)), norm=$(round(norm(g_solution), digits=6))")

# ============================================================================
# Summary
# ============================================================================
println("\n" * "="^70)
println("SUMMARY")
println("="^70)
println("  TRUE params: $(round.(true_params, digits=4)), obj=$(round(obj_true, digits=2))")
println("  Solution from TRUE: $(round.(solution_from_true, digits=4)), obj=$(round(obj_solution_true, digits=2))")
println("  Solution from DEFAULT: $(round.(solution_from_default, digits=4)), obj=$(round(obj_solution_default, digits=2))")
println("  WRONG params: $(round.(wrong_params, digits=4)), obj=$(round(obj_wrong, digits=2))")
println("="^70)
