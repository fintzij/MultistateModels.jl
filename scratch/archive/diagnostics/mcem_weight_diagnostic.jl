# MCEM Weight Diagnostic for Spline Panel Tests
# Goal: Understand why MCEM converges to wrong solution with spline hazards

using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using MultistateModels
using DataFrames
using Random
using Statistics
import MultistateModels: Hazard, @formula, multistatemodel, fit, set_parameters!, 
    simulate, get_parameters, draw_paths, is_markov, unflatten_parameters

println("="^70)
println("MCEM IMPORTANCE WEIGHT DIAGNOSTIC")
println("="^70)

const N_SUBJECTS = 500  # Smaller for faster diagnostics
const MAX_TIME = 5.0
const DEGREE = 2
const knots = [2.5]
const boundaryknots = [0.0, MAX_TIME]
const PANEL_TIMES = collect(0.0:1.0:5.0)  # Coarser panel for simplicity

Random.seed!(54321)

# ============================================================================
# Step 1: Create ground truth (calibrated spline from Weibull exact data)
# ============================================================================
println("\n[1] Creating ground truth from Weibull exact data...")

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
println("  Generated $(nrow(exact_data)) exact obs from Weibull κ=1.3, λ=0.15")

# Calibrate spline to exact data
sp_h12_calib = Hazard(@formula(0 ~ 1), "sp", 1, 2;
    degree=DEGREE, knots=knots, boundaryknots=boundaryknots, extrapolation="constant")
calib_model = multistatemodel(sp_h12_calib; data=exact_data)
calib_fit = fit(calib_model; verbose=false, compute_vcov=false, penalty=:none)

true_params = get_parameters(calib_fit, 1, scale=:natural)
println("  Calibrated spline coefficients: $(round.(true_params, digits=4))")

# ============================================================================
# Step 2: Generate panel data from calibrated spline
# ============================================================================
println("\n[2] Generating panel data from calibrated spline...")

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
println("  Generated $(nrow(panel_data)) panel observations")

# Count transitions
n_transitioned = sum(panel_data.statefrom .!= panel_data.stateto)
println("  Subjects who transitioned: ~$(n_transitioned) transitions observed")

# ============================================================================
# Step 3: Build fitting model with surrogate
# ============================================================================
println("\n[3] Building fitting model with Markov surrogate...")

fit_h12 = Hazard(@formula(0 ~ 1), "sp", 1, 2;
    degree=DEGREE, knots=knots, boundaryknots=boundaryknots, extrapolation="constant")
fit_model = multistatemodel(fit_h12; data=panel_data, surrogate=:markov, initialize=false)

# Set parameters to true values for diagnostics
set_parameters!(fit_model, (h12 = true_params,))
println("  Model parameters set to true: $(round.(true_params, digits=4))")

# ============================================================================
# Step 4: Draw paths and examine importance weights AT TRUE PARAMETERS
# ============================================================================
println("\n[4] Drawing paths at TRUE parameters...")

result_true = draw_paths(fit_model; npaths=200, return_logliks=true)
println("  Drew $(length(result_true.samplepaths)) subjects × $(length(result_true.samplepaths[1])) paths each")

# ESS statistics
ess_values = result_true.subj_ess
println("  ESS per subject: min=$(round(minimum(ess_values), digits=1)), median=$(round(median(ess_values), digits=1)), max=$(round(maximum(ess_values), digits=1))")

# Log importance weight statistics (before normalization)
all_log_weights = Float64[]
for i in eachindex(result_true.loglik_target)
    # log w = log p_target - log p_surrogate
    log_w = result_true.loglik_target[i] .- result_true.loglik_surrog[i]
    append!(all_log_weights, log_w)
end
println("  Log importance weights: mean=$(round(mean(all_log_weights), digits=2)), std=$(round(std(all_log_weights), digits=2)), range=[$(round(minimum(all_log_weights), digits=2)), $(round(maximum(all_log_weights), digits=2))]")

# ============================================================================
# Step 5: Perturb parameters and examine weight distribution
# ============================================================================
println("\n[5] Examining weight behavior with perturbed parameters...")

# Test with wrong parameters (the kind MCEM converges to)
wrong_params = [0.25, 0.05, 1.2]  # Approximately what MCEM finds
set_parameters!(fit_model, (h12 = wrong_params,))

result_wrong = draw_paths(fit_model; npaths=200, return_logliks=true)
println("  With WRONG params $(round.(wrong_params, digits=2)):")
ess_wrong = result_wrong.subj_ess
println("  ESS: min=$(round(minimum(ess_wrong), digits=1)), median=$(round(median(ess_wrong), digits=1)), max=$(round(maximum(ess_wrong), digits=1))")

# Compare log-likelihoods at true vs wrong params
# For each subject, compute average weighted LL
set_parameters!(fit_model, (h12 = true_params,))
result_true2 = draw_paths(fit_model; npaths=200, return_logliks=true)

ll_at_true = sum([sum(result_true2.loglik_target[i] .* result_true2.ImportanceWeightsNormalized[i]) for i in eachindex(result_true2.loglik_target)])
ll_at_wrong = sum([sum(result_wrong.loglik_target[i] .* result_wrong.ImportanceWeightsNormalized[i]) for i in eachindex(result_wrong.loglik_target)])
println("  Weighted LL at true: $(round(ll_at_true, digits=2))")
println("  Weighted LL at wrong: $(round(ll_at_wrong, digits=2))")

# ============================================================================
# Step 6: Check surrogate fit quality
# ============================================================================
println("\n[6] Checking Markov surrogate quality...")

surrogate = fit_model.markovsurrogate
println("  Surrogate fitted: $(surrogate.fitted)")
surrogate_rate = surrogate.parameters.flat[1]
println("  Surrogate exponential rate: $(round(surrogate_rate, digits=4))")

# Compare hazard shapes
eval_times = [0.5, 1.0, 2.0, 3.0, 4.0, 4.5]
println("\n  Hazard comparison (spline truth vs Markov surrogate):")
println("  Time    Spline(true)  Surrogate   Ratio")
set_parameters!(fit_model, (h12 = true_params,))
for t in eval_times
    h_spline = fit_model.hazards[1](t, true_params, NamedTuple())
    # Surrogate is exponential (constant hazard)
    h_surrog = surrogate_rate
    ratio = h_spline / h_surrog
    println("  $(rpad(t, 6)) $(rpad(round(h_spline, digits=4), 13)) $(rpad(round(h_surrog, digits=4), 11)) $(round(ratio, digits=3))")
end

# ============================================================================
# Step 7: Compute expected Q function at true vs wrong
# ============================================================================
println("\n[7] Computing MCEM Q function at different parameter values...")

# Draw paths once at current surrogate
set_parameters!(fit_model, (h12 = true_params,))
result_base = draw_paths(fit_model; npaths=300, return_logliks=true)
paths = result_base.samplepaths
weights_normalized = result_base.ImportanceWeightsNormalized

# Function to compute Q (expected complete-data LL)
function compute_Q(model, params, paths, weights)
    set_parameters!(model, (h12 = params,))
    pars_nested = unflatten_parameters(params, model)
    
    Q = 0.0
    for i in eachindex(paths)
        for j in eachindex(paths[i])
            path = paths[i][j]
            # Compute log-likelihood of path under params
            ll = MultistateModels.loglik(pars_nested, path, model.hazards, model)
            Q += ll * weights[i][j]
        end
    end
    return Q
end

Q_true = compute_Q(fit_model, true_params, paths, weights_normalized)
Q_wrong = compute_Q(fit_model, wrong_params, paths, weights_normalized)

println("  Q(true params) = $(round(Q_true, digits=2))")
println("  Q(wrong params) = $(round(Q_wrong, digits=2))")
println("  ΔQ = Q(true) - Q(wrong) = $(round(Q_true - Q_wrong, digits=2))")

# Test intermediate parameters
println("\n  Testing parameter interpolation:")
for α in [0.0, 0.25, 0.5, 0.75, 1.0]
    interp_params = α .* true_params .+ (1-α) .* wrong_params
    Q_interp = compute_Q(fit_model, interp_params, paths, weights_normalized)
    println("  α=$α params=$(round.(interp_params, digits=3)) Q=$(round(Q_interp, digits=2))")
end

# ============================================================================
# Step 8: Check if paths cover the transition time distribution
# ============================================================================
println("\n[8] Examining sampled path distribution...")

# Extract transition times from sampled paths
transition_times = Float64[]
for subj_paths in paths
    for path in subj_paths
        for k in 1:(length(path.times)-1)
            if path.states[k] != path.states[k+1]
                push!(transition_times, path.times[k+1])
            end
        end
    end
end

if !isempty(transition_times)
    println("  Sampled transition times: n=$(length(transition_times)), mean=$(round(mean(transition_times), digits=2)), range=[$(round(minimum(transition_times), digits=2)), $(round(maximum(transition_times), digits=2))]")
else
    println("  No transitions in sampled paths!")
end

println("\n" * "="^70)
println("DIAGNOSTIC COMPLETE")
println("="^70)
