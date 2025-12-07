#!/usr/bin/env julia
"""
Run Long Tests with Enhanced Diagnostics

This script runs all long tests for a 3-state illness-death model:
    Healthy (1) → Ill (2) → Dead (3)
    Healthy (1) → Dead (3)

Tests include:
- Parametric hazards: Exponential, Weibull, Gompertz
- Covariate effects: Proportional hazards (PH) and Accelerated failure time (AFT)
- Phase-type approximations: Exact and panel data

Generates comprehensive diagnostic reports including:
1. Parameter comparison tables with relative errors
2. State prevalence plots: Expected (true), Observed (data), Fitted (estimated)
3. Cumulative incidence plots for state transitions

Usage:
    julia --threads=auto --project=. test/run_longtests_with_diagnostics.jl

Output:
    - test/reports/inference_longtests.md
    - test/reports/assets/diagnostics/*.png
"""

using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using Test
using MultistateModels
using DataFrames
using Random
using Statistics
using LinearAlgebra
using Distributions
using Printf
using Dates

# Check for CairoMakie (optional for plots)
const HAS_PLOTS = try
    using CairoMakie
    true
catch
    @warn "CairoMakie not available. Plots will be skipped."
    false
end

if HAS_PLOTS
    CairoMakie.activate!(type = "png", px_per_unit = 2.0)
end

# Import internal functions
import MultistateModels: Hazard, multistatemodel, fit, set_parameters!, simulate,
    get_parameters_flat, get_parameters, SamplePath, @formula,
    fit_surrogate, needs_phasetype_proposal, resolve_proposal_config,
    loglik_exact, loglik_markov,
    PhaseTypeConfig, build_phasetype_model, build_phasetype_hazards

# =============================================================================
# Configuration
# =============================================================================
const OUTPUT_DIR = joinpath(@__DIR__, "reports", "assets", "diagnostics")
mkpath(OUTPUT_DIR)

const REPORT_FILE = joinpath(@__DIR__, "reports", "inference_longtests.md")

# Test configuration
const RNG_SEED = 0xABCD2025
const N_SUBJECTS = 1000            # Sample size for most tests
const N_SUBJECTS_GOMPERTZ = 5000   # Larger sample for Gompertz (shape/scale correlation)
const N_SUBJECTS_COVARIATE = 2000  # Larger sample for covariate model (rare 1→3 transition)
const N_SIM_TRAJ = 5000            # Trajectories for distributional comparison
const MAX_TIME = 15.0              # Maximum follow-up time
const EVAL_TIMES = collect(0.0:0.5:MAX_TIME)

# MCEM settings
const MCEM_TOL = 0.01
const MCEM_ESS_INITIAL = 100
const MCEM_ESS_MAX = 2000
const MAX_ITER = 50

# =============================================================================
# Results Storage
# =============================================================================
mutable struct TestResult
    test_name::String
    hazard_family::String
    true_params::Dict{String, Float64}
    est_params::Dict{String, Float64}
    rel_errors::Dict{String, Float64}
    se_params::Dict{String, Float64}
    ci_lower::Dict{String, Float64}
    ci_upper::Dict{String, Float64}
    converged::Bool
    loglik::Float64
    runtime_seconds::Float64
    n_subjects::Int
    # Prevalence data
    eval_times::Vector{Float64}
    prevalence_true::Union{Nothing, Matrix{Float64}}      # Expected (from true parameters)
    prevalence_observed::Union{Nothing, Matrix{Float64}}  # Observed (from data)
    prevalence_fitted::Union{Nothing, Matrix{Float64}}    # Fitted (from estimated parameters)
    # Cumulative incidence
    cumincid_true::Union{Nothing, Dict{String, Vector{Float64}}}
    cumincid_observed::Union{Nothing, Dict{String, Vector{Float64}}}
    cumincid_fitted::Union{Nothing, Dict{String, Vector{Float64}}}
end

TestResult(name::String, family::String) = TestResult(
    name, family,
    Dict{String, Float64}(), Dict{String, Float64}(), Dict{String, Float64}(),
    Dict{String, Float64}(), Dict{String, Float64}(), Dict{String, Float64}(),
    false, NaN, 0.0, 0,
    Float64[], nothing, nothing, nothing, nothing, nothing, nothing
)

const ALL_RESULTS = TestResult[]

# =============================================================================
# Helper Functions
# =============================================================================

"""
    compute_relative_error(true_val, est_val)

Compute relative error as percentage. For values near zero, returns absolute error × 100.
"""
function compute_relative_error(true_val::Float64, est_val::Float64)
    if abs(true_val) > 0.01
        return 100.0 * (est_val - true_val) / true_val
    else
        return 100.0 * (est_val - true_val)
    end
end

"""
    compute_state_prevalence(paths, eval_times, n_states)

Compute state prevalence at each evaluation time from a collection of sample paths.
"""
function compute_state_prevalence(paths, eval_times::Vector{Float64}, n_states::Int)
    n_times = length(eval_times)
    prevalence = zeros(Float64, n_times, n_states)
    n_paths = length(paths)
    
    for path in paths
        for (t_idx, t) in enumerate(eval_times)
            state_idx = searchsortedlast(path.times, t)
            if state_idx >= 1
                state = path.states[state_idx]
                prevalence[t_idx, state] += 1.0
            end
        end
    end
    
    prevalence ./= n_paths
    return prevalence
end

"""
    compute_cumulative_incidence(paths, eval_times, from_state, to_state)

Compute cumulative incidence of transition from_state → to_state.
"""
function compute_cumulative_incidence(paths, eval_times::Vector{Float64},
                                       from_state::Int, to_state::Int)
    n_times = length(eval_times)
    cumincid = zeros(Float64, n_times)
    n_paths = length(paths)
    
    for path in paths
        transition_time = Inf
        for i in 1:(length(path.states) - 1)
            if path.states[i] == from_state && path.states[i+1] == to_state
                transition_time = path.times[i+1]
                break
            end
        end
        
        for (t_idx, t) in enumerate(eval_times)
            if transition_time <= t
                cumincid[t_idx] += 1.0
            end
        end
    end
    
    cumincid ./= n_paths
    return cumincid
end

"""
    compute_prevalence_from_data(exact_data, eval_times, n_states)

Compute state prevalence at each evaluation time from exact observed data.
Assumes data has statefrom, stateto, tstart, tstop columns with obstype=1 (exact observations).
"""
function compute_prevalence_from_data(exact_data::DataFrame, eval_times::Vector{Float64}, n_states::Int)
    n_times = length(eval_times)
    prevalence = zeros(Float64, n_times, n_states)
    
    # Group by subject ID
    subjects = unique(exact_data.id)
    n_subjects = length(subjects)
    
    for subj_id in subjects
        subj_data = filter(row -> row.id == subj_id, exact_data)
        
        for (t_idx, t) in enumerate(eval_times)
            # Find state at time t
            state = nothing
            for row in eachrow(subj_data)
                if row.tstart <= t < row.tstop
                    state = row.statefrom
                    break
                elseif t >= row.tstop
                    state = row.stateto
                end
            end
            
            if !isnothing(state) && state <= n_states
                prevalence[t_idx, state] += 1.0
            end
        end
    end
    
    prevalence ./= n_subjects
    return prevalence
end

"""
    compute_cumincid_from_data(exact_data, eval_times, from_state, to_state)

Compute cumulative incidence of transition from_state → to_state from exact observed data.
"""
function compute_cumincid_from_data(exact_data::DataFrame, eval_times::Vector{Float64},
                                    from_state::Int, to_state::Int)
    n_times = length(eval_times)
    cumincid = zeros(Float64, n_times)
    
    subjects = unique(exact_data.id)
    n_subjects = length(subjects)
    
    for subj_id in subjects
        subj_data = filter(row -> row.id == subj_id, exact_data)
        
        # Find first transition from_state → to_state
        transition_time = Inf
        for row in eachrow(subj_data)
            if row.statefrom == from_state && row.stateto == to_state
                transition_time = row.tstop
                break
            end
        end
        
        for (t_idx, t) in enumerate(eval_times)
            if transition_time <= t
                cumincid[t_idx] += 1.0
            end
        end
    end
    
    cumincid ./= n_subjects
    return cumincid
end

"""
    extract_ci(se, est; level=0.95)

Compute confidence interval from standard error and estimate.
"""
function extract_ci(se::Float64, est::Float64; level=0.95)
    z = quantile(Normal(), 1 - (1 - level) / 2)
    return (est - z * se, est + z * se)
end

# =============================================================================
# Plotting Functions
# =============================================================================

if HAS_PLOTS

"""
    plot_prevalence_comparison(result, n_states)

Create state prevalence comparison plot for illness-death model.
Shows Expected (true parameters), Observed (data), and Fitted (estimated parameters).
"""
function plot_prevalence_comparison(result::TestResult, n_states::Int)
    fig = Figure(size = (900, 350))
    
    state_labels = ["Healthy", "Ill", "Dead"]
    
    for s in 1:n_states
        ax = Axis(fig[1, s], 
                  title = state_labels[s],
                  xlabel = s == 2 ? "Time" : "",
                  ylabel = s == 1 ? "Prevalence" : "")
        
        # Expected (from true parameters) - solid black
        if !isnothing(result.prevalence_true)
            lines!(ax, result.eval_times, result.prevalence_true[:, s], 
                   color = :black, linewidth = 2.5, label = "Expected")
        end
        
        # Observed (from data) - gray dots
        if !isnothing(result.prevalence_observed)
            scatter!(ax, result.eval_times, result.prevalence_observed[:, s], 
                     color = :gray, markersize = 6, label = "Observed")
        end
        
        # Fitted (from estimated parameters) - dashed blue
        if !isnothing(result.prevalence_fitted)
            lines!(ax, result.eval_times, result.prevalence_fitted[:, s], 
                   color = :steelblue, linewidth = 2, linestyle = :dash, label = "Fitted")
        end
        
        if s == n_states
            axislegend(ax, position = :rt)
        end
        
        ylims!(ax, 0, 1)
    end
    
    Label(fig[0, :], result.test_name, fontsize = 16, font = :bold)
    
    return fig
end

"""
    plot_cumulative_incidence(result)

Create cumulative incidence comparison plot.
Shows Expected (true parameters), Observed (data), and Fitted (estimated parameters).
"""
function plot_cumulative_incidence(result::TestResult)
    if isnothing(result.cumincid_true) || isnothing(result.cumincid_fitted)
        return nothing
    end
    
    transitions = sort(collect(keys(result.cumincid_true)))
    n_trans = length(transitions)
    
    fig = Figure(size = (300 * n_trans, 350))
    
    trans_labels = Dict(
        "1→2" => "Healthy → Ill",
        "1→3" => "Healthy → Dead", 
        "2→3" => "Ill → Dead"
    )
    
    for (i, trans) in enumerate(transitions)
        label = get(trans_labels, trans, trans)
        ax = Axis(fig[1, i],
                  title = label,
                  xlabel = "Time",
                  ylabel = i == 1 ? "Cumulative Incidence" : "")
        
        # Expected (from true parameters) - solid black
        lines!(ax, result.eval_times, result.cumincid_true[trans],
               color = :black, linewidth = 2.5, label = "Expected")
        
        # Observed (from data) - gray dots
        if !isnothing(result.cumincid_observed) && haskey(result.cumincid_observed, trans)
            scatter!(ax, result.eval_times, result.cumincid_observed[trans],
                     color = :gray, markersize = 6, label = "Observed")
        end
        
        # Fitted (from estimated parameters) - dashed blue
        lines!(ax, result.eval_times, result.cumincid_fitted[trans],
               color = :steelblue, linewidth = 2, linestyle = :dash, label = "Fitted")
        
        if i == n_trans
            axislegend(ax, position = :rb)
        end
        
        ylims!(ax, 0, 1)
    end
    
    Label(fig[0, :], "$(result.test_name) - Cumulative Incidence", fontsize = 14, font = :bold)
    
    return fig
end

end  # if HAS_PLOTS

# =============================================================================
# Test Functions - All use 3-state illness-death model
# =============================================================================

"""
    run_exponential_test()

Test exponential hazards (illness-death model) with exact data.
States: Healthy (1) → Ill (2) → Dead (3), Healthy (1) → Dead (3)
"""
function run_exponential_test()
    result = TestResult("Exponential", "exp")
    
    Random.seed!(RNG_SEED)
    
    # True parameters (log-rates) for illness-death model
    true_rate_12 = 0.15   # Healthy → Ill
    true_rate_23 = 0.20   # Ill → Dead
    true_rate_13 = 0.05   # Healthy → Dead
    
    result.true_params["h12 (Healthy→Ill)"] = log(true_rate_12)
    result.true_params["h23 (Ill→Dead)"] = log(true_rate_23)
    result.true_params["h13 (Healthy→Dead)"] = log(true_rate_13)
    
    true_params = (
        h12 = [log(true_rate_12)],
        h23 = [log(true_rate_23)],
        h13 = [log(true_rate_13)]
    )
    
    # Define hazards for illness-death model
    h12 = Hazard(@formula(0 ~ 1), "exp", 1, 2)  # Healthy → Ill
    h23 = Hazard(@formula(0 ~ 1), "exp", 2, 3)  # Ill → Dead
    h13 = Hazard(@formula(0 ~ 1), "exp", 1, 3)  # Healthy → Dead
    
    # Generate data
    template = DataFrame(
        id = 1:N_SUBJECTS,
        tstart = zeros(N_SUBJECTS),
        tstop = fill(MAX_TIME, N_SUBJECTS),
        statefrom = ones(Int, N_SUBJECTS),
        stateto = ones(Int, N_SUBJECTS),
        obstype = ones(Int, N_SUBJECTS)
    )
    
    model_sim = multistatemodel(h12, h23, h13; data=template)
    set_parameters!(model_sim, true_params)
    
    sim_result = simulate(model_sim; paths=false, data=true, nsim=1)
    exact_data = sim_result[1]
    result.n_subjects = length(unique(exact_data.id))
    
    # Fit model
    model_fit = multistatemodel(h12, h23, h13; data=exact_data)
    
    start_time = time()
    fitted = fit(model_fit; parallel=true, verbose=false, 
                 compute_vcov=true, compute_ij_vcov=true)
    result.runtime_seconds = time() - start_time
    
    # Extract estimates and compute relative errors
    # Note: get_parameters_flat returns in transition matrix order: 1→2, 1→3, 2→3
    params_flat = get_parameters_flat(fitted)
    param_names = ["h12 (Healthy→Ill)", "h13 (Healthy→Dead)", "h23 (Ill→Dead)"]
    
    vcov = fitted.vcov
    ses = !isnothing(vcov) ? sqrt.(diag(vcov)) : fill(NaN, 3)
    
    for (i, pname) in enumerate(param_names)
        result.est_params[pname] = params_flat[i]
        result.rel_errors[pname] = compute_relative_error(result.true_params[pname], params_flat[i])
        result.se_params[pname] = ses[i]
        ci = extract_ci(ses[i], params_flat[i])
        result.ci_lower[pname] = ci[1]
        result.ci_upper[pname] = ci[2]
    end
    
    result.converged = true
    result.loglik = fitted.loglik.loglik
    result.eval_times = EVAL_TIMES
    
    # Compute observed prevalence and cumulative incidence from the data
    result.prevalence_observed = compute_prevalence_from_data(exact_data, EVAL_TIMES, 3)
    result.cumincid_observed = Dict(
        "1→2" => compute_cumincid_from_data(exact_data, EVAL_TIMES, 1, 2),
        "1→3" => compute_cumincid_from_data(exact_data, EVAL_TIMES, 1, 3),
        "2→3" => compute_cumincid_from_data(exact_data, EVAL_TIMES, 2, 3)
    )
    
    # Simulate trajectories for prevalence comparison
    # simulate returns Vector{Vector{SamplePath}}, need to flatten for nsim=1
    true_paths_nested = simulate(model_sim; paths=true, data=false, nsim=1)
    true_paths = true_paths_nested[1]  # Get first simulation's paths
    result.prevalence_true = compute_state_prevalence(true_paths, EVAL_TIMES, 3)
    
    fitted_paths_nested = simulate(fitted; paths=true, data=false, nsim=1, tmax=MAX_TIME)
    fitted_paths = fitted_paths_nested[1]
    result.prevalence_fitted = compute_state_prevalence(fitted_paths, EVAL_TIMES, 3)
    
    # Cumulative incidence
    result.cumincid_true = Dict(
        "1→2" => compute_cumulative_incidence(true_paths, EVAL_TIMES, 1, 2),
        "1→3" => compute_cumulative_incidence(true_paths, EVAL_TIMES, 1, 3),
        "2→3" => compute_cumulative_incidence(true_paths, EVAL_TIMES, 2, 3)
    )
    result.cumincid_fitted = Dict(
        "1→2" => compute_cumulative_incidence(fitted_paths, EVAL_TIMES, 1, 2),
        "1→3" => compute_cumulative_incidence(fitted_paths, EVAL_TIMES, 1, 3),
        "2→3" => compute_cumulative_incidence(fitted_paths, EVAL_TIMES, 2, 3)
    )
    
    push!(ALL_RESULTS, result)
    return result
end

"""
    run_weibull_test()

Test Weibull hazards (illness-death model) with exact data.
"""
function run_weibull_test()
    result = TestResult("Weibull", "wei")
    
    Random.seed!(RNG_SEED + 1)
    
    # True parameters: (log_shape, log_scale) - NOTE: shape comes first in Weibull
    # Using shape values further from 1 for better identifiability
    true_scale_12, true_shape_12 = 0.12, 1.8   # Healthy → Ill (shape=1.8)
    true_scale_23, true_shape_23 = 0.15, 2.0   # Ill → Dead (shape=2.0)
    true_scale_13, true_shape_13 = 0.04, 1.5   # Healthy → Dead (shape=1.5)
    
    result.true_params["h12_shape"] = log(true_shape_12)
    result.true_params["h12_scale"] = log(true_scale_12)
    result.true_params["h13_shape"] = log(true_shape_13)
    result.true_params["h13_scale"] = log(true_scale_13)
    result.true_params["h23_shape"] = log(true_shape_23)
    result.true_params["h23_scale"] = log(true_scale_23)
    
    true_params = (
        h12 = [log(true_shape_12), log(true_scale_12)],
        h23 = [log(true_shape_23), log(true_scale_23)],
        h13 = [log(true_shape_13), log(true_scale_13)]
    )
    
    h12 = Hazard(@formula(0 ~ 1), "wei", 1, 2)
    h23 = Hazard(@formula(0 ~ 1), "wei", 2, 3)
    h13 = Hazard(@formula(0 ~ 1), "wei", 1, 3)
    
    template = DataFrame(
        id = 1:N_SUBJECTS,
        tstart = zeros(N_SUBJECTS),
        tstop = fill(MAX_TIME, N_SUBJECTS),
        statefrom = ones(Int, N_SUBJECTS),
        stateto = ones(Int, N_SUBJECTS),
        obstype = ones(Int, N_SUBJECTS)
    )
    
    model_sim = multistatemodel(h12, h23, h13; data=template)
    set_parameters!(model_sim, true_params)
    
    sim_result = simulate(model_sim; paths=false, data=true, nsim=1)
    exact_data = sim_result[1]
    result.n_subjects = length(unique(exact_data.id))
    
    model_fit = multistatemodel(h12, h23, h13; data=exact_data)
    
    start_time = time()
    fitted = fit(model_fit; parallel=true, verbose=false,
                 compute_vcov=true, compute_ij_vcov=true)
    result.runtime_seconds = time() - start_time
    
    params_flat = get_parameters_flat(fitted)
    param_names = ["h12_shape", "h12_scale", "h13_shape", "h13_scale", "h23_shape", "h23_scale"]
    
    vcov = fitted.vcov
    ses = !isnothing(vcov) ? sqrt.(diag(vcov)) : fill(NaN, 6)
    
    for (i, pname) in enumerate(param_names)
        result.est_params[pname] = params_flat[i]
        result.rel_errors[pname] = compute_relative_error(result.true_params[pname], params_flat[i])
        result.se_params[pname] = ses[i]
        ci = extract_ci(ses[i], params_flat[i])
        result.ci_lower[pname] = ci[1]
        result.ci_upper[pname] = ci[2]
    end
    
    result.converged = true
    result.loglik = fitted.loglik.loglik
    result.eval_times = EVAL_TIMES
    
    # Compute observed prevalence and cumulative incidence from the data
    result.prevalence_observed = compute_prevalence_from_data(exact_data, EVAL_TIMES, 3)
    result.cumincid_observed = Dict(
        "1→2" => compute_cumincid_from_data(exact_data, EVAL_TIMES, 1, 2),
        "1→3" => compute_cumincid_from_data(exact_data, EVAL_TIMES, 1, 3),
        "2→3" => compute_cumincid_from_data(exact_data, EVAL_TIMES, 2, 3)
    )
    
    true_paths_nested = simulate(model_sim; paths=true, data=false, nsim=1)
    true_paths = true_paths_nested[1]
    result.prevalence_true = compute_state_prevalence(true_paths, EVAL_TIMES, 3)
    
    fitted_paths_nested = simulate(fitted; paths=true, data=false, nsim=1, tmax=MAX_TIME)
    fitted_paths = fitted_paths_nested[1]
    result.prevalence_fitted = compute_state_prevalence(fitted_paths, EVAL_TIMES, 3)
    
    result.cumincid_true = Dict(
        "1→2" => compute_cumulative_incidence(true_paths, EVAL_TIMES, 1, 2),
        "1→3" => compute_cumulative_incidence(true_paths, EVAL_TIMES, 1, 3),
        "2→3" => compute_cumulative_incidence(true_paths, EVAL_TIMES, 2, 3)
    )
    result.cumincid_fitted = Dict(
        "1→2" => compute_cumulative_incidence(fitted_paths, EVAL_TIMES, 1, 2),
        "1→3" => compute_cumulative_incidence(fitted_paths, EVAL_TIMES, 1, 3),
        "2→3" => compute_cumulative_incidence(fitted_paths, EVAL_TIMES, 2, 3)
    )
    
    push!(ALL_RESULTS, result)
    return result
end

"""
    run_gompertz_test()

Test Gompertz hazards (illness-death model) with exact data.
"""
function run_gompertz_test()
    result = TestResult("Gompertz", "gom")
    
    Random.seed!(RNG_SEED + 2)
    
    # True parameters: (log_shape, log_scale) - both log-transformed
    # Gompertz: h(t) = scale * shape * exp(shape * t)
    # At t=0: h(0) = scale * shape
    # Using shape=0.1 for clear time-dependence, adjust scale for baseline rates
    true_shape_12, true_scale_12 = 0.1, 1.5    # h(0) = 0.15 for Healthy → Ill
    true_shape_23, true_scale_23 = 0.1, 2.0    # h(0) = 0.20 for Ill → Dead
    true_shape_13, true_scale_13 = 0.08, 0.625 # h(0) = 0.05 for Healthy → Dead
    
    result.true_params["h12_shape"] = log(true_shape_12)
    result.true_params["h12_scale"] = log(true_scale_12)
    result.true_params["h13_shape"] = log(true_shape_13)
    result.true_params["h13_scale"] = log(true_scale_13)
    result.true_params["h23_shape"] = log(true_shape_23)
    result.true_params["h23_scale"] = log(true_scale_23)
    
    true_params = (
        h12 = [log(true_shape_12), log(true_scale_12)],
        h23 = [log(true_shape_23), log(true_scale_23)],
        h13 = [log(true_shape_13), log(true_scale_13)]
    )
    
    h12 = Hazard(@formula(0 ~ 1), "gom", 1, 2)
    h23 = Hazard(@formula(0 ~ 1), "gom", 2, 3)
    h13 = Hazard(@formula(0 ~ 1), "gom", 1, 3)
    
    # Use larger sample size for Gompertz due to shape/scale correlation
    N = N_SUBJECTS_GOMPERTZ
    template = DataFrame(
        id = 1:N,
        tstart = zeros(N),
        tstop = fill(MAX_TIME, N),
        statefrom = ones(Int, N),
        stateto = ones(Int, N),
        obstype = ones(Int, N)
    )
    
    model_sim = multistatemodel(h12, h23, h13; data=template)
    set_parameters!(model_sim, true_params)
    
    sim_result = simulate(model_sim; paths=false, data=true, nsim=1)
    exact_data = sim_result[1]
    result.n_subjects = length(unique(exact_data.id))
    
    model_fit = multistatemodel(h12, h23, h13; data=exact_data)
    
    start_time = time()
    fitted = fit(model_fit; parallel=true, verbose=false,
                 compute_vcov=true, compute_ij_vcov=true)
    result.runtime_seconds = time() - start_time
    
    params_flat = get_parameters_flat(fitted)
    param_names = ["h12_shape", "h12_scale", "h13_shape", "h13_scale", "h23_shape", "h23_scale"]
    
    vcov = fitted.vcov
    ses = !isnothing(vcov) ? sqrt.(diag(vcov)) : fill(NaN, 6)
    
    for (i, pname) in enumerate(param_names)
        result.est_params[pname] = params_flat[i]
        result.rel_errors[pname] = compute_relative_error(result.true_params[pname], params_flat[i])
        result.se_params[pname] = ses[i]
        ci = extract_ci(ses[i], params_flat[i])
        result.ci_lower[pname] = ci[1]
        result.ci_upper[pname] = ci[2]
    end
    
    result.converged = true
    result.loglik = fitted.loglik.loglik
    result.eval_times = EVAL_TIMES
    
    # Compute observed prevalence and cumulative incidence from the data
    result.prevalence_observed = compute_prevalence_from_data(exact_data, EVAL_TIMES, 3)
    result.cumincid_observed = Dict(
        "1→2" => compute_cumincid_from_data(exact_data, EVAL_TIMES, 1, 2),
        "1→3" => compute_cumincid_from_data(exact_data, EVAL_TIMES, 1, 3),
        "2→3" => compute_cumincid_from_data(exact_data, EVAL_TIMES, 2, 3)
    )
    
    true_paths_nested = simulate(model_sim; paths=true, data=false, nsim=1)
    true_paths = true_paths_nested[1]
    result.prevalence_true = compute_state_prevalence(true_paths, EVAL_TIMES, 3)
    
    fitted_paths_nested = simulate(fitted; paths=true, data=false, nsim=1, tmax=MAX_TIME)
    fitted_paths = fitted_paths_nested[1]
    result.prevalence_fitted = compute_state_prevalence(fitted_paths, EVAL_TIMES, 3)
    
    result.cumincid_true = Dict(
        "1→2" => compute_cumulative_incidence(true_paths, EVAL_TIMES, 1, 2),
        "1→3" => compute_cumulative_incidence(true_paths, EVAL_TIMES, 1, 3),
        "2→3" => compute_cumulative_incidence(true_paths, EVAL_TIMES, 2, 3)
    )
    result.cumincid_fitted = Dict(
        "1→2" => compute_cumulative_incidence(fitted_paths, EVAL_TIMES, 1, 2),
        "1→3" => compute_cumulative_incidence(fitted_paths, EVAL_TIMES, 1, 3),
        "2→3" => compute_cumulative_incidence(fitted_paths, EVAL_TIMES, 2, 3)
    )
    
    push!(ALL_RESULTS, result)
    return result
end

"""
    run_exponential_covariate_test()

Test exponential hazards with covariate (illness-death model).
"""
function run_exponential_covariate_test()
    result = TestResult("Exponential + Covariate", "exp")
    
    Random.seed!(RNG_SEED + 3)
    
    # True parameters: (log_intercept, beta)
    # Using larger beta values for better estimation precision
    true_int_12, true_beta_12 = 0.12, 0.5   # Healthy → Ill
    true_int_23, true_beta_23 = 0.15, 0.4   # Ill → Dead
    true_int_13, true_beta_13 = 0.04, 0.5   # Healthy → Dead (larger effect for rare transition)
    
    result.true_params["h12_intercept"] = log(true_int_12)
    result.true_params["h12_beta"] = true_beta_12
    result.true_params["h13_intercept"] = log(true_int_13)
    result.true_params["h13_beta"] = true_beta_13
    result.true_params["h23_intercept"] = log(true_int_23)
    result.true_params["h23_beta"] = true_beta_23
    
    true_params = (
        h12 = [log(true_int_12), true_beta_12],
        h13 = [log(true_int_13), true_beta_13],
        h23 = [log(true_int_23), true_beta_23]
    )
    
    h12 = Hazard(@formula(0 ~ 1 + x), "exp", 1, 2)
    h23 = Hazard(@formula(0 ~ 1 + x), "exp", 2, 3)
    h13 = Hazard(@formula(0 ~ 1 + x), "exp", 1, 3)
    
    # Use larger sample size for covariate model (rare 1→3 transition)
    N = N_SUBJECTS_COVARIATE
    
    # Generate covariate data (binary treatment indicator)
    x_vals = rand([0.0, 1.0], N)
    
    template = DataFrame(
        id = 1:N,
        tstart = zeros(N),
        tstop = fill(MAX_TIME, N),
        statefrom = ones(Int, N),
        stateto = ones(Int, N),
        obstype = ones(Int, N),
        x = x_vals
    )
    
    model_sim = multistatemodel(h12, h23, h13; data=template)
    set_parameters!(model_sim, true_params)
    
    sim_result = simulate(model_sim; paths=false, data=true, nsim=1)
    exact_data = sim_result[1]
    result.n_subjects = length(unique(exact_data.id))
    
    model_fit = multistatemodel(h12, h23, h13; data=exact_data)
    
    start_time = time()
    fitted = fit(model_fit; parallel=true, verbose=false,
                 compute_vcov=true, compute_ij_vcov=true)
    result.runtime_seconds = time() - start_time
    
    params_flat = get_parameters_flat(fitted)
    param_names = ["h12_intercept", "h12_beta", "h13_intercept", "h13_beta", "h23_intercept", "h23_beta"]
    
    vcov = fitted.vcov
    ses = !isnothing(vcov) ? sqrt.(diag(vcov)) : fill(NaN, 6)
    
    for (i, pname) in enumerate(param_names)
        result.est_params[pname] = params_flat[i]
        result.rel_errors[pname] = compute_relative_error(result.true_params[pname], params_flat[i])
        result.se_params[pname] = ses[i]
        ci = extract_ci(ses[i], params_flat[i])
        result.ci_lower[pname] = ci[1]
        result.ci_upper[pname] = ci[2]
    end
    
    result.converged = true
    result.loglik = fitted.loglik.loglik
    result.eval_times = EVAL_TIMES
    
    # Compute observed prevalence and cumulative incidence from the data
    result.prevalence_observed = compute_prevalence_from_data(exact_data, EVAL_TIMES, 3)
    result.cumincid_observed = Dict(
        "1→2" => compute_cumincid_from_data(exact_data, EVAL_TIMES, 1, 2),
        "1→3" => compute_cumincid_from_data(exact_data, EVAL_TIMES, 1, 3),
        "2→3" => compute_cumincid_from_data(exact_data, EVAL_TIMES, 2, 3)
    )
    
    # Simulate from true model (using x=0 for prevalence)
    template_x0 = DataFrame(
        id = 1:N_SIM_TRAJ,
        tstart = zeros(N_SIM_TRAJ),
        tstop = fill(MAX_TIME, N_SIM_TRAJ),
        statefrom = ones(Int, N_SIM_TRAJ),
        stateto = ones(Int, N_SIM_TRAJ),
        obstype = ones(Int, N_SIM_TRAJ),
        x = zeros(N_SIM_TRAJ)
    )
    model_sim_x0 = multistatemodel(h12, h23, h13; data=template_x0)
    set_parameters!(model_sim_x0, true_params)
    
    true_paths_nested = simulate(model_sim_x0; paths=true, data=false, nsim=1)
    true_paths = true_paths_nested[1]
    result.prevalence_true = compute_state_prevalence(true_paths, EVAL_TIMES, 3)
    
    fitted_paths_nested = simulate(fitted; paths=true, data=false, nsim=1, tmax=MAX_TIME)
    fitted_paths = fitted_paths_nested[1]
    result.prevalence_fitted = compute_state_prevalence(fitted_paths, EVAL_TIMES, 3)
    
    result.cumincid_true = Dict(
        "1→2" => compute_cumulative_incidence(true_paths, EVAL_TIMES, 1, 2),
        "1→3" => compute_cumulative_incidence(true_paths, EVAL_TIMES, 1, 3),
        "2→3" => compute_cumulative_incidence(true_paths, EVAL_TIMES, 2, 3)
    )
    result.cumincid_fitted = Dict(
        "1→2" => compute_cumulative_incidence(fitted_paths, EVAL_TIMES, 1, 2),
        "1→3" => compute_cumulative_incidence(fitted_paths, EVAL_TIMES, 1, 3),
        "2→3" => compute_cumulative_incidence(fitted_paths, EVAL_TIMES, 2, 3)
    )
    
    push!(ALL_RESULTS, result)
    return result
end

# =============================================================================
# AFT (Accelerated Failure Time) Tests
# =============================================================================

"""
    run_weibull_aft_test()

Test Weibull hazards with AFT covariate parameterization (illness-death model).
AFT: covariates scale time rather than multiply hazard.
"""
function run_weibull_aft_test()
    result = TestResult("Weibull AFT", "wei-aft")
    
    Random.seed!(RNG_SEED + 10)
    
    # True parameters: (log_shape, log_scale, beta_aft)
    # Weibull AFT: h(t|x) = h_0(t * exp(-x'β)) * exp(-x'β)
    # β > 0 means longer survival for x=1 (time is stretched)
    true_shape_12, true_scale_12, true_beta_12 = 1.8, 0.12, 0.3   # Healthy → Ill
    true_shape_23, true_scale_23, true_beta_23 = 2.0, 0.15, 0.2   # Ill → Dead
    true_shape_13, true_scale_13, true_beta_13 = 1.5, 0.04, 0.4   # Healthy → Dead
    
    result.true_params["h12_shape"] = log(true_shape_12)
    result.true_params["h12_scale"] = log(true_scale_12)
    result.true_params["h12_beta"] = true_beta_12
    result.true_params["h13_shape"] = log(true_shape_13)
    result.true_params["h13_scale"] = log(true_scale_13)
    result.true_params["h13_beta"] = true_beta_13
    result.true_params["h23_shape"] = log(true_shape_23)
    result.true_params["h23_scale"] = log(true_scale_23)
    result.true_params["h23_beta"] = true_beta_23
    
    true_params = (
        h12 = [log(true_shape_12), log(true_scale_12), true_beta_12],
        h13 = [log(true_shape_13), log(true_scale_13), true_beta_13],
        h23 = [log(true_shape_23), log(true_scale_23), true_beta_23]
    )
    
    # Define hazards with AFT parameterization
    h12 = Hazard(@formula(0 ~ 1 + x), "wei", 1, 2; linpred_effect=:aft)
    h23 = Hazard(@formula(0 ~ 1 + x), "wei", 2, 3; linpred_effect=:aft)
    h13 = Hazard(@formula(0 ~ 1 + x), "wei", 1, 3; linpred_effect=:aft)
    
    N = N_SUBJECTS_COVARIATE
    
    # Generate covariate data (binary treatment indicator)
    x_vals = rand([0.0, 1.0], N)
    
    template = DataFrame(
        id = 1:N,
        tstart = zeros(N),
        tstop = fill(MAX_TIME, N),
        statefrom = ones(Int, N),
        stateto = ones(Int, N),
        obstype = ones(Int, N),
        x = x_vals
    )
    
    model_sim = multistatemodel(h12, h23, h13; data=template)
    set_parameters!(model_sim, true_params)
    
    sim_result = simulate(model_sim; paths=false, data=true, nsim=1)
    exact_data = sim_result[1]
    result.n_subjects = length(unique(exact_data.id))
    
    model_fit = multistatemodel(h12, h23, h13; data=exact_data)
    
    start_time = time()
    fitted = fit(model_fit; parallel=true, verbose=false,
                 compute_vcov=true, compute_ij_vcov=true)
    result.runtime_seconds = time() - start_time
    
    params_flat = get_parameters_flat(fitted)
    param_names = ["h12_shape", "h12_scale", "h12_beta", "h13_shape", "h13_scale", "h13_beta", "h23_shape", "h23_scale", "h23_beta"]
    
    vcov = fitted.vcov
    ses = !isnothing(vcov) ? sqrt.(diag(vcov)) : fill(NaN, 9)
    
    for (i, pname) in enumerate(param_names)
        result.est_params[pname] = params_flat[i]
        result.rel_errors[pname] = compute_relative_error(result.true_params[pname], params_flat[i])
        result.se_params[pname] = ses[i]
        ci = extract_ci(ses[i], params_flat[i])
        result.ci_lower[pname] = ci[1]
        result.ci_upper[pname] = ci[2]
    end
    
    result.converged = true
    result.loglik = fitted.loglik.loglik
    result.eval_times = EVAL_TIMES
    
    # Compute observed prevalence and cumulative incidence from the data
    result.prevalence_observed = compute_prevalence_from_data(exact_data, EVAL_TIMES, 3)
    result.cumincid_observed = Dict(
        "1→2" => compute_cumincid_from_data(exact_data, EVAL_TIMES, 1, 2),
        "1→3" => compute_cumincid_from_data(exact_data, EVAL_TIMES, 1, 3),
        "2→3" => compute_cumincid_from_data(exact_data, EVAL_TIMES, 2, 3)
    )
    
    # Simulate from true model (using x=0 for prevalence comparison)
    template_x0 = DataFrame(
        id = 1:N_SIM_TRAJ,
        tstart = zeros(N_SIM_TRAJ),
        tstop = fill(MAX_TIME, N_SIM_TRAJ),
        statefrom = ones(Int, N_SIM_TRAJ),
        stateto = ones(Int, N_SIM_TRAJ),
        obstype = ones(Int, N_SIM_TRAJ),
        x = zeros(N_SIM_TRAJ)
    )
    model_sim_x0 = multistatemodel(h12, h23, h13; data=template_x0)
    set_parameters!(model_sim_x0, true_params)
    
    true_paths_nested = simulate(model_sim_x0; paths=true, data=false, nsim=1)
    true_paths = true_paths_nested[1]
    result.prevalence_true = compute_state_prevalence(true_paths, EVAL_TIMES, 3)
    
    fitted_paths_nested = simulate(fitted; paths=true, data=false, nsim=1, tmax=MAX_TIME)
    fitted_paths = fitted_paths_nested[1]
    result.prevalence_fitted = compute_state_prevalence(fitted_paths, EVAL_TIMES, 3)
    
    result.cumincid_true = Dict(
        "1→2" => compute_cumulative_incidence(true_paths, EVAL_TIMES, 1, 2),
        "1→3" => compute_cumulative_incidence(true_paths, EVAL_TIMES, 1, 3),
        "2→3" => compute_cumulative_incidence(true_paths, EVAL_TIMES, 2, 3)
    )
    result.cumincid_fitted = Dict(
        "1→2" => compute_cumulative_incidence(fitted_paths, EVAL_TIMES, 1, 2),
        "1→3" => compute_cumulative_incidence(fitted_paths, EVAL_TIMES, 1, 3),
        "2→3" => compute_cumulative_incidence(fitted_paths, EVAL_TIMES, 2, 3)
    )
    
    push!(ALL_RESULTS, result)
    return result
end

# =============================================================================
# Phase-Type Hazard Model Tests
# =============================================================================

"""
Helper to map phases to observed states for phase-type models.
For illness-death with n_phases phases per transient state:
- Phases 1:n_phases → State 1 (Healthy)
- Phases (n_phases+1):2*n_phases → State 2 (Ill)
- Phase 2*n_phases+1 → State 3 (Dead, absorbing)
"""
function make_phase_to_state_map(n_phases::Int)
    n_dead_phase = 2 * n_phases + 1
    phase_to_state = Dict{Int, Int}()
    for p in 1:n_phases
        phase_to_state[p] = 1  # Healthy phases
    end
    for p in (n_phases + 1):(2 * n_phases)
        phase_to_state[p] = 2  # Ill phases
    end
    phase_to_state[n_dead_phase] = 3  # Dead
    return phase_to_state
end

"""
Compute prevalence from phase-type data, mapping phases to observed states.
"""
function compute_prevalence_from_data_phasetype(exact_data::DataFrame, eval_times::Vector{Float64}, 
                                                 n_states::Int, phase_to_state::Dict{Int, Int})
    n_times = length(eval_times)
    prevalence = zeros(Float64, n_times, n_states)
    
    subjects = unique(exact_data.id)
    n_subjects = length(subjects)
    
    for subj_id in subjects
        subj_data = filter(row -> row.id == subj_id, exact_data)
        
        for (t_idx, t) in enumerate(eval_times)
            state = nothing
            for row in eachrow(subj_data)
                if row.tstart <= t < row.tstop
                    phase = row.statefrom
                    state = get(phase_to_state, phase, phase)
                    break
                elseif t >= row.tstop
                    phase = row.stateto
                    state = get(phase_to_state, phase, phase)
                end
            end
            
            if !isnothing(state) && state <= n_states
                prevalence[t_idx, state] += 1.0
            end
        end
    end
    
    prevalence ./= n_subjects
    return prevalence
end

"""
Compute cumulative incidence from phase-type data for transitions between observed state groups.
"""
function compute_cumincid_from_data_phasetype(exact_data::DataFrame, eval_times::Vector{Float64},
                                               from_phases::Vector{Int}, to_phases::Vector{Int},
                                               phase_to_state::Dict{Int, Int})
    n_times = length(eval_times)
    cumincid = zeros(Float64, n_times)
    
    subjects = unique(exact_data.id)
    n_subjects = length(subjects)
    
    for subj_id in subjects
        subj_data = filter(row -> row.id == subj_id, exact_data)
        
        # Find first transition from from_phases to to_phases
        transition_time = Inf
        for row in eachrow(subj_data)
            if row.statefrom in from_phases && row.stateto in to_phases
                transition_time = row.tstop
                break
            end
        end
        
        for (t_idx, t) in enumerate(eval_times)
            if transition_time <= t
                cumincid[t_idx] += 1.0
            end
        end
    end
    
    cumincid ./= n_subjects
    return cumincid
end

"""
Compute state prevalence collapsing phase space to observed states.
"""
function compute_state_prevalence_phasetype(paths::Vector{SamplePath}, eval_times::Vector{Float64}, 
                                            n_states::Int, phase_to_state::Dict{Int, Int})
    n_times = length(eval_times)
    prevalence = zeros(Float64, n_times, n_states)
    n_paths = length(paths)
    
    for path in paths
        for (t_idx, t) in enumerate(eval_times)
            state_idx = searchsortedlast(path.times, t)
            if state_idx >= 1
                phase = path.states[state_idx]
                state = get(phase_to_state, phase, phase)  # Map or keep as-is
                if state <= n_states
                    prevalence[t_idx, state] += 1.0
                end
            end
        end
    end
    
    prevalence ./= n_paths
    return prevalence
end

"""
    run_phasetype_exact_test()

Test phase-type hazard model with exact data (2 phases per transient state).
Phase-type models are Markov on expanded state space → direct MLE.
"""
function run_phasetype_exact_test()
    result = TestResult("Phase-Type Exact (2-phase)", "phasetype")
    
    Random.seed!(RNG_SEED + 100)
    
    n_phases = 2
    n_subjects = N_SUBJECTS
    
    # True parameters for 2-phase Coxian illness-death
    # Phase progression rates (within-state)
    lambda_1 = 0.8   # Phase 1→2 within Healthy
    lambda_2 = 0.6   # Phase 3→4 within Ill
    
    # Exit rates (between-state) from each phase
    mu_12_p1 = 0.3   # Healthy phase 1 → Ill phase 1
    mu_12_p2 = 0.4   # Healthy phase 2 → Ill phase 1  
    mu_13_p1 = 0.1   # Healthy phase 1 → Dead
    mu_13_p2 = 0.15  # Healthy phase 2 → Dead
    mu_23_p3 = 0.25  # Ill phase 1 → Dead
    mu_23_p4 = 0.35  # Ill phase 2 → Dead
    
    # Build phase-type hazards for illness-death model
    # Transitions in expanded space: phases 1,2 (Healthy), phases 3,4 (Ill), phase 5 (Dead)
    h12_exp = Hazard(@formula(0 ~ 1), "exp", 1, 2)  # Phase 1 → Phase 2
    h13_exp = Hazard(@formula(0 ~ 1), "exp", 1, 3)  # Phase 1 → Phase 3 (first Ill phase)
    h15_exp = Hazard(@formula(0 ~ 1), "exp", 1, 5)  # Phase 1 → Dead
    h23_exp = Hazard(@formula(0 ~ 1), "exp", 2, 3)  # Phase 2 → Phase 3
    h25_exp = Hazard(@formula(0 ~ 1), "exp", 2, 5)  # Phase 2 → Dead
    h34_exp = Hazard(@formula(0 ~ 1), "exp", 3, 4)  # Phase 3 → Phase 4
    h35_exp = Hazard(@formula(0 ~ 1), "exp", 3, 5)  # Phase 3 → Dead
    h45_exp = Hazard(@formula(0 ~ 1), "exp", 4, 5)  # Phase 4 → Dead
    
    true_params = (
        h12 = [log(lambda_1)],
        h13 = [log(mu_12_p1)],
        h15 = [log(mu_13_p1)],
        h23 = [log(mu_12_p2)],
        h25 = [log(mu_13_p2)],
        h34 = [log(lambda_2)],
        h35 = [log(mu_23_p3)],
        h45 = [log(mu_23_p4)]
    )
    
    # Store true params in result
    result.true_params["λ₁ (Healthy 1→2)"] = log(lambda_1)
    result.true_params["μ₁₂_p1 (H1→Ill)"] = log(mu_12_p1)
    result.true_params["μ₁₃_p1 (H1→Dead)"] = log(mu_13_p1)
    result.true_params["μ₁₂_p2 (H2→Ill)"] = log(mu_12_p2)
    result.true_params["μ₁₃_p2 (H2→Dead)"] = log(mu_13_p2)
    result.true_params["λ₂ (Ill 3→4)"] = log(lambda_2)
    result.true_params["μ₂₃_p3 (I3→Dead)"] = log(mu_23_p3)
    result.true_params["μ₂₃_p4 (I4→Dead)"] = log(mu_23_p4)
    
    # Generate template
    template = DataFrame(
        id = 1:n_subjects,
        tstart = zeros(n_subjects),
        tstop = fill(MAX_TIME, n_subjects),
        statefrom = ones(Int, n_subjects),
        stateto = ones(Int, n_subjects),
        obstype = ones(Int, n_subjects)
    )
    
    # Build and simulate
    model_sim = multistatemodel(h12_exp, h13_exp, h15_exp, h23_exp, h25_exp, 
                                 h34_exp, h35_exp, h45_exp; data=template)
    set_parameters!(model_sim, true_params)
    
    sim_result = simulate(model_sim; paths=false, data=true, nsim=1)
    exact_data = sim_result[1]
    result.n_subjects = length(unique(exact_data.id))
    
    # Fit model
    model_fit = multistatemodel(h12_exp, h13_exp, h15_exp, h23_exp, h25_exp,
                                 h34_exp, h35_exp, h45_exp; data=exact_data)
    
    start_time = time()
    fitted = fit(model_fit; parallel=true, verbose=false, 
                 compute_vcov=true, compute_ij_vcov=false)
    result.runtime_seconds = time() - start_time
    
    # Extract estimates
    params_flat = get_parameters_flat(fitted)
    param_names = ["λ₁ (Healthy 1→2)", "μ₁₂_p1 (H1→Ill)", "μ₁₃_p1 (H1→Dead)",
                   "μ₁₂_p2 (H2→Ill)", "μ₁₃_p2 (H2→Dead)", "λ₂ (Ill 3→4)",
                   "μ₂₃_p3 (I3→Dead)", "μ₂₃_p4 (I4→Dead)"]
    
    vcov = fitted.vcov
    ses = !isnothing(vcov) ? sqrt.(diag(vcov)) : fill(NaN, length(params_flat))
    
    for (i, pname) in enumerate(param_names)
        result.est_params[pname] = params_flat[i]
        result.rel_errors[pname] = compute_relative_error(result.true_params[pname], params_flat[i])
        result.se_params[pname] = ses[i]
        ci = extract_ci(ses[i], params_flat[i])
        result.ci_lower[pname] = ci[1]
        result.ci_upper[pname] = ci[2]
    end
    
    result.converged = true
    result.loglik = fitted.loglik.loglik
    result.eval_times = EVAL_TIMES
    
    # Simulate paths for prevalence (need to map phases to observed states)
    phase_to_state = make_phase_to_state_map(n_phases)
    
    # Compute observed prevalence and cumulative incidence from the data (phase space → observed space)
    result.prevalence_observed = compute_prevalence_from_data_phasetype(exact_data, EVAL_TIMES, 3, phase_to_state)
    result.cumincid_observed = Dict(
        "1→2" => compute_cumincid_from_data_phasetype(exact_data, EVAL_TIMES, [1, 2], [3, 4], phase_to_state),
        "1→3" => compute_cumincid_from_data_phasetype(exact_data, EVAL_TIMES, [1, 2], [5], phase_to_state),
        "2→3" => compute_cumincid_from_data_phasetype(exact_data, EVAL_TIMES, [3, 4], [5], phase_to_state)
    )
    
    true_paths_nested = simulate(model_sim; paths=true, data=false, nsim=1)
    true_paths = true_paths_nested[1]
    result.prevalence_true = compute_state_prevalence_phasetype(true_paths, EVAL_TIMES, 3, phase_to_state)
    
    fitted_paths_nested = simulate(fitted; paths=true, data=false, nsim=1, tmax=MAX_TIME)
    fitted_paths = fitted_paths_nested[1]
    result.prevalence_fitted = compute_state_prevalence_phasetype(fitted_paths, EVAL_TIMES, 3, phase_to_state)
    
    # For phase-type, cumulative incidence needs special handling (transitions between observed states)
    # We track when individuals first enter Ill (phase 3 or 4) and Dead (phase 5)
    result.cumincid_true = Dict(
        "1→2" => compute_phasetype_cumincid(true_paths, EVAL_TIMES, [1, 2], [3, 4]),
        "1→3" => compute_phasetype_cumincid(true_paths, EVAL_TIMES, [1, 2], [5]),
        "2→3" => compute_phasetype_cumincid(true_paths, EVAL_TIMES, [3, 4], [5])
    )
    result.cumincid_fitted = Dict(
        "1→2" => compute_phasetype_cumincid(fitted_paths, EVAL_TIMES, [1, 2], [3, 4]),
        "1→3" => compute_phasetype_cumincid(fitted_paths, EVAL_TIMES, [1, 2], [5]),
        "2→3" => compute_phasetype_cumincid(fitted_paths, EVAL_TIMES, [3, 4], [5])
    )
    
    push!(ALL_RESULTS, result)
    return result
end

"""
Compute cumulative incidence for phase-type models.
Track proportion who transitioned from any phase in `from_phases` to any phase in `to_phases`.
"""
function compute_phasetype_cumincid(paths::Vector{SamplePath}, eval_times::Vector{Float64},
                                     from_phases::Vector{Int}, to_phases::Vector{Int})
    n_times = length(eval_times)
    cumincid = zeros(Float64, n_times)
    n_paths = length(paths)
    
    for path in paths
        # Find first transition from from_phases to to_phases
        first_trans_time = Inf
        for i in 1:(length(path.states) - 1)
            if path.states[i] in from_phases && path.states[i + 1] in to_phases
                first_trans_time = path.times[i + 1]
                break
            end
        end
        
        # Count as event at each eval time >= first_trans_time
        for (t_idx, t) in enumerate(eval_times)
            if t >= first_trans_time
                cumincid[t_idx] += 1.0
            end
        end
    end
    
    cumincid ./= n_paths
    return cumincid
end

"""
    run_phasetype_panel_test()

Test phase-type hazard model with panel data (observations at fixed times).
"""
function run_phasetype_panel_test()
    result = TestResult("Phase-Type Panel (2-phase)", "phasetype")
    
    Random.seed!(RNG_SEED + 200)
    
    n_phases = 2
    n_subjects = N_SUBJECTS
    panel_times = [0.0, 2.5, 5.0, 7.5, 10.0]
    
    # Same true parameters as exact test
    lambda_1 = 0.8
    lambda_2 = 0.6
    mu_12_p1 = 0.3
    mu_12_p2 = 0.4  
    mu_13_p1 = 0.1
    mu_13_p2 = 0.15
    mu_23_p3 = 0.25
    mu_23_p4 = 0.35
    
    h12_exp = Hazard(@formula(0 ~ 1), "exp", 1, 2)
    h13_exp = Hazard(@formula(0 ~ 1), "exp", 1, 3)
    h15_exp = Hazard(@formula(0 ~ 1), "exp", 1, 5)
    h23_exp = Hazard(@formula(0 ~ 1), "exp", 2, 3)
    h25_exp = Hazard(@formula(0 ~ 1), "exp", 2, 5)
    h34_exp = Hazard(@formula(0 ~ 1), "exp", 3, 4)
    h35_exp = Hazard(@formula(0 ~ 1), "exp", 3, 5)
    h45_exp = Hazard(@formula(0 ~ 1), "exp", 4, 5)
    
    true_params = (
        h12 = [log(lambda_1)],
        h13 = [log(mu_12_p1)],
        h15 = [log(mu_13_p1)],
        h23 = [log(mu_12_p2)],
        h25 = [log(mu_13_p2)],
        h34 = [log(lambda_2)],
        h35 = [log(mu_23_p3)],
        h45 = [log(mu_23_p4)]
    )
    
    result.true_params["λ₁ (Healthy 1→2)"] = log(lambda_1)
    result.true_params["μ₁₂_p1 (H1→Ill)"] = log(mu_12_p1)
    result.true_params["μ₁₃_p1 (H1→Dead)"] = log(mu_13_p1)
    result.true_params["μ₁₂_p2 (H2→Ill)"] = log(mu_12_p2)
    result.true_params["μ₁₃_p2 (H2→Dead)"] = log(mu_13_p2)
    result.true_params["λ₂ (Ill 3→4)"] = log(lambda_2)
    result.true_params["μ₂₃_p3 (I3→Dead)"] = log(mu_23_p3)
    result.true_params["μ₂₃_p4 (I4→Dead)"] = log(mu_23_p4)
    
    # Generate exact data template
    template = DataFrame(
        id = 1:n_subjects,
        tstart = zeros(n_subjects),
        tstop = fill(MAX_TIME, n_subjects),
        statefrom = ones(Int, n_subjects),
        stateto = ones(Int, n_subjects),
        obstype = ones(Int, n_subjects)
    )
    
    model_sim = multistatemodel(h12_exp, h13_exp, h15_exp, h23_exp, h25_exp,
                                 h34_exp, h35_exp, h45_exp; data=template)
    set_parameters!(model_sim, true_params)
    
    # Simulate paths and convert to panel data
    sim_paths = simulate(model_sim; paths=true, data=false, nsim=1)[1]
    
    # Create panel data by observing at fixed times
    phase_to_state = make_phase_to_state_map(n_phases)
    
    panel_rows = DataFrame[]
    for (subj_id, path) in enumerate(sim_paths)
        for i in 1:(length(panel_times) - 1)
            t_start = panel_times[i]
            t_stop = panel_times[i + 1]
            
            # Get state at t_start and t_stop
            idx_start = searchsortedlast(path.times, t_start)
            idx_stop = searchsortedlast(path.times, t_stop)
            
            if idx_start >= 1 && idx_stop >= 1
                phase_start = path.states[idx_start]
                phase_stop = path.states[idx_stop]
                state_start = phase_to_state[phase_start]
                state_stop = phase_to_state[phase_stop]
                
                # Only include if not already absorbed at start
                if state_start < 3
                    push!(panel_rows, DataFrame(
                        id = subj_id,
                        tstart = t_start,
                        tstop = t_stop,
                        statefrom = state_start,
                        stateto = state_stop,
                        obstype = 2  # Panel observation
                    ))
                end
            end
        end
    end
    
    panel_data = vcat(panel_rows...)
    result.n_subjects = length(unique(panel_data.id))
    
    # For panel data with phase-type, we need to fit on observed state space
    # and use Markov surrogate (since we observe collapsed states)
    h12_obs = Hazard(@formula(0 ~ 1), "exp", 1, 2)
    h13_obs = Hazard(@formula(0 ~ 1), "exp", 1, 3)
    h23_obs = Hazard(@formula(0 ~ 1), "exp", 2, 3)
    
    model_fit = multistatemodel(h12_obs, h13_obs, h23_obs; data=panel_data)
    
    start_time = time()
    fitted = fit(model_fit; parallel=true, verbose=false,
                 compute_vcov=true, compute_ij_vcov=false)
    result.runtime_seconds = time() - start_time
    
    # For panel test, we compare observed-space rates (not phase-space)
    # Clear and reset true_params for observed space
    empty!(result.true_params)
    empty!(result.est_params)
    empty!(result.rel_errors)
    empty!(result.se_params)
    empty!(result.ci_lower)
    empty!(result.ci_upper)
    
    # Approximate observed-space rates from phase-type structure
    # These are effective rates (not directly comparable to phase rates)
    result.true_params["h12 (Healthy→Ill)"] = NaN  # Complex function of phase params
    result.true_params["h13 (Healthy→Dead)"] = NaN
    result.true_params["h23 (Ill→Dead)"] = NaN
    
    params_flat = get_parameters_flat(fitted)
    param_names = ["h12 (Healthy→Ill)", "h13 (Healthy→Dead)", "h23 (Ill→Dead)"]
    
    vcov = fitted.vcov
    ses = !isnothing(vcov) ? sqrt.(diag(vcov)) : fill(NaN, length(params_flat))
    
    for (i, pname) in enumerate(param_names)
        result.est_params[pname] = params_flat[i]
        result.rel_errors[pname] = NaN  # Can't compute rel error without true observed-space rates
        result.se_params[pname] = ses[i]
        ci = extract_ci(ses[i], params_flat[i])
        result.ci_lower[pname] = ci[1]
        result.ci_upper[pname] = ci[2]
    end
    
    result.converged = true
    result.loglik = fitted.loglik.loglik
    result.eval_times = EVAL_TIMES
    
    # Compute observed from original simulated phase-type paths (true process)
    # Note: panel_data is only at panel observation times, not the full trajectory
    result.prevalence_observed = compute_state_prevalence_phasetype(sim_paths, EVAL_TIMES, 3, phase_to_state)
    result.cumincid_observed = Dict(
        "1→2" => compute_phasetype_cumincid(sim_paths, EVAL_TIMES, [1, 2], [3, 4]),
        "1→3" => compute_phasetype_cumincid(sim_paths, EVAL_TIMES, [1, 2], [5]),
        "2→3" => compute_phasetype_cumincid(sim_paths, EVAL_TIMES, [3, 4], [5])
    )
    
    # Prevalence comparison
    result.prevalence_true = compute_state_prevalence_phasetype(sim_paths, EVAL_TIMES, 3, phase_to_state)
    
    # Simulate from fitted (observed-space Markov) model
    fitted_sim_template = DataFrame(
        id = 1:N_SIM_TRAJ,
        tstart = zeros(N_SIM_TRAJ),
        tstop = fill(MAX_TIME, N_SIM_TRAJ),
        statefrom = ones(Int, N_SIM_TRAJ),
        stateto = ones(Int, N_SIM_TRAJ),
        obstype = ones(Int, N_SIM_TRAJ)
    )
    model_fitted_sim = multistatemodel(h12_obs, h13_obs, h23_obs; data=fitted_sim_template)
    set_parameters!(model_fitted_sim, get_parameters(fitted))
    
    fitted_paths = simulate(model_fitted_sim; paths=true, data=false, nsim=1, tmax=MAX_TIME)[1]
    result.prevalence_fitted = compute_state_prevalence(fitted_paths, EVAL_TIMES, 3)
    
    # Cumulative incidence
    result.cumincid_true = Dict(
        "1→2" => compute_phasetype_cumincid(sim_paths, EVAL_TIMES, [1, 2], [3, 4]),
        "1→3" => compute_phasetype_cumincid(sim_paths, EVAL_TIMES, [1, 2], [5]),
        "2→3" => compute_phasetype_cumincid(sim_paths, EVAL_TIMES, [3, 4], [5])
    )
    result.cumincid_fitted = Dict(
        "1→2" => compute_cumulative_incidence(fitted_paths, EVAL_TIMES, 1, 2),
        "1→3" => compute_cumulative_incidence(fitted_paths, EVAL_TIMES, 1, 3),
        "2→3" => compute_cumulative_incidence(fitted_paths, EVAL_TIMES, 2, 3)
    )
    
    push!(ALL_RESULTS, result)
    return result
end

# =============================================================================
# Report Generation
# =============================================================================

"""
    generate_parameter_table(results)

Generate markdown table of parameter comparisons with relative errors.
"""
function generate_parameter_table(results::Vector{TestResult})
    lines = String[]
    
    push!(lines, "| Test | Parameter | True | Estimate | Rel. Error (%) | SE | 95% CI |")
    push!(lines, "|------|-----------|------|----------|----------------|-----|--------|")
    
    for result in results
        first_param = true
        for param in sort(collect(keys(result.true_params)))
            true_val = result.true_params[param]
            est_val = get(result.est_params, param, NaN)
            rel_err = get(result.rel_errors, param, NaN)
            se_val = get(result.se_params, param, NaN)
            ci_lo = get(result.ci_lower, param, NaN)
            ci_hi = get(result.ci_upper, param, NaN)
            
            test_name = first_param ? result.test_name : ""
            
            push!(lines, @sprintf("| %s | %s | %.4f | %.4f | %.2f | %.4f | (%.4f, %.4f) |",
                test_name, param, true_val, est_val, rel_err, se_val, ci_lo, ci_hi))
            
            first_param = false
        end
    end
    
    return join(lines, "\n")
end

"""
    generate_summary_table(results)

Generate markdown summary table.
"""
function generate_summary_table(results::Vector{TestResult})
    lines = String[]
    
    push!(lines, "| Test | Family | N | Log-lik | Runtime (s) | Max |Rel. Error| (%) |")
    push!(lines, "|------|--------|---|---------|-------------|---------------------|")
    
    for result in results
        max_err = maximum(abs.(values(result.rel_errors)))
        push!(lines, @sprintf("| %s | %s | %d | %.2f | %.2f | %.2f |",
            result.test_name, result.hazard_family, result.n_subjects,
            result.loglik, result.runtime_seconds, max_err))
    end
    
    return join(lines, "\n")
end

"""
    generate_report(results)

Generate the full markdown report.
"""
function generate_report(results::Vector{TestResult})
    timestamp = Dates.format(now(), "yyyy-mm-dd HH:MM:SS")
    julia_version = string(VERSION)
    nthreads = Threads.nthreads()
    
    report = """
---
title: "Inference Long Tests - Illness-Death Model"
format:
    html:
        theme:
            light: flatly
            dark: darkly
        highlight-style: atom-one-dark
---

# Inference Long Tests - Illness-Death Model

_Generated: $(timestamp)_

_Julia: $(julia_version), Threads: $(nthreads)_

This document contains results from inference validation tests using a 3-state 
illness-death model:

```
Healthy (1) ──→ Ill (2) ──→ Dead (3)
     │                         ↑
     └─────────────────────────┘
```

All tests use n = $(N_SUBJECTS) subjects.

---

## Summary

$(generate_summary_table(results))

---

## Parameter Estimates and Relative Errors

$(generate_parameter_table(results))

---

## Diagnostic Plots

"""
    
    # Add plot references
    for result in results
        slug = replace(lowercase(result.test_name), " " => "_", "+" => "")
        slug = replace(slug, "__" => "_")
        
        report *= """
### $(result.test_name)

"""
        
        if HAS_PLOTS
            report *= """
#### State Prevalence
![](assets/diagnostics/prevalence_$(slug).png)

#### Cumulative Incidence
![](assets/diagnostics/cumincid_$(slug).png)

"""
        else
            report *= "_Plots not generated (CairoMakie not available)_\n\n"
        end
    end
    
    report *= """

---

## Test Configuration

- RNG Seed: $(RNG_SEED)
- Sample Size: $(N_SUBJECTS)
- Simulation Trajectories: $(N_SIM_TRAJ)
- Max Follow-up Time: $(MAX_TIME)
- Parallel Enabled: $(Threads.nthreads() > 1 ? "Yes ($(Threads.nthreads()) threads)" : "No")

"""
    
    return report
end

# =============================================================================
# Main Execution
# =============================================================================

function main()
    println("=" ^ 70)
    println("Running Inference Long Tests - Illness-Death Model")
    println("Julia $(VERSION), Threads: $(Threads.nthreads())")
    println("Sample sizes: $(N_SUBJECTS) (exp/wei), $(N_SUBJECTS_GOMPERTZ) (gom), $(N_SUBJECTS_COVARIATE) (cov)")
    println("=" ^ 70)
    println()
    flush(stdout)
    
    # Run tests
    println("1. Running Exponential test...")
    flush(stdout)
    run_exponential_test()
    max_err = maximum(abs.(values(ALL_RESULTS[end].rel_errors)))
    println("   ✓ Completed in $(round(ALL_RESULTS[end].runtime_seconds; digits=2))s, max |rel. error| = $(round(max_err; digits=2))%")
    flush(stdout)
    
    println("2. Running Weibull test...")
    flush(stdout)
    run_weibull_test()
    max_err = maximum(abs.(values(ALL_RESULTS[end].rel_errors)))
    println("   ✓ Completed in $(round(ALL_RESULTS[end].runtime_seconds; digits=2))s, max |rel. error| = $(round(max_err; digits=2))%")
    flush(stdout)
    
    println("3. Running Gompertz test...")
    flush(stdout)
    run_gompertz_test()
    max_err = maximum(abs.(values(ALL_RESULTS[end].rel_errors)))
    println("   ✓ Completed in $(round(ALL_RESULTS[end].runtime_seconds; digits=2))s, max |rel. error| = $(round(max_err; digits=2))%")
    flush(stdout)
    
    println("4. Running Exponential + Covariate test (PH)...")
    flush(stdout)
    run_exponential_covariate_test()
    max_err = maximum(abs.(values(ALL_RESULTS[end].rel_errors)))
    println("   ✓ Completed in $(round(ALL_RESULTS[end].runtime_seconds; digits=2))s, max |rel. error| = $(round(max_err; digits=2))%")
    flush(stdout)
    
    println("5. Running Weibull AFT test...")
    flush(stdout)
    run_weibull_aft_test()
    max_err = maximum(abs.(values(ALL_RESULTS[end].rel_errors)))
    println("   ✓ Completed in $(round(ALL_RESULTS[end].runtime_seconds; digits=2))s, max |rel. error| = $(round(max_err; digits=2))%")
    flush(stdout)
    
    println("6. Running Phase-Type Exact test...")
    flush(stdout)
    run_phasetype_exact_test()
    max_err = maximum(abs.(values(ALL_RESULTS[end].rel_errors)))
    println("   ✓ Completed in $(round(ALL_RESULTS[end].runtime_seconds; digits=2))s, max |rel. error| = $(round(max_err; digits=2))%")
    flush(stdout)
    
    println("7. Running Phase-Type Panel test...")
    flush(stdout)
    run_phasetype_panel_test()
    # Panel test may have NaN rel errors (can't compare to true observed-space rates)
    rel_errs = collect(filter(!isnan, collect(values(ALL_RESULTS[end].rel_errors))))
    if !isempty(rel_errs)
        max_err = maximum(abs.(rel_errs))
        println("   ✓ Completed in $(round(ALL_RESULTS[end].runtime_seconds; digits=2))s, max |rel. error| = $(round(max_err; digits=2))%")
    else
        println("   ✓ Completed in $(round(ALL_RESULTS[end].runtime_seconds; digits=2))s (phase-type panel: rel. error N/A)")
    end
    flush(stdout)
    
    println()
    println("Generating plots...")
    flush(stdout)
    
    # Generate plots if CairoMakie available
    if HAS_PLOTS
        for result in ALL_RESULTS
            slug = replace(lowercase(result.test_name), " " => "_", "+" => "")
            slug = replace(slug, "__" => "_")
            
            # Prevalence plot
            if !isnothing(result.prevalence_true)
                fig = plot_prevalence_comparison(result, 3)
                save(joinpath(OUTPUT_DIR, "prevalence_$(slug).png"), fig)
                println("   Saved prevalence_$(slug).png")
            end
            
            # Cumulative incidence plot
            if !isnothing(result.cumincid_true)
                fig = plot_cumulative_incidence(result)
                if !isnothing(fig)
                    save(joinpath(OUTPUT_DIR, "cumincid_$(slug).png"), fig)
                    println("   Saved cumincid_$(slug).png")
                end
            end
        end
    end
    flush(stdout)
    
    println()
    println("Generating report...")
    flush(stdout)
    
    # Generate report
    report = generate_report(ALL_RESULTS)
    open(REPORT_FILE, "w") do f
        write(f, report)
    end
    println("   Report written to: $REPORT_FILE")
    
    println()
    println("=" ^ 70)
    println("All tests completed!")
    println("=" ^ 70)
    
    # Print summary
    println()
    println("Summary of Relative Errors:")
    for result in ALL_RESULTS
        max_err = maximum(abs.(values(result.rel_errors)))
        println("  $(result.test_name): max |rel. error| = $(round(max_err; digits=2))%")
    end
    flush(stdout)
end

# Always run main when included
main()
