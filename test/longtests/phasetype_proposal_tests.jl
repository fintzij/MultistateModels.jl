# =============================================================================
# Phase-Type PROPOSAL Tests (for Semi-Markov MCEM)
# =============================================================================
#
# These tests validate phase-type PROPOSALS for fitting semi-Markov models
# (Weibull, Gompertz) using MCEM with importance sampling.
#
# Key distinction:
# - This file tests phase-type PROPOSALS → Weibull/Gompertz targets → MCEM
# - phasetype_hazard_tests.jl tests :pt HAZARD FAMILY → Direct MLE
#
# The target models here are Weibull/Gompertz (semi-Markov), NOT phase-type.
# The phase-type is used as a proposal distribution for importance sampling.
# =============================================================================

using MultistateModels
using Distributions
using DataFrames
using Random
using Test

# include("common.jl")

# =============================================================================
# Test 31: Weibull, No Covariates, Panel Data, Heuristic Phase-Type Proposal
# =============================================================================

"""
    run_ph_wei_nocov_panel_heuristic()

Test Weibull model with no covariates and panel data.
Fitted using MCEM with Phase-Type proposal (heuristic).
"""
function run_ph_wei_nocov_panel_heuristic()
    test_name = "ph_wei_nocov_panel_heuristic"
    @info "Running $test_name"

    # True parameters
    α_12_true = 1.5
    λ_12_true = 0.15
    α_23_true = 1.3
    λ_23_true = 0.20

    # Create model for simulation
    h12 = Hazard(@formula(0 ~ 1), "wei", 1, 2)
    h23 = Hazard(@formula(0 ~ 1), "wei", 2, 3)

    Random.seed!(12370)
    dat = create_baseline_template(N_SUBJECTS; max_time=MAX_TIME)

    msm_sim = multistatemodel(h12, h23; data=dat)
    set_parameters!(msm_sim,
        (h12 = [log(λ_12_true), log(α_12_true)],
         h23 = [log(λ_23_true), log(α_23_true)]))

    # Simulate exact data and convert to panel
    paths = simulate(msm_sim; paths=true, data=false)[1]
    panel_data = create_panel_data(paths, collect(PANEL_TIMES), 3)

    # Create fitting model
    msm_fit = multistatemodel(h12, h23; data=panel_data)

    # Configure Phase-Type Proposal
    proposal_config = ProposalConfig(
        type = :phasetype,
        n_phases = :heuristic,
        structure = :unstructured,
        max_phases = 5
    )

    # Fit model using MCEM
    fitted = fit(msm_fit;
        proposal=proposal_config,
        tol=MCEM_TOL,
        ess_target_initial=MCEM_ESS_INITIAL,
        ess_target_max=MCEM_ESS_MAX,
        maxiter=MCEM_MAXITER)

    # Extract estimates
    params = get_parameters(fitted)
    λ_12_est = exp(params[:h12][1])
    α_12_est = exp(params[:h12][2])
    λ_23_est = exp(params[:h23][1])
    α_23_est = exp(params[:h23][2])

    # Compute relative errors
    rel_errs = Dict(
        "λ_12" => compute_relative_error(λ_12_true, λ_12_est),
        "α_12" => compute_relative_error(α_12_true, α_12_est),
        "λ_23" => compute_relative_error(λ_23_true, λ_23_est),
        "α_23" => compute_relative_error(α_23_true, α_23_est),
    )

    # Compute prevalence and cumulative incidence
    n_states = 3
    prev_true = compute_state_prevalence(paths, EVAL_TIMES, n_states)
    prev_obs = compute_prevalence_from_data(panel_data, EVAL_TIMES, n_states)

    # Simulate from fitted model
    dat_fit = create_baseline_template(N_SUBJECTS; max_time=MAX_TIME)
    msm_fit_sim = multistatemodel(h12, h23; data=dat_fit)
    set_parameters!(msm_fit_sim, get_parameters(fitted))
    paths_fitted = simulate(msm_fit_sim; paths=true, data=false)[1]
    prev_fitted = compute_state_prevalence(paths_fitted, EVAL_TIMES, n_states)

    cumincid_12_true = compute_cumulative_incidence(paths, EVAL_TIMES, 1, 2)
    cumincid_12_obs = compute_cumincid_from_data(panel_data, EVAL_TIMES, 1, 2)
    cumincid_12_fitted = compute_cumulative_incidence(paths_fitted, EVAL_TIMES, 1, 2)

    cumincid_23_true = compute_cumulative_incidence(paths, EVAL_TIMES, 2, 3)
    cumincid_23_obs = compute_cumincid_from_data(panel_data, EVAL_TIMES, 2, 3)
    cumincid_23_fitted = compute_cumulative_incidence(paths_fitted, EVAL_TIMES, 2, 3)

    # Create result
    result = TestResult(
        name = test_name,
        family = "weibull",
        parameterization = :ph,
        covariates = :none,
        data_type = :panel,
        n_subjects = N_SUBJECTS,
        true_params = Dict(
            "λ_12" => λ_12_true, "α_12" => α_12_true,
            "λ_23" => λ_23_true, "α_23" => α_23_true,
        ),
        estimated_params = Dict(
            "λ_12" => λ_12_est, "α_12" => α_12_est,
            "λ_23" => λ_23_est, "α_23" => α_23_est,
        ),
        rel_errors = rel_errs,
        max_rel_error = NaN,
        passed = false,
        eval_times = EVAL_TIMES,
        prevalence_true = prev_true,
        prevalence_observed = prev_obs,
        prevalence_fitted = prev_fitted,
        cumincid_12_true = cumincid_12_true,
        cumincid_12_observed = cumincid_12_obs,
        cumincid_12_fitted = cumincid_12_fitted,
        cumincid_23_true = cumincid_23_true,
        cumincid_23_observed = cumincid_23_obs,
        cumincid_23_fitted = cumincid_23_fitted,
    )

    finalize_result!(result)
    push!(ALL_RESULTS, result)

    @info "$test_name completed" passed=result.passed max_rel_error=result.max_rel_error
    return result
end

# =============================================================================
# Test 32: Weibull, No Covariates, Panel Data, Auto Phase-Type Proposal
# =============================================================================

"""
    run_ph_wei_nocov_panel_auto()

Test Weibull model with no covariates and panel data.
Fitted using MCEM with Phase-Type proposal (auto selection).
"""
function run_ph_wei_nocov_panel_auto()
    test_name = "ph_wei_nocov_panel_auto"
    @info "Running $test_name"

    # True parameters
    α_12_true = 1.5
    λ_12_true = 0.15
    α_23_true = 1.3
    λ_23_true = 0.20

    # Create model for simulation
    h12 = Hazard(@formula(0 ~ 1), "wei", 1, 2)
    h23 = Hazard(@formula(0 ~ 1), "wei", 2, 3)

    Random.seed!(12371)
    dat = create_baseline_template(N_SUBJECTS; max_time=MAX_TIME)

    msm_sim = multistatemodel(h12, h23; data=dat)
    set_parameters!(msm_sim,
        (h12 = [log(λ_12_true), log(α_12_true)],
         h23 = [log(λ_23_true), log(α_23_true)]))

    # Simulate exact data and convert to panel
    paths = simulate(msm_sim; paths=true, data=false)[1]
    panel_data = create_panel_data(paths, collect(PANEL_TIMES), 3)

    # Create fitting model
    msm_fit = multistatemodel(h12, h23; data=panel_data)

    # Configure Phase-Type Proposal
    proposal_config = ProposalConfig(
        type = :phasetype,
        n_phases = :auto,
        structure = :unstructured,
        max_phases = 5
    )

    # Fit model using MCEM
    fitted = fit(msm_fit;
        proposal=proposal_config,
        tol=MCEM_TOL,
        ess_target_initial=MCEM_ESS_INITIAL,
        ess_target_max=MCEM_ESS_MAX,
        maxiter=MCEM_MAXITER)

    # Extract estimates
    params = get_parameters(fitted)
    λ_12_est = exp(params[:h12][1])
    α_12_est = exp(params[:h12][2])
    λ_23_est = exp(params[:h23][1])
    α_23_est = exp(params[:h23][2])

    # Compute relative errors
    rel_errs = Dict(
        "λ_12" => compute_relative_error(λ_12_true, λ_12_est),
        "α_12" => compute_relative_error(α_12_true, α_12_est),
        "λ_23" => compute_relative_error(λ_23_true, λ_23_est),
        "α_23" => compute_relative_error(α_23_true, α_23_est),
    )

    # Compute prevalence and cumulative incidence
    n_states = 3
    prev_true = compute_state_prevalence(paths, EVAL_TIMES, n_states)
    prev_obs = compute_prevalence_from_data(panel_data, EVAL_TIMES, n_states)

    # Simulate from fitted model
    dat_fit = create_baseline_template(N_SUBJECTS; max_time=MAX_TIME)
    msm_fit_sim = multistatemodel(h12, h23; data=dat_fit)
    set_parameters!(msm_fit_sim, get_parameters(fitted))
    paths_fitted = simulate(msm_fit_sim; paths=true, data=false)[1]
    prev_fitted = compute_state_prevalence(paths_fitted, EVAL_TIMES, n_states)

    cumincid_12_true = compute_cumulative_incidence(paths, EVAL_TIMES, 1, 2)
    cumincid_12_obs = compute_cumincid_from_data(panel_data, EVAL_TIMES, 1, 2)
    cumincid_12_fitted = compute_cumulative_incidence(paths_fitted, EVAL_TIMES, 1, 2)

    cumincid_23_true = compute_cumulative_incidence(paths, EVAL_TIMES, 2, 3)
    cumincid_23_obs = compute_cumincid_from_data(panel_data, EVAL_TIMES, 2, 3)
    cumincid_23_fitted = compute_cumulative_incidence(paths_fitted, EVAL_TIMES, 2, 3)

    # Create result
    result = TestResult(
        name = test_name,
        family = "weibull",
        parameterization = :ph,
        covariates = :none,
        data_type = :panel,
        n_subjects = N_SUBJECTS,
        true_params = Dict(
            "λ_12" => λ_12_true, "α_12" => α_12_true,
            "λ_23" => λ_23_true, "α_23" => α_23_true,
        ),
        estimated_params = Dict(
            "λ_12" => λ_12_est, "α_12" => α_12_est,
            "λ_23" => λ_23_est, "α_23" => α_23_est,
        ),
        rel_errors = rel_errs,
        max_rel_error = NaN,
        passed = false,
        eval_times = EVAL_TIMES,
        prevalence_true = prev_true,
        prevalence_observed = prev_obs,
        prevalence_fitted = prev_fitted,
        cumincid_12_true = cumincid_12_true,
        cumincid_12_observed = cumincid_12_obs,
        cumincid_12_fitted = cumincid_12_fitted,
        cumincid_23_true = cumincid_23_true,
        cumincid_23_observed = cumincid_23_obs,
        cumincid_23_fitted = cumincid_23_fitted,
    )

    finalize_result!(result)
    push!(ALL_RESULTS, result)

    @info "$test_name completed" passed=result.passed max_rel_error=result.max_rel_error
    return result
end

# =============================================================================
# Test 33: Weibull, No Covariates, Panel Data, Manual Phase-Type Proposal (Int)
# =============================================================================

"""
    run_ph_wei_nocov_panel_manual_int()

Test Weibull model with no covariates and panel data.
Fitted using MCEM with Phase-Type proposal (manual n_phases=2).
"""
function run_ph_wei_nocov_panel_manual_int()
    test_name = "ph_wei_nocov_panel_manual_int"
    @info "Running $test_name"

    # True parameters
    α_12_true = 1.5
    λ_12_true = 0.15
    α_23_true = 1.3
    λ_23_true = 0.20

    # Create model for simulation
    h12 = Hazard(@formula(0 ~ 1), "wei", 1, 2)
    h23 = Hazard(@formula(0 ~ 1), "wei", 2, 3)

    Random.seed!(12372)
    dat = create_baseline_template(N_SUBJECTS; max_time=MAX_TIME)

    msm_sim = multistatemodel(h12, h23; data=dat)
    set_parameters!(msm_sim,
        (h12 = [log(λ_12_true), log(α_12_true)],
         h23 = [log(λ_23_true), log(α_23_true)]))

    # Simulate exact data and convert to panel
    paths = simulate(msm_sim; paths=true, data=false)[1]
    panel_data = create_panel_data(paths, collect(PANEL_TIMES), 3)

    # Create fitting model
    msm_fit = multistatemodel(h12, h23; data=panel_data)

    # Create result
    result = TestResult(
        name = test_name,
        family = "weibull",
        parameterization = :ph,
        covariates = :none,
        data_type = :panel,
        n_subjects = N_SUBJECTS,
        true_params = Dict(
            "λ_12" => λ_12_true, "α_12" => α_12_true,
            "λ_23" => λ_23_true, "α_23" => α_23_true,
        ),
        estimated_params = Dict(
            "λ_12" => λ_12_est, "α_12" => α_12_est,
            "λ_23" => λ_23_est, "α_23" => α_23_est,
        ),
        rel_errors = rel_errs,
        max_rel_error = NaN,
        passed = false,
        eval_times = EVAL_TIMES,
        prevalence_true = prev_true,
        prevalence_observed = prev_obs,
        prevalence_fitted = prev_fitted,
        cumincid_12_true = cumincid_12_true,
        cumincid_12_observed = cumincid_12_obs,
        cumincid_12_fitted = cumincid_12_fitted,
        cumincid_23_true = cumincid_23_true,
        cumincid_23_observed = cumincid_23_obs,
        cumincid_23_fitted = cumincid_23_fitted,
    )

    finalize_result!(result)
    push!(ALL_RESULTS, result)

    @info "$test_name completed" passed=result.passed max_rel_error=result.max_rel_error
    return result
end

# =============================================================================
# Test 34: Weibull, No Covariates, Panel Data, Manual Phase-Type Proposal (Vec)
# =============================================================================

"""
    run_ph_wei_nocov_panel_manual_vec()

Test Weibull model with no covariates and panel data.
Fitted using MCEM with Phase-Type proposal (manual n_phases=[2, 2]).
"""
function run_ph_wei_nocov_panel_manual_vec()
    test_name = "ph_wei_nocov_panel_manual_vec"
    @info "Running $test_name"

    # True parameters
    α_12_true = 1.5
    λ_12_true = 0.15
    α_23_true = 1.3
    λ_23_true = 0.20

    # Create model for simulation
    h12 = Hazard(@formula(0 ~ 1), "wei", 1, 2)
    h23 = Hazard(@formula(0 ~ 1), "wei", 2, 3)

    Random.seed!(12373)
    dat = create_baseline_template(N_SUBJECTS; max_time=MAX_TIME)

    msm_sim = multistatemodel(h12, h23; data=dat)
    set_parameters!(msm_sim,
        (h12 = [log(λ_12_true), log(α_12_true)],
         h23 = [log(λ_23_true), log(α_23_true)]))

    # Simulate exact data and convert to panel
    paths = simulate(msm_sim; paths=true, data=false)[1]
    panel_data = create_panel_data(paths, collect(PANEL_TIMES), 3)

    # Create fitting model
    msm_fit = multistatemodel(h12, h23; data=panel_data)

    # Configure Phase-Type Proposal
    # Note: n_phases vector length must match number of transient states?
    # Or number of hazards?
    # Based on ProposalConfig doc: "Vector{Int}: Per-state specification (length = number of transient states)"
    # Here we have states 1 and 2 as transient (assuming 3 is absorbing).
    # So [2, 2] means 2 phases for state 1, 2 phases for state 2.
    proposal_config = ProposalConfig(
        type = :phasetype,
        n_phases = [2, 2],
        structure = :unstructured,
        max_phases = 5
    )

    # Fit model using MCEM
    fitted = fit(msm_fit;
        proposal=proposal_config,
        tol=MCEM_TOL,
        ess_target_initial=MCEM_ESS_INITIAL,
        ess_target_max=MCEM_ESS_MAX,
        maxiter=MCEM_MAXITER)

    # Extract estimates
    params = get_parameters(fitted)
    λ_12_est = exp(params[:h12][1])
    α_12_est = exp(params[:h12][2])
    λ_23_est = exp(params[:h23][1])
    α_23_est = exp(params[:h23][2])

    # Compute relative errors
    rel_errs = Dict(
        "λ_12" => compute_relative_error(λ_12_true, λ_12_est),
        "α_12" => compute_relative_error(α_12_true, α_12_est),
        "λ_23" => compute_relative_error(λ_23_true, λ_23_est),
        "α_23" => compute_relative_error(α_23_true, α_23_est),
    )

    # Compute prevalence and cumulative incidence
    n_states = 3
    prev_true = compute_state_prevalence(paths, EVAL_TIMES, n_states)
    prev_obs = compute_prevalence_from_data(panel_data, EVAL_TIMES, n_states)

    # Simulate from fitted model
    dat_fit = create_baseline_template(N_SUBJECTS; max_time=MAX_TIME)
    msm_fit_sim = multistatemodel(h12, h23; data=dat_fit)
    set_parameters!(msm_fit_sim, get_parameters(fitted))
    paths_fitted = simulate(msm_fit_sim; paths=true, data=false)[1]
    prev_fitted = compute_state_prevalence(paths_fitted, EVAL_TIMES, n_states)

    cumincid_12_true = compute_cumulative_incidence(paths, EVAL_TIMES, 1, 2)
    cumincid_12_obs = compute_cumincid_from_data(panel_data, EVAL_TIMES, 1, 2)
    cumincid_12_fitted = compute_cumulative_incidence(paths_fitted, EVAL_TIMES, 1, 2)

    cumincid_23_true = compute_cumulative_incidence(paths, EVAL_TIMES, 2, 3)
    cumincid_23_obs = compute_cumincid_from_data(panel_data, EVAL_TIMES, 2, 3)
    cumincid_23_fitted = compute_cumulative_incidence(paths_fitted, EVAL_TIMES, 2, 3)

    # Create result
    result = TestResult(
        name = test_name,
        family = "weibull",
        parameterization = :ph,
        covariates = :none,
        data_type = :panel,
        n_subjects = N_SUBJECTS,
        true_params = Dict(
            "λ_12" => λ_12_true, "α_12" => α_12_true,
            "λ_23" => λ_23_true, "α_23" => α_23_true,
        ),
        estimated_params = Dict(
            "λ_12" => λ_12_est, "α_12" => α_12_est,
            "λ_23" => λ_23_est, "α_23" => α_23_est,
        ),
        rel_errors = rel_errs,
        max_rel_error = NaN,
        passed = false,
        eval_times = EVAL_TIMES,
        prevalence_true = prev_true,
        prevalence_observed = prev_obs,
        prevalence_fitted = prev_fitted,
        cumincid_12_true = cumincid_12_true,
        cumincid_12_observed = cumincid_12_obs,
        cumincid_12_fitted = cumincid_12_fitted,
        cumincid_23_true = cumincid_23_true,
        cumincid_23_observed = cumincid_23_obs,
        cumincid_23_fitted = cumincid_23_fitted,
    )

    finalize_result!(result)
    push!(ALL_RESULTS, result)

    @info "$test_name completed" passed=result.passed max_rel_error=result.max_rel_error
    return result
end

# =============================================================================
# Test 35: Weibull, PH + Time-Fixed Covariate, Panel Data, Heuristic Phase-Type
# =============================================================================

"""
    run_ph_wei_ph_tfc_panel_heuristic()

Test Weibull model with proportional hazards and time-fixed covariate, panel data.
Fitted using MCEM with Phase-Type proposal (heuristic).
"""
function run_ph_wei_ph_tfc_panel_heuristic()
    test_name = "ph_wei_ph_tfc_panel_heuristic"
    @info "Running $test_name"

    # True parameters
    α_12_true = 1.5
    λ_12_true = 0.15
    β_12_true = 0.5
    α_23_true = 1.3
    λ_23_true = 0.20
    β_23_true = 0.3

    # Create model for simulation
    h12 = Hazard(@formula(0 ~ 1 + x), "wei", 1, 2; linpred_effect=:ph)
    h23 = Hazard(@formula(0 ~ 1 + x), "wei", 2, 3; linpred_effect=:ph)

    Random.seed!(12374)
    dat = create_tfc_template(N_SUBJECTS; max_time=MAX_TIME)
    x_vals = dat.x

    msm_sim = multistatemodel(h12, h23; data=dat)
    set_parameters!(msm_sim,
        (h12 = [log(λ_12_true), log(α_12_true), β_12_true],
         h23 = [log(λ_23_true), log(α_23_true), β_23_true]))

    # Simulate and convert to panel
    paths = simulate(msm_sim; paths=true, data=false)[1]
    panel_data = create_panel_data_with_covariate(paths, collect(PANEL_TIMES), 3, x_vals)

    # Create fitting model
    msm_fit = multistatemodel(h12, h23; data=panel_data)

    # Configure Phase-Type Proposal
    proposal_config = ProposalConfig(
        type = :phasetype,
        n_phases = :heuristic,
        structure = :unstructured,
        max_phases = 5
    )

    # Fit model using MCEM
    fitted = fit(msm_fit;
        proposal=proposal_config,
        tol=MCEM_TOL,
        ess_target_initial=MCEM_ESS_INITIAL,
        ess_target_max=MCEM_ESS_MAX,
        maxiter=MCEM_MAXITER)

    # Extract estimates
    params = get_parameters(fitted)
    λ_12_est = exp(params[:h12][1])
    α_12_est = exp(params[:h12][2])
    β_12_est = params[:h12][3]
    λ_23_est = exp(params[:h23][1])
    α_23_est = exp(params[:h23][2])
    β_23_est = params[:h23][3]

    # Compute relative errors
    rel_errs = Dict(
        "λ_12" => compute_relative_error(λ_12_true, λ_12_est),
        "α_12" => compute_relative_error(α_12_true, α_12_est),
        "β_12" => compute_relative_error(β_12_true, β_12_est),
        "λ_23" => compute_relative_error(λ_23_true, λ_23_est),
        "α_23" => compute_relative_error(α_23_true, α_23_est),
        "β_23" => compute_relative_error(β_23_true, β_23_est),
    )

    # Compute prevalence and cumulative incidence
    n_states = 3
    prev_true = compute_state_prevalence(paths, EVAL_TIMES, n_states)
    prev_obs = compute_prevalence_from_data(panel_data, EVAL_TIMES, n_states)

    # Simulate from fitted model
    dat_fit = create_tfc_template(N_SUBJECTS; max_time=MAX_TIME)
    msm_fit_sim = multistatemodel(h12, h23; data=dat_fit)
    set_parameters!(msm_fit_sim, get_parameters(fitted))
    paths_fitted = simulate(msm_fit_sim; paths=true, data=false)[1]
    prev_fitted = compute_state_prevalence(paths_fitted, EVAL_TIMES, n_states)

    cumincid_12_true = compute_cumulative_incidence(paths, EVAL_TIMES, 1, 2)
    cumincid_12_obs = compute_cumincid_from_data(panel_data, EVAL_TIMES, 1, 2)
    cumincid_12_fitted = compute_cumulative_incidence(paths_fitted, EVAL_TIMES, 1, 2)

    cumincid_23_true = compute_cumulative_incidence(paths, EVAL_TIMES, 2, 3)
    cumincid_23_obs = compute_cumincid_from_data(panel_data, EVAL_TIMES, 2, 3)
    cumincid_23_fitted = compute_cumulative_incidence(paths_fitted, EVAL_TIMES, 2, 3)

    # Create result
    result = TestResult(
        name = test_name,
        family = "weibull",
        parameterization = :ph,
        covariates = :tfc,
        data_type = :panel,
        n_subjects = N_SUBJECTS,
        true_params = Dict(
            "λ_12" => λ_12_true, "α_12" => α_12_true, "β_12" => β_12_true,
            "λ_23" => λ_23_true, "α_23" => α_23_true, "β_23" => β_23_true,
        ),
        estimated_params = Dict(
            "λ_12" => λ_12_est, "α_12" => α_12_est, "β_12" => β_12_est,
            "λ_23" => λ_23_est, "α_23" => α_23_est, "β_23" => β_23_est,
        ),
        rel_errors = rel_errs,
        max_rel_error = NaN,
        passed = false,
        eval_times = EVAL_TIMES,
        prevalence_true = prev_true,
        prevalence_observed = prev_obs,
        prevalence_fitted = prev_fitted,
        cumincid_12_true = cumincid_12_true,
        cumincid_12_observed = cumincid_12_obs,
        cumincid_12_fitted = cumincid_12_fitted,
        cumincid_23_true = cumincid_23_true,
        cumincid_23_observed = cumincid_23_obs,
        cumincid_23_fitted = cumincid_23_fitted,
    )

    finalize_result!(result)
    push!(ALL_RESULTS, result)

    @info "$test_name completed" passed=result.passed max_rel_error=result.max_rel_error
    return result
end

# =============================================================================
# Test 36: Weibull, AFT + Time-Fixed Covariate, Panel Data, Heuristic Phase-Type
# =============================================================================

"""
    run_ph_wei_aft_tfc_panel_heuristic()

Test Weibull model with accelerated failure time and time-fixed covariate, panel data.
Fitted using MCEM with Phase-Type proposal (heuristic).
"""
function run_ph_wei_aft_tfc_panel_heuristic()
    test_name = "ph_wei_aft_tfc_panel_heuristic"
    @info "Running $test_name"

    # True parameters
    α_12_true = 1.5
    λ_12_true = 0.15
    β_12_true = 0.5
    α_23_true = 1.3
    λ_23_true = 0.20
    β_23_true = 0.3

    # Create model for simulation
    h12 = Hazard(@formula(0 ~ 1 + x), "wei", 1, 2; linpred_effect=:aft)
    h23 = Hazard(@formula(0 ~ 1 + x), "wei", 2, 3; linpred_effect=:aft)

    Random.seed!(12375)
    dat = create_tfc_template(N_SUBJECTS; max_time=MAX_TIME)
    x_vals = dat.x

    msm_sim = multistatemodel(h12, h23; data=dat)
    set_parameters!(msm_sim,
        (h12 = [log(λ_12_true), log(α_12_true), β_12_true],
         h23 = [log(λ_23_true), log(α_23_true), β_23_true]))

    # Simulate and convert to panel
    paths = simulate(msm_sim; paths=true, data=false)[1]
    panel_data = create_panel_data_with_covariate(paths, collect(PANEL_TIMES), 3, x_vals)

    # Create fitting model
    msm_fit = multistatemodel(h12, h23; data=panel_data)

    # Configure Phase-Type Proposal
    proposal_config = ProposalConfig(
        type = :phasetype,
        n_phases = :heuristic,
        structure = :unstructured,
        max_phases = 5
    )

    # Fit model using MCEM
    fitted = fit(msm_fit;
        proposal=proposal_config,
        tol=MCEM_TOL,
        ess_target_initial=MCEM_ESS_INITIAL,
        ess_target_max=MCEM_ESS_MAX,
        maxiter=MCEM_MAXITER)

    # Extract estimates
    params = get_parameters(fitted)
    λ_12_est = exp(params[:h12][1])
    α_12_est = exp(params[:h12][2])
    β_12_est = params[:h12][3]
    λ_23_est = exp(params[:h23][1])
    α_23_est = exp(params[:h23][2])
    β_23_est = params[:h23][3]

    # Compute relative errors
    rel_errs = Dict(
        "λ_12" => compute_relative_error(λ_12_true, λ_12_est),
        "α_12" => compute_relative_error(α_12_true, α_12_est),
        "β_12" => compute_relative_error(β_12_true, β_12_est),
        "λ_23" => compute_relative_error(λ_23_true, λ_23_est),
        "α_23" => compute_relative_error(α_23_true, α_23_est),
        "β_23" => compute_relative_error(β_23_true, β_23_est),
    )

    # Compute prevalence and cumulative incidence
    n_states = 3
    prev_true = compute_state_prevalence(paths, EVAL_TIMES, n_states)
    prev_obs = compute_prevalence_from_data(panel_data, EVAL_TIMES, n_states)

    # Simulate from fitted model
    dat_fit = create_tfc_template(N_SUBJECTS; max_time=MAX_TIME)
    msm_fit_sim = multistatemodel(h12, h23; data=dat_fit)
    set_parameters!(msm_fit_sim, get_parameters(fitted))
    paths_fitted = simulate(msm_fit_sim; paths=true, data=false)[1]
    prev_fitted = compute_state_prevalence(paths_fitted, EVAL_TIMES, n_states)

    cumincid_12_true = compute_cumulative_incidence(paths, EVAL_TIMES, 1, 2)
    cumincid_12_obs = compute_cumincid_from_data(panel_data, EVAL_TIMES, 1, 2)
    cumincid_12_fitted = compute_cumulative_incidence(paths_fitted, EVAL_TIMES, 1, 2)

    cumincid_23_true = compute_cumulative_incidence(paths, EVAL_TIMES, 2, 3)
    cumincid_23_obs = compute_cumincid_from_data(panel_data, EVAL_TIMES, 2, 3)
    cumincid_23_fitted = compute_cumulative_incidence(paths_fitted, EVAL_TIMES, 2, 3)

    # Create result
    result = TestResult(
        name = test_name,
        family = "weibull",
        parameterization = :aft,
        covariates = :tfc,
        data_type = :panel,
        n_subjects = N_SUBJECTS,
        true_params = Dict(
            "λ_12" => λ_12_true, "α_12" => α_12_true, "β_12" => β_12_true,
            "λ_23" => λ_23_true, "α_23" => α_23_true, "β_23" => β_23_true,
        ),
        estimated_params = Dict(
            "λ_12" => λ_12_est, "α_12" => α_12_est, "β_12" => β_12_est,
            "λ_23" => λ_23_est, "α_23" => α_23_est, "β_23" => β_23_est,
        ),
        rel_errors = rel_errs,
        max_rel_error = NaN,
        passed = false,
        eval_times = EVAL_TIMES,
        prevalence_true = prev_true,
        prevalence_observed = prev_obs,
        prevalence_fitted = prev_fitted,
        cumincid_12_true = cumincid_12_true,
        cumincid_12_observed = cumincid_12_obs,
        cumincid_12_fitted = cumincid_12_fitted,
        cumincid_23_true = cumincid_23_true,
        cumincid_23_observed = cumincid_23_obs,
        cumincid_23_fitted = cumincid_23_fitted,
    )

    finalize_result!(result)
    push!(ALL_RESULTS, result)

    @info "$test_name completed" passed=result.passed max_rel_error=result.max_rel_error
    return result
end

# =============================================================================
# Test 37: Gompertz, No Covariates, Panel Data, Heuristic Phase-Type Proposal
# =============================================================================

"""
    run_ph_gom_nocov_panel_heuristic()

Test Gompertz model with no covariates and panel data.
Fitted using MCEM with Phase-Type proposal (heuristic).
"""
function run_ph_gom_nocov_panel_heuristic()
    test_name = "ph_gom_nocov_panel_heuristic"
    @info "Running $test_name"

    # True parameters
    η_12_true = 0.1
    γ_12_true = 0.15
    η_23_true = 0.15
    γ_23_true = 0.1

    # Create model for simulation
    h12 = Hazard(@formula(0 ~ 1), "gom", 1, 2)
    h23 = Hazard(@formula(0 ~ 1), "gom", 2, 3)

    Random.seed!(12376)
    dat = create_baseline_template(N_SUBJECTS; max_time=MAX_TIME)

    msm_sim = multistatemodel(h12, h23; data=dat)
    set_parameters!(msm_sim,
        (h12 = [log(η_12_true), log(γ_12_true)],
         h23 = [log(η_23_true), log(γ_23_true)]))

    # Simulate exact data and convert to panel
    paths = simulate(msm_sim; paths=true, data=false)[1]
    panel_data = create_panel_data(paths, collect(PANEL_TIMES), 3)

    # Create fitting model
    msm_fit = multistatemodel(h12, h23; data=panel_data)

    # Configure Phase-Type Proposal
    proposal_config = ProposalConfig(
        type = :phasetype,
        n_phases = :heuristic,
        structure = :unstructured,
        max_phases = 5
    )

    # Fit model using MCEM
    fitted = fit(msm_fit;
        proposal=proposal_config,
        tol=MCEM_TOL,
        ess_target_initial=MCEM_ESS_INITIAL,
        ess_target_max=MCEM_ESS_MAX,
        maxiter=MCEM_MAXITER)

    # Extract estimates
    params = get_parameters(fitted)
    η_12_est = exp(params[:h12][1])
    γ_12_est = exp(params[:h12][2])
    η_23_est = exp(params[:h23][1])
    γ_23_est = exp(params[:h23][2])

    # Compute relative errors
    rel_errs = Dict(
        "η_12" => compute_relative_error(η_12_true, η_12_est),
        "γ_12" => compute_relative_error(γ_12_true, γ_12_est),
        "η_23" => compute_relative_error(η_23_true, η_23_est),
        "γ_23" => compute_relative_error(γ_23_true, γ_23_est),
    )

    # Compute prevalence and cumulative incidence
    n_states = 3
    prev_true = compute_state_prevalence(paths, EVAL_TIMES, n_states)
    prev_obs = compute_prevalence_from_data(panel_data, EVAL_TIMES, n_states)

    # Simulate from fitted model
    dat_fit = create_baseline_template(N_SUBJECTS; max_time=MAX_TIME)
    msm_fit_sim = multistatemodel(h12, h23; data=dat_fit)
    set_parameters!(msm_fit_sim, get_parameters(fitted))
    paths_fitted = simulate(msm_fit_sim; paths=true, data=false)[1]
    prev_fitted = compute_state_prevalence(paths_fitted, EVAL_TIMES, n_states)

    cumincid_12_true = compute_cumulative_incidence(paths, EVAL_TIMES, 1, 2)
    cumincid_12_obs = compute_cumincid_from_data(panel_data, EVAL_TIMES, 1, 2)
    cumincid_12_fitted = compute_cumulative_incidence(paths_fitted, EVAL_TIMES, 1, 2)

    cumincid_23_true = compute_cumulative_incidence(paths, EVAL_TIMES, 2, 3)
    cumincid_23_obs = compute_cumincid_from_data(panel_data, EVAL_TIMES, 2, 3)
    cumincid_23_fitted = compute_cumulative_incidence(paths_fitted, EVAL_TIMES, 2, 3)

    # Create result
    result = TestResult(
        name = test_name,
        family = "gompertz",
        parameterization = :ph,
        covariates = :none,
        data_type = :panel,
        n_subjects = N_SUBJECTS,
        true_params = Dict(
            "η_12" => η_12_true, "γ_12" => γ_12_true,
            "η_23" => η_23_true, "γ_23" => γ_23_true,
        ),
        estimated_params = Dict(
            "η_12" => η_12_est, "γ_12" => γ_12_est,
            "η_23" => η_23_est, "γ_23" => γ_23_est,
        ),
        rel_errors = rel_errs,
        max_rel_error = NaN,
        passed = false,
        eval_times = EVAL_TIMES,
        prevalence_true = prev_true,
        prevalence_observed = prev_obs,
        prevalence_fitted = prev_fitted,
        cumincid_12_true = cumincid_12_true,
        cumincid_12_observed = cumincid_12_obs,
        cumincid_12_fitted = cumincid_12_fitted,
        cumincid_23_true = cumincid_23_true,
        cumincid_23_observed = cumincid_23_obs,
        cumincid_23_fitted = cumincid_23_fitted,
    )

    finalize_result!(result)
    push!(ALL_RESULTS, result)

    @info "$test_name completed" passed=result.passed max_rel_error=result.max_rel_error
    return result
end

# =============================================================================
# Test 38: Gompertz, PH + Time-Varying Covariate, Panel Data, Heuristic Phase-Type
# =============================================================================

"""
    run_ph_gom_ph_tvc_panel_heuristic()

Test Gompertz model with proportional hazards and time-varying covariate, panel data.
Fitted using MCEM with Phase-Type proposal (heuristic).
"""
function run_ph_gom_ph_tvc_panel_heuristic()
    test_name = "ph_gom_ph_tvc_panel_heuristic"
    @info "Running $test_name"

    # True parameters
    η_12_true = 0.1
    γ_12_true = 0.15
    β_12_true = 0.5
    η_23_true = 0.15
    γ_23_true = 0.1
    β_23_true = 0.3

    # Create model for simulation
    h12 = Hazard(@formula(0 ~ 1 + x), "gom", 1, 2; linpred_effect=:ph)
    h23 = Hazard(@formula(0 ~ 1 + x), "gom", 2, 3; linpred_effect=:ph)

    Random.seed!(12377)
    dat = create_tvc_template(N_SUBJECTS; max_time=MAX_TIME)
    x_vals = dat.x

    msm_sim = multistatemodel(h12, h23; data=dat)
    set_parameters!(msm_sim,
        (h12 = [log(η_12_true), log(γ_12_true), β_12_true],
         h23 = [log(η_23_true), log(γ_23_true), β_23_true]))

    # Simulate and convert to panel
    paths = simulate(msm_sim; paths=true, data=false)[1]
    panel_data = create_panel_data_with_tvc(paths, collect(PANEL_TIMES), 3, x_vals)

    # Create fitting model
    msm_fit = multistatemodel(h12, h23; data=panel_data)

    # Configure Phase-Type Proposal
    proposal_config = ProposalConfig(
        type = :phasetype,
        n_phases = :heuristic,
        structure = :unstructured,
        max_phases = 5
    )

    # Fit model using MCEM
    fitted = fit(msm_fit;
        proposal=proposal_config,
        tol=MCEM_TOL,
        ess_target_initial=MCEM_ESS_INITIAL,
        ess_target_max=MCEM_ESS_MAX,
        maxiter=MCEM_MAXITER)

    # Extract estimates
    params = get_parameters(fitted)
    η_12_est = exp(params[:h12][1])
    γ_12_est = exp(params[:h12][2])
    β_12_est = params[:h12][3]
    η_23_est = exp(params[:h23][1])
    γ_23_est = exp(params[:h23][2])
    β_23_est = params[:h23][3]

    # Compute relative errors
    rel_errs = Dict(
        "η_12" => compute_relative_error(η_12_true, η_12_est),
        "γ_12" => compute_relative_error(γ_12_true, γ_12_est),
        "β_12" => compute_relative_error(β_12_true, β_12_est),
        "η_23" => compute_relative_error(η_23_true, η_23_est),
        "γ_23" => compute_relative_error(γ_23_true, γ_23_est),
        "β_23" => compute_relative_error(β_23_true, β_23_est),
    )

    # Compute prevalence and cumulative incidence
    n_states = 3
    prev_true = compute_state_prevalence(paths, EVAL_TIMES, n_states)
    prev_obs = compute_prevalence_from_data(panel_data, EVAL_TIMES, n_states)

    # Simulate from fitted model
    dat_fit = create_tvc_template(N_SUBJECTS; max_time=MAX_TIME)
    msm_fit_sim = multistatemodel(h12, h23; data=dat_fit)
    set_parameters!(msm_fit_sim, get_parameters(fitted))
    paths_fitted = simulate(msm_fit_sim; paths=true, data=false)[1]
    prev_fitted = compute_state_prevalence(paths_fitted, EVAL_TIMES, n_states)

    cumincid_12_true = compute_cumulative_incidence(paths, EVAL_TIMES, 1, 2)
    cumincid_12_obs = compute_cumincid_from_data(panel_data, EVAL_TIMES, 1, 2)
    cumincid_12_fitted = compute_cumulative_incidence(paths_fitted, EVAL_TIMES, 1, 2)

    cumincid_23_true = compute_cumulative_incidence(paths, EVAL_TIMES, 2, 3)
    cumincid_23_obs = compute_cumincid_from_data(panel_data, EVAL_TIMES, 2, 3)
    cumincid_23_fitted = compute_cumulative_incidence(paths_fitted, EVAL_TIMES, 2, 3)

    # Create result
    result = TestResult(
        name = test_name,
        family = "gompertz",
        parameterization = :ph,
        covariates = :tvc,
        data_type = :panel,
        n_subjects = N_SUBJECTS,
        true_params = Dict(
            "η_12" => η_12_true, "γ_12" => γ_12_true, "β_12" => β_12_true,
            "η_23" => η_23_true, "γ_23" => γ_23_true, "β_23" => β_23_true,
        ),
        estimated_params = Dict(
            "η_12" => η_12_est, "γ_12" => γ_12_est, "β_12" => β_12_est,
            "η_23" => η_23_est, "γ_23" => γ_23_est, "β_23" => β_23_est,
        ),
        rel_errors = rel_errs,
        max_rel_error = NaN,
        passed = false,
        eval_times = EVAL_TIMES,
        prevalence_true = prev_true,
        prevalence_observed = prev_obs,
        prevalence_fitted = prev_fitted,
        cumincid_12_true = cumincid_12_true,
        cumincid_12_observed = cumincid_12_obs,
        cumincid_12_fitted = cumincid_12_fitted,
        cumincid_23_true = cumincid_23_true,
        cumincid_23_observed = cumincid_23_obs,
        cumincid_23_fitted = cumincid_23_fitted,
    )

    finalize_result!(result)
    push!(ALL_RESULTS, result)

    @info "$test_name completed" passed=result.passed max_rel_error=result.max_rel_error
    return result
end

# =============================================================================
# Test 39: Gompertz, AFT + Time-Varying Covariate, Panel Data, Heuristic Phase-Type
# =============================================================================

"""
    run_ph_gom_aft_tvc_panel_heuristic()

Test Gompertz model with accelerated failure time and time-varying covariate, panel data.
Fitted using MCEM with Phase-Type proposal (heuristic).
"""
function run_ph_gom_aft_tvc_panel_heuristic()
    test_name = "ph_gom_aft_tvc_panel_heuristic"
    @info "Running $test_name"

    # True parameters
    η_12_true = 0.1
    γ_12_true = 0.15
    β_12_true = 0.5
    η_23_true = 0.15
    γ_23_true = 0.1
    β_23_true = 0.3

    # Create model for simulation
    h12 = Hazard(@formula(0 ~ 1 + x), "gom", 1, 2; linpred_effect=:aft)
    h23 = Hazard(@formula(0 ~ 1 + x), "gom", 2, 3; linpred_effect=:aft)

    Random.seed!(12378)
    dat = create_tvc_template(N_SUBJECTS; max_time=MAX_TIME)
    x_vals = dat.x

    msm_sim = multistatemodel(h12, h23; data=dat)
    set_parameters!(msm_sim,
        (h12 = [log(η_12_true), log(γ_12_true), β_12_true],
         h23 = [log(η_23_true), log(γ_23_true), β_23_true]))

    # Simulate and convert to panel
    paths = simulate(msm_sim; paths=true, data=false)[1]
    panel_data = create_panel_data_with_tvc(paths, collect(PANEL_TIMES), 3, x_vals)

    # Create fitting model
    msm_fit = multistatemodel(h12, h23; data=panel_data)

    # Configure Phase-Type Proposal
    proposal_config = ProposalConfig(
        type = :phasetype,
        n_phases = :heuristic,
        structure = :unstructured,
        max_phases = 5
    )

    # Fit model using MCEM
    fitted = fit(msm_fit;
        proposal=proposal_config,
        tol=MCEM_TOL,
        ess_target_initial=MCEM_ESS_INITIAL,
        ess_target_max=MCEM_ESS_MAX,
        maxiter=MCEM_MAXITER)

    # Extract estimates
    params = get_parameters(fitted)
    η_12_est = exp(params[:h12][1])
    γ_12_est = exp(params[:h12][2])
    β_12_est = params[:h12][3]
    η_23_est = exp(params[:h23][1])
    γ_23_est = exp(params[:h23][2])
    β_23_est = params[:h23][3]

    # Compute relative errors
    rel_errs = Dict(
        "η_12" => compute_relative_error(η_12_true, η_12_est),
        "γ_12" => compute_relative_error(γ_12_true, γ_12_est),
        "β_12" => compute_relative_error(β_12_true, β_12_est),
        "η_23" => compute_relative_error(η_23_true, η_23_est),
        "γ_23" => compute_relative_error(γ_23_true, γ_23_est),
        "β_23" => compute_relative_error(β_23_true, β_23_est),
    )

    # Compute prevalence and cumulative incidence
    n_states = 3
    prev_true = compute_state_prevalence(paths, EVAL_TIMES, n_states)
    prev_obs = compute_prevalence_from_data(panel_data, EVAL_TIMES, n_states)

    # Simulate from fitted model
    dat_fit = create_tvc_template(N_SUBJECTS; max_time=MAX_TIME)
    msm_fit_sim = multistatemodel(h12, h23; data=dat_fit)
    set_parameters!(msm_fit_sim, get_parameters(fitted))
    paths_fitted = simulate(msm_fit_sim; paths=true, data=false)[1]
    prev_fitted = compute_state_prevalence(paths_fitted, EVAL_TIMES, n_states)

    cumincid_12_true = compute_cumulative_incidence(paths, EVAL_TIMES, 1, 2)
    cumincid_12_obs = compute_cumincid_from_data(panel_data, EVAL_TIMES, 1, 2)
    cumincid_12_fitted = compute_cumulative_incidence(paths_fitted, EVAL_TIMES, 1, 2)

    cumincid_23_true = compute_cumulative_incidence(paths, EVAL_TIMES, 2, 3)
    cumincid_23_obs = compute_cumincid_from_data(panel_data, EVAL_TIMES, 2, 3)
    cumincid_23_fitted = compute_cumulative_incidence(paths_fitted, EVAL_TIMES, 2, 3)

    # Create result
    result = TestResult(
        name = test_name,
        family = "gompertz",
        parameterization = :aft,
        covariates = :tvc,
        data_type = :panel,
        n_subjects = N_SUBJECTS,
        true_params = Dict(
            "η_12" => η_12_true, "γ_12" => γ_12_true, "β_12" => β_12_true,
            "η_23" => η_23_true, "γ_23" => γ_23_true, "β_23" => β_23_true,
        ),
        estimated_params = Dict(
            "η_12" => η_12_est, "γ_12" => γ_12_est, "β_12" => β_12_est,
            "η_23" => η_23_est, "γ_23" => γ_23_est, "β_23" => β_23_est,
        ),
        rel_errors = rel_errs,
        max_rel_error = NaN,
        passed = false,
        eval_times = EVAL_TIMES,
        prevalence_true = prev_true,
        prevalence_observed = prev_obs,
        prevalence_fitted = prev_fitted,
        cumincid_12_true = cumincid_12_true,
        cumincid_12_observed = cumincid_12_obs,
        cumincid_12_fitted = cumincid_12_fitted,
        cumincid_23_true = cumincid_23_true,
        cumincid_23_observed = cumincid_23_obs,
        cumincid_23_fitted = cumincid_23_fitted,
    )

    finalize_result!(result)
    push!(ALL_RESULTS, result)

    @info "$test_name completed" passed=result.passed max_rel_error=result.max_rel_error
    return result
end

# =============================================================================
# Test 40: Weibull, No Covariates, Panel Data, Phase-Type (Prop to Prog)
# =============================================================================

"""
    run_ph_wei_nocov_panel_structure_prop()

Test Weibull model with no covariates and panel data.
Fitted using MCEM with Phase-Type proposal (structure=:prop_to_prog).
"""
function run_ph_wei_nocov_panel_structure_prop()
    test_name = "ph_wei_nocov_panel_structure_prop"
    @info "Running $test_name"

    # True parameters
    α_12_true = 1.5
    λ_12_true = 0.15
    α_23_true = 1.3
    λ_23_true = 0.20

    # Create model for simulation
    h12 = Hazard(@formula(0 ~ 1), "wei", 1, 2)
    h23 = Hazard(@formula(0 ~ 1), "wei", 2, 3)

    Random.seed!(12379)
    dat = create_baseline_template(N_SUBJECTS; max_time=MAX_TIME)

    msm_sim = multistatemodel(h12, h23; data=dat)
    set_parameters!(msm_sim,
        (h12 = [log(λ_12_true), log(α_12_true)],
         h23 = [log(λ_23_true), log(α_23_true)]))

    # Simulate exact data and convert to panel
    paths = simulate(msm_sim; paths=true, data=false)[1]
    panel_data = create_panel_data(paths, collect(PANEL_TIMES), 3)

    # Create fitting model
    msm_fit = multistatemodel(h12, h23; data=panel_data)

    # Configure Phase-Type Proposal
    proposal_config = ProposalConfig(
        type = :phasetype,
        n_phases = :heuristic,
        structure = :prop_to_prog,
        max_phases = 5
    )

    # Fit model using MCEM
    fitted = fit(msm_fit;
        proposal=proposal_config,
        tol=MCEM_TOL,
        ess_target_initial=MCEM_ESS_INITIAL,
        ess_target_max=MCEM_ESS_MAX,
        maxiter=MCEM_MAXITER)

    # Extract estimates
    params = get_parameters(fitted)
    λ_12_est = exp(params[:h12][1])
    α_12_est = exp(params[:h12][2])
    λ_23_est = exp(params[:h23][1])
    α_23_est = exp(params[:h23][2])

    # Compute relative errors
    rel_errs = Dict(
        "λ_12" => compute_relative_error(λ_12_true, λ_12_est),
        "α_12" => compute_relative_error(α_12_true, α_12_est),
        "λ_23" => compute_relative_error(λ_23_true, λ_23_est),
        "α_23" => compute_relative_error(α_23_true, α_23_est),
    )

    # Compute prevalence and cumulative incidence
    n_states = 3
    prev_true = compute_state_prevalence(paths, EVAL_TIMES, n_states)
    prev_obs = compute_prevalence_from_data(panel_data, EVAL_TIMES, n_states)

    # Simulate from fitted model
    dat_fit = create_baseline_template(N_SUBJECTS; max_time=MAX_TIME)
    msm_fit_sim = multistatemodel(h12, h23; data=dat_fit)
    set_parameters!(msm_fit_sim, get_parameters(fitted))
    paths_fitted = simulate(msm_fit_sim; paths=true, data=false)[1]
    prev_fitted = compute_state_prevalence(paths_fitted, EVAL_TIMES, n_states)

    cumincid_12_true = compute_cumulative_incidence(paths, EVAL_TIMES, 1, 2)
    cumincid_12_obs = compute_cumincid_from_data(panel_data, EVAL_TIMES, 1, 2)
    cumincid_12_fitted = compute_cumulative_incidence(paths_fitted, EVAL_TIMES, 1, 2)

    cumincid_23_true = compute_cumulative_incidence(paths, EVAL_TIMES, 2, 3)
    cumincid_23_obs = compute_cumincid_from_data(panel_data, EVAL_TIMES, 2, 3)
    cumincid_23_fitted = compute_cumulative_incidence(paths_fitted, EVAL_TIMES, 2, 3)

    # Create result
    result = TestResult(
        name = test_name,
        family = "weibull",
        parameterization = :ph,
        covariates = :none,
        data_type = :panel,
        n_subjects = N_SUBJECTS,
        true_params = Dict(
            "λ_12" => λ_12_true, "α_12" => α_12_true,
            "λ_23" => λ_23_true, "α_23" => α_23_true,
        ),
        estimated_params = Dict(
            "λ_12" => λ_12_est, "α_12" => α_12_est,
            "λ_23" => λ_23_est, "α_23" => α_23_est,
        ),
        rel_errors = rel_errs,
        max_rel_error = NaN,
        passed = false,
        eval_times = EVAL_TIMES,
        prevalence_true = prev_true,
        prevalence_observed = prev_obs,
        prevalence_fitted = prev_fitted,
        cumincid_12_true = cumincid_12_true,
        cumincid_12_observed = cumincid_12_obs,
        cumincid_12_fitted = cumincid_12_fitted,
        cumincid_23_true = cumincid_23_true,
        cumincid_23_observed = cumincid_23_obs,
        cumincid_23_fitted = cumincid_23_fitted,
    )

    finalize_result!(result)
    push!(ALL_RESULTS, result)

    @info "$test_name completed" passed=result.passed max_rel_error=result.max_rel_error
    return result
end

"""
    run_all_phasetype_proposal_tests()

Run all 10 Phase-Type PROPOSAL tests.
These test phase-type proposals for semi-Markov MCEM, NOT the :pt hazard family.
"""
function run_all_phasetype_proposal_tests()
    @info "Running all Phase-Type PROPOSAL tests (10 tests)"
    results = TestResult[]
    
    push!(results, run_ph_wei_nocov_panel_heuristic())
    push!(results, run_ph_wei_nocov_panel_auto())
    push!(results, run_ph_wei_nocov_panel_manual_int())
    push!(results, run_ph_wei_nocov_panel_manual_vec())
    push!(results, run_ph_wei_ph_tfc_panel_heuristic())
    push!(results, run_ph_wei_aft_tfc_panel_heuristic())
    push!(results, run_ph_gom_nocov_panel_heuristic())
    push!(results, run_ph_gom_ph_tvc_panel_heuristic())
    push!(results, run_ph_gom_aft_tvc_panel_heuristic())
    push!(results, run_ph_wei_nocov_panel_structure_prop())
    
    return results
end

