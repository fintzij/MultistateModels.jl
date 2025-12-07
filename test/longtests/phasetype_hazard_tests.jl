# =============================================================================
# Phase-Type Hazard Family Long Tests
# =============================================================================
#
# Tests for the :pt phase-type HAZARD FAMILY (Coxian phase-type distributions).
# These test inference when the TARGET MODEL uses :pt hazards.
#
# Key distinction:
# - This file tests :pt HAZARD MODELS → Markov on expanded space → Direct MLE
# - phasetype_tests.jl tests phase-type PROPOSALS for semi-Markov MCEM
#
# 3-state progressive model: 1 → 2 → 3 (absorbing)
# Phase-type hazards create an expanded state space with exponential hazards.
#
# Since phase-type models are Markov on the expanded space, we use direct
# panel likelihood fitting - no MCEM required.
# =============================================================================

# =============================================================================
# Test PT-1: Phase-Type (2-phase), No Covariates, Exact Data
# =============================================================================

"""
    run_pt2_nocov_exact()

Test 2-phase Coxian phase-type model with no covariates and exact data.
Both transitions (1→2 and 2→3) use pt2 hazards.

Parameters per transition (2-phase Coxian with unstructured):
- λ (progression rate through phases)
- μ₁ (exit rate from phase 1)
- μ₂ (exit rate from phase 2)
"""
function run_pt2_nocov_exact()
    test_name = "pt2_nocov_exact"
    @info "Running $test_name"
    
    Random.seed!(RNG_SEED)
    
    # True parameters (natural scale for interpretability)
    # For h12: 2-phase Coxian
    λ_12_true = 0.5    # Progression rate
    μ1_12_true = 0.2   # Exit from phase 1
    μ2_12_true = 0.3   # Exit from phase 2
    
    # For h23: 2-phase Coxian  
    λ_23_true = 0.4
    μ1_23_true = 0.25
    μ2_23_true = 0.35
    
    # Create hazards with phase-type family
    h12 = Hazard(@formula(0 ~ 1), "pt", 1, 2; n_phases=2, coxian_structure=:unstructured)
    h23 = Hazard(@formula(0 ~ 1), "pt", 2, 3; n_phases=2, coxian_structure=:unstructured)
    
    # Create data template
    dat = create_baseline_template(N_SUBJECTS; max_time=MAX_TIME)
    
    # Build model for simulation
    msm_sim = multistatemodel(h12, h23; data=dat)
    
    # Set true parameters (user-facing: natural scale)
    set_parameters!(msm_sim, (
        h12 = [λ_12_true, μ1_12_true, μ2_12_true],
        h23 = [λ_23_true, μ1_23_true, μ2_23_true]
    ))
    
    # Simulate exact data
    paths = simulate(msm_sim; paths=true, data=false)[1]
    exact_data = simulate(msm_sim; paths=false, data=true)[1]
    
    # Create fitting model with SAME specification
    msm_fit = multistatemodel(h12, h23; data=exact_data)
    
    # Fit model (direct MLE, no MCEM needed)
    fitted = fit(msm_fit; verbose=false, compute_vcov=false)
    
    # Extract estimates (user-facing parameters, natural scale)
    params = get_parameters(fitted)
    λ_12_est = params[:h12][1]
    μ1_12_est = params[:h12][2]
    μ2_12_est = params[:h12][3]
    λ_23_est = params[:h23][1]
    μ1_23_est = params[:h23][2]
    μ2_23_est = params[:h23][3]
    
    # Compute relative errors
    rel_errs = Dict(
        "λ_12" => compute_relative_error(λ_12_true, λ_12_est),
        "μ1_12" => compute_relative_error(μ1_12_true, μ1_12_est),
        "μ2_12" => compute_relative_error(μ2_12_true, μ2_12_est),
        "λ_23" => compute_relative_error(λ_23_true, λ_23_est),
        "μ1_23" => compute_relative_error(μ1_23_true, μ1_23_est),
        "μ2_23" => compute_relative_error(μ2_23_true, μ2_23_est),
    )
    
    # Compute prevalence and cumulative incidence
    n_states = 3
    prev_true = compute_state_prevalence(paths, EVAL_TIMES, n_states)
    prev_obs = compute_prevalence_from_data(exact_data, EVAL_TIMES, n_states)
    
    # Simulate from fitted model
    dat_fit = create_baseline_template(N_SUBJECTS; max_time=MAX_TIME)
    msm_fit_sim = multistatemodel(h12, h23; data=dat_fit)
    set_parameters!(msm_fit_sim, get_parameters(fitted))
    paths_fitted = simulate(msm_fit_sim; paths=true, data=false)[1]
    prev_fitted = compute_state_prevalence(paths_fitted, EVAL_TIMES, n_states)
    
    cumincid_12_true = compute_cumulative_incidence(paths, EVAL_TIMES, 1, 2)
    cumincid_12_obs = compute_cumincid_from_data(exact_data, EVAL_TIMES, 1, 2)
    cumincid_12_fitted = compute_cumulative_incidence(paths_fitted, EVAL_TIMES, 1, 2)
    
    cumincid_23_true = compute_cumulative_incidence(paths, EVAL_TIMES, 2, 3)
    cumincid_23_obs = compute_cumincid_from_data(exact_data, EVAL_TIMES, 2, 3)
    cumincid_23_fitted = compute_cumulative_incidence(paths_fitted, EVAL_TIMES, 2, 3)
    
    # Create result
    result = TestResult(
        name = test_name,
        family = "phasetype",
        parameterization = :none,
        covariates = :none,
        data_type = :exact,
        n_subjects = N_SUBJECTS,
        true_params = Dict(
            "λ_12" => λ_12_true, "μ1_12" => μ1_12_true, "μ2_12" => μ2_12_true,
            "λ_23" => λ_23_true, "μ1_23" => μ1_23_true, "μ2_23" => μ2_23_true,
        ),
        estimated_params = Dict(
            "λ_12" => λ_12_est, "μ1_12" => μ1_12_est, "μ2_12" => μ2_12_est,
            "λ_23" => λ_23_est, "μ1_23" => μ1_23_est, "μ2_23" => μ2_23_est,
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
        cumincid_23_fitted = cumincid_23_fitted
    )
    
    finalize_result!(result)
    push!(ALL_RESULTS, result)
    
    @info "$test_name completed" passed=result.passed max_rel_error=result.max_rel_error
    return result
end

# =============================================================================
# Test PT-2: Phase-Type (2-phase), No Covariates, Panel Data
# =============================================================================

"""
    run_pt2_nocov_panel()

Test 2-phase Coxian phase-type model with no covariates and panel data.
Panel data fitting uses Markov likelihood on expanded state space.
"""
function run_pt2_nocov_panel()
    test_name = "pt2_nocov_panel"
    @info "Running $test_name"
    
    Random.seed!(RNG_SEED + 1)
    
    # True parameters (natural scale)
    λ_12_true = 0.5
    μ1_12_true = 0.2
    μ2_12_true = 0.3
    
    λ_23_true = 0.4
    μ1_23_true = 0.25
    μ2_23_true = 0.35
    
    # Create hazards
    h12 = Hazard(@formula(0 ~ 1), "pt", 1, 2; n_phases=2, coxian_structure=:unstructured)
    h23 = Hazard(@formula(0 ~ 1), "pt", 2, 3; n_phases=2, coxian_structure=:unstructured)
    
    dat = create_baseline_template(N_SUBJECTS; max_time=MAX_TIME)
    
    msm_sim = multistatemodel(h12, h23; data=dat)
    set_parameters!(msm_sim, (
        h12 = [λ_12_true, μ1_12_true, μ2_12_true],
        h23 = [λ_23_true, μ1_23_true, μ2_23_true]
    ))
    
    # Simulate and convert to panel
    paths = simulate(msm_sim; paths=true, data=false)[1]
    panel_data = create_panel_data(paths, collect(PANEL_TIMES), 3)
    
    # Create fitting model
    msm_fit = multistatemodel(h12, h23; data=panel_data)
    
    # Fit using Markov likelihood (no MCEM)
    fitted = fit(msm_fit; verbose=false, compute_vcov=false)
    
    # Extract estimates
    params = get_parameters(fitted)
    λ_12_est = params[:h12][1]
    μ1_12_est = params[:h12][2]
    μ2_12_est = params[:h12][3]
    λ_23_est = params[:h23][1]
    μ1_23_est = params[:h23][2]
    μ2_23_est = params[:h23][3]
    
    rel_errs = Dict(
        "λ_12" => compute_relative_error(λ_12_true, λ_12_est),
        "μ1_12" => compute_relative_error(μ1_12_true, μ1_12_est),
        "μ2_12" => compute_relative_error(μ2_12_true, μ2_12_est),
        "λ_23" => compute_relative_error(λ_23_true, λ_23_est),
        "μ1_23" => compute_relative_error(μ1_23_true, μ1_23_est),
        "μ2_23" => compute_relative_error(μ2_23_true, μ2_23_est),
    )
    
    n_states = 3
    prev_true = compute_state_prevalence(paths, EVAL_TIMES, n_states)
    prev_obs = compute_prevalence_from_data(panel_data, EVAL_TIMES, n_states)
    
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
    
    result = TestResult(
        name = test_name,
        family = "phasetype",
        parameterization = :none,
        covariates = :none,
        data_type = :panel,
        n_subjects = N_SUBJECTS,
        true_params = Dict(
            "λ_12" => λ_12_true, "μ1_12" => μ1_12_true, "μ2_12" => μ2_12_true,
            "λ_23" => λ_23_true, "μ1_23" => μ1_23_true, "μ2_23" => μ2_23_true,
        ),
        estimated_params = Dict(
            "λ_12" => λ_12_est, "μ1_12" => μ1_12_est, "μ2_12" => μ2_12_est,
            "λ_23" => λ_23_est, "μ1_23" => μ1_23_est, "μ2_23" => μ2_23_est,
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
        cumincid_23_fitted = cumincid_23_fitted
    )
    
    finalize_result!(result)
    push!(ALL_RESULTS, result)
    
    @info "$test_name completed" passed=result.passed max_rel_error=result.max_rel_error
    return result
end

# =============================================================================
# Test PT-3: Phase-Type (3-phase), No Covariates, Exact Data
# =============================================================================

"""
    run_pt3_nocov_exact()

Test 3-phase Coxian phase-type model with no covariates and exact data.

Parameters per transition (3-phase Coxian with unstructured):
- λ₁, λ₂ (progression rates)
- μ₁, μ₂, μ₃ (exit rates from each phase)
"""
function run_pt3_nocov_exact()
    test_name = "pt3_nocov_exact"
    @info "Running $test_name"
    
    Random.seed!(RNG_SEED + 2)
    
    # True parameters for h12 (3-phase): [λ₁, λ₂, μ₁, μ₂, μ₃]
    λ1_12_true = 0.6
    λ2_12_true = 0.5
    μ1_12_true = 0.15
    μ2_12_true = 0.20
    μ3_12_true = 0.25
    
    # True parameters for h23 (3-phase)
    λ1_23_true = 0.5
    λ2_23_true = 0.4
    μ1_23_true = 0.18
    μ2_23_true = 0.22
    μ3_23_true = 0.30
    
    h12 = Hazard(@formula(0 ~ 1), "pt", 1, 2; n_phases=3, coxian_structure=:unstructured)
    h23 = Hazard(@formula(0 ~ 1), "pt", 2, 3; n_phases=3, coxian_structure=:unstructured)
    
    dat = create_baseline_template(N_SUBJECTS; max_time=MAX_TIME)
    
    msm_sim = multistatemodel(h12, h23; data=dat)
    set_parameters!(msm_sim, (
        h12 = [λ1_12_true, λ2_12_true, μ1_12_true, μ2_12_true, μ3_12_true],
        h23 = [λ1_23_true, λ2_23_true, μ1_23_true, μ2_23_true, μ3_23_true]
    ))
    
    paths = simulate(msm_sim; paths=true, data=false)[1]
    exact_data = simulate(msm_sim; paths=false, data=true)[1]
    
    msm_fit = multistatemodel(h12, h23; data=exact_data)
    fitted = fit(msm_fit; verbose=false, compute_vcov=false)
    
    params = get_parameters(fitted)
    λ1_12_est = params[:h12][1]
    λ2_12_est = params[:h12][2]
    μ1_12_est = params[:h12][3]
    μ2_12_est = params[:h12][4]
    μ3_12_est = params[:h12][5]
    λ1_23_est = params[:h23][1]
    λ2_23_est = params[:h23][2]
    μ1_23_est = params[:h23][3]
    μ2_23_est = params[:h23][4]
    μ3_23_est = params[:h23][5]
    
    rel_errs = Dict(
        "λ1_12" => compute_relative_error(λ1_12_true, λ1_12_est),
        "λ2_12" => compute_relative_error(λ2_12_true, λ2_12_est),
        "μ1_12" => compute_relative_error(μ1_12_true, μ1_12_est),
        "μ2_12" => compute_relative_error(μ2_12_true, μ2_12_est),
        "μ3_12" => compute_relative_error(μ3_12_true, μ3_12_est),
        "λ1_23" => compute_relative_error(λ1_23_true, λ1_23_est),
        "λ2_23" => compute_relative_error(λ2_23_true, λ2_23_est),
        "μ1_23" => compute_relative_error(μ1_23_true, μ1_23_est),
        "μ2_23" => compute_relative_error(μ2_23_true, μ2_23_est),
        "μ3_23" => compute_relative_error(μ3_23_true, μ3_23_est),
    )
    
    n_states = 3
    prev_true = compute_state_prevalence(paths, EVAL_TIMES, n_states)
    prev_obs = compute_prevalence_from_data(exact_data, EVAL_TIMES, n_states)
    
    dat_fit = create_baseline_template(N_SUBJECTS; max_time=MAX_TIME)
    msm_fit_sim = multistatemodel(h12, h23; data=dat_fit)
    set_parameters!(msm_fit_sim, get_parameters(fitted))
    paths_fitted = simulate(msm_fit_sim; paths=true, data=false)[1]
    prev_fitted = compute_state_prevalence(paths_fitted, EVAL_TIMES, n_states)
    
    cumincid_12_true = compute_cumulative_incidence(paths, EVAL_TIMES, 1, 2)
    cumincid_12_obs = compute_cumincid_from_data(exact_data, EVAL_TIMES, 1, 2)
    cumincid_12_fitted = compute_cumulative_incidence(paths_fitted, EVAL_TIMES, 1, 2)
    
    cumincid_23_true = compute_cumulative_incidence(paths, EVAL_TIMES, 2, 3)
    cumincid_23_obs = compute_cumincid_from_data(exact_data, EVAL_TIMES, 2, 3)
    cumincid_23_fitted = compute_cumulative_incidence(paths_fitted, EVAL_TIMES, 2, 3)
    
    result = TestResult(
        name = test_name,
        family = "phasetype",
        parameterization = :none,
        covariates = :none,
        data_type = :exact,
        n_subjects = N_SUBJECTS,
        true_params = Dict(
            "λ1_12" => λ1_12_true, "λ2_12" => λ2_12_true,
            "μ1_12" => μ1_12_true, "μ2_12" => μ2_12_true, "μ3_12" => μ3_12_true,
            "λ1_23" => λ1_23_true, "λ2_23" => λ2_23_true,
            "μ1_23" => μ1_23_true, "μ2_23" => μ2_23_true, "μ3_23" => μ3_23_true,
        ),
        estimated_params = Dict(
            "λ1_12" => λ1_12_est, "λ2_12" => λ2_12_est,
            "μ1_12" => μ1_12_est, "μ2_12" => μ2_12_est, "μ3_12" => μ3_12_est,
            "λ1_23" => λ1_23_est, "λ2_23" => λ2_23_est,
            "μ1_23" => μ1_23_est, "μ2_23" => μ2_23_est, "μ3_23" => μ3_23_est,
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
        cumincid_23_fitted = cumincid_23_fitted
    )
    
    finalize_result!(result)
    push!(ALL_RESULTS, result)
    
    @info "$test_name completed" passed=result.passed max_rel_error=result.max_rel_error
    return result
end

# =============================================================================
# Test PT-4: Phase-Type (3-phase), No Covariates, Panel Data
# =============================================================================

"""
    run_pt3_nocov_panel()

Test 3-phase Coxian phase-type model with no covariates and panel data.
"""
function run_pt3_nocov_panel()
    test_name = "pt3_nocov_panel"
    @info "Running $test_name"
    
    Random.seed!(RNG_SEED + 3)
    
    # True parameters
    λ1_12_true = 0.6
    λ2_12_true = 0.5
    μ1_12_true = 0.15
    μ2_12_true = 0.20
    μ3_12_true = 0.25
    
    λ1_23_true = 0.5
    λ2_23_true = 0.4
    μ1_23_true = 0.18
    μ2_23_true = 0.22
    μ3_23_true = 0.30
    
    h12 = Hazard(@formula(0 ~ 1), "pt", 1, 2; n_phases=3, coxian_structure=:unstructured)
    h23 = Hazard(@formula(0 ~ 1), "pt", 2, 3; n_phases=3, coxian_structure=:unstructured)
    
    dat = create_baseline_template(N_SUBJECTS; max_time=MAX_TIME)
    
    msm_sim = multistatemodel(h12, h23; data=dat)
    set_parameters!(msm_sim, (
        h12 = [λ1_12_true, λ2_12_true, μ1_12_true, μ2_12_true, μ3_12_true],
        h23 = [λ1_23_true, λ2_23_true, μ1_23_true, μ2_23_true, μ3_23_true]
    ))
    
    paths = simulate(msm_sim; paths=true, data=false)[1]
    panel_data = create_panel_data(paths, collect(PANEL_TIMES), 3)
    
    msm_fit = multistatemodel(h12, h23; data=panel_data)
    fitted = fit(msm_fit; verbose=false, compute_vcov=false)
    
    params = get_parameters(fitted)
    λ1_12_est = params[:h12][1]
    λ2_12_est = params[:h12][2]
    μ1_12_est = params[:h12][3]
    μ2_12_est = params[:h12][4]
    μ3_12_est = params[:h12][5]
    λ1_23_est = params[:h23][1]
    λ2_23_est = params[:h23][2]
    μ1_23_est = params[:h23][3]
    μ2_23_est = params[:h23][4]
    μ3_23_est = params[:h23][5]
    
    rel_errs = Dict(
        "λ1_12" => compute_relative_error(λ1_12_true, λ1_12_est),
        "λ2_12" => compute_relative_error(λ2_12_true, λ2_12_est),
        "μ1_12" => compute_relative_error(μ1_12_true, μ1_12_est),
        "μ2_12" => compute_relative_error(μ2_12_true, μ2_12_est),
        "μ3_12" => compute_relative_error(μ3_12_true, μ3_12_est),
        "λ1_23" => compute_relative_error(λ1_23_true, λ1_23_est),
        "λ2_23" => compute_relative_error(λ2_23_true, λ2_23_est),
        "μ1_23" => compute_relative_error(μ1_23_true, μ1_23_est),
        "μ2_23" => compute_relative_error(μ2_23_true, μ2_23_est),
        "μ3_23" => compute_relative_error(μ3_23_true, μ3_23_est),
    )
    
    n_states = 3
    prev_true = compute_state_prevalence(paths, EVAL_TIMES, n_states)
    prev_obs = compute_prevalence_from_data(panel_data, EVAL_TIMES, n_states)
    
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
    
    result = TestResult(
        name = test_name,
        family = "phasetype",
        parameterization = :none,
        covariates = :none,
        data_type = :panel,
        n_subjects = N_SUBJECTS,
        true_params = Dict(
            "λ1_12" => λ1_12_true, "λ2_12" => λ2_12_true,
            "μ1_12" => μ1_12_true, "μ2_12" => μ2_12_true, "μ3_12" => μ3_12_true,
            "λ1_23" => λ1_23_true, "λ2_23" => λ2_23_true,
            "μ1_23" => μ1_23_true, "μ2_23" => μ2_23_true, "μ3_23" => μ3_23_true,
        ),
        estimated_params = Dict(
            "λ1_12" => λ1_12_est, "λ2_12" => λ2_12_est,
            "μ1_12" => μ1_12_est, "μ2_12" => μ2_12_est, "μ3_12" => μ3_12_est,
            "λ1_23" => λ1_23_est, "λ2_23" => λ2_23_est,
            "μ1_23" => μ1_23_est, "μ2_23" => μ2_23_est, "μ3_23" => μ3_23_est,
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
        cumincid_23_fitted = cumincid_23_fitted
    )
    
    finalize_result!(result)
    push!(ALL_RESULTS, result)
    
    @info "$test_name completed" passed=result.passed max_rel_error=result.max_rel_error
    return result
end

# =============================================================================
# Test PT-5: Phase-Type (2-phase) with allequal structure, Exact Data
# =============================================================================

"""
    run_pt2_allequal_exact()

Test 2-phase Coxian with :allequal structure.
The structure constraint is applied during optimization.
User still provides all 2n-1 parameters, but set them equal for the DGP.
"""
function run_pt2_allequal_exact()
    test_name = "pt2_allequal_exact"
    @info "Running $test_name"
    
    Random.seed!(RNG_SEED + 4)
    
    # With :allequal, all rates are constrained equal during optimization
    # But we still provide 2n-1 parameters for the DGP
    # For 2-phase: [λ, μ₁, μ₂] where we set λ = μ₁ = μ₂ = rate
    rate_12_true = 0.4
    rate_23_true = 0.5
    
    h12 = Hazard(@formula(0 ~ 1), "pt", 1, 2; n_phases=2, coxian_structure=:allequal)
    h23 = Hazard(@formula(0 ~ 1), "pt", 2, 3; n_phases=2, coxian_structure=:allequal)
    
    dat = create_baseline_template(N_SUBJECTS; max_time=MAX_TIME)
    
    msm_sim = multistatemodel(h12, h23; data=dat)
    # Set all rates equal (consistent with :allequal structure)
    set_parameters!(msm_sim, (
        h12 = [rate_12_true, rate_12_true, rate_12_true],
        h23 = [rate_23_true, rate_23_true, rate_23_true]
    ))
    
    paths = simulate(msm_sim; paths=true, data=false)[1]
    exact_data = simulate(msm_sim; paths=false, data=true)[1]
    
    msm_fit = multistatemodel(h12, h23; data=exact_data)
    fitted = fit(msm_fit; verbose=false, compute_vcov=false)
    
    # Extract estimates - should all be approximately equal
    params = get_parameters(fitted)
    λ_12_est = params[:h12][1]
    μ1_12_est = params[:h12][2]
    μ2_12_est = params[:h12][3]
    λ_23_est = params[:h23][1]
    μ1_23_est = params[:h23][2]
    μ2_23_est = params[:h23][3]
    
    # Average the estimates for comparison (should all be equal due to constraint)
    rate_12_est = (λ_12_est + μ1_12_est + μ2_12_est) / 3
    rate_23_est = (λ_23_est + μ1_23_est + μ2_23_est) / 3
    
    rel_errs = Dict(
        "rate_12" => compute_relative_error(rate_12_true, rate_12_est),
        "rate_23" => compute_relative_error(rate_23_true, rate_23_est),
    )
    
    n_states = 3
    prev_true = compute_state_prevalence(paths, EVAL_TIMES, n_states)
    prev_obs = compute_prevalence_from_data(exact_data, EVAL_TIMES, n_states)
    
    dat_fit = create_baseline_template(N_SUBJECTS; max_time=MAX_TIME)
    msm_fit_sim = multistatemodel(h12, h23; data=dat_fit)
    set_parameters!(msm_fit_sim, get_parameters(fitted))
    paths_fitted = simulate(msm_fit_sim; paths=true, data=false)[1]
    prev_fitted = compute_state_prevalence(paths_fitted, EVAL_TIMES, n_states)
    
    cumincid_12_true = compute_cumulative_incidence(paths, EVAL_TIMES, 1, 2)
    cumincid_12_obs = compute_cumincid_from_data(exact_data, EVAL_TIMES, 1, 2)
    cumincid_12_fitted = compute_cumulative_incidence(paths_fitted, EVAL_TIMES, 1, 2)
    
    cumincid_23_true = compute_cumulative_incidence(paths, EVAL_TIMES, 2, 3)
    cumincid_23_obs = compute_cumincid_from_data(exact_data, EVAL_TIMES, 2, 3)
    cumincid_23_fitted = compute_cumulative_incidence(paths_fitted, EVAL_TIMES, 2, 3)
    
    result = TestResult(
        name = test_name,
        family = "phasetype",
        parameterization = :none,
        covariates = :none,
        data_type = :exact,
        n_subjects = N_SUBJECTS,
        true_params = Dict("rate_12" => rate_12_true, "rate_23" => rate_23_true),
        estimated_params = Dict("rate_12" => rate_12_est, "rate_23" => rate_23_est),
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
        cumincid_23_fitted = cumincid_23_fitted
    )
    
    finalize_result!(result)
    push!(ALL_RESULTS, result)
    
    @info "$test_name completed" passed=result.passed max_rel_error=result.max_rel_error
    return result
end

# =============================================================================
# Test PT-6: Phase-Type (2-phase) with allequal structure, Panel Data
# =============================================================================

"""
    run_pt2_allequal_panel()

Test 2-phase Coxian with :allequal structure and panel data.
"""
function run_pt2_allequal_panel()
    test_name = "pt2_allequal_panel"
    @info "Running $test_name"
    
    Random.seed!(RNG_SEED + 5)
    
    rate_12_true = 0.4
    rate_23_true = 0.5
    
    h12 = Hazard(@formula(0 ~ 1), "pt", 1, 2; n_phases=2, coxian_structure=:allequal)
    h23 = Hazard(@formula(0 ~ 1), "pt", 2, 3; n_phases=2, coxian_structure=:allequal)
    
    dat = create_baseline_template(N_SUBJECTS; max_time=MAX_TIME)
    
    msm_sim = multistatemodel(h12, h23; data=dat)
    # Set all rates equal (consistent with :allequal structure)
    set_parameters!(msm_sim, (
        h12 = [rate_12_true, rate_12_true, rate_12_true],
        h23 = [rate_23_true, rate_23_true, rate_23_true]
    ))
    
    paths = simulate(msm_sim; paths=true, data=false)[1]
    panel_data = create_panel_data(paths, collect(PANEL_TIMES), 3)
    
    msm_fit = multistatemodel(h12, h23; data=panel_data)
    fitted = fit(msm_fit; verbose=false, compute_vcov=false)
    
    params = get_parameters(fitted)
    λ_12_est = params[:h12][1]
    μ1_12_est = params[:h12][2]
    μ2_12_est = params[:h12][3]
    λ_23_est = params[:h23][1]
    μ1_23_est = params[:h23][2]
    μ2_23_est = params[:h23][3]
    
    # Average the estimates
    rate_12_est = (λ_12_est + μ1_12_est + μ2_12_est) / 3
    rate_23_est = (λ_23_est + μ1_23_est + μ2_23_est) / 3
    
    rel_errs = Dict(
        "rate_12" => compute_relative_error(rate_12_true, rate_12_est),
        "rate_23" => compute_relative_error(rate_23_true, rate_23_est),
    )
    
    n_states = 3
    prev_true = compute_state_prevalence(paths, EVAL_TIMES, n_states)
    prev_obs = compute_prevalence_from_data(panel_data, EVAL_TIMES, n_states)
    
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
    
    result = TestResult(
        name = test_name,
        family = "phasetype",
        parameterization = :none,
        covariates = :none,
        data_type = :panel,
        n_subjects = N_SUBJECTS,
        true_params = Dict("rate_12" => rate_12_true, "rate_23" => rate_23_true),
        estimated_params = Dict("rate_12" => rate_12_est, "rate_23" => rate_23_est),
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
        cumincid_23_fitted = cumincid_23_fitted
    )
    
    finalize_result!(result)
    push!(ALL_RESULTS, result)
    
    @info "$test_name completed" passed=result.passed max_rel_error=result.max_rel_error
    return result
end

# =============================================================================
# Test PT-7: Mixed Phase-Type and Exponential, Exact Data
# =============================================================================

"""
    run_pt_exp_mixed_exact()

Test model with h12 using phase-type and h23 using exponential.
This tests that phase-type hazards work correctly alongside standard hazards.
"""
function run_pt_exp_mixed_exact()
    test_name = "pt_exp_mixed_exact"
    @info "Running $test_name"
    
    Random.seed!(RNG_SEED + 6)
    
    # h12: 2-phase Coxian
    λ_12_true = 0.5
    μ1_12_true = 0.2
    μ2_12_true = 0.3
    
    # h23: Exponential
    λ_23_true = 0.4
    
    h12 = Hazard(@formula(0 ~ 1), "pt", 1, 2; n_phases=2, coxian_structure=:unstructured)
    h23 = Hazard(@formula(0 ~ 1), "exp", 2, 3)
    
    dat = create_baseline_template(N_SUBJECTS; max_time=MAX_TIME)
    
    msm_sim = multistatemodel(h12, h23; data=dat)
    set_parameters!(msm_sim, (
        h12 = [λ_12_true, μ1_12_true, μ2_12_true],
        h23 = [log(λ_23_true)]  # Exponential uses log scale
    ))
    
    paths = simulate(msm_sim; paths=true, data=false)[1]
    exact_data = simulate(msm_sim; paths=false, data=true)[1]
    
    msm_fit = multistatemodel(h12, h23; data=exact_data)
    fitted = fit(msm_fit; verbose=false, compute_vcov=false)
    
    params = get_parameters(fitted)
    λ_12_est = params[:h12][1]
    μ1_12_est = params[:h12][2]
    μ2_12_est = params[:h12][3]
    λ_23_est = exp(params[:h23][1])  # Exponential returns log scale
    
    rel_errs = Dict(
        "λ_12" => compute_relative_error(λ_12_true, λ_12_est),
        "μ1_12" => compute_relative_error(μ1_12_true, μ1_12_est),
        "μ2_12" => compute_relative_error(μ2_12_true, μ2_12_est),
        "λ_23" => compute_relative_error(λ_23_true, λ_23_est),
    )
    
    n_states = 3
    prev_true = compute_state_prevalence(paths, EVAL_TIMES, n_states)
    prev_obs = compute_prevalence_from_data(exact_data, EVAL_TIMES, n_states)
    
    dat_fit = create_baseline_template(N_SUBJECTS; max_time=MAX_TIME)
    msm_fit_sim = multistatemodel(h12, h23; data=dat_fit)
    set_parameters!(msm_fit_sim, get_parameters(fitted))
    paths_fitted = simulate(msm_fit_sim; paths=true, data=false)[1]
    prev_fitted = compute_state_prevalence(paths_fitted, EVAL_TIMES, n_states)
    
    cumincid_12_true = compute_cumulative_incidence(paths, EVAL_TIMES, 1, 2)
    cumincid_12_obs = compute_cumincid_from_data(exact_data, EVAL_TIMES, 1, 2)
    cumincid_12_fitted = compute_cumulative_incidence(paths_fitted, EVAL_TIMES, 1, 2)
    
    cumincid_23_true = compute_cumulative_incidence(paths, EVAL_TIMES, 2, 3)
    cumincid_23_obs = compute_cumincid_from_data(exact_data, EVAL_TIMES, 2, 3)
    cumincid_23_fitted = compute_cumulative_incidence(paths_fitted, EVAL_TIMES, 2, 3)
    
    result = TestResult(
        name = test_name,
        family = "phasetype",
        parameterization = :none,
        covariates = :none,
        data_type = :exact,
        n_subjects = N_SUBJECTS,
        true_params = Dict(
            "λ_12" => λ_12_true, "μ1_12" => μ1_12_true, "μ2_12" => μ2_12_true,
            "λ_23" => λ_23_true,
        ),
        estimated_params = Dict(
            "λ_12" => λ_12_est, "μ1_12" => μ1_12_est, "μ2_12" => μ2_12_est,
            "λ_23" => λ_23_est,
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
        cumincid_23_fitted = cumincid_23_fitted
    )
    
    finalize_result!(result)
    push!(ALL_RESULTS, result)
    
    @info "$test_name completed" passed=result.passed max_rel_error=result.max_rel_error
    return result
end

# =============================================================================
# Test PT-8: Mixed Phase-Type and Exponential, Panel Data
# =============================================================================

"""
    run_pt_exp_mixed_panel()

Test model with h12 using phase-type and h23 using exponential, panel data.
"""
function run_pt_exp_mixed_panel()
    test_name = "pt_exp_mixed_panel"
    @info "Running $test_name"
    
    Random.seed!(RNG_SEED + 7)
    
    λ_12_true = 0.5
    μ1_12_true = 0.2
    μ2_12_true = 0.3
    λ_23_true = 0.4
    
    h12 = Hazard(@formula(0 ~ 1), "pt", 1, 2; n_phases=2, coxian_structure=:unstructured)
    h23 = Hazard(@formula(0 ~ 1), "exp", 2, 3)
    
    dat = create_baseline_template(N_SUBJECTS; max_time=MAX_TIME)
    
    msm_sim = multistatemodel(h12, h23; data=dat)
    set_parameters!(msm_sim, (
        h12 = [λ_12_true, μ1_12_true, μ2_12_true],
        h23 = [log(λ_23_true)]
    ))
    
    paths = simulate(msm_sim; paths=true, data=false)[1]
    panel_data = create_panel_data(paths, collect(PANEL_TIMES), 3)
    
    msm_fit = multistatemodel(h12, h23; data=panel_data)
    fitted = fit(msm_fit; verbose=false, compute_vcov=false)
    
    params = get_parameters(fitted)
    λ_12_est = params[:h12][1]
    μ1_12_est = params[:h12][2]
    μ2_12_est = params[:h12][3]
    λ_23_est = exp(params[:h23][1])
    
    rel_errs = Dict(
        "λ_12" => compute_relative_error(λ_12_true, λ_12_est),
        "μ1_12" => compute_relative_error(μ1_12_true, μ1_12_est),
        "μ2_12" => compute_relative_error(μ2_12_true, μ2_12_est),
        "λ_23" => compute_relative_error(λ_23_true, λ_23_est),
    )
    
    n_states = 3
    prev_true = compute_state_prevalence(paths, EVAL_TIMES, n_states)
    prev_obs = compute_prevalence_from_data(panel_data, EVAL_TIMES, n_states)
    
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
    
    result = TestResult(
        name = test_name,
        family = "phasetype",
        parameterization = :none,
        covariates = :none,
        data_type = :panel,
        n_subjects = N_SUBJECTS,
        true_params = Dict(
            "λ_12" => λ_12_true, "μ1_12" => μ1_12_true, "μ2_12" => μ2_12_true,
            "λ_23" => λ_23_true,
        ),
        estimated_params = Dict(
            "λ_12" => λ_12_est, "μ1_12" => μ1_12_est, "μ2_12" => μ2_12_est,
            "λ_23" => λ_23_est,
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
        cumincid_23_fitted = cumincid_23_fitted
    )
    
    finalize_result!(result)
    push!(ALL_RESULTS, result)
    
    @info "$test_name completed" passed=result.passed max_rel_error=result.max_rel_error
    return result
end

# =============================================================================
# Test PT-9: Phase-Type (2-phase), PH + Time-Fixed Covariate, Exact Data
# =============================================================================

"""
    run_pt2_ph_tfc_exact()

Test 2-phase Coxian phase-type model with PH covariate effect and exact data.

For phase-type hazards:
- Baseline parameters: λ (progression), μ₁, μ₂ (exit rates)
- Covariate β applies to exit hazards via PH effect
- Parameters: [λ, μ₁, μ₂, β]
"""
function run_pt2_ph_tfc_exact()
    test_name = "pt2_ph_tfc_exact"
    @info "Running $test_name"
    
    Random.seed!(RNG_SEED + 10)
    
    # True parameters (natural scale)
    # h12: 2-phase Coxian with covariate
    λ_12_true = 0.5     # Progression rate (no covariate)
    μ1_12_true = 0.2    # Exit from phase 1
    μ2_12_true = 0.3    # Exit from phase 2
    β_12_true = 0.4     # Covariate effect on exit hazards
    
    # h23: 2-phase Coxian with covariate
    λ_23_true = 0.4
    μ1_23_true = 0.25
    μ2_23_true = 0.35
    β_23_true = 0.3
    
    # Create hazards with PH covariate effect
    h12 = Hazard(@formula(0 ~ 1 + x), "pt", 1, 2; n_phases=2, coxian_structure=:unstructured, linpred_effect=:ph)
    h23 = Hazard(@formula(0 ~ 1 + x), "pt", 2, 3; n_phases=2, coxian_structure=:unstructured, linpred_effect=:ph)
    
    # Create data template with covariate
    dat = create_tfc_template(N_SUBJECTS; max_time=MAX_TIME)
    
    msm_sim = multistatemodel(h12, h23; data=dat)
    
    # Set parameters: [λ, μ₁, μ₂, β]
    set_parameters!(msm_sim, (
        h12 = [λ_12_true, μ1_12_true, μ2_12_true, β_12_true],
        h23 = [λ_23_true, μ1_23_true, μ2_23_true, β_23_true]
    ))
    
    # Simulate exact data
    paths = simulate(msm_sim; paths=true, data=false)[1]
    exact_data = simulate(msm_sim; paths=false, data=true)[1]
    
    # Fit model
    msm_fit = multistatemodel(h12, h23; data=exact_data)
    fitted = fit(msm_fit; verbose=false, compute_vcov=false)
    
    # Extract estimates
    params = get_parameters(fitted)
    λ_12_est = params[:h12][1]
    μ1_12_est = params[:h12][2]
    μ2_12_est = params[:h12][3]
    β_12_est = params[:h12][4]
    λ_23_est = params[:h23][1]
    μ1_23_est = params[:h23][2]
    μ2_23_est = params[:h23][3]
    β_23_est = params[:h23][4]
    
    rel_errs = Dict(
        "λ_12" => compute_relative_error(λ_12_true, λ_12_est),
        "μ1_12" => compute_relative_error(μ1_12_true, μ1_12_est),
        "μ2_12" => compute_relative_error(μ2_12_true, μ2_12_est),
        "β_12" => compute_relative_error(β_12_true, β_12_est),
        "λ_23" => compute_relative_error(λ_23_true, λ_23_est),
        "μ1_23" => compute_relative_error(μ1_23_true, μ1_23_est),
        "μ2_23" => compute_relative_error(μ2_23_true, μ2_23_est),
        "β_23" => compute_relative_error(β_23_true, β_23_est),
    )
    
    n_states = 3
    prev_true = compute_state_prevalence(paths, EVAL_TIMES, n_states)
    prev_obs = compute_prevalence_from_data(exact_data, EVAL_TIMES, n_states)
    
    dat_fit = create_tfc_template(N_SUBJECTS; max_time=MAX_TIME)
    msm_fit_sim = multistatemodel(h12, h23; data=dat_fit)
    set_parameters!(msm_fit_sim, get_parameters(fitted))
    paths_fitted = simulate(msm_fit_sim; paths=true, data=false)[1]
    prev_fitted = compute_state_prevalence(paths_fitted, EVAL_TIMES, n_states)
    
    cumincid_12_true = compute_cumulative_incidence(paths, EVAL_TIMES, 1, 2)
    cumincid_12_obs = compute_cumincid_from_data(exact_data, EVAL_TIMES, 1, 2)
    cumincid_12_fitted = compute_cumulative_incidence(paths_fitted, EVAL_TIMES, 1, 2)
    
    cumincid_23_true = compute_cumulative_incidence(paths, EVAL_TIMES, 2, 3)
    cumincid_23_obs = compute_cumincid_from_data(exact_data, EVAL_TIMES, 2, 3)
    cumincid_23_fitted = compute_cumulative_incidence(paths_fitted, EVAL_TIMES, 2, 3)
    
    result = TestResult(
        name = test_name,
        family = "phasetype",
        parameterization = :ph,
        covariates = :tfc,
        data_type = :exact,
        n_subjects = N_SUBJECTS,
        true_params = Dict(
            "λ_12" => λ_12_true, "μ1_12" => μ1_12_true, "μ2_12" => μ2_12_true, "β_12" => β_12_true,
            "λ_23" => λ_23_true, "μ1_23" => μ1_23_true, "μ2_23" => μ2_23_true, "β_23" => β_23_true,
        ),
        estimated_params = Dict(
            "λ_12" => λ_12_est, "μ1_12" => μ1_12_est, "μ2_12" => μ2_12_est, "β_12" => β_12_est,
            "λ_23" => λ_23_est, "μ1_23" => μ1_23_est, "μ2_23" => μ2_23_est, "β_23" => β_23_est,
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
        cumincid_23_fitted = cumincid_23_fitted
    )
    
    finalize_result!(result)
    push!(ALL_RESULTS, result)
    
    @info "$test_name completed" passed=result.passed max_rel_error=result.max_rel_error
    return result
end

# =============================================================================
# Test PT-10: Phase-Type (2-phase), PH + Time-Fixed Covariate, Panel Data
# =============================================================================

"""
    run_pt2_ph_tfc_panel()

Test 2-phase Coxian phase-type model with PH covariate effect and panel data.
"""
function run_pt2_ph_tfc_panel()
    test_name = "pt2_ph_tfc_panel"
    @info "Running $test_name"
    
    Random.seed!(RNG_SEED + 11)
    
    λ_12_true = 0.5
    μ1_12_true = 0.2
    μ2_12_true = 0.3
    β_12_true = 0.4
    
    λ_23_true = 0.4
    μ1_23_true = 0.25
    μ2_23_true = 0.35
    β_23_true = 0.3
    
    h12 = Hazard(@formula(0 ~ 1 + x), "pt", 1, 2; n_phases=2, coxian_structure=:unstructured, linpred_effect=:ph)
    h23 = Hazard(@formula(0 ~ 1 + x), "pt", 2, 3; n_phases=2, coxian_structure=:unstructured, linpred_effect=:ph)
    
    dat = create_tfc_template(N_SUBJECTS; max_time=MAX_TIME)
    x_vals = dat.x
    
    msm_sim = multistatemodel(h12, h23; data=dat)
    set_parameters!(msm_sim, (
        h12 = [λ_12_true, μ1_12_true, μ2_12_true, β_12_true],
        h23 = [λ_23_true, μ1_23_true, μ2_23_true, β_23_true]
    ))
    
    paths = simulate(msm_sim; paths=true, data=false)[1]
    panel_data = create_panel_data_with_covariate(paths, collect(PANEL_TIMES), 3, x_vals)
    
    msm_fit = multistatemodel(h12, h23; data=panel_data)
    fitted = fit(msm_fit; verbose=false, compute_vcov=false)
    
    params = get_parameters(fitted)
    λ_12_est = params[:h12][1]
    μ1_12_est = params[:h12][2]
    μ2_12_est = params[:h12][3]
    β_12_est = params[:h12][4]
    λ_23_est = params[:h23][1]
    μ1_23_est = params[:h23][2]
    μ2_23_est = params[:h23][3]
    β_23_est = params[:h23][4]
    
    rel_errs = Dict(
        "λ_12" => compute_relative_error(λ_12_true, λ_12_est),
        "μ1_12" => compute_relative_error(μ1_12_true, μ1_12_est),
        "μ2_12" => compute_relative_error(μ2_12_true, μ2_12_est),
        "β_12" => compute_relative_error(β_12_true, β_12_est),
        "λ_23" => compute_relative_error(λ_23_true, λ_23_est),
        "μ1_23" => compute_relative_error(μ1_23_true, μ1_23_est),
        "μ2_23" => compute_relative_error(μ2_23_true, μ2_23_est),
        "β_23" => compute_relative_error(β_23_true, β_23_est),
    )
    
    n_states = 3
    prev_true = compute_state_prevalence(paths, EVAL_TIMES, n_states)
    prev_obs = compute_prevalence_from_data(panel_data, EVAL_TIMES, n_states)
    
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
    
    result = TestResult(
        name = test_name,
        family = "phasetype",
        parameterization = :ph,
        covariates = :tfc,
        data_type = :panel,
        n_subjects = N_SUBJECTS,
        true_params = Dict(
            "λ_12" => λ_12_true, "μ1_12" => μ1_12_true, "μ2_12" => μ2_12_true, "β_12" => β_12_true,
            "λ_23" => λ_23_true, "μ1_23" => μ1_23_true, "μ2_23" => μ2_23_true, "β_23" => β_23_true,
        ),
        estimated_params = Dict(
            "λ_12" => λ_12_est, "μ1_12" => μ1_12_est, "μ2_12" => μ2_12_est, "β_12" => β_12_est,
            "λ_23" => λ_23_est, "μ1_23" => μ1_23_est, "μ2_23" => μ2_23_est, "β_23" => β_23_est,
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
        cumincid_23_fitted = cumincid_23_fitted
    )
    
    finalize_result!(result)
    push!(ALL_RESULTS, result)
    
    @info "$test_name completed" passed=result.passed max_rel_error=result.max_rel_error
    return result
end

# =============================================================================
# Test PT-11: Phase-Type (2-phase), AFT + Time-Fixed Covariate, Exact Data
# =============================================================================

"""
    run_pt2_aft_tfc_exact()

Test 2-phase Coxian phase-type model with AFT covariate effect and exact data.
"""
function run_pt2_aft_tfc_exact()
    test_name = "pt2_aft_tfc_exact"
    @info "Running $test_name"
    
    Random.seed!(RNG_SEED + 12)
    
    λ_12_true = 0.5
    μ1_12_true = 0.2
    μ2_12_true = 0.3
    β_12_true = 0.3
    
    λ_23_true = 0.4
    μ1_23_true = 0.25
    μ2_23_true = 0.35
    β_23_true = 0.25
    
    h12 = Hazard(@formula(0 ~ 1 + x), "pt", 1, 2; n_phases=2, coxian_structure=:unstructured, linpred_effect=:aft)
    h23 = Hazard(@formula(0 ~ 1 + x), "pt", 2, 3; n_phases=2, coxian_structure=:unstructured, linpred_effect=:aft)
    
    dat = create_tfc_template(N_SUBJECTS; max_time=MAX_TIME)
    
    msm_sim = multistatemodel(h12, h23; data=dat)
    set_parameters!(msm_sim, (
        h12 = [λ_12_true, μ1_12_true, μ2_12_true, β_12_true],
        h23 = [λ_23_true, μ1_23_true, μ2_23_true, β_23_true]
    ))
    
    paths = simulate(msm_sim; paths=true, data=false)[1]
    exact_data = simulate(msm_sim; paths=false, data=true)[1]
    
    msm_fit = multistatemodel(h12, h23; data=exact_data)
    fitted = fit(msm_fit; verbose=false, compute_vcov=false)
    
    params = get_parameters(fitted)
    λ_12_est = params[:h12][1]
    μ1_12_est = params[:h12][2]
    μ2_12_est = params[:h12][3]
    β_12_est = params[:h12][4]
    λ_23_est = params[:h23][1]
    μ1_23_est = params[:h23][2]
    μ2_23_est = params[:h23][3]
    β_23_est = params[:h23][4]
    
    rel_errs = Dict(
        "λ_12" => compute_relative_error(λ_12_true, λ_12_est),
        "μ1_12" => compute_relative_error(μ1_12_true, μ1_12_est),
        "μ2_12" => compute_relative_error(μ2_12_true, μ2_12_est),
        "β_12" => compute_relative_error(β_12_true, β_12_est),
        "λ_23" => compute_relative_error(λ_23_true, λ_23_est),
        "μ1_23" => compute_relative_error(μ1_23_true, μ1_23_est),
        "μ2_23" => compute_relative_error(μ2_23_true, μ2_23_est),
        "β_23" => compute_relative_error(β_23_true, β_23_est),
    )
    
    n_states = 3
    prev_true = compute_state_prevalence(paths, EVAL_TIMES, n_states)
    prev_obs = compute_prevalence_from_data(exact_data, EVAL_TIMES, n_states)
    
    dat_fit = create_tfc_template(N_SUBJECTS; max_time=MAX_TIME)
    msm_fit_sim = multistatemodel(h12, h23; data=dat_fit)
    set_parameters!(msm_fit_sim, get_parameters(fitted))
    paths_fitted = simulate(msm_fit_sim; paths=true, data=false)[1]
    prev_fitted = compute_state_prevalence(paths_fitted, EVAL_TIMES, n_states)
    
    cumincid_12_true = compute_cumulative_incidence(paths, EVAL_TIMES, 1, 2)
    cumincid_12_obs = compute_cumincid_from_data(exact_data, EVAL_TIMES, 1, 2)
    cumincid_12_fitted = compute_cumulative_incidence(paths_fitted, EVAL_TIMES, 1, 2)
    
    cumincid_23_true = compute_cumulative_incidence(paths, EVAL_TIMES, 2, 3)
    cumincid_23_obs = compute_cumincid_from_data(exact_data, EVAL_TIMES, 2, 3)
    cumincid_23_fitted = compute_cumulative_incidence(paths_fitted, EVAL_TIMES, 2, 3)
    
    result = TestResult(
        name = test_name,
        family = "phasetype",
        parameterization = :aft,
        covariates = :tfc,
        data_type = :exact,
        n_subjects = N_SUBJECTS,
        true_params = Dict(
            "λ_12" => λ_12_true, "μ1_12" => μ1_12_true, "μ2_12" => μ2_12_true, "β_12" => β_12_true,
            "λ_23" => λ_23_true, "μ1_23" => μ1_23_true, "μ2_23" => μ2_23_true, "β_23" => β_23_true,
        ),
        estimated_params = Dict(
            "λ_12" => λ_12_est, "μ1_12" => μ1_12_est, "μ2_12" => μ2_12_est, "β_12" => β_12_est,
            "λ_23" => λ_23_est, "μ1_23" => μ1_23_est, "μ2_23" => μ2_23_est, "β_23" => β_23_est,
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
        cumincid_23_fitted = cumincid_23_fitted
    )
    
    finalize_result!(result)
    push!(ALL_RESULTS, result)
    
    @info "$test_name completed" passed=result.passed max_rel_error=result.max_rel_error
    return result
end

# =============================================================================
# Test PT-12: Phase-Type (2-phase), AFT + Time-Fixed Covariate, Panel Data
# =============================================================================

"""
    run_pt2_aft_tfc_panel()

Test 2-phase Coxian phase-type model with AFT covariate effect and panel data.
"""
function run_pt2_aft_tfc_panel()
    test_name = "pt2_aft_tfc_panel"
    @info "Running $test_name"
    
    Random.seed!(RNG_SEED + 13)
    
    λ_12_true = 0.5
    μ1_12_true = 0.2
    μ2_12_true = 0.3
    β_12_true = 0.3
    
    λ_23_true = 0.4
    μ1_23_true = 0.25
    μ2_23_true = 0.35
    β_23_true = 0.25
    
    h12 = Hazard(@formula(0 ~ 1 + x), "pt", 1, 2; n_phases=2, coxian_structure=:unstructured, linpred_effect=:aft)
    h23 = Hazard(@formula(0 ~ 1 + x), "pt", 2, 3; n_phases=2, coxian_structure=:unstructured, linpred_effect=:aft)
    
    dat = create_tfc_template(N_SUBJECTS; max_time=MAX_TIME)
    x_vals = dat.x
    
    msm_sim = multistatemodel(h12, h23; data=dat)
    set_parameters!(msm_sim, (
        h12 = [λ_12_true, μ1_12_true, μ2_12_true, β_12_true],
        h23 = [λ_23_true, μ1_23_true, μ2_23_true, β_23_true]
    ))
    
    paths = simulate(msm_sim; paths=true, data=false)[1]
    panel_data = create_panel_data_with_covariate(paths, collect(PANEL_TIMES), 3, x_vals)
    
    msm_fit = multistatemodel(h12, h23; data=panel_data)
    fitted = fit(msm_fit; verbose=false, compute_vcov=false)
    
    params = get_parameters(fitted)
    λ_12_est = params[:h12][1]
    μ1_12_est = params[:h12][2]
    μ2_12_est = params[:h12][3]
    β_12_est = params[:h12][4]
    λ_23_est = params[:h23][1]
    μ1_23_est = params[:h23][2]
    μ2_23_est = params[:h23][3]
    β_23_est = params[:h23][4]
    
    rel_errs = Dict(
        "λ_12" => compute_relative_error(λ_12_true, λ_12_est),
        "μ1_12" => compute_relative_error(μ1_12_true, μ1_12_est),
        "μ2_12" => compute_relative_error(μ2_12_true, μ2_12_est),
        "β_12" => compute_relative_error(β_12_true, β_12_est),
        "λ_23" => compute_relative_error(λ_23_true, λ_23_est),
        "μ1_23" => compute_relative_error(μ1_23_true, μ1_23_est),
        "μ2_23" => compute_relative_error(μ2_23_true, μ2_23_est),
        "β_23" => compute_relative_error(β_23_true, β_23_est),
    )
    
    n_states = 3
    prev_true = compute_state_prevalence(paths, EVAL_TIMES, n_states)
    prev_obs = compute_prevalence_from_data(panel_data, EVAL_TIMES, n_states)
    
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
    
    result = TestResult(
        name = test_name,
        family = "phasetype",
        parameterization = :aft,
        covariates = :tfc,
        data_type = :panel,
        n_subjects = N_SUBJECTS,
        true_params = Dict(
            "λ_12" => λ_12_true, "μ1_12" => μ1_12_true, "μ2_12" => μ2_12_true, "β_12" => β_12_true,
            "λ_23" => λ_23_true, "μ1_23" => μ1_23_true, "μ2_23" => μ2_23_true, "β_23" => β_23_true,
        ),
        estimated_params = Dict(
            "λ_12" => λ_12_est, "μ1_12" => μ1_12_est, "μ2_12" => μ2_12_est, "β_12" => β_12_est,
            "λ_23" => λ_23_est, "μ1_23" => μ1_23_est, "μ2_23" => μ2_23_est, "β_23" => β_23_est,
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
        cumincid_23_fitted = cumincid_23_fitted
    )
    
    finalize_result!(result)
    push!(ALL_RESULTS, result)
    
    @info "$test_name completed" passed=result.passed max_rel_error=result.max_rel_error
    return result
end

# =============================================================================
# Test PT-13: Phase-Type (2-phase), PH + Time-Varying Covariate, Exact Data
# =============================================================================

"""
    run_pt2_ph_tvc_exact()

Test 2-phase Coxian phase-type model with PH effect and time-varying covariate, exact data.
"""
function run_pt2_ph_tvc_exact()
    test_name = "pt2_ph_tvc_exact"
    @info "Running $test_name"
    
    Random.seed!(RNG_SEED + 14)
    
    λ_12_true = 0.5
    μ1_12_true = 0.2
    μ2_12_true = 0.3
    β_12_true = 0.4
    
    λ_23_true = 0.4
    μ1_23_true = 0.25
    μ2_23_true = 0.35
    β_23_true = 0.3
    
    h12 = Hazard(@formula(0 ~ 1 + x), "pt", 1, 2; n_phases=2, coxian_structure=:unstructured, linpred_effect=:ph)
    h23 = Hazard(@formula(0 ~ 1 + x), "pt", 2, 3; n_phases=2, coxian_structure=:unstructured, linpred_effect=:ph)
    
    dat = create_tvc_template(N_SUBJECTS; max_time=MAX_TIME)
    
    msm_sim = multistatemodel(h12, h23; data=dat)
    set_parameters!(msm_sim, (
        h12 = [λ_12_true, μ1_12_true, μ2_12_true, β_12_true],
        h23 = [λ_23_true, μ1_23_true, μ2_23_true, β_23_true]
    ))
    
    paths = simulate(msm_sim; paths=true, data=false)[1]
    exact_data = simulate(msm_sim; paths=false, data=true)[1]
    
    msm_fit = multistatemodel(h12, h23; data=exact_data)
    fitted = fit(msm_fit; verbose=false, compute_vcov=false)
    
    params = get_parameters(fitted)
    λ_12_est = params[:h12][1]
    μ1_12_est = params[:h12][2]
    μ2_12_est = params[:h12][3]
    β_12_est = params[:h12][4]
    λ_23_est = params[:h23][1]
    μ1_23_est = params[:h23][2]
    μ2_23_est = params[:h23][3]
    β_23_est = params[:h23][4]
    
    rel_errs = Dict(
        "λ_12" => compute_relative_error(λ_12_true, λ_12_est),
        "μ1_12" => compute_relative_error(μ1_12_true, μ1_12_est),
        "μ2_12" => compute_relative_error(μ2_12_true, μ2_12_est),
        "β_12" => compute_relative_error(β_12_true, β_12_est),
        "λ_23" => compute_relative_error(λ_23_true, λ_23_est),
        "μ1_23" => compute_relative_error(μ1_23_true, μ1_23_est),
        "μ2_23" => compute_relative_error(μ2_23_true, μ2_23_est),
        "β_23" => compute_relative_error(β_23_true, β_23_est),
    )
    
    n_states = 3
    prev_true = compute_state_prevalence(paths, EVAL_TIMES, n_states)
    prev_obs = compute_prevalence_from_data(exact_data, EVAL_TIMES, n_states)
    
    dat_fit = create_tvc_template(N_SUBJECTS; max_time=MAX_TIME)
    msm_fit_sim = multistatemodel(h12, h23; data=dat_fit)
    set_parameters!(msm_fit_sim, get_parameters(fitted))
    paths_fitted = simulate(msm_fit_sim; paths=true, data=false)[1]
    prev_fitted = compute_state_prevalence(paths_fitted, EVAL_TIMES, n_states)
    
    cumincid_12_true = compute_cumulative_incidence(paths, EVAL_TIMES, 1, 2)
    cumincid_12_obs = compute_cumincid_from_data(exact_data, EVAL_TIMES, 1, 2)
    cumincid_12_fitted = compute_cumulative_incidence(paths_fitted, EVAL_TIMES, 1, 2)
    
    cumincid_23_true = compute_cumulative_incidence(paths, EVAL_TIMES, 2, 3)
    cumincid_23_obs = compute_cumincid_from_data(exact_data, EVAL_TIMES, 2, 3)
    cumincid_23_fitted = compute_cumulative_incidence(paths_fitted, EVAL_TIMES, 2, 3)
    
    result = TestResult(
        name = test_name,
        family = "phasetype",
        parameterization = :ph,
        covariates = :tvc,
        data_type = :exact,
        n_subjects = N_SUBJECTS,
        true_params = Dict(
            "λ_12" => λ_12_true, "μ1_12" => μ1_12_true, "μ2_12" => μ2_12_true, "β_12" => β_12_true,
            "λ_23" => λ_23_true, "μ1_23" => μ1_23_true, "μ2_23" => μ2_23_true, "β_23" => β_23_true,
        ),
        estimated_params = Dict(
            "λ_12" => λ_12_est, "μ1_12" => μ1_12_est, "μ2_12" => μ2_12_est, "β_12" => β_12_est,
            "λ_23" => λ_23_est, "μ1_23" => μ1_23_est, "μ2_23" => μ2_23_est, "β_23" => β_23_est,
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
        cumincid_23_fitted = cumincid_23_fitted
    )
    
    finalize_result!(result)
    push!(ALL_RESULTS, result)
    
    @info "$test_name completed" passed=result.passed max_rel_error=result.max_rel_error
    return result
end

# =============================================================================
# Test PT-14: Phase-Type (2-phase), PH + Time-Varying Covariate, Panel Data
# =============================================================================

"""
    run_pt2_ph_tvc_panel()

Test 2-phase Coxian phase-type model with PH effect and time-varying covariate, panel data.
"""
function run_pt2_ph_tvc_panel()
    test_name = "pt2_ph_tvc_panel"
    @info "Running $test_name"
    
    Random.seed!(RNG_SEED + 15)
    
    λ_12_true = 0.5
    μ1_12_true = 0.2
    μ2_12_true = 0.3
    β_12_true = 0.4
    
    λ_23_true = 0.4
    μ1_23_true = 0.25
    μ2_23_true = 0.35
    β_23_true = 0.3
    
    h12 = Hazard(@formula(0 ~ 1 + x), "pt", 1, 2; n_phases=2, coxian_structure=:unstructured, linpred_effect=:ph)
    h23 = Hazard(@formula(0 ~ 1 + x), "pt", 2, 3; n_phases=2, coxian_structure=:unstructured, linpred_effect=:ph)
    
    dat = create_tvc_template(N_SUBJECTS; max_time=MAX_TIME)
    
    msm_sim = multistatemodel(h12, h23; data=dat)
    set_parameters!(msm_sim, (
        h12 = [λ_12_true, μ1_12_true, μ2_12_true, β_12_true],
        h23 = [λ_23_true, μ1_23_true, μ2_23_true, β_23_true]
    ))
    
    paths = simulate(msm_sim; paths=true, data=false)[1]
    panel_data = create_panel_data_with_tvc(paths, collect(PANEL_TIMES), 3, dat)
    
    msm_fit = multistatemodel(h12, h23; data=panel_data)
    fitted = fit(msm_fit; verbose=false, compute_vcov=false)
    
    params = get_parameters(fitted)
    λ_12_est = params[:h12][1]
    μ1_12_est = params[:h12][2]
    μ2_12_est = params[:h12][3]
    β_12_est = params[:h12][4]
    λ_23_est = params[:h23][1]
    μ1_23_est = params[:h23][2]
    μ2_23_est = params[:h23][3]
    β_23_est = params[:h23][4]
    
    rel_errs = Dict(
        "λ_12" => compute_relative_error(λ_12_true, λ_12_est),
        "μ1_12" => compute_relative_error(μ1_12_true, μ1_12_est),
        "μ2_12" => compute_relative_error(μ2_12_true, μ2_12_est),
        "β_12" => compute_relative_error(β_12_true, β_12_est),
        "λ_23" => compute_relative_error(λ_23_true, λ_23_est),
        "μ1_23" => compute_relative_error(μ1_23_true, μ1_23_est),
        "μ2_23" => compute_relative_error(μ2_23_true, μ2_23_est),
        "β_23" => compute_relative_error(β_23_true, β_23_est),
    )
    
    n_states = 3
    prev_true = compute_state_prevalence(paths, EVAL_TIMES, n_states)
    prev_obs = compute_prevalence_from_data(panel_data, EVAL_TIMES, n_states)
    
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
    
    result = TestResult(
        name = test_name,
        family = "phasetype",
        parameterization = :ph,
        covariates = :tvc,
        data_type = :panel,
        n_subjects = N_SUBJECTS,
        true_params = Dict(
            "λ_12" => λ_12_true, "μ1_12" => μ1_12_true, "μ2_12" => μ2_12_true, "β_12" => β_12_true,
            "λ_23" => λ_23_true, "μ1_23" => μ1_23_true, "μ2_23" => μ2_23_true, "β_23" => β_23_true,
        ),
        estimated_params = Dict(
            "λ_12" => λ_12_est, "μ1_12" => μ1_12_est, "μ2_12" => μ2_12_est, "β_12" => β_12_est,
            "λ_23" => λ_23_est, "μ1_23" => μ1_23_est, "μ2_23" => μ2_23_est, "β_23" => β_23_est,
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
        cumincid_23_fitted = cumincid_23_fitted
    )
    
    finalize_result!(result)
    push!(ALL_RESULTS, result)
    
    @info "$test_name completed" passed=result.passed max_rel_error=result.max_rel_error
    return result
end

# =============================================================================
# Test PT-15: Phase-Type (2-phase), AFT + Time-Varying Covariate, Exact Data
# =============================================================================

"""
    run_pt2_aft_tvc_exact()

Test 2-phase Coxian phase-type model with AFT effect and time-varying covariate, exact data.
"""
function run_pt2_aft_tvc_exact()
    test_name = "pt2_aft_tvc_exact"
    @info "Running $test_name"
    
    Random.seed!(RNG_SEED + 16)
    
    λ_12_true = 0.5
    μ1_12_true = 0.2
    μ2_12_true = 0.3
    β_12_true = 0.3
    
    λ_23_true = 0.4
    μ1_23_true = 0.25
    μ2_23_true = 0.35
    β_23_true = 0.25
    
    h12 = Hazard(@formula(0 ~ 1 + x), "pt", 1, 2; n_phases=2, coxian_structure=:unstructured, linpred_effect=:aft)
    h23 = Hazard(@formula(0 ~ 1 + x), "pt", 2, 3; n_phases=2, coxian_structure=:unstructured, linpred_effect=:aft)
    
    dat = create_tvc_template(N_SUBJECTS; max_time=MAX_TIME)
    
    msm_sim = multistatemodel(h12, h23; data=dat)
    set_parameters!(msm_sim, (
        h12 = [λ_12_true, μ1_12_true, μ2_12_true, β_12_true],
        h23 = [λ_23_true, μ1_23_true, μ2_23_true, β_23_true]
    ))
    
    paths = simulate(msm_sim; paths=true, data=false)[1]
    exact_data = simulate(msm_sim; paths=false, data=true)[1]
    
    msm_fit = multistatemodel(h12, h23; data=exact_data)
    fitted = fit(msm_fit; verbose=false, compute_vcov=false)
    
    params = get_parameters(fitted)
    λ_12_est = params[:h12][1]
    μ1_12_est = params[:h12][2]
    μ2_12_est = params[:h12][3]
    β_12_est = params[:h12][4]
    λ_23_est = params[:h23][1]
    μ1_23_est = params[:h23][2]
    μ2_23_est = params[:h23][3]
    β_23_est = params[:h23][4]
    
    rel_errs = Dict(
        "λ_12" => compute_relative_error(λ_12_true, λ_12_est),
        "μ1_12" => compute_relative_error(μ1_12_true, μ1_12_est),
        "μ2_12" => compute_relative_error(μ2_12_true, μ2_12_est),
        "β_12" => compute_relative_error(β_12_true, β_12_est),
        "λ_23" => compute_relative_error(λ_23_true, λ_23_est),
        "μ1_23" => compute_relative_error(μ1_23_true, μ1_23_est),
        "μ2_23" => compute_relative_error(μ2_23_true, μ2_23_est),
        "β_23" => compute_relative_error(β_23_true, β_23_est),
    )
    
    n_states = 3
    prev_true = compute_state_prevalence(paths, EVAL_TIMES, n_states)
    prev_obs = compute_prevalence_from_data(exact_data, EVAL_TIMES, n_states)
    
    dat_fit = create_tvc_template(N_SUBJECTS; max_time=MAX_TIME)
    msm_fit_sim = multistatemodel(h12, h23; data=dat_fit)
    set_parameters!(msm_fit_sim, get_parameters(fitted))
    paths_fitted = simulate(msm_fit_sim; paths=true, data=false)[1]
    prev_fitted = compute_state_prevalence(paths_fitted, EVAL_TIMES, n_states)
    
    cumincid_12_true = compute_cumulative_incidence(paths, EVAL_TIMES, 1, 2)
    cumincid_12_obs = compute_cumincid_from_data(exact_data, EVAL_TIMES, 1, 2)
    cumincid_12_fitted = compute_cumulative_incidence(paths_fitted, EVAL_TIMES, 1, 2)
    
    cumincid_23_true = compute_cumulative_incidence(paths, EVAL_TIMES, 2, 3)
    cumincid_23_obs = compute_cumincid_from_data(exact_data, EVAL_TIMES, 2, 3)
    cumincid_23_fitted = compute_cumulative_incidence(paths_fitted, EVAL_TIMES, 2, 3)
    
    result = TestResult(
        name = test_name,
        family = "phasetype",
        parameterization = :aft,
        covariates = :tvc,
        data_type = :exact,
        n_subjects = N_SUBJECTS,
        true_params = Dict(
            "λ_12" => λ_12_true, "μ1_12" => μ1_12_true, "μ2_12" => μ2_12_true, "β_12" => β_12_true,
            "λ_23" => λ_23_true, "μ1_23" => μ1_23_true, "μ2_23" => μ2_23_true, "β_23" => β_23_true,
        ),
        estimated_params = Dict(
            "λ_12" => λ_12_est, "μ1_12" => μ1_12_est, "μ2_12" => μ2_12_est, "β_12" => β_12_est,
            "λ_23" => λ_23_est, "μ1_23" => μ1_23_est, "μ2_23" => μ2_23_est, "β_23" => β_23_est,
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
        cumincid_23_fitted = cumincid_23_fitted
    )
    
    finalize_result!(result)
    push!(ALL_RESULTS, result)
    
    @info "$test_name completed" passed=result.passed max_rel_error=result.max_rel_error
    return result
end

# =============================================================================
# Test PT-16: Phase-Type (2-phase), AFT + Time-Varying Covariate, Panel Data
# =============================================================================

"""
    run_pt2_aft_tvc_panel()

Test 2-phase Coxian phase-type model with AFT effect and time-varying covariate, panel data.
"""
function run_pt2_aft_tvc_panel()
    test_name = "pt2_aft_tvc_panel"
    @info "Running $test_name"
    
    Random.seed!(RNG_SEED + 17)
    
    λ_12_true = 0.5
    μ1_12_true = 0.2
    μ2_12_true = 0.3
    β_12_true = 0.3
    
    λ_23_true = 0.4
    μ1_23_true = 0.25
    μ2_23_true = 0.35
    β_23_true = 0.25
    
    h12 = Hazard(@formula(0 ~ 1 + x), "pt", 1, 2; n_phases=2, coxian_structure=:unstructured, linpred_effect=:aft)
    h23 = Hazard(@formula(0 ~ 1 + x), "pt", 2, 3; n_phases=2, coxian_structure=:unstructured, linpred_effect=:aft)
    
    dat = create_tvc_template(N_SUBJECTS; max_time=MAX_TIME)
    
    msm_sim = multistatemodel(h12, h23; data=dat)
    set_parameters!(msm_sim, (
        h12 = [λ_12_true, μ1_12_true, μ2_12_true, β_12_true],
        h23 = [λ_23_true, μ1_23_true, μ2_23_true, β_23_true]
    ))
    
    paths = simulate(msm_sim; paths=true, data=false)[1]
    panel_data = create_panel_data_with_tvc(paths, collect(PANEL_TIMES), 3, dat)
    
    msm_fit = multistatemodel(h12, h23; data=panel_data)
    fitted = fit(msm_fit; verbose=false, compute_vcov=false)
    
    params = get_parameters(fitted)
    λ_12_est = params[:h12][1]
    μ1_12_est = params[:h12][2]
    μ2_12_est = params[:h12][3]
    β_12_est = params[:h12][4]
    λ_23_est = params[:h23][1]
    μ1_23_est = params[:h23][2]
    μ2_23_est = params[:h23][3]
    β_23_est = params[:h23][4]
    
    rel_errs = Dict(
        "λ_12" => compute_relative_error(λ_12_true, λ_12_est),
        "μ1_12" => compute_relative_error(μ1_12_true, μ1_12_est),
        "μ2_12" => compute_relative_error(μ2_12_true, μ2_12_est),
        "β_12" => compute_relative_error(β_12_true, β_12_est),
        "λ_23" => compute_relative_error(λ_23_true, λ_23_est),
        "μ1_23" => compute_relative_error(μ1_23_true, μ1_23_est),
        "μ2_23" => compute_relative_error(μ2_23_true, μ2_23_est),
        "β_23" => compute_relative_error(β_23_true, β_23_est),
    )
    
    n_states = 3
    prev_true = compute_state_prevalence(paths, EVAL_TIMES, n_states)
    prev_obs = compute_prevalence_from_data(panel_data, EVAL_TIMES, n_states)
    
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
    
    result = TestResult(
        name = test_name,
        family = "phasetype",
        parameterization = :aft,
        covariates = :tvc,
        data_type = :panel,
        n_subjects = N_SUBJECTS,
        true_params = Dict(
            "λ_12" => λ_12_true, "μ1_12" => μ1_12_true, "μ2_12" => μ2_12_true, "β_12" => β_12_true,
            "λ_23" => λ_23_true, "μ1_23" => μ1_23_true, "μ2_23" => μ2_23_true, "β_23" => β_23_true,
        ),
        estimated_params = Dict(
            "λ_12" => λ_12_est, "μ1_12" => μ1_12_est, "μ2_12" => μ2_12_est, "β_12" => β_12_est,
            "λ_23" => λ_23_est, "μ1_23" => μ1_23_est, "μ2_23" => μ2_23_est, "β_23" => β_23_est,
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
        cumincid_23_fitted = cumincid_23_fitted
    )
    
    finalize_result!(result)
    push!(ALL_RESULTS, result)
    
    @info "$test_name completed" passed=result.passed max_rel_error=result.max_rel_error
    return result
end

# =============================================================================
# Runner Function
# =============================================================================

"""
    run_all_phasetype_hazard_tests()

Run all phase-type HAZARD FAMILY tests.
These test the :pt hazard family, not phase-type proposals.
"""
function run_all_phasetype_hazard_tests()
    @info "Running all Phase-Type HAZARD FAMILY tests"
    results = TestResult[]
    
    # 2-phase unstructured, no covariates
    push!(results, run_pt2_nocov_exact())
    push!(results, run_pt2_nocov_panel())
    
    # 3-phase unstructured, no covariates
    push!(results, run_pt3_nocov_exact())
    push!(results, run_pt3_nocov_panel())
    
    # 2-phase with allequal constraint
    push!(results, run_pt2_allequal_exact())
    push!(results, run_pt2_allequal_panel())
    
    # Mixed phase-type + exponential
    push!(results, run_pt_exp_mixed_exact())
    push!(results, run_pt_exp_mixed_panel())
    
    # PH with time-fixed covariates
    push!(results, run_pt2_ph_tfc_exact())
    push!(results, run_pt2_ph_tfc_panel())
    
    # AFT with time-fixed covariates
    push!(results, run_pt2_aft_tfc_exact())
    push!(results, run_pt2_aft_tfc_panel())
    
    # PH with time-varying covariates
    push!(results, run_pt2_ph_tvc_exact())
    push!(results, run_pt2_ph_tvc_panel())
    
    # AFT with time-varying covariates
    push!(results, run_pt2_aft_tvc_exact())
    push!(results, run_pt2_aft_tvc_panel())
    
    return results
end
