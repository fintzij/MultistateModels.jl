# =============================================================================
# Gompertz Family Long Tests
# =============================================================================
#
# Tests 21-30: Gompertz hazard with various covariate settings and data types.
# Fitting: MLE for exact data, MCEM (Markov proposal) for panel data.
#
# 3-state progressive model: 1 → 2 → 3 (absorbing)
# =============================================================================

# =============================================================================
# Test 21: Gompertz, No Covariates, Exact Data
# =============================================================================

"""
    run_gom_nocov_exact()

Test Gompertz model with no covariates and exact data.
True parameters: η₁₂ = 0.1, γ₁₂ = 0.15, η₂₃ = 0.15, γ₂₃ = 0.1
"""
function run_gom_nocov_exact()
    test_name = "gom_nocov_exact"
    @info "Running $test_name"

    # True parameters
    η_12_true = 0.1
    γ_12_true = 0.15
    η_23_true = 0.15
    γ_23_true = 0.1

    # Create model for simulation
    h12 = Hazard(@formula(0 ~ 1), "gom", 1, 2)
    h23 = Hazard(@formula(0 ~ 1), "gom", 2, 3)

    dat = create_baseline_template(N_SUBJECTS; max_time=MAX_TIME)

    msm_sim = multistatemodel(h12, h23; data=dat)
    set_parameters!(msm_sim,
        (h12 = [log(γ_12_true), log(η_12_true)],
         h23 = [log(γ_23_true), log(η_23_true)]))

    # Simulate exact data
    paths = simulate(msm_sim; paths=true, data=false)[1]
    exact_data = simulate(msm_sim; paths=false, data=true)[1]

    # Create fitting model
    msm_fit = multistatemodel(h12, h23; data=exact_data)

    # Fit model
    fitted = fit(msm_fit)

    # Extract estimates
    params = get_parameters(fitted)
    γ_12_est = params[:h12][1]
    η_12_est = params[:h12][2]
    γ_23_est = params[:h23][1]
    η_23_est = params[:h23][2]

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
    prev_obs = compute_prevalence_from_data(exact_data, EVAL_TIMES, n_states)

    # Simulate from fitted model for prevalence
    set_parameters!(msm_sim, get_parameters(fitted))
    paths_fitted = simulate(msm_sim; paths=true, data=false)[1]
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
        family = "gompertz",
        parameterization = :none,
        covariates = :none,
        data_type = :exact,
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
# Test 22: Gompertz, No Covariates, Panel Data
# =============================================================================

"""
    run_gom_nocov_panel()

Test Gompertz model with no covariates and panel data.
Fitted using MCEM with Markov proposal.
"""
function run_gom_nocov_panel()
    test_name = "gom_nocov_panel"
    @info "Running $test_name"

    # True parameters
    η_12_true = 0.1
    γ_12_true = 0.15
    η_23_true = 0.15
    γ_23_true = 0.1

    # Create model for simulation
    h12 = Hazard(@formula(0 ~ 1), "gom", 1, 2)
    h23 = Hazard(@formula(0 ~ 1), "gom", 2, 3)

    dat = create_baseline_template(N_SUBJECTS; max_time=MAX_TIME)

    msm_sim = multistatemodel(h12, h23; data=dat)
    set_parameters!(msm_sim,
        (h12 = [log(γ_12_true), log(η_12_true)],
         h23 = [log(γ_23_true), log(η_23_true)]))

    # Simulate exact data and convert to panel
    paths = simulate(msm_sim; paths=true, data=false)[1]
    panel_data = create_panel_data(paths, collect(PANEL_TIMES), 3)

    # Create fitting model
    msm_fit = multistatemodel(h12, h23; data=panel_data, surrogate=:markov)

    # Fit model using MCEM
    fitted = fit(msm_fit;
        tol=MCEM_TOL,
        ess_target_initial=MCEM_ESS_INITIAL,
        ess_target_max=MCEM_ESS_MAX,
        maxiter=MCEM_MAX_ITER)

    # Extract estimates
    params = get_parameters(fitted)
    γ_12_est = params[:h12][1]
    η_12_est = params[:h12][2]
    γ_23_est = params[:h23][1]
    η_23_est = params[:h23][2]

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
        parameterization = :none,
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
# Test 23: Gompertz, PH + Time-Fixed Covariate, Exact Data
# =============================================================================

"""
    run_gom_ph_tfc_exact()

Test Gompertz model with proportional hazards and time-fixed covariate, exact data.
Model: h(t|x) = η * exp(γ * t) * exp(β * x)
True parameters: η₁₂ = 0.1, γ₁₂ = 0.15, β₁₂ = 0.5
                 η₂₃ = 0.15, γ₂₃ = 0.1, β₂₃ = 0.3
"""
function run_gom_ph_tfc_exact()
    test_name = "gom_ph_tfc_exact"
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

    Random.seed!(12353)
    dat = create_tfc_template(N_SUBJECTS; max_time=MAX_TIME)

    msm_sim = multistatemodel(h12, h23; data=dat)
    set_parameters!(msm_sim,
        (h12 = [log(γ_12_true), log(η_12_true), β_12_true],
         h23 = [log(γ_23_true), log(η_23_true), β_23_true]))

    # Simulate exact data
    paths = simulate(msm_sim; paths=true, data=false)[1]
    exact_data = simulate(msm_sim; paths=false, data=true)[1]

    # Create fitting model
    msm_fit = multistatemodel(h12, h23; data=exact_data)

    # Fit model
    fitted = fit(msm_fit)

    # Extract estimates
    params = get_parameters(fitted)
    γ_12_est = params[:h12][1]
    η_12_est = params[:h12][2]
    β_12_est = params[:h12][3]
    γ_23_est = params[:h23][1]
    η_23_est = params[:h23][2]
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
    prev_obs = compute_prevalence_from_data(exact_data, EVAL_TIMES, n_states)

    # Simulate from fitted model
    set_parameters!(msm_sim, get_parameters(fitted))
    paths_fitted = simulate(msm_sim; paths=true, data=false)[1]
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
        family = "gompertz",
        parameterization = :ph,
        covariates = :tfc,
        data_type = :exact,
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
# Test 24: Gompertz, PH + Time-Fixed Covariate, Panel Data
# =============================================================================

"""
    run_gom_ph_tfc_panel()

Test Gompertz model with proportional hazards and time-fixed covariate, panel data.
Fitted using MCEM with Markov proposal.
"""
function run_gom_ph_tfc_panel()
    test_name = "gom_ph_tfc_panel"
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

    Random.seed!(12354)
    dat = create_tfc_template(N_SUBJECTS; max_time=MAX_TIME)
    x_vals = dat.x

    msm_sim = multistatemodel(h12, h23; data=dat)
    set_parameters!(msm_sim,
        (h12 = [log(γ_12_true), log(η_12_true), β_12_true],
         h23 = [log(γ_23_true), log(η_23_true), β_23_true]))

    # Simulate and convert to panel
    paths = simulate(msm_sim; paths=true, data=false)[1]
    panel_data = create_panel_data_with_covariate(paths, collect(PANEL_TIMES), 3, x_vals)

    # Create fitting model
    msm_fit = multistatemodel(h12, h23; data=panel_data, surrogate=:markov)

    # Fit model using MCEM
    fitted = fit(msm_fit;
        tol=MCEM_TOL,
        ess_target_initial=MCEM_ESS_INITIAL,
        ess_target_max=MCEM_ESS_MAX,
        maxiter=MCEM_MAX_ITER)

    # Extract estimates
    params = get_parameters(fitted)
    γ_12_est = params[:h12][1]
    η_12_est = params[:h12][2]
    β_12_est = params[:h12][3]
    γ_23_est = params[:h23][1]
    η_23_est = params[:h23][2]
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
        family = "gompertz",
        parameterization = :ph,
        covariates = :tfc,
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
# Test 25: Gompertz, AFT + Time-Fixed Covariate, Exact Data
# =============================================================================

"""
    run_gom_aft_tfc_exact()

Test Gompertz model with accelerated failure time and time-fixed covariate, exact data.
"""
function run_gom_aft_tfc_exact()
    test_name = "gom_aft_tfc_exact"
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

    Random.seed!(12355)
    dat = create_tfc_template(N_SUBJECTS; max_time=MAX_TIME)

    msm_sim = multistatemodel(h12, h23; data=dat)
    set_parameters!(msm_sim,
        (h12 = [log(γ_12_true), log(η_12_true), β_12_true],
         h23 = [log(γ_23_true), log(η_23_true), β_23_true]))

    # Simulate exact data
    paths = simulate(msm_sim; paths=true, data=false)[1]
    exact_data = simulate(msm_sim; paths=false, data=true)[1]

    # Create fitting model
    msm_fit = multistatemodel(h12, h23; data=exact_data)

    # Fit model
    fitted = fit(msm_fit)

    # Extract estimates
    params = get_parameters(fitted)
    γ_12_est = params[:h12][1]
    η_12_est = params[:h12][2]
    β_12_est = params[:h12][3]
    γ_23_est = params[:h23][1]
    η_23_est = params[:h23][2]
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
    prev_obs = compute_prevalence_from_data(exact_data, EVAL_TIMES, n_states)

    # Simulate from fitted model
    set_parameters!(msm_sim, get_parameters(fitted))
    paths_fitted = simulate(msm_sim; paths=true, data=false)[1]
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
        family = "gompertz",
        parameterization = :aft,
        covariates = :tfc,
        data_type = :exact,
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
# Test 26: Gompertz, AFT + Time-Fixed Covariate, Panel Data
# =============================================================================

"""
    run_gom_aft_tfc_panel()

Test Gompertz model with accelerated failure time and time-fixed covariate, panel data.
Fitted using MCEM with Markov proposal.
"""
function run_gom_aft_tfc_panel()
    test_name = "gom_aft_tfc_panel"
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

    Random.seed!(12356)
    dat = create_tfc_template(N_SUBJECTS; max_time=MAX_TIME)
    x_vals = dat.x

    msm_sim = multistatemodel(h12, h23; data=dat)
    set_parameters!(msm_sim,
        (h12 = [log(γ_12_true), log(η_12_true), β_12_true],
         h23 = [log(γ_23_true), log(η_23_true), β_23_true]))

    # Simulate and convert to panel
    paths = simulate(msm_sim; paths=true, data=false)[1]
    panel_data = create_panel_data_with_covariate(paths, collect(PANEL_TIMES), 3, x_vals)

    # Create fitting model
    msm_fit = multistatemodel(h12, h23; data=panel_data, surrogate=:markov)

    # Fit model using MCEM
    fitted = fit(msm_fit;
        tol=MCEM_TOL,
        ess_target_initial=MCEM_ESS_INITIAL,
        ess_target_max=MCEM_ESS_MAX,
        maxiter=MCEM_MAX_ITER)

    # Extract estimates
    params = get_parameters(fitted)
    γ_12_est = params[:h12][1]
    η_12_est = params[:h12][2]
    β_12_est = params[:h12][3]
    γ_23_est = params[:h23][1]
    η_23_est = params[:h23][2]
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
        family = "gompertz",
        parameterization = :aft,
        covariates = :tfc,
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
# Test 27: Gompertz, PH + Time-Varying Covariate, Exact Data
# =============================================================================

"""
    run_gom_ph_tvc_exact()

Test Gompertz model with proportional hazards and time-varying covariate, exact data.
Covariate x(t) changes at t=5.
"""
function run_gom_ph_tvc_exact()
    test_name = "gom_ph_tvc_exact"
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

    Random.seed!(12357)
    dat = create_tvc_template(N_SUBJECTS; max_time=MAX_TIME)

    msm_sim = multistatemodel(h12, h23; data=dat)
    set_parameters!(msm_sim,
        (h12 = [log(γ_12_true), log(η_12_true), β_12_true],
         h23 = [log(γ_23_true), log(η_23_true), β_23_true]))

    # Simulate exact data
    paths = simulate(msm_sim; paths=true, data=false)[1]
    exact_data = simulate(msm_sim; paths=false, data=true)[1]

    # Create fitting model
    msm_fit = multistatemodel(h12, h23; data=exact_data)

    # Fit model
    fitted = fit(msm_fit)

    # Extract estimates
    params = get_parameters(fitted)
    γ_12_est = params[:h12][1]
    η_12_est = params[:h12][2]
    β_12_est = params[:h12][3]
    γ_23_est = params[:h23][1]
    η_23_est = params[:h23][2]
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
    prev_obs = compute_prevalence_from_data(exact_data, EVAL_TIMES, n_states)

    # Simulate from fitted model
    set_parameters!(msm_sim, get_parameters(fitted))
    paths_fitted = simulate(msm_sim; paths=true, data=false)[1]
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
        family = "gompertz",
        parameterization = :ph,
        covariates = :tvc,
        data_type = :exact,
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
# Test 28: Gompertz, PH + Time-Varying Covariate, Panel Data
# =============================================================================

"""
    run_gom_ph_tvc_panel()

Test Gompertz model with proportional hazards and time-varying covariate, panel data.
Fitted using MCEM with Markov proposal.
"""
function run_gom_ph_tvc_panel()
    test_name = "gom_ph_tvc_panel"
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

    Random.seed!(12358)
    dat = create_tvc_template(N_SUBJECTS; max_time=MAX_TIME)
    x_vals = dat.x

    msm_sim = multistatemodel(h12, h23; data=dat)
    set_parameters!(msm_sim,
        (h12 = [log(γ_12_true), log(η_12_true), β_12_true],
         h23 = [log(γ_23_true), log(η_23_true), β_23_true]))

    # Simulate and convert to panel
    paths = simulate(msm_sim; paths=true, data=false)[1]
    panel_data = create_panel_data_with_tvc(paths, collect(PANEL_TIMES), 3, x_vals)

    # Create fitting model
    msm_fit = multistatemodel(h12, h23; data=panel_data, surrogate=:markov)

    # Fit model using MCEM
    fitted = fit(msm_fit;
        tol=MCEM_TOL,
        ess_target_initial=MCEM_ESS_INITIAL,
        ess_target_max=MCEM_ESS_MAX,
        maxiter=MCEM_MAX_ITER)

    # Extract estimates
    params = get_parameters(fitted)
    γ_12_est = params[:h12][1]
    η_12_est = params[:h12][2]
    β_12_est = params[:h12][3]
    γ_23_est = params[:h23][1]
    η_23_est = params[:h23][2]
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
# Test 29: Gompertz, AFT + Time-Varying Covariate, Exact Data
# =============================================================================

"""
    run_gom_aft_tvc_exact()

Test Gompertz model with accelerated failure time and time-varying covariate, exact data.
"""
function run_gom_aft_tvc_exact()
    test_name = "gom_aft_tvc_exact"
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

    Random.seed!(12359)
    dat = create_tvc_template(N_SUBJECTS; max_time=MAX_TIME)

    msm_sim = multistatemodel(h12, h23; data=dat)
    set_parameters!(msm_sim,
        (h12 = [log(γ_12_true), log(η_12_true), β_12_true],
         h23 = [log(γ_23_true), log(η_23_true), β_23_true]))

    # Simulate exact data
    paths = simulate(msm_sim; paths=true, data=false)[1]
    exact_data = simulate(msm_sim; paths=false, data=true)[1]

    # Create fitting model
    msm_fit = multistatemodel(h12, h23; data=exact_data)

    # Fit model
    fitted = fit(msm_fit)

    # Extract estimates
    params = get_parameters(fitted)
    γ_12_est = params[:h12][1]
    η_12_est = params[:h12][2]
    β_12_est = params[:h12][3]
    γ_23_est = params[:h23][1]
    η_23_est = params[:h23][2]
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
    prev_obs = compute_prevalence_from_data(exact_data, EVAL_TIMES, n_states)

    # Simulate from fitted model
    set_parameters!(msm_sim, get_parameters(fitted))
    paths_fitted = simulate(msm_sim; paths=true, data=false)[1]
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
        family = "gompertz",
        parameterization = :aft,
        covariates = :tvc,
        data_type = :exact,
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
# Test 30: Gompertz, AFT + Time-Varying Covariate, Panel Data
# =============================================================================

"""
    run_gom_aft_tvc_panel()

Test Gompertz model with accelerated failure time and time-varying covariate, panel data.
Fitted using MCEM with Markov proposal.
"""
function run_gom_aft_tvc_panel()
    test_name = "gom_aft_tvc_panel"
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

    Random.seed!(12360)
    dat = create_tvc_template(N_SUBJECTS; max_time=MAX_TIME)
    x_vals = dat.x

    msm_sim = multistatemodel(h12, h23; data=dat)
    set_parameters!(msm_sim,
        (h12 = [log(γ_12_true), log(η_12_true), β_12_true],
         h23 = [log(γ_23_true), log(η_23_true), β_23_true]))

    # Simulate and convert to panel
    paths = simulate(msm_sim; paths=true, data=false)[1]
    panel_data = create_panel_data_with_tvc(paths, collect(PANEL_TIMES), 3, x_vals)

    # Create fitting model
    msm_fit = multistatemodel(h12, h23; data=panel_data, surrogate=:markov)

    # Fit model using MCEM
    fitted = fit(msm_fit;
        tol=MCEM_TOL,
        ess_target_initial=MCEM_ESS_INITIAL,
        ess_target_max=MCEM_ESS_MAX,
        maxiter=MCEM_MAX_ITER)

    # Extract estimates
    params = get_parameters(fitted)
    γ_12_est = params[:h12][1]
    η_12_est = params[:h12][2]
    β_12_est = params[:h12][3]
    γ_23_est = params[:h23][1]
    η_23_est = params[:h23][2]
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

"""
    run_all_gompertz_tests()

Run all 10 Gompertz family tests.
"""
function run_all_gompertz_tests()
    @info "Running all Gompertz family tests (10 tests)"
    results = TestResult[]
    
    push!(results, run_gom_nocov_exact())
    push!(results, run_gom_nocov_panel())
    push!(results, run_gom_ph_tfc_exact())
    push!(results, run_gom_ph_tfc_panel())
    push!(results, run_gom_aft_tfc_exact())
    push!(results, run_gom_aft_tfc_panel())
    push!(results, run_gom_ph_tvc_exact())
    push!(results, run_gom_ph_tvc_panel())
    push!(results, run_gom_aft_tvc_exact())
    push!(results, run_gom_aft_tvc_panel())
    
    return results
end





