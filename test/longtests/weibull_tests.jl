# =============================================================================
# Weibull Family Long Tests
# =============================================================================
#
# Tests 11-20: Weibull hazard with various covariate settings and data types.
# Fitting: MLE for exact data, MCEM (Markov proposal) for panel data.
#
# 3-state progressive model: 1 → 2 → 3 (absorbing)
# =============================================================================

# =============================================================================
# Test 11: Weibull, No Covariates, Exact Data
# =============================================================================

"""
    run_wei_nocov_exact()

Test Weibull model with no covariates and exact data.
True parameters: α₁₂ = 1.5, λ₁₂ = 0.15, α₂₃ = 1.3, λ₂₃ = 0.20
"""
function run_wei_nocov_exact()
    test_name = "wei_nocov_exact"
    @info "Running $test_name"

    # True parameters
    α_12_true = 1.5
    λ_12_true = 0.15
    α_23_true = 1.3
    λ_23_true = 0.20

    # Create model for simulation
    h12 = Hazard(@formula(0 ~ 1), "wei", 1, 2)
    h23 = Hazard(@formula(0 ~ 1), "wei", 2, 3)

    dat = create_baseline_template(N_SUBJECTS; max_time=MAX_TIME)

    msm_sim = multistatemodel(h12, h23; data=dat)
    set_parameters!(msm_sim,
        (h12 = [log(α_12_true), log(λ_12_true)],
         h23 = [log(α_23_true), log(λ_23_true)]))

    # Simulate exact data
    paths = simulate(msm_sim; paths=true, data=false)[1]
    exact_data = simulate(msm_sim; paths=false, data=true)[1]

    # Create fitting model
    msm_fit = multistatemodel(h12, h23; data=exact_data)

    # Fit model
    fitted = fit(msm_fit)

    # Extract estimates
    params = get_parameters(fitted)
    α_12_est = exp(params[:h12][1])
    λ_12_est = exp(params[:h12][2])
    α_23_est = exp(params[:h23][1])
    λ_23_est = exp(params[:h23][2])

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
        family = "weibull",
        parameterization = :none,
        covariates = :none,
        data_type = :exact,
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
# Test 12: Weibull, No Covariates, Panel Data
# =============================================================================

"""
    run_wei_nocov_panel()

Test Weibull model with no covariates and panel data.
Fitted using MCEM with Markov proposal.
"""
function run_wei_nocov_panel()
    test_name = "wei_nocov_panel"
    @info "Running $test_name"

    # True parameters
    α_12_true = 1.5
    λ_12_true = 0.15
    α_23_true = 1.3
    λ_23_true = 0.20

    # Create model for simulation
    h12 = Hazard(@formula(0 ~ 1), "wei", 1, 2)
    h23 = Hazard(@formula(0 ~ 1), "wei", 2, 3)

    dat = create_baseline_template(N_SUBJECTS; max_time=MAX_TIME)

    msm_sim = multistatemodel(h12, h23; data=dat)
    set_parameters!(msm_sim,
        (h12 = [log(α_12_true), log(λ_12_true)],
         h23 = [log(α_23_true), log(λ_23_true)]))

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
    α_12_est = exp(params[:h12][1])
    λ_12_est = exp(params[:h12][2])
    α_23_est = exp(params[:h23][1])
    λ_23_est = exp(params[:h23][2])

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
        parameterization = :none,
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
# Test 13: Weibull, PH + Time-Fixed Covariate, Exact Data
# =============================================================================

"""
    run_wei_ph_tfc_exact()

Test Weibull model with proportional hazards and time-fixed covariate, exact data.
Model: h(t|x) = λ * α * t^(α-1) * exp(β * x)
True parameters: α₁₂ = 1.5, λ₁₂ = 0.15, β₁₂ = 0.5
                 α₂₃ = 1.3, λ₂₃ = 0.20, β₂₃ = 0.3
"""
function run_wei_ph_tfc_exact()
    test_name = "wei_ph_tfc_exact"
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

    Random.seed!(12345)
    dat = create_tfc_template(N_SUBJECTS; max_time=MAX_TIME)

    msm_sim = multistatemodel(h12, h23; data=dat)
    set_parameters!(msm_sim,
        (h12 = [log(α_12_true), log(λ_12_true)], β_12_true],
         h23 = [log(α_23_true), log(λ_23_true)], β_23_true]))

    # Simulate exact data
    paths = simulate(msm_sim; paths=true, data=false)[1]
    exact_data = simulate(msm_sim; paths=false, data=true)[1]

    # Create fitting model
    msm_fit = multistatemodel(h12, h23; data=exact_data)

    # Fit model
    fitted = fit(msm_fit)

    # Extract estimates
    params = get_parameters(fitted)
    α_12_est = exp(params[:h12][1])
    λ_12_est = exp(params[:h12][2])
    β_12_est = params[:h12][3]
    α_23_est = exp(params[:h23][1])
    λ_23_est = exp(params[:h23][2])
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
        family = "weibull",
        parameterization = :ph,
        covariates = :tfc,
        data_type = :exact,
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
# Test 14: Weibull, PH + Time-Fixed Covariate, Panel Data
# =============================================================================

"""
    run_wei_ph_tfc_panel()

Test Weibull model with proportional hazards and time-fixed covariate, panel data.
Fitted using MCEM with Markov proposal.
"""
function run_wei_ph_tfc_panel()
    test_name = "wei_ph_tfc_panel"
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

    Random.seed!(12346)
    dat = create_tfc_template(N_SUBJECTS; max_time=MAX_TIME)
    x_vals = dat.x

    msm_sim = multistatemodel(h12, h23; data=dat)
    set_parameters!(msm_sim,
        (h12 = [log(α_12_true), log(λ_12_true)], β_12_true],
         h23 = [log(α_23_true), log(λ_23_true)], β_23_true]))

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
    α_12_est = exp(params[:h12][1])
    λ_12_est = exp(params[:h12][2])
    β_12_est = params[:h12][3]
    α_23_est = exp(params[:h23][1])
    λ_23_est = exp(params[:h23][2])
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
# Test 15: Weibull, AFT + Time-Fixed Covariate, Exact Data
# =============================================================================

"""
    run_wei_aft_tfc_exact()

Test Weibull model with accelerated failure time and time-fixed covariate, exact data.
"""
function run_wei_aft_tfc_exact()
    test_name = "wei_aft_tfc_exact"
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

    Random.seed!(12347)
    dat = create_tfc_template(N_SUBJECTS; max_time=MAX_TIME)

    msm_sim = multistatemodel(h12, h23; data=dat)
    set_parameters!(msm_sim,
        (h12 = [log(α_12_true), log(λ_12_true)], β_12_true],
         h23 = [log(α_23_true), log(λ_23_true)], β_23_true]))

    # Simulate exact data
    paths = simulate(msm_sim; paths=true, data=false)[1]
    exact_data = simulate(msm_sim; paths=false, data=true)[1]

    # Create fitting model
    msm_fit = multistatemodel(h12, h23; data=exact_data)

    # Fit model
    fitted = fit(msm_fit)

    # Extract estimates
    params = get_parameters(fitted)
    α_12_est = exp(params[:h12][1])
    λ_12_est = exp(params[:h12][2])
    β_12_est = params[:h12][3]
    α_23_est = exp(params[:h23][1])
    λ_23_est = exp(params[:h23][2])
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
        family = "weibull",
        parameterization = :aft,
        covariates = :tfc,
        data_type = :exact,
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
# Test 16: Weibull, AFT + Time-Fixed Covariate, Panel Data
# =============================================================================

"""
    run_wei_aft_tfc_panel()

Test Weibull model with accelerated failure time and time-fixed covariate, panel data.
Fitted using MCEM with Markov proposal.
"""
function run_wei_aft_tfc_panel()
    test_name = "wei_aft_tfc_panel"
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

    Random.seed!(12348)
    dat = create_tfc_template(N_SUBJECTS; max_time=MAX_TIME)
    x_vals = dat.x

    msm_sim = multistatemodel(h12, h23; data=dat)
    set_parameters!(msm_sim,
        (h12 = [log(α_12_true), log(λ_12_true)], β_12_true],
         h23 = [log(α_23_true), log(λ_23_true)], β_23_true]))

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
    α_12_est = exp(params[:h12][1])
    λ_12_est = exp(params[:h12][2])
    β_12_est = params[:h12][3]
    α_23_est = exp(params[:h23][1])
    λ_23_est = exp(params[:h23][2])
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
# Test 17: Weibull, PH + Time-Varying Covariate, Exact Data
# =============================================================================

"""
    run_wei_ph_tvc_exact()

Test Weibull model with proportional hazards and time-varying covariate, exact data.
Covariate x(t) changes at t=5.
"""
function run_wei_ph_tvc_exact()
    test_name = "wei_ph_tvc_exact"
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

    Random.seed!(12349)
    dat = create_tvc_template(N_SUBJECTS; max_time=MAX_TIME)

    msm_sim = multistatemodel(h12, h23; data=dat)
    set_parameters!(msm_sim,
        (h12 = [log(α_12_true), log(λ_12_true)], β_12_true],
         h23 = [log(α_23_true), log(λ_23_true)], β_23_true]))

    # Simulate exact data
    paths = simulate(msm_sim; paths=true, data=false)[1]
    exact_data = simulate(msm_sim; paths=false, data=true)[1]

    # Create fitting model
    msm_fit = multistatemodel(h12, h23; data=exact_data)

    # Fit model
    fitted = fit(msm_fit)

    # Extract estimates
    params = get_parameters(fitted)
    α_12_est = exp(params[:h12][1])
    λ_12_est = exp(params[:h12][2])
    β_12_est = params[:h12][3]
    α_23_est = exp(params[:h23][1])
    λ_23_est = exp(params[:h23][2])
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
        family = "weibull",
        parameterization = :ph,
        covariates = :tvc,
        data_type = :exact,
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
# Test 18: Weibull, PH + Time-Varying Covariate, Panel Data
# =============================================================================

"""
    run_wei_ph_tvc_panel()

Test Weibull model with proportional hazards and time-varying covariate, panel data.
Fitted using MCEM with Markov proposal.
"""
function run_wei_ph_tvc_panel()
    test_name = "wei_ph_tvc_panel"
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

    Random.seed!(12350)
    dat = create_tvc_template(N_SUBJECTS; max_time=MAX_TIME)
    x_vals = dat.x

    msm_sim = multistatemodel(h12, h23; data=dat)
    set_parameters!(msm_sim,
        (h12 = [log(α_12_true), log(λ_12_true)], β_12_true],
         h23 = [log(α_23_true), log(λ_23_true)], β_23_true]))

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
    α_12_est = exp(params[:h12][1])
    λ_12_est = exp(params[:h12][2])
    β_12_est = params[:h12][3]
    α_23_est = exp(params[:h23][1])
    λ_23_est = exp(params[:h23][2])
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
        family = "weibull",
        parameterization = :ph,
        covariates = :tvc,
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
# Test 19: Weibull, AFT + Time-Varying Covariate, Exact Data
# =============================================================================

"""
    run_wei_aft_tvc_exact()

Test Weibull model with accelerated failure time and time-varying covariate, exact data.
"""
function run_wei_aft_tvc_exact()
    test_name = "wei_aft_tvc_exact"
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

    Random.seed!(12351)
    dat = create_tvc_template(N_SUBJECTS; max_time=MAX_TIME)

    msm_sim = multistatemodel(h12, h23; data=dat)
    set_parameters!(msm_sim,
        (h12 = [log(α_12_true), log(λ_12_true)], β_12_true],
         h23 = [log(α_23_true), log(λ_23_true)], β_23_true]))

    # Simulate exact data
    paths = simulate(msm_sim; paths=true, data=false)[1]
    exact_data = simulate(msm_sim; paths=false, data=true)[1]

    # Create fitting model
    msm_fit = multistatemodel(h12, h23; data=exact_data)

    # Fit model
    fitted = fit(msm_fit)

    # Extract estimates
    params = get_parameters(fitted)
    α_12_est = exp(params[:h12][1])
    λ_12_est = exp(params[:h12][2])
    β_12_est = params[:h12][3]
    α_23_est = exp(params[:h23][1])
    λ_23_est = exp(params[:h23][2])
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
        family = "weibull",
        parameterization = :aft,
        covariates = :tvc,
        data_type = :exact,
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
# Test 20: Weibull, AFT + Time-Varying Covariate, Panel Data
# =============================================================================

"""
    run_wei_aft_tvc_panel()

Test Weibull model with accelerated failure time and time-varying covariate, panel data.
Fitted using MCEM with Markov proposal.
"""
function run_wei_aft_tvc_panel()
    test_name = "wei_aft_tvc_panel"
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

    Random.seed!(12352)
    dat = create_tvc_template(N_SUBJECTS; max_time=MAX_TIME)
    x_vals = dat.x

    msm_sim = multistatemodel(h12, h23; data=dat)
    set_parameters!(msm_sim,
        (h12 = [log(α_12_true), log(λ_12_true)], β_12_true],
         h23 = [log(α_23_true), log(λ_23_true)], β_23_true]))

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
    α_12_est = exp(params[:h12][1])
    λ_12_est = exp(params[:h12][2])
    β_12_est = params[:h12][3]
    α_23_est = exp(params[:h23][1])
    λ_23_est = exp(params[:h23][2])
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
        family = "weibull",
        parameterization = :aft,
        covariates = :tvc,
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

"""
    run_all_weibull_tests()

Run all 10 Weibull family tests.
"""
function run_all_weibull_tests()
    @info "Running all Weibull family tests (10 tests)"
    results = TestResult[]
    
    push!(results, run_wei_nocov_exact())
    push!(results, run_wei_nocov_panel())
    push!(results, run_wei_ph_tfc_exact())
    push!(results, run_wei_ph_tfc_panel())
    push!(results, run_wei_aft_tfc_exact())
    push!(results, run_wei_aft_tfc_panel())
    push!(results, run_wei_ph_tvc_exact())
    push!(results, run_wei_ph_tvc_panel())
    push!(results, run_wei_aft_tvc_exact())
    push!(results, run_wei_aft_tvc_panel())
    
    return results
end





