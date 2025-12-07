# =============================================================================
# Exponential Family Long Tests
# =============================================================================
#
# Tests 1-10: Exponential hazard with various covariate settings and data types.
# Fitting: MLE for exact data, Markov for panel data.
#
# 3-state progressive model: 1 → 2 → 3 (absorbing)
# =============================================================================

# =============================================================================
# Test 1: Exponential, No Covariates, Exact Data
# =============================================================================

"""
    run_exp_nocov_exact()

Test exponential model with no covariates and exact data.
True parameters: λ₁₂ = 0.2, λ₂₃ = 0.3
"""
function run_exp_nocov_exact()
    test_name = "exp_nocov_exact"
    @info "Running $test_name"
    
    # True parameters
    λ_12_true = 0.2
    λ_23_true = 0.3
    
    # Create model for simulation
    h12 = Hazard(@formula(0 ~ 1), "exp", 1, 2)
    h23 = Hazard(@formula(0 ~ 1), "exp", 2, 3)
    
    dat = create_baseline_template(N_SUBJECTS; max_time=MAX_TIME)
    
    msm_sim = multistatemodel(h12, h23; data=dat)
    set_parameters!(msm_sim, 
        (h12 = [log(λ_12_true)],
         h23 = [log(λ_23_true)]))
    
    # Simulate exact data
    paths = simulate(msm_sim; paths=true, data=false)[1]
    exact_data = simulate(msm_sim; paths=false, data=true)[1]
    
    # Create fitting model
    msm_fit = multistatemodel(h12, h23; data=exact_data)
    
    # Fit model
    fitted = fit(msm_fit)
    
    # Extract estimates
    params = get_parameters(fitted)
    λ_12_est = exp(params[:h12][1])
    λ_23_est = exp(params[:h23][1])
    
    # Compute relative errors
    rel_err_12 = compute_relative_error(λ_12_true, λ_12_est)
    rel_err_23 = compute_relative_error(λ_23_true, λ_23_est)
    
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
        family = "exponential",
        parameterization = :none,
        covariates = :none,
        data_type = :exact,
        n_subjects = N_SUBJECTS,
        true_params = Dict("λ_12" => λ_12_true, "λ_23" => λ_23_true),
        estimated_params = Dict("λ_12" => λ_12_est, "λ_23" => λ_23_est),
        rel_errors = Dict("λ_12" => rel_err_12, "λ_23" => rel_err_23),
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
# Test 2: Exponential, No Covariates, Panel Data
# =============================================================================

"""
    run_exp_nocov_panel()

Test exponential model with no covariates and panel data.
Fitted using Markov likelihood.
"""
function run_exp_nocov_panel()
    test_name = "exp_nocov_panel"
    @info "Running $test_name"
    
    # True parameters
    λ_12_true = 0.2
    λ_23_true = 0.3
    
    # Create model for simulation
    h12 = Hazard(@formula(0 ~ 1), "exp", 1, 2)
    h23 = Hazard(@formula(0 ~ 1), "exp", 2, 3)
    
    dat = create_baseline_template(N_SUBJECTS; max_time=MAX_TIME)
    
    msm_sim = multistatemodel(h12, h23; data=dat)
    set_parameters!(msm_sim, 
        (h12 = [log(λ_12_true)],
         h23 = [log(λ_23_true)]))
    
    # Simulate exact data and convert to panel
    paths = simulate(msm_sim; paths=true, data=false)[1]
    panel_data = create_panel_data(paths, collect(PANEL_TIMES), 3)
    
    # Create fitting model
    msm_fit = multistatemodel(h12, h23; data=panel_data)
    
    # Fit model using Markov likelihood
    fitted = fit(msm_fit)
    
    # Extract estimates
    params = get_parameters(fitted)
    λ_12_est = exp(params[:h12][1])
    λ_23_est = exp(params[:h23][1])
    
    # Compute relative errors
    rel_err_12 = compute_relative_error(λ_12_true, λ_12_est)
    rel_err_23 = compute_relative_error(λ_23_true, λ_23_est)
    
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
        family = "exponential",
        parameterization = :none,
        covariates = :none,
        data_type = :panel,
        n_subjects = N_SUBJECTS,
        true_params = Dict("λ_12" => λ_12_true, "λ_23" => λ_23_true),
        estimated_params = Dict("λ_12" => λ_12_est, "λ_23" => λ_23_est),
        rel_errors = Dict("λ_12" => rel_err_12, "λ_23" => rel_err_23),
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
# Test 3: Exponential, PH + Time-Fixed Covariate, Exact Data
# =============================================================================

"""
    run_exp_ph_tfc_exact()

Test exponential model with proportional hazards and time-fixed covariate, exact data.
Model: h(t|x) = λ * exp(β * x)
True parameters: λ₁₂ = 0.2, β₁₂ = 0.5, λ₂₃ = 0.3, β₂₃ = 0.3
"""
function run_exp_ph_tfc_exact()
    test_name = "exp_ph_tfc_exact"
    @info "Running $test_name"
    
    # True parameters
    λ_12_true = 0.2
    β_12_true = 0.5
    λ_23_true = 0.3
    β_23_true = 0.3
    
    # Create model for simulation
    h12 = Hazard(@formula(0 ~ 1 + x), "exp", 1, 2; linpred_effect=:ph)
    h23 = Hazard(@formula(0 ~ 1 + x), "exp", 2, 3; linpred_effect=:ph)
    
    Random.seed!(12345)
    dat = create_tfc_template(N_SUBJECTS; max_time=MAX_TIME)
    
    msm_sim = multistatemodel(h12, h23; data=dat)
    set_parameters!(msm_sim, 
        (h12 = [log(λ_12_true), β_12_true],
         h23 = [log(λ_23_true), β_23_true]))
    
    # Simulate exact data
    paths = simulate(msm_sim; paths=true, data=false)[1]
    exact_data = simulate(msm_sim; paths=false, data=true)[1]
    
    # Create fitting model
    msm_fit = multistatemodel(h12, h23; data=exact_data)
    
    # Fit model
    fitted = fit(msm_fit)
    
    # Extract estimates
    params = get_parameters(fitted)
    λ_12_est = exp(params[:h12][1])
    β_12_est = params[:h12][2]
    λ_23_est = exp(params[:h23][1])
    β_23_est = params[:h23][2]
    
    # Compute relative errors
    rel_errs = Dict(
        "λ_12" => compute_relative_error(λ_12_true, λ_12_est),
        "β_12" => compute_relative_error(β_12_true, β_12_est),
        "λ_23" => compute_relative_error(λ_23_true, λ_23_est),
        "β_23" => compute_relative_error(β_23_true, β_23_est)
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
        family = "exponential",
        parameterization = :ph,
        covariates = :tfc,
        data_type = :exact,
        n_subjects = N_SUBJECTS,
        true_params = Dict("λ_12" => λ_12_true, "β_12" => β_12_true, 
                          "λ_23" => λ_23_true, "β_23" => β_23_true),
        estimated_params = Dict("λ_12" => λ_12_est, "β_12" => β_12_est,
                               "λ_23" => λ_23_est, "β_23" => β_23_est),
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
# Test 4: Exponential, PH + Time-Fixed Covariate, Panel Data
# =============================================================================

"""
    run_exp_ph_tfc_panel()

Test exponential model with proportional hazards and time-fixed covariate, panel data.
Fitted using Markov likelihood.
"""
function run_exp_ph_tfc_panel()
    test_name = "exp_ph_tfc_panel"
    @info "Running $test_name"
    
    # True parameters
    λ_12_true = 0.2
    β_12_true = 0.5
    λ_23_true = 0.3
    β_23_true = 0.3
    
    # Create model for simulation
    h12 = Hazard(@formula(0 ~ 1 + x), "exp", 1, 2; linpred_effect=:ph)
    h23 = Hazard(@formula(0 ~ 1 + x), "exp", 2, 3; linpred_effect=:ph)
    
    Random.seed!(12346)
    dat = create_tfc_template(N_SUBJECTS; max_time=MAX_TIME)
    x_vals = dat.x
    
    msm_sim = multistatemodel(h12, h23; data=dat)
    set_parameters!(msm_sim, 
        (h12 = [log(λ_12_true), β_12_true],
         h23 = [log(λ_23_true), β_23_true]))
    
    # Simulate and convert to panel
    paths = simulate(msm_sim; paths=true, data=false)[1]
    panel_data = create_panel_data_with_covariate(paths, collect(PANEL_TIMES), 3, x_vals)
    
    # Create fitting model
    msm_fit = multistatemodel(h12, h23; data=panel_data)
    
    # Fit model
    fitted = fit(msm_fit)
    
    # Extract estimates
    params = get_parameters(fitted)
    λ_12_est = exp(params[:h12][1])
    β_12_est = params[:h12][2]
    λ_23_est = exp(params[:h23][1])
    β_23_est = params[:h23][2]
    
    # Compute relative errors
    rel_errs = Dict(
        "λ_12" => compute_relative_error(λ_12_true, λ_12_est),
        "β_12" => compute_relative_error(β_12_true, β_12_est),
        "λ_23" => compute_relative_error(λ_23_true, λ_23_est),
        "β_23" => compute_relative_error(β_23_true, β_23_est)
    )
    
    # Compute prevalence and cumulative incidence
    n_states = 3
    prev_true = compute_state_prevalence(paths, EVAL_TIMES, n_states)
    prev_obs = compute_prevalence_from_data(panel_data, EVAL_TIMES, n_states)
    
    # Simulate from fitted model
    set_parameters!(msm_sim, get_parameters(fitted))
    paths_fitted = simulate(msm_sim; paths=true, data=false)[1]
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
        family = "exponential",
        parameterization = :ph,
        covariates = :tfc,
        data_type = :panel,
        n_subjects = N_SUBJECTS,
        true_params = Dict("λ_12" => λ_12_true, "β_12" => β_12_true, 
                          "λ_23" => λ_23_true, "β_23" => β_23_true),
        estimated_params = Dict("λ_12" => λ_12_est, "β_12" => β_12_est,
                               "λ_23" => λ_23_est, "β_23" => β_23_est),
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
# Test 5: Exponential, AFT + Time-Fixed Covariate, Exact Data
# =============================================================================

"""
    run_exp_aft_tfc_exact()

Test exponential model with AFT parameterization and time-fixed covariate, exact data.
Model: h(t|x) = λ * exp(-β * x) at time t * exp(β * x)
True parameters: λ₁₂ = 0.2, β₁₂ = 0.5, λ₂₃ = 0.3, β₂₃ = 0.3

Note: For exponential, AFT and PH are equivalent (AFT coefficient = -PH coefficient).
"""
function run_exp_aft_tfc_exact()
    test_name = "exp_aft_tfc_exact"
    @info "Running $test_name"
    
    # True parameters (in AFT parameterization)
    λ_12_true = 0.2
    β_12_true = 0.5  # AFT parameter (negative of PH parameter)
    λ_23_true = 0.3
    β_23_true = 0.3
    
    # Create model for simulation
    h12 = Hazard(@formula(0 ~ 1 + x), "exp", 1, 2; linpred_effect=:aft)
    h23 = Hazard(@formula(0 ~ 1 + x), "exp", 2, 3; linpred_effect=:aft)
    
    Random.seed!(12347)
    dat = create_tfc_template(N_SUBJECTS; max_time=MAX_TIME)
    
    msm_sim = multistatemodel(h12, h23; data=dat)
    set_parameters!(msm_sim, 
        (h12 = [log(λ_12_true), β_12_true],
         h23 = [log(λ_23_true), β_23_true]))
    
    # Simulate exact data
    paths = simulate(msm_sim; paths=true, data=false)[1]
    exact_data = simulate(msm_sim; paths=false, data=true)[1]
    
    # Create fitting model
    msm_fit = multistatemodel(h12, h23; data=exact_data)
    
    # Fit model
    fitted = fit(msm_fit)
    
    # Extract estimates
    params = get_parameters(fitted)
    λ_12_est = exp(params[:h12][1])
    β_12_est = params[:h12][2]
    λ_23_est = exp(params[:h23][1])
    β_23_est = params[:h23][2]
    
    # Compute relative errors
    rel_errs = Dict(
        "λ_12" => compute_relative_error(λ_12_true, λ_12_est),
        "β_12" => compute_relative_error(β_12_true, β_12_est),
        "λ_23" => compute_relative_error(λ_23_true, λ_23_est),
        "β_23" => compute_relative_error(β_23_true, β_23_est)
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
        family = "exponential",
        parameterization = :aft,
        covariates = :tfc,
        data_type = :exact,
        n_subjects = N_SUBJECTS,
        true_params = Dict("λ_12" => λ_12_true, "β_12" => β_12_true, 
                          "λ_23" => λ_23_true, "β_23" => β_23_true),
        estimated_params = Dict("λ_12" => λ_12_est, "β_12" => β_12_est,
                               "λ_23" => λ_23_est, "β_23" => β_23_est),
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
# Test 6: Exponential, AFT + Time-Fixed Covariate, Panel Data
# =============================================================================

"""
    run_exp_aft_tfc_panel()

Test exponential model with AFT parameterization and time-fixed covariate, panel data.
"""
function run_exp_aft_tfc_panel()
    test_name = "exp_aft_tfc_panel"
    @info "Running $test_name"
    
    # True parameters
    λ_12_true = 0.2
    β_12_true = 0.5
    λ_23_true = 0.3
    β_23_true = 0.3
    
    # Create model for simulation
    h12 = Hazard(@formula(0 ~ 1 + x), "exp", 1, 2; linpred_effect=:aft)
    h23 = Hazard(@formula(0 ~ 1 + x), "exp", 2, 3; linpred_effect=:aft)
    
    Random.seed!(12348)
    dat = create_tfc_template(N_SUBJECTS; max_time=MAX_TIME)
    x_vals = dat.x
    
    msm_sim = multistatemodel(h12, h23; data=dat)
    set_parameters!(msm_sim, 
        (h12 = [log(λ_12_true), β_12_true],
         h23 = [log(λ_23_true), β_23_true]))
    
    # Simulate and convert to panel
    paths = simulate(msm_sim; paths=true, data=false)[1]
    panel_data = create_panel_data_with_covariate(paths, collect(PANEL_TIMES), 3, x_vals)
    
    # Create fitting model
    msm_fit = multistatemodel(h12, h23; data=panel_data)
    
    # Fit model
    fitted = fit(msm_fit)
    
    # Extract estimates
    params = get_parameters(fitted)
    λ_12_est = exp(params[:h12][1])
    β_12_est = params[:h12][2]
    λ_23_est = exp(params[:h23][1])
    β_23_est = params[:h23][2]
    
    # Compute relative errors
    rel_errs = Dict(
        "λ_12" => compute_relative_error(λ_12_true, λ_12_est),
        "β_12" => compute_relative_error(β_12_true, β_12_est),
        "λ_23" => compute_relative_error(λ_23_true, λ_23_est),
        "β_23" => compute_relative_error(β_23_true, β_23_est)
    )
    
    # Compute prevalence and cumulative incidence
    n_states = 3
    prev_true = compute_state_prevalence(paths, EVAL_TIMES, n_states)
    prev_obs = compute_prevalence_from_data(panel_data, EVAL_TIMES, n_states)
    
    # Simulate from fitted model
    set_parameters!(msm_sim, get_parameters(fitted))
    paths_fitted = simulate(msm_sim; paths=true, data=false)[1]
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
        family = "exponential",
        parameterization = :aft,
        covariates = :tfc,
        data_type = :panel,
        n_subjects = N_SUBJECTS,
        true_params = Dict("λ_12" => λ_12_true, "β_12" => β_12_true, 
                          "λ_23" => λ_23_true, "β_23" => β_23_true),
        estimated_params = Dict("λ_12" => λ_12_est, "β_12" => β_12_est,
                               "λ_23" => λ_23_est, "β_23" => β_23_est),
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
# Test 7: Exponential, PH + Time-Varying Covariate, Exact Data
# =============================================================================

"""
    run_exp_ph_tvc_exact()

Test exponential model with PH parameterization and time-varying covariate, exact data.
Covariate x changes from 0 to 1 at time TVC_CHANGEPOINT.
"""
function run_exp_ph_tvc_exact()
    test_name = "exp_ph_tvc_exact"
    @info "Running $test_name"
    
    # True parameters
    λ_12_true = 0.2
    β_12_true = 0.5
    λ_23_true = 0.3
    β_23_true = 0.3
    
    # Create model for simulation
    h12 = Hazard(@formula(0 ~ 1 + x), "exp", 1, 2; linpred_effect=:ph)
    h23 = Hazard(@formula(0 ~ 1 + x), "exp", 2, 3; linpred_effect=:ph)
    
    dat = create_tvc_template(N_SUBJECTS; max_time=MAX_TIME)
    
    msm_sim = multistatemodel(h12, h23; data=dat)
    set_parameters!(msm_sim, 
        (h12 = [log(λ_12_true), β_12_true],
         h23 = [log(λ_23_true), β_23_true]))
    
    # Simulate exact data
    paths = simulate(msm_sim; paths=true, data=false)[1]
    exact_data = simulate(msm_sim; paths=false, data=true)[1]
    
    # Create fitting model
    msm_fit = multistatemodel(h12, h23; data=exact_data)
    
    # Fit model
    fitted = fit(msm_fit)
    
    # Extract estimates
    params = get_parameters(fitted)
    λ_12_est = exp(params[:h12][1])
    β_12_est = params[:h12][2]
    λ_23_est = exp(params[:h23][1])
    β_23_est = params[:h23][2]
    
    # Compute relative errors
    rel_errs = Dict(
        "λ_12" => compute_relative_error(λ_12_true, λ_12_est),
        "β_12" => compute_relative_error(β_12_true, β_12_est),
        "λ_23" => compute_relative_error(λ_23_true, λ_23_est),
        "β_23" => compute_relative_error(β_23_true, β_23_est)
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
        family = "exponential",
        parameterization = :ph,
        covariates = :tvc,
        data_type = :exact,
        n_subjects = N_SUBJECTS,
        true_params = Dict("λ_12" => λ_12_true, "β_12" => β_12_true, 
                          "λ_23" => λ_23_true, "β_23" => β_23_true),
        estimated_params = Dict("λ_12" => λ_12_est, "β_12" => β_12_est,
                               "λ_23" => λ_23_est, "β_23" => β_23_est),
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
# Test 8: Exponential, PH + Time-Varying Covariate, Panel Data
# =============================================================================

"""
    run_exp_ph_tvc_panel()

Test exponential model with PH parameterization and time-varying covariate, panel data.
"""
function run_exp_ph_tvc_panel()
    test_name = "exp_ph_tvc_panel"
    @info "Running $test_name"
    
    # True parameters
    λ_12_true = 0.2
    β_12_true = 0.5
    λ_23_true = 0.3
    β_23_true = 0.3
    
    # Create model for simulation
    h12 = Hazard(@formula(0 ~ 1 + x), "exp", 1, 2; linpred_effect=:ph)
    h23 = Hazard(@formula(0 ~ 1 + x), "exp", 2, 3; linpred_effect=:ph)
    
    dat = create_tvc_template(N_SUBJECTS; max_time=MAX_TIME)
    
    msm_sim = multistatemodel(h12, h23; data=dat)
    set_parameters!(msm_sim, 
        (h12 = [log(λ_12_true), β_12_true],
         h23 = [log(λ_23_true), β_23_true]))
    
    # Simulate and convert to panel
    paths = simulate(msm_sim; paths=true, data=false)[1]
    panel_data = create_panel_data_with_tvc(paths, collect(PANEL_TIMES), 3)
    
    # Create fitting model
    msm_fit = multistatemodel(h12, h23; data=panel_data)
    
    # Fit model
    fitted = fit(msm_fit)
    
    # Extract estimates
    params = get_parameters(fitted)
    λ_12_est = exp(params[:h12][1])
    β_12_est = params[:h12][2]
    λ_23_est = exp(params[:h23][1])
    β_23_est = params[:h23][2]
    
    # Compute relative errors
    rel_errs = Dict(
        "λ_12" => compute_relative_error(λ_12_true, λ_12_est),
        "β_12" => compute_relative_error(β_12_true, β_12_est),
        "λ_23" => compute_relative_error(λ_23_true, λ_23_est),
        "β_23" => compute_relative_error(β_23_true, β_23_est)
    )
    
    # Compute prevalence and cumulative incidence
    n_states = 3
    prev_true = compute_state_prevalence(paths, EVAL_TIMES, n_states)
    prev_obs = compute_prevalence_from_data(panel_data, EVAL_TIMES, n_states)
    
    # Simulate from fitted model
    set_parameters!(msm_sim, get_parameters(fitted))
    paths_fitted = simulate(msm_sim; paths=true, data=false)[1]
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
        family = "exponential",
        parameterization = :ph,
        covariates = :tvc,
        data_type = :panel,
        n_subjects = N_SUBJECTS,
        true_params = Dict("λ_12" => λ_12_true, "β_12" => β_12_true, 
                          "λ_23" => λ_23_true, "β_23" => β_23_true),
        estimated_params = Dict("λ_12" => λ_12_est, "β_12" => β_12_est,
                               "λ_23" => λ_23_est, "β_23" => β_23_est),
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
# Test 9: Exponential, AFT + Time-Varying Covariate, Exact Data
# =============================================================================

"""
    run_exp_aft_tvc_exact()

Test exponential model with AFT parameterization and time-varying covariate, exact data.
"""
function run_exp_aft_tvc_exact()
    test_name = "exp_aft_tvc_exact"
    @info "Running $test_name"
    
    # True parameters
    λ_12_true = 0.2
    β_12_true = 0.5
    λ_23_true = 0.3
    β_23_true = 0.3
    
    # Create model for simulation
    h12 = Hazard(@formula(0 ~ 1 + x), "exp", 1, 2; linpred_effect=:aft)
    h23 = Hazard(@formula(0 ~ 1 + x), "exp", 2, 3; linpred_effect=:aft)
    
    dat = create_tvc_template(N_SUBJECTS; max_time=MAX_TIME)
    
    msm_sim = multistatemodel(h12, h23; data=dat)
    set_parameters!(msm_sim, 
        (h12 = [log(λ_12_true), β_12_true],
         h23 = [log(λ_23_true), β_23_true]))
    
    # Simulate exact data
    paths = simulate(msm_sim; paths=true, data=false)[1]
    exact_data = simulate(msm_sim; paths=false, data=true)[1]
    
    # Create fitting model
    msm_fit = multistatemodel(h12, h23; data=exact_data)
    
    # Fit model
    fitted = fit(msm_fit)
    
    # Extract estimates
    params = get_parameters(fitted)
    λ_12_est = exp(params[:h12][1])
    β_12_est = params[:h12][2]
    λ_23_est = exp(params[:h23][1])
    β_23_est = params[:h23][2]
    
    # Compute relative errors
    rel_errs = Dict(
        "λ_12" => compute_relative_error(λ_12_true, λ_12_est),
        "β_12" => compute_relative_error(β_12_true, β_12_est),
        "λ_23" => compute_relative_error(λ_23_true, λ_23_est),
        "β_23" => compute_relative_error(β_23_true, β_23_est)
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
        family = "exponential",
        parameterization = :aft,
        covariates = :tvc,
        data_type = :exact,
        n_subjects = N_SUBJECTS,
        true_params = Dict("λ_12" => λ_12_true, "β_12" => β_12_true, 
                          "λ_23" => λ_23_true, "β_23" => β_23_true),
        estimated_params = Dict("λ_12" => λ_12_est, "β_12" => β_12_est,
                               "λ_23" => λ_23_est, "β_23" => β_23_est),
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
# Test 10: Exponential, AFT + Time-Varying Covariate, Panel Data
# =============================================================================

"""
    run_exp_aft_tvc_panel()

Test exponential model with AFT parameterization and time-varying covariate, panel data.
"""
function run_exp_aft_tvc_panel()
    test_name = "exp_aft_tvc_panel"
    @info "Running $test_name"
    
    # True parameters
    λ_12_true = 0.2
    β_12_true = 0.5
    λ_23_true = 0.3
    β_23_true = 0.3
    
    # Create model for simulation
    h12 = Hazard(@formula(0 ~ 1 + x), "exp", 1, 2; linpred_effect=:aft)
    h23 = Hazard(@formula(0 ~ 1 + x), "exp", 2, 3; linpred_effect=:aft)
    
    dat = create_tvc_template(N_SUBJECTS; max_time=MAX_TIME)
    
    msm_sim = multistatemodel(h12, h23; data=dat)
    set_parameters!(msm_sim, 
        (h12 = [log(λ_12_true), β_12_true],
         h23 = [log(λ_23_true), β_23_true]))
    
    # Simulate and convert to panel
    paths = simulate(msm_sim; paths=true, data=false)[1]
    panel_data = create_panel_data_with_tvc(paths, collect(PANEL_TIMES), 3)
    
    # Create fitting model
    msm_fit = multistatemodel(h12, h23; data=panel_data)
    
    # Fit model
    fitted = fit(msm_fit)
    
    # Extract estimates
    params = get_parameters(fitted)
    λ_12_est = exp(params[:h12][1])
    β_12_est = params[:h12][2]
    λ_23_est = exp(params[:h23][1])
    β_23_est = params[:h23][2]
    
    # Compute relative errors
    rel_errs = Dict(
        "λ_12" => compute_relative_error(λ_12_true, λ_12_est),
        "β_12" => compute_relative_error(β_12_true, β_12_est),
        "λ_23" => compute_relative_error(λ_23_true, λ_23_est),
        "β_23" => compute_relative_error(β_23_true, β_23_est)
    )
    
    # Compute prevalence and cumulative incidence
    n_states = 3
    prev_true = compute_state_prevalence(paths, EVAL_TIMES, n_states)
    prev_obs = compute_prevalence_from_data(panel_data, EVAL_TIMES, n_states)
    
    # Simulate from fitted model
    set_parameters!(msm_sim, get_parameters(fitted))
    paths_fitted = simulate(msm_sim; paths=true, data=false)[1]
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
        family = "exponential",
        parameterization = :aft,
        covariates = :tvc,
        data_type = :panel,
        n_subjects = N_SUBJECTS,
        true_params = Dict("λ_12" => λ_12_true, "β_12" => β_12_true, 
                          "λ_23" => λ_23_true, "β_23" => β_23_true),
        estimated_params = Dict("λ_12" => λ_12_est, "β_12" => β_12_est,
                               "λ_23" => λ_23_est, "β_23" => β_23_est),
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
# Run All Exponential Tests
# =============================================================================

"""
    run_all_exponential_tests()

Run all 10 exponential family tests.
"""
function run_all_exponential_tests()
    @info "Running all exponential family tests (10 tests)"
    
    results = TestResult[]
    
    push!(results, run_exp_nocov_exact())
    push!(results, run_exp_nocov_panel())
    push!(results, run_exp_ph_tfc_exact())
    push!(results, run_exp_ph_tfc_panel())
    push!(results, run_exp_aft_tfc_exact())
    push!(results, run_exp_aft_tfc_panel())
    push!(results, run_exp_ph_tvc_exact())
    push!(results, run_exp_ph_tvc_panel())
    push!(results, run_exp_aft_tvc_exact())
    push!(results, run_exp_aft_tvc_panel())
    
    n_passed = count(r -> r.passed, results)
    @info "Exponential tests completed" passed="$n_passed/10"
    
    return results
end
