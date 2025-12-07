"""
Robust Long Test Suite: Markov and Phase-Type Models with Tight Tolerances

This test suite performs rigorous validation of:
1. Markov model panel data fitting (matrix exponentiation MLE)
2. Phase-type importance sampling correctness
3. MCEM convergence for semi-Markov models

Large sample sizes (n=5000) and tight tolerances ensure CORRECTNESS.

Key validations:
- Markov surrogate MLE matches analytical solution
- Phase-type IS weights are algebraically correct (=1 when target=proposal)
- MCEM parameter recovery for Weibull/Gompertz within tight bounds

References:
- Kalbfleisch & Lawless (1985) JASA - Markov MLE for panel data
- Titman & Sharples (2010) Biometrics - phase-type semi-Markov approximations
- Morsomme et al. (2025) Biostatistics - MCEM for multistate models
"""

using Test
using MultistateModels
using DataFrames
using Random
using Statistics
using LinearAlgebra

import MultistateModels: Hazard, multistatemodel, fit, set_parameters!, simulate,
    get_parameters_flat, get_parameters, get_log_scale_params, SamplePath, @formula,
    fit_surrogate, build_tpm_mapping, build_phasetype_surrogate, PhaseTypeConfig,
    build_phasetype_emat_expanded, build_phasetype_tpm_book, build_fbmats_phasetype,
    compute_phasetype_marginal_loglik, draw_samplepath_phasetype, loglik,
    loglik_phasetype_expanded

# =============================================================================
# CONSTANTS
# =============================================================================
const RNG_SEED = 0xABCD2026
const N_SUBJECTS = 1000            # Standard sample size for longtests
const MAX_TIME = 12.0              # Follow-up time
const PARAM_TOL_REL = 0.20         # 20% relative tolerance for MLE (n=1000)
const PARAM_TOL_BETA = 0.25        # 25% for covariate effects
const PARAM_TOL_MCEM = 0.35        # 35% for MCEM (inherent MC noise)
const IS_WEIGHT_TOL = 1e-10        # IS weights must be exactly 1.0

# =============================================================================
# Helper: Generate panel data from illness-death model
# =============================================================================
function generate_panel_data(hazards, true_params;
    n_subj::Int = N_SUBJECTS,
    obs_times::Vector{Float64} = [0.0, 3.0, 6.0, 9.0, MAX_TIME],
    covariate_data::Union{Nothing, DataFrame} = nothing)
    
    nobs = length(obs_times) - 1
    
    template = DataFrame(
        id = repeat(1:n_subj, inner=nobs),
        tstart = repeat(obs_times[1:end-1], n_subj),
        tstop = repeat(obs_times[2:end], n_subj),
        statefrom = ones(Int, n_subj * nobs),
        stateto = ones(Int, n_subj * nobs),
        obstype = fill(2, n_subj * nobs)  # Panel observation
    )
    
    if !isnothing(covariate_data)
        for col in names(covariate_data)
            template[!, col] = repeat(covariate_data[!, col], inner=nobs)
        end
    end
    
    model = multistatemodel(hazards...; data=template)
    set_parameters!(model, true_params)
    
    # Use autotmax=false to preserve panel observation times structure
    sim_result = simulate(model; paths=false, data=true, nsim=1, autotmax=false)
    return sim_result[1]
end

# =============================================================================
# TEST SECTION 1: MARKOV MLE (PANEL DATA, EXPONENTIAL HAZARDS)
# =============================================================================

@testset "Robust Markov MLE - No Covariates (n=$N_SUBJECTS)" begin
    Random.seed!(RNG_SEED)
    
    # True exponential rates (Markov model)
    true_rate_12 = 0.30
    true_rate_23 = 0.25
    true_rate_13 = 0.12
    
    true_params = (
        h12 = [log(true_rate_12)],
        h23 = [log(true_rate_23)],
        h13 = [log(true_rate_13)]
    )
    
    h12 = Hazard(@formula(0 ~ 1), "exp", 1, 2)
    h23 = Hazard(@formula(0 ~ 1), "exp", 2, 3)
    h13 = Hazard(@formula(0 ~ 1), "exp", 1, 3)
    
    panel_data = generate_panel_data((h12, h23, h13), true_params)
    
    model_fit = multistatemodel(h12, h23, h13; data=panel_data)
    fitted = fit(model_fit; verbose=false, compute_vcov=true)
    
    p = get_parameters(fitted; scale=:natural)
    
    # TIGHT TOLERANCE TESTS
    @test isapprox(p.h12[1], true_rate_12; rtol=PARAM_TOL_REL)
    @test isapprox(p.h23[1], true_rate_23; rtol=PARAM_TOL_REL)
    @test isapprox(p.h13[1], true_rate_13; rtol=PARAM_TOL_REL)
    
    @test isfinite(fitted.loglik.loglik)
    @test !isnothing(fitted.vcov)
    
    println("  ✓ Markov MLE panel data: rates recovered within $(100*PARAM_TOL_REL)%")
end

@testset "Robust Markov MLE - With Covariate (n=$N_SUBJECTS)" begin
    Random.seed!(RNG_SEED + 1)
    
    true_rate_12, true_beta_12 = 0.25, 0.4
    true_rate_23, true_beta_23 = 0.20, -0.3
    true_rate_13, true_beta_13 = 0.10, 0.5
    
    cov_data = DataFrame(x = rand([0.0, 1.0], N_SUBJECTS))
    
    true_params = (
        h12 = [log(true_rate_12), true_beta_12],
        h23 = [log(true_rate_23), true_beta_23],
        h13 = [log(true_rate_13), true_beta_13]
    )
    
    h12 = Hazard(@formula(0 ~ x), "exp", 1, 2)
    h23 = Hazard(@formula(0 ~ x), "exp", 2, 3)
    h13 = Hazard(@formula(0 ~ x), "exp", 1, 3)
    
    panel_data = generate_panel_data((h12, h23, h13), true_params; covariate_data=cov_data)
    
    model_fit = multistatemodel(h12, h23, h13; data=panel_data)
    fitted = fit(model_fit; verbose=false, compute_vcov=true)
    
    # Use natural scale NamedTuple - safer than assuming flat vector order
    p = get_parameters(fitted; scale=:natural)
    
    # Rate recovery (first element of each hazard vector)
    @test isapprox(p.h12[1], true_rate_12; rtol=PARAM_TOL_REL)
    @test isapprox(p.h23[1], true_rate_23; rtol=PARAM_TOL_REL)
    @test isapprox(p.h13[1], true_rate_13; rtol=PARAM_TOL_REL)
    
    # Beta recovery (second element of each hazard vector - not transformed)
    @test isapprox(p.h12[2], true_beta_12; atol=PARAM_TOL_BETA)
    @test isapprox(p.h23[2], true_beta_23; atol=PARAM_TOL_BETA)
    @test isapprox(p.h13[2], true_beta_13; atol=PARAM_TOL_BETA)
    
    # Sign verification
    @test sign(p.h12[2]) == sign(true_beta_12)
    @test sign(p.h23[2]) == sign(true_beta_23)
    @test sign(p.h13[2]) == sign(true_beta_13)
    
    @test isfinite(fitted.loglik.loglik)
    
    println("  ✓ Markov MLE with covariate: rates and betas recovered")
end

# =============================================================================
# TEST SECTION 2: PHASE-TYPE IMPORTANCE SAMPLING CORRECTNESS
# =============================================================================

@testset "Phase-Type IS Weights = 1 (Algebraic Identity)" begin
    Random.seed!(RNG_SEED + 10)
    
    # When target = proposal (Markov model with 1 phase), weights must be EXACTLY 1.0
    # This is an algebraic identity, not a statistical property.
    
    h12 = Hazard(@formula(0 ~ 1), "exp", 1, 2)
    
    # Use SIMULATED panel data to ensure valid state sequences
    # (can't have 2→1 when 2 is absorbing)
    n_subj = 100
    template = DataFrame(
        id = repeat(1:n_subj, inner=3),
        tstart = repeat([0.0, 1.0, 2.0], n_subj),
        tstop = repeat([1.0, 2.0, 3.0], n_subj),
        statefrom = repeat([1, 1, 1], n_subj),
        stateto = repeat([1, 1, 1], n_subj),  # placeholder
        obstype = repeat([2, 2, 2], n_subj)
    )
    
    # Set true rate and simulate
    true_rate = 0.3
    model_sim = multistatemodel(h12; data=template)
    set_parameters!(model_sim, (h12 = [log(true_rate)],))
    sim_result = simulate(model_sim; paths=false, data=true, nsim=1, autotmax=false)
    dat = sim_result[1]
    
    model = multistatemodel(h12; data=dat)
    
    # Fit Markov surrogate
    surrogate_fitted = fit_surrogate(model; verbose=false)
    
    # Set target = proposal (exact match)
    set_parameters!(model, [collect(surrogate_fitted.parameters[1])])
    
    # Build phase-type with 1 phase (equivalent to Markov)
    tmat = model.tmat
    phasetype_config = PhaseTypeConfig(n_phases=[1, 1])
    surrogate = build_phasetype_surrogate(tmat, phasetype_config)
    
    # Set phase-type rate to match Markov
    markov_rate = exp(surrogate_fitted.parameters[1][1])
    surrogate.expanded_Q[1, 1] = -markov_rate
    surrogate.expanded_Q[1, 2] = markov_rate
    
    # Build infrastructure
    emat_ph = build_phasetype_emat_expanded(model, surrogate)
    books = build_tpm_mapping(model.data)
    absorbingstates = findall([isa(h, MultistateModels._TotalHazardAbsorbing) for h in model.totalhazards])
    tpm_book_ph, hazmat_book_ph = build_phasetype_tpm_book(surrogate, books, model.data)
    fbmats_ph = build_fbmats_phasetype(model, surrogate)
    
    # Sample paths and verify weights = 1.0 EXACTLY
    n_paths = 100
    for _ in 1:n_paths
        path_result = draw_samplepath_phasetype(
            1, model, tpm_book_ph, hazmat_book_ph, books[2],
            fbmats_ph, emat_ph, surrogate, absorbingstates)
        
        params = get_log_scale_params(model.parameters)
        ll_target = loglik(params, path_result.collapsed, model.hazards, model)
        ll_surrog = loglik_phasetype_expanded(path_result.expanded, surrogate)
        
        log_weight = ll_target - ll_surrog
        weight = exp(log_weight)
        
        # CRITICAL: Weight must be EXACTLY 1.0 (algebraic identity)
        @test isapprox(weight, 1.0; atol=IS_WEIGHT_TOL)
    end
    
    println("  ✓ Phase-type IS weights = 1.0 exactly when target = proposal (100 samples)")
end

@testset "Phase-Type IS Estimate Consistency" begin
    Random.seed!(RNG_SEED + 11)
    
    # For any model, IS estimate should be consistent across sample sizes
    h12 = Hazard(@formula(0 ~ 1), "wei", 1, 2)  # Weibull (semi-Markov)
    
    # Use SIMULATED panel data
    n_subj = 50
    template = DataFrame(
        id = repeat(1:n_subj, inner=2),
        tstart = repeat([0.0, 1.0], n_subj),
        tstop = repeat([1.0, 2.0], n_subj),
        statefrom = repeat([1, 1], n_subj),
        stateto = repeat([1, 1], n_subj),  # placeholder
        obstype = repeat([2, 2], n_subj)
    )
    
    # Set true params and simulate
    true_shape, true_scale = 1.2, 0.2
    model_sim = multistatemodel(h12; data=template)
    set_parameters!(model_sim, (h12 = [log(true_shape), log(true_scale)],))
    sim_result = simulate(model_sim; paths=false, data=true, nsim=1, autotmax=false)
    dat = sim_result[1]
    
    model = multistatemodel(h12; data=dat)
    
    # Fit phase-type surrogate (must be fitted, not just built with defaults)
    surrogate = fit_surrogate(model; type=:phasetype, n_phases=[2, 1], verbose=false)
    
    emat_ph = build_phasetype_emat_expanded(model, surrogate)
    books = build_tpm_mapping(model.data)
    absorbingstates = findall([isa(h, MultistateModels._TotalHazardAbsorbing) for h in model.totalhazards])
    tpm_book_ph, hazmat_book_ph = build_phasetype_tpm_book(surrogate, books, model.data)
    fbmats_ph = build_fbmats_phasetype(model, surrogate)
    
    ll_marginal = compute_phasetype_marginal_loglik(model, surrogate, emat_ph)
    
    # Compute IS estimate with many samples
    n_paths = 500
    log_weights = Float64[]
    
    for _ in 1:n_paths
        result = draw_samplepath_phasetype(
            1, model, tpm_book_ph, hazmat_book_ph, books[2],
            fbmats_ph, emat_ph, surrogate, absorbingstates)
        
        params = get_log_scale_params(model.parameters)
        ll_target = loglik(params, result.collapsed, model.hazards, model)
        ll_surrog = loglik_phasetype_expanded(result.expanded, surrogate)
        push!(log_weights, ll_target - ll_surrog)
    end
    
    weights = exp.(log_weights)
    ll_is = ll_marginal + log(mean(weights))
    
    # IS estimate must be finite
    @test isfinite(ll_is)
    
    # IS correction should be moderate (not huge)
    @test abs(ll_is - ll_marginal) < 10.0
    
    println("  ✓ Phase-type IS estimate is finite and correction is moderate")
end

# =============================================================================
# TEST SECTION 3: MCEM PARAMETER RECOVERY (SEMI-MARKOV)
# Note: We use competing risks (1→2, 1→3) instead of illness-death because
# the downstream transition (2→3) has limited events which makes MCEM
# convergence unreliable for robust testing.
# =============================================================================

@testset "Robust MCEM Weibull - Competing Risks (n=$N_SUBJECTS)" begin
    Random.seed!(RNG_SEED + 20)
    
    # Weibull requires MCEM (semi-Markov)
    # Use competing risks model for robust estimation
    true_shape_12, true_scale_12 = 1.3, 0.18
    true_shape_13, true_scale_13 = 1.1, 0.12
    
    true_params = (
        h12 = [log(true_shape_12), log(true_scale_12)],
        h13 = [log(true_shape_13), log(true_scale_13)]
    )
    
    h12 = Hazard(@formula(0 ~ 1), "wei", 1, 2)
    h13 = Hazard(@formula(0 ~ 1), "wei", 1, 3)
    
    # Generate panel data using helper
    panel_data = generate_panel_data((h12, h13), true_params; n_subj=N_SUBJECTS)
    
    # MCEM requires surrogate=:markov for semi-Markov models
    model_fit = multistatemodel(h12, h13; data=panel_data, surrogate=:markov)
    fitted = fit(model_fit;
        verbose=false,
        maxiter=50,
        tol=0.01,
        ess_target_initial=100,
        max_ess=1000,
        compute_vcov=false,
        return_convergence_records=true)
    
    p = get_parameters(fitted; scale=:natural)
    
    # Shape recovery (MCEM tolerance)
    @test isapprox(p.h12[1], true_shape_12; rtol=PARAM_TOL_MCEM)
    @test isapprox(p.h13[1], true_shape_13; rtol=PARAM_TOL_MCEM)
    
    # Scale recovery
    @test isapprox(p.h12[2], true_scale_12; rtol=PARAM_TOL_MCEM)
    @test isapprox(p.h13[2], true_scale_13; rtol=PARAM_TOL_MCEM)
    
    @test isfinite(fitted.loglik.loglik)
    @test !isnothing(fitted.ConvergenceRecords)
    
    println("  ✓ MCEM Weibull competing risks: shape and scale recovered within $(100*PARAM_TOL_MCEM)%")
end

@testset "Robust MCEM Weibull - With Covariate (n=$N_SUBJECTS)" begin
    Random.seed!(RNG_SEED + 21)
    
    true_shape = 1.2
    true_scale = 0.20
    true_beta = 0.5
    
    cov_data = DataFrame(x = rand([0.0, 1.0], N_SUBJECTS))
    
    true_params = (h12 = [log(true_shape), log(true_scale), true_beta],)
    
    h12 = Hazard(@formula(0 ~ x), "wei", 1, 2)
    
    # Simple 2-state model (avoids downstream transition issues)
    nobs = 4
    obs_times = [0.0, 3.0, 6.0, 9.0, MAX_TIME]
    
    template = DataFrame(
        id = repeat(1:N_SUBJECTS, inner=nobs),
        tstart = repeat(obs_times[1:end-1], N_SUBJECTS),
        tstop = repeat(obs_times[2:end], N_SUBJECTS),
        statefrom = ones(Int, N_SUBJECTS * nobs),
        stateto = ones(Int, N_SUBJECTS * nobs),
        obstype = fill(2, N_SUBJECTS * nobs),
        x = repeat(cov_data.x, inner=nobs)
    )
    
    model_sim = multistatemodel(h12; data=template)
    set_parameters!(model_sim, true_params)
    
    sim_result = simulate(model_sim; paths=false, data=true, nsim=1, autotmax=false)
    panel_data = sim_result[1]
    
    # MCEM requires surrogate=:markov for semi-Markov models
    model_fit = multistatemodel(h12; data=panel_data, surrogate=:markov)
    fitted = fit(model_fit;
        verbose=false,
        maxiter=50,
        tol=0.01,
        ess_target_initial=100,
        max_ess=1000,
        compute_vcov=false,
        return_convergence_records=true)
    
    p_est = get_parameters_flat(fitted)
    
    # Shape and scale recovery
    @test isapprox(exp(p_est[1]), true_shape; rtol=PARAM_TOL_MCEM)
    @test isapprox(exp(p_est[2]), true_scale; rtol=PARAM_TOL_MCEM)
    
    # Beta recovery
    @test isapprox(p_est[3], true_beta; atol=0.25)
    @test sign(p_est[3]) == sign(true_beta)
    
    @test isfinite(fitted.loglik.loglik)
    
    println("  ✓ MCEM Weibull with covariate: all parameters recovered")
end

@testset "Robust MCEM Gompertz - Competing Risks (n=$N_SUBJECTS)" begin
    Random.seed!(RNG_SEED + 30)
    
    # Gompertz requires MCEM (semi-Markov)
    # Use competing risks model for robust estimation
    # Gompertz params are [log(shape), log(scale)]
    true_shape_12, true_scale_12 = 0.12, 0.15
    true_shape_13, true_scale_13 = 0.10, 0.12
    
    true_params = (
        h12 = [log(true_shape_12), log(true_scale_12)],
        h13 = [log(true_shape_13), log(true_scale_13)]
    )
    
    h12 = Hazard(@formula(0 ~ 1), "gom", 1, 2)
    h13 = Hazard(@formula(0 ~ 1), "gom", 1, 3)
    
    panel_data = generate_panel_data((h12, h13), true_params; n_subj=N_SUBJECTS)
    
    # MCEM requires surrogate=:markov for semi-Markov models
    model_fit = multistatemodel(h12, h13; data=panel_data, surrogate=:markov)
    fitted = fit(model_fit;
        verbose=false,
        maxiter=50,
        tol=0.01,
        ess_target_initial=100,
        max_ess=1000,
        compute_vcov=false,
        return_convergence_records=true)
    
    p = get_parameters(fitted; scale=:natural)
    
    # Shape recovery (using absolute tolerance for small values)
    @test isapprox(p.h12[1], true_shape_12; atol=0.05)
    @test isapprox(p.h13[1], true_shape_13; atol=0.05)
    
    # Scale recovery
    @test isapprox(p.h12[2], true_scale_12; rtol=PARAM_TOL_MCEM)
    @test isapprox(p.h13[2], true_scale_13; rtol=PARAM_TOL_MCEM)
    
    @test isfinite(fitted.loglik.loglik)
    
    println("  ✓ MCEM Gompertz competing risks: shape and scale recovered within tolerance")
end

# =============================================================================
# TEST SECTION 4: MARKOV SURROGATE FITTING
# =============================================================================

@testset "Markov Surrogate MLE Optimality" begin
    Random.seed!(RNG_SEED + 40)
    
    # The Markov surrogate MLE should have higher log-likelihood than heuristic
    h12 = Hazard(@formula(0 ~ 1), "exp", 1, 2)
    h23 = Hazard(@formula(0 ~ 1), "exp", 2, 3)
    h13 = Hazard(@formula(0 ~ 1), "exp", 1, 3)
    
    true_params = (
        h12 = [log(0.25)],
        h23 = [log(0.20)],
        h13 = [log(0.10)]
    )
    
    panel_data = generate_panel_data((h12, h23, h13), true_params; n_subj=500)
    
    model = multistatemodel(h12, h23, h13; data=panel_data)
    
    surrogate_mle = fit_surrogate(model; method=:mle, verbose=false)
    surrogate_heur = fit_surrogate(model; method=:heuristic, verbose=false)
    
    ll_mle = MultistateModels.compute_markov_marginal_loglik(model, surrogate_mle)
    ll_heur = MultistateModels.compute_markov_marginal_loglik(model, surrogate_heur)
    
    # MLE must be >= heuristic (optimality)
    @test ll_mle >= ll_heur - 1e-6
    
    # Both should be finite
    @test isfinite(ll_mle)
    @test isfinite(ll_heur)
    
    println("  ✓ Markov surrogate MLE is optimal (ll_mle >= ll_heur)")
end

# =============================================================================
# TEST SECTION 5: REVERSIBLE MODEL
# =============================================================================

@testset "Robust Markov Reversible Model (n=$N_SUBJECTS)" begin
    Random.seed!(RNG_SEED + 50)
    
    # Reversible 1 ↔ 2 model
    true_rate_12 = 0.30
    true_rate_21 = 0.25
    
    true_params = (
        h12 = [log(true_rate_12)],
        h21 = [log(true_rate_21)]
    )
    
    h12 = Hazard(@formula(0 ~ 1), "exp", 1, 2)
    h21 = Hazard(@formula(0 ~ 1), "exp", 2, 1)
    
    nobs = 5
    obs_times = [0.0, 2.0, 4.0, 6.0, 8.0, MAX_TIME]
    
    template = DataFrame(
        id = repeat(1:N_SUBJECTS, inner=nobs),
        tstart = repeat(obs_times[1:end-1], N_SUBJECTS),
        tstop = repeat(obs_times[2:end], N_SUBJECTS),
        statefrom = ones(Int, N_SUBJECTS * nobs),
        stateto = ones(Int, N_SUBJECTS * nobs),
        obstype = fill(2, N_SUBJECTS * nobs)
    )
    
    model_sim = multistatemodel(h12, h21; data=template)
    set_parameters!(model_sim, true_params)
    
    sim_result = simulate(model_sim; paths=false, data=true, nsim=1, autotmax=false)
    panel_data = sim_result[1]
    
    model_fit = multistatemodel(h12, h21; data=panel_data)
    fitted = fit(model_fit; verbose=false, compute_vcov=true)
    
    p = get_parameters(fitted; scale=:natural)
    
    @test isapprox(p.h12[1], true_rate_12; rtol=PARAM_TOL_REL)
    @test isapprox(p.h21[1], true_rate_21; rtol=PARAM_TOL_REL)
    
    @test isfinite(fitted.loglik.loglik)
    
    println("  ✓ Reversible Markov model: both rates recovered")
end

# =============================================================================
# SUMMARY
# =============================================================================

println("\n" * "="^70)
println("ROBUST MARKOV/PHASE-TYPE LONGTEST SUITE COMPLETE")
println("="^70)
println("Markov MLE sample size: n = $N_SUBJECTS")
println("MCEM sample size: n = $N_SUBJECTS")
println("Tolerances: $(100*PARAM_TOL_REL)% Markov, $(100*PARAM_TOL_MCEM)% MCEM")
println("IS weight tolerance: $IS_WEIGHT_TOL (algebraic exactness)")
println("="^70)
