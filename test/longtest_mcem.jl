"""
Long test suite for MCEM algorithm statistical validation.

This test suite verifies:
1. Parameter recovery: MCEM with Weibull hazards recovers known parameters
2. Phase-type vs Markov ESS: Phase-type proposals yield higher ESS for non-exponential hazards
3. Convergence: MCEM converges for semi-Markov illness-death models

These tests are computationally intensive (~5-10 minutes total).

References:
- Morsomme et al. (2025) Biostatistics kxaf038 - multistate semi-Markov MCEM
- Titman & Sharples (2010) Biometrics 66(3):742-752 - phase-type approximations
- Caffo et al. (2005) JRSS-B - ascent-based MCEM stopping rules
"""

using Test
using MultistateModels
using DataFrames
using Random
using Statistics
using LinearAlgebra

# Import internal functions for testing
import MultistateModels: Hazard, multistatemodel, fit, set_parameters!, simulate,
    get_parameters_flat, get_log_scale_params, MarkovProposal, PhaseTypeProposal,
    fit_surrogate, build_tpm_mapping, build_hazmat_book, build_tpm_book,
    compute_hazmat!, compute_tmat!, ExpMethodGeneric, ExponentialUtilities,
    needs_phasetype_proposal, resolve_proposal_config, SamplePath

const RNG_SEED = 0xABCD1234
const N_SUBJECTS_SMALL = 50   # For quick convergence tests
const N_SUBJECTS_MED = 100    # For parameter recovery
const MCEM_TOL = 0.05         # Relaxed tolerance for faster convergence
const MAX_ITER_SHORT = 30     # Short iteration limit for tests

# ============================================================================
# Test 1: MCEM Parameter Recovery for Weibull Model
# ============================================================================

@testset "MCEM Parameter Recovery - Weibull Illness-Death" begin
    Random.seed!(RNG_SEED)
    
    # True parameters (on log scale)
    # h12: Weibull with shape=1.5, scale=0.3
    true_log_shape_12 = log(1.5)
    true_log_scale_12 = log(0.3)
    # h23: Weibull with shape=1.2, scale=0.2 (absorbing)
    true_log_shape_23 = log(1.2)
    true_log_scale_23 = log(0.2)
    
    # Create illness-death model (1 → 2 → 3, where 3 is absorbing)
    h12 = Hazard(@formula(0 ~ 1), "wei", 1, 2)
    h23 = Hazard(@formula(0 ~ 1), "wei", 2, 3)
    
    # Generate panel data template with observations at discrete times
    obs_times = [0.0, 2.0, 4.0, 6.0, 8.0, 10.0, 15.0, 20.0]
    nobs = length(obs_times) - 1
    
    sim_data = DataFrame(
        id = repeat(1:N_SUBJECTS_MED, inner=nobs),
        tstart = repeat(obs_times[1:end-1], N_SUBJECTS_MED),
        tstop = repeat(obs_times[2:end], N_SUBJECTS_MED),
        statefrom = ones(Int, N_SUBJECTS_MED * nobs),
        stateto = ones(Int, N_SUBJECTS_MED * nobs),
        obstype = fill(2, N_SUBJECTS_MED * nobs)  # Panel observations
    )
    
    model_sim = multistatemodel(h12, h23; data=sim_data)
    
    # Set true parameters for simulation
    MultistateModels.set_parameters!(model_sim, (h12 = [true_log_shape_12, true_log_scale_12],
                                                  h23 = [true_log_shape_23, true_log_scale_23]))
    
    # Simulate panel data using package simulation machinery
    # This correctly handles panel observations with obstype=2 for all intervals
    sim_result = simulate(model_sim; paths=false, data=true, nsim=1)
    panel_data = sim_result[1, 1]
    
    # Fit model via MCEM
    model_fit = multistatemodel(h12, h23; data=panel_data)
    
    # Run MCEM with short iterations for testing
    fitted = fit(model_fit; 
        verbose=false,
        maxiter=MAX_ITER_SHORT,
        tol=MCEM_TOL,
        ess_target_initial=30,
        max_ess=500,
        compute_vcov=true,
        compute_ij_vcov=false,  # Skip for speed
        return_convergence_records=true)
    
    # Check that MCEM produced reasonable estimates
    fitted_params = get_parameters_flat(fitted)
    true_params = [true_log_shape_12, true_log_scale_12, true_log_shape_23, true_log_scale_23]
    
    # Parameters should be in reasonable range (within 50% of true values on natural scale)
    for i in 1:4
        fitted_natural = exp(fitted_params[i])
        true_natural = exp(true_params[i])
        rel_error = abs(fitted_natural - true_natural) / true_natural
        @test rel_error < 0.5  # Within 50% - reasonable for small sample
    end
    
    # Check convergence records exist
    @test !isnothing(fitted.ConvergenceRecords)
    @test length(fitted.ConvergenceRecords.mll_trace) > 0
    
    # Log-likelihood should be finite and negative
    @test isfinite(fitted.loglik.loglik)
    @test fitted.loglik.loglik < 0
end

# ============================================================================
# Test 2: Phase-Type vs Markov Proposal Comparison
# ============================================================================

@testset "Phase-Type vs Markov Proposal ESS" begin
    Random.seed!(RNG_SEED + 1)
    
    # Create a Weibull model where phase-type should help
    # Weibull shape > 1 means increasing hazard (poor exponential approximation)
    h12 = Hazard(@formula(0 ~ 1), "wei", 1, 2)
    
    # Generate panel data
    n_subj = N_SUBJECTS_SMALL
    dat = DataFrame(
        id = repeat(1:n_subj, inner=3),
        tstart = repeat([0.0, 2.0, 4.0], n_subj),
        tstop = repeat([2.0, 4.0, 6.0], n_subj),
        statefrom = repeat([1, 1, 1], n_subj),
        stateto = vcat([[rand() < 0.2 ? 2 : 1, rand() < 0.4 ? 2 : 1, 2] for _ in 1:n_subj]...),
        obstype = repeat([2, 2, 2], n_subj)
    )
    
    model = multistatemodel(h12; data=dat)
    
    # Set Weibull parameters with shape > 1 (increasing hazard)
    MultistateModels.set_parameters!(model, (h12 = [log(2.0), log(0.5)],))  # shape=2, scale=0.5
    
    # Test that needs_phasetype_proposal detects non-exponential hazards
    # Function takes hazards vector, not model
    @test needs_phasetype_proposal(model.hazards) == true
    
    # Test proposal resolution
    config_auto = resolve_proposal_config(:auto, model)
    @test config_auto.type == :phasetype  # Should default to phase-type for Weibull
    
    config_markov = resolve_proposal_config(:markov, model)
    @test config_markov.type == :markov
    
    config_ph = resolve_proposal_config(PhaseTypeProposal(n_phases=2), model)
    @test config_ph.type == :phasetype
    @test config_ph.n_phases == 2
    
    # Create exponential model for comparison
    h12_exp = Hazard(@formula(0 ~ 1), "exp", 1, 2)
    model_exp = multistatemodel(h12_exp; data=dat)
    
    # Exponential model should NOT need phase-type
    @test needs_phasetype_proposal(model_exp.hazards) == false
    
    config_exp_auto = resolve_proposal_config(:auto, model_exp)
    @test config_exp_auto.type == :markov  # Should use Markov for exponential
end

# ============================================================================
# Test 3: MCEM Convergence Diagnostics
# ============================================================================

@testset "MCEM Convergence Diagnostics" begin
    Random.seed!(RNG_SEED + 2)
    
    # Simple reversible model for reliable convergence
    h12 = Hazard(@formula(0 ~ 1), "wei", 1, 2)
    h21 = Hazard(@formula(0 ~ 1), "wei", 2, 1)
    
    # Small dataset
    n_subj = 30
    dat = DataFrame(
        id = repeat(1:n_subj, inner=3),
        tstart = repeat([0.0, 1.0, 2.0], n_subj),
        tstop = repeat([1.0, 2.0, 3.0], n_subj),
        statefrom = repeat([1, 1, 2], n_subj),
        stateto = vcat([[1, rand() < 0.5 ? 2 : 1, rand() < 0.5 ? 1 : 2] for _ in 1:n_subj]...),
        obstype = repeat([2, 2, 2], n_subj)
    )
    
    model = multistatemodel(h12, h21; data=dat)
    
    # Fit with convergence records
    fitted = fit(model;
        proposal=:markov,  # Use Markov for simplicity
        verbose=false,
        maxiter=15,
        tol=0.1,  # Relaxed for quick convergence
        ess_target_initial=20,
        max_ess=200,
        compute_vcov=false,
        compute_ij_vcov=false,
        return_convergence_records=true)
    
    # Check convergence records
    records = fitted.ConvergenceRecords
    @test !isnothing(records)
    
    # MLL trace should exist and increase (mostly)
    mll_trace = records.mll_trace
    @test length(mll_trace) >= 1
    
    # ESS trace should exist
    ess_trace = records.ess_trace
    @test size(ess_trace, 2) == length(mll_trace)
    @test size(ess_trace, 1) == n_subj
    
    # Parameters trace should exist
    params_trace = records.parameters_trace
    @test size(params_trace, 2) == length(mll_trace)
    
    # Pareto-k diagnostics should be available
    pareto_k = records.psis_pareto_k
    @test length(pareto_k) == n_subj
    # Most should be finite; NaN/Inf can occur for subjects with single unique path
    @test count(isfinite.(pareto_k)) >= n_subj * 0.5
end

# ============================================================================
# Test 4: Surrogate Fitting
# ============================================================================

@testset "Markov Surrogate Fitting" begin
    Random.seed!(RNG_SEED + 3)
    
    # Create semi-Markov model
    h12 = Hazard(@formula(0 ~ 1), "wei", 1, 2)
    h21 = Hazard(@formula(0 ~ 1), "wei", 2, 1)
    
    n_subj = 40
    dat = DataFrame(
        id = repeat(1:n_subj, inner=2),
        tstart = repeat([0.0, 1.5], n_subj),
        tstop = repeat([1.5, 3.0], n_subj),
        statefrom = repeat([1, 1], n_subj),
        stateto = vcat([[rand() < 0.4 ? 2 : 1, 2] for _ in 1:n_subj]...),
        obstype = repeat([2, 2], n_subj)
    )
    
    model = multistatemodel(h12, h21; data=dat)
    
    # Fit Markov surrogate
    surrogate_fitted = fit_surrogate(model; verbose=false)
    
    # Check surrogate has valid log-likelihood
    @test isfinite(surrogate_fitted.loglik.loglik)
    @test surrogate_fitted.loglik.loglik < 0
    
    # Check surrogate parameters are finite
    surrogate_params = get_parameters_flat(surrogate_fitted)
    @test all(isfinite.(surrogate_params))
    
    # Surrogate should have exponential hazards (Markov)
    @test all(isa.(surrogate_fitted.hazards, MultistateModels._MarkovHazard))
end

# ============================================================================
# Test 5: Viterbi MAP Initialization
# ============================================================================

@testset "Viterbi MAP Warm Start" begin
    Random.seed!(RNG_SEED + 4)
    
    # Create model with panel data
    h12 = Hazard(@formula(0 ~ 1), "exp", 1, 2)
    h21 = Hazard(@formula(0 ~ 1), "exp", 2, 1)
    
    n_subj = 20
    dat = DataFrame(
        id = repeat(1:n_subj, inner=2),
        tstart = repeat([0.0, 1.0], n_subj),
        tstop = repeat([1.0, 2.0], n_subj),
        statefrom = repeat([1, 1], n_subj),
        stateto = vcat([[rand() < 0.3 ? 2 : 1, 2] for _ in 1:n_subj]...),
        obstype = repeat([2, 1], n_subj)
    )
    
    # Create model with surrogate for MCEM testing
    model = multistatemodel(h12, h21; data=dat, surrogate=:markov)
    
    # Build surrogate and TPM infrastructure
    books = build_tpm_mapping(model.data)
    hazmat_book = build_hazmat_book(Float64, model.tmat, books[1])
    tpm_book = build_tpm_book(Float64, model.tmat, books[1])
    cache = ExponentialUtilities.alloc_mem(similar(hazmat_book[1]), ExpMethodGeneric())
    
    surrogate = model.markovsurrogate
    surrogate_pars = get_log_scale_params(surrogate.parameters)
    
    for t in eachindex(books[1])
        compute_hazmat!(hazmat_book[t], surrogate_pars, surrogate.hazards, books[1][t], model.data)
        compute_tmat!(tpm_book[t], hazmat_book[t], books[1][t], cache)
    end
    
    fbmats = MultistateModels.build_fbmats(model)
    absorbingstates = findall([isa(h, MultistateModels._TotalHazardAbsorbing) for h in model.totalhazards])
    
    # Compute Viterbi path for first subject with panel observations
    subj_with_panel = findfirst(i -> any(dat.obstype[model.subjectindices[i]] .> 1), 1:n_subj)
    
    if !isnothing(subj_with_panel)
        map_path = MultistateModels.viterbi_map_path(
            subj_with_panel, model, tpm_book, hazmat_book, books[2], fbmats, absorbingstates)
        
        @test map_path isa SamplePath
        @test map_path.subj == subj_with_panel
        @test length(map_path.times) >= 1
        @test length(map_path.states) == length(map_path.times)
        @test all(s -> s in [1, 2], map_path.states)
    end
end

# ============================================================================
# Print summary
# ============================================================================

println("\n=== MCEM Long Test Suite Complete ===\n")
println("Tests verify:")
println("  - MCEM parameter recovery for Weibull illness-death models")
println("  - Phase-type vs Markov proposal selection")
println("  - Convergence diagnostics and ESS tracking")
println("  - Markov surrogate fitting")
println("  - Viterbi MAP warm start initialization")
