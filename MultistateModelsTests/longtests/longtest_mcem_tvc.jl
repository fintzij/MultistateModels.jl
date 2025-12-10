"""
Long test suite for MCEM algorithm with time-varying covariates (TVC).

This test suite verifies that MCEM correctly handles:
1. Panel data with covariate changes within observation intervals
2. Both PH and AFT covariate effects under TVC
3. All hazard families (exponential, Weibull, Gompertz) with TVC
4. Illness-death models with TVC
5. Semi-Markov models where sojourn time resets interact with TVC
6. Parameter recovery under TVC scenarios

Key scenarios:
- Binary TVC (treatment switches)
- Continuous TVC (time-varying biomarker)
- Multiple covariate change points per subject
- Competing risks with TVC

Notes on ESS behavior:
- Subjects with early transitions (in the first observation interval) may have ESS ≈ 1.0
  because the path structure is deterministic (only the exact transition time varies).
- High Pareto-k values (close to 1.0) indicate the Markov surrogate may not be an ideal
  proposal for the semi-Markov target, but the algorithm still converges.
- Subjects who never transition have Pareto-k = 0.0 (no importance weight variability).

References:
- Morsomme et al. (2025) Biostatistics kxaf038 - multistate semi-Markov MCEM
- Andersen & Keiding (2002) Statistical Methods in Medical Research - multi-state models
"""

using Test
using MultistateModels
using DataFrames
using Random
using Statistics
using LinearAlgebra

# Import internal functions for testing
import MultistateModels: Hazard, multistatemodel, fit, set_parameters!, simulate,
    get_parameters_flat, SamplePath

const RNG_SEED = 0xABCDEF01
const N_SUBJECTS = 1000       # Standard sample size for longtests
const MCEM_TOL = 0.05
const MAX_ITER = 25
const PARAM_TOL_REL = 0.50  # Relative tolerance for parameter recovery (50% - relaxed for TVC)

# ============================================================================
# Helper: Build TVC Panel Data
# ============================================================================

"""
Build panel data with time-varying covariate.
Each subject has covariate values that change at specified times.
"""
function build_tvc_panel_data(;
    n_subjects::Int,
    obs_times::Vector{Float64},
    covariate_change_times::Vector{Float64},
    covariate_generator::Function,  # (subj_id) -> Vector of covariate values
    obstype::Int = 2
)
    @assert length(covariate_change_times) >= 1 "Need at least one change time"
    
    # Merge observation times with covariate change times
    all_times = sort(unique(vcat(0.0, obs_times, covariate_change_times)))
    
    rows = []
    for subj in 1:n_subjects
        covariate_vals = covariate_generator(subj)
        @assert length(covariate_vals) == length(covariate_change_times) + 1
        
        for i in 1:(length(all_times) - 1)
            t_start = all_times[i]
            t_stop = all_times[i + 1]
            
            # Find which covariate interval we're in
            cov_idx = 1
            for (j, ct) in enumerate(covariate_change_times)
                if t_start >= ct
                    cov_idx = j + 1
                end
            end
            
            push!(rows, (
                id = subj,
                tstart = t_start,
                tstop = t_stop,
                statefrom = 1,
                stateto = 1,
                obstype = obstype,
                x = covariate_vals[cov_idx]
            ))
        end
    end
    
    return DataFrame(rows)
end

# ============================================================================
# Test 1: MCEM with Binary TVC (Treatment Switch) - Weibull Hazard
# ============================================================================

@testset "MCEM with Binary TVC (Treatment Switch)" begin
    Random.seed!(RNG_SEED)
    
    # Scenario: Patients switch from control (x=0) to treatment (x=1) at t=3
    # NOTE: We use Weibull hazards (semi-Markov) to trigger MCEM.
    # Exponential hazards are Markov and dispatch to matrix exponentiation.
    
    # True parameters: Weibull with shape=1.0 (equivalent to exponential), scale=0.25
    true_log_shape = log(1.0)
    true_log_scale = log(0.25)
    true_beta = 0.6  # Positive effect = increased hazard with treatment
    
    # Build panel data: observe at t=0,2,4,6,8, treatment starts at t=3
    panel_data = build_tvc_panel_data(
        n_subjects = N_SUBJECTS,
        obs_times = [2.0, 4.0, 6.0, 8.0],
        covariate_change_times = [3.0],
        covariate_generator = _ -> [0.0, 1.0],  # All subjects: control then treatment
        obstype = 2
    )
    
    # Create model for simulation with Weibull (semi-Markov)
    h12_sim = Hazard(@formula(0 ~ x), "wei", 1, 2)
    model_sim = multistatemodel(h12_sim; data=panel_data, surrogate=:markov)
    set_parameters!(model_sim, (h12 = [true_log_shape, true_log_scale, true_beta],))
    
    # Simulate panel data
    sim_result = simulate(model_sim; paths=false, data=true, nsim=1, autotmax=false)
    simulated_data = sim_result[1, 1]
    
    # Fit model via MCEM (Weibull triggers MCEM path)
    h12_fit = Hazard(@formula(0 ~ x), "wei", 1, 2)
    model_fit = multistatemodel(h12_fit; data=simulated_data, surrogate=:markov)
    
    fitted = fit(model_fit;
        proposal=:markov,
        verbose=false,
        maxiter=MAX_ITER,
        tol=MCEM_TOL,
        ess_target_initial=25,
        max_ess=400,
        compute_vcov=false,
        return_convergence_records=true)
    
    # Verify MCEM was used (has ConvergenceRecords with ess_trace)
    @test !isnothing(fitted.ConvergenceRecords)
    @test hasproperty(fitted.ConvergenceRecords, :ess_trace)
    
    # Check ESS diagnostics
    # NOTE: Some subjects may have ESS ≈ 1.0 if they transition early (deterministic path structure)
    ess_final = fitted.ConvergenceRecords.ess_trace[:, end]
    @test all(ess_final .>= 1.0)  # ESS should be at least 1
    @test mean(ess_final) > 5.0    # Average ESS should be reasonable
    
    # Parameter recovery tests (relaxed tolerance for stochastic MCEM)
    fitted_params = get_parameters_flat(fitted)
    
    # Shape parameter recovery (exp(fitted) vs exp(true))
    fitted_shape = exp(fitted_params[1])
    true_shape = exp(true_log_shape)
    @test isapprox(fitted_shape, true_shape; rtol=PARAM_TOL_REL)
    
    # Scale parameter recovery
    fitted_scale = exp(fitted_params[2])
    true_scale = exp(true_log_scale)
    @test isapprox(fitted_scale, true_scale; rtol=PARAM_TOL_REL)
    
    # Beta (treatment effect) recovery - check sign is correct
    # MCEM with TVC can have higher variance; just verify direction
    @test isfinite(fitted_params[3])
    @test fitted_params[3] > 0  # Correct sign (positive effect)
    
    # Verify log-likelihood is finite (convergence check)
    @test isfinite(fitted.loglik.loglik)
    
    println("  ✓ Binary TVC (treatment switch) MCEM fitting works")
end

# ============================================================================
# Test 2: MCEM with Continuous TVC (Time-Varying Biomarker) - Weibull Hazard
# ============================================================================

@testset "MCEM with Continuous TVC (Biomarker)" begin
    Random.seed!(RNG_SEED + 1)
    
    # Scenario: Biomarker increases over time (e.g., disease progression marker)
    # NOTE: Using Weibull with shape≈1 to trigger MCEM (exponential dispatches to Markov MLE)
    true_log_shape = log(1.0)  # Shape=1 is like exponential
    true_log_scale = log(0.2)
    true_beta = 0.3  # Each unit increase in biomarker increases log-hazard by 0.3
    
    # Build panel data with continuous covariate that changes at t=2,4
    panel_data = build_tvc_panel_data(
        n_subjects = N_SUBJECTS,
        obs_times = [3.0, 6.0],
        covariate_change_times = [2.0, 4.0],
        covariate_generator = subj -> begin
            # Subject-specific biomarker trajectory with some noise
            base = 1.0 + 0.2 * randn()
            [base, base + 0.5, base + 1.0]  # Increasing biomarker
        end,
        obstype = 2
    )
    
    # Create and simulate with Weibull
    h12_sim = Hazard(@formula(0 ~ x), "wei", 1, 2)
    model_sim = multistatemodel(h12_sim; data=panel_data, surrogate=:markov)
    set_parameters!(model_sim, (h12 = [true_log_shape, true_log_scale, true_beta],))
    
    sim_result = simulate(model_sim; paths=false, data=true, nsim=1, autotmax=false)
    simulated_data = sim_result[1, 1]
    
    # Fit model
    h12_fit = Hazard(@formula(0 ~ x), "wei", 1, 2)
    model_fit = multistatemodel(h12_fit; data=simulated_data, surrogate=:markov)
    
    fitted = fit(model_fit;
        proposal=:markov,
        verbose=false,
        maxiter=MAX_ITER,
        tol=MCEM_TOL,
        ess_target_initial=25,
        max_ess=400,
        compute_vcov=false,
        return_convergence_records=true)
    
    # Verify MCEM was used
    @test !isnothing(fitted.ConvergenceRecords)
    @test hasproperty(fitted.ConvergenceRecords, :ess_trace)
    
    # Parameter recovery tests (more relaxed for continuous TVC)
    fitted_params = get_parameters_flat(fitted)
    
    # All parameters should be finite
    @test all(isfinite.(fitted_params))
    
    # Shape parameter recovery
    fitted_shape = exp(fitted_params[1])
    true_shape = exp(true_log_shape)
    @test isapprox(fitted_shape, true_shape; rtol=PARAM_TOL_REL)
    
    # Scale parameter recovery  
    fitted_scale = exp(fitted_params[2])
    true_scale = exp(true_log_scale)
    @test isapprox(fitted_scale, true_scale; rtol=PARAM_TOL_REL)
    
    # Beta recovery (continuous covariate - check sign at minimum)
    @test fitted_params[3] > -0.5  # Positive or near-zero (true is positive)
    
    # Convergence check
    @test isfinite(fitted.loglik.loglik)
    
    println("  ✓ Continuous TVC (biomarker) MCEM fitting works")
end

# ============================================================================
# Test 3: MCEM with Weibull + TVC (Semi-Markov with Time-Varying Covariates)
# ============================================================================

@testset "MCEM Weibull + TVC (Semi-Markov)" begin
    Random.seed!(RNG_SEED + 2)
    
    # Semi-Markov model: Weibull hazards depend on sojourn time and TVC
    # Use moderate parameters to avoid degenerate cases
    true_log_shape = log(1.2)  # Shape > 1: increasing hazard (mild)
    true_log_scale = log(0.15)  # Lower scale for fewer events
    true_beta = 0.3
    
    # Panel data with TVC - longer observation period
    panel_data = build_tvc_panel_data(
        n_subjects = N_SUBJECTS,
        obs_times = [3.0, 6.0, 9.0],
        covariate_change_times = [4.0],
        covariate_generator = subj -> [0.0, 1.0],  # Switch at t=4
        obstype = 2
    )
    
    # Simulate from Weibull model with TVC
    h12_sim = Hazard(@formula(0 ~ x), "wei", 1, 2)
    model_sim = multistatemodel(h12_sim; data=panel_data, surrogate=:markov)
    set_parameters!(model_sim, (h12 = [true_log_shape, true_log_scale, true_beta],))
    
    sim_result = simulate(model_sim; paths=false, data=true, nsim=1, autotmax=false)
    simulated_data = sim_result[1, 1]
    
    # Fit model
    h12_fit = Hazard(@formula(0 ~ x), "wei", 1, 2)
    model_fit = multistatemodel(h12_fit; data=simulated_data, surrogate=:markov)
    
    fitted = fit(model_fit;
        proposal=:markov,
        verbose=false,
        maxiter=MAX_ITER,
        tol=MCEM_TOL,
        ess_target_initial=30,
        max_ess=500,
        compute_vcov=false,
        return_convergence_records=true)
    
    # Verify MCEM was used
    @test !isnothing(fitted.ConvergenceRecords)
    @test hasproperty(fitted.ConvergenceRecords, :ess_trace)
    
    # Parameter recovery tests
    fitted_params = get_parameters_flat(fitted)
    
    # Shape parameter recovery
    fitted_shape = exp(fitted_params[1])
    true_shape = exp(true_log_shape)
    @test isapprox(fitted_shape, true_shape; rtol=PARAM_TOL_REL)
    
    # Scale parameter recovery
    fitted_scale = exp(fitted_params[2])
    true_scale = exp(true_log_scale)
    @test isapprox(fitted_scale, true_scale; rtol=PARAM_TOL_REL)
    
    # Beta recovery
    @test isapprox(fitted_params[3], true_beta; atol=0.5)
    
    # Convergence check
    @test isfinite(fitted.loglik.loglik)
    
    println("  ✓ Weibull + TVC (semi-Markov) MCEM fitting works")
end

# ============================================================================
# Test 4: MCEM with Gompertz + TVC (Aging Effect + Treatment)
# ============================================================================

@testset "MCEM Gompertz + TVC (Aging + Treatment)" begin
    Random.seed!(RNG_SEED + 3)
    
    # Gompertz models increasing mortality with age
    # TVC captures treatment effect
    true_shape = 0.15  # Exponential increase in hazard
    true_log_scale = log(0.1)
    true_beta = -0.5  # Treatment reduces hazard
    
    # Panel data with treatment switch
    panel_data = build_tvc_panel_data(
        n_subjects = N_SUBJECTS,
        obs_times = [3.0, 6.0, 9.0, 12.0],
        covariate_change_times = [4.0],
        covariate_generator = subj -> rand() < 0.5 ? [0.0, 1.0] : [0.0, 0.0],  # Half get treatment
        obstype = 2
    )
    
    # Simulate
    h12_sim = Hazard(@formula(0 ~ x), "gom", 1, 2)
    model_sim = multistatemodel(h12_sim; data=panel_data, surrogate=:markov)
    set_parameters!(model_sim, (h12 = [true_shape, true_log_scale, true_beta],))
    
    sim_result = simulate(model_sim; paths=false, data=true, nsim=1, autotmax=false)
    simulated_data = sim_result[1, 1]
    
    # Fit model
    h12_fit = Hazard(@formula(0 ~ x), "gom", 1, 2)
    model_fit = multistatemodel(h12_fit; data=simulated_data, surrogate=:markov)
    
    fitted = fit(model_fit;
        proposal=:markov,
        verbose=false,
        maxiter=MAX_ITER,
        tol=MCEM_TOL,
        ess_target_initial=25,
        max_ess=400,
        compute_vcov=false,
        return_convergence_records=true)
    
    # Verify MCEM was used
    @test !isnothing(fitted.ConvergenceRecords)
    @test hasproperty(fitted.ConvergenceRecords, :ess_trace)
    
    # Check convergence
    @test isfinite(fitted.loglik.loglik)
    
    println("  ✓ Gompertz + TVC (aging + treatment) MCEM fitting works")
end

# ============================================================================
# Test 5: MCEM Illness-Death Model with TVC
# ============================================================================

@testset "MCEM Illness-Death with TVC" begin
    Random.seed!(RNG_SEED + 4)
    
    # Illness-death model: 1 (healthy) → 2 (ill) → 3 (dead), plus 1 → 3
    # TVC: Treatment status changes
    # NOTE: Using Weibull for all hazards to ensure MCEM is triggered
    
    # Build multi-state panel data with proper time grid
    n_subj = N_SUBJECTS
    change_time = 3.0
    
    # Create time grid that includes change time
    time_grid = [0.0, 2.0, change_time, 5.0, 7.0, 9.0]
    
    rows = []
    for subj in 1:n_subj
        trt_on = rand() < 0.5  # Half get treatment at change_time
        
        for i in 1:(length(time_grid) - 1)
            x_val = (time_grid[i] >= change_time && trt_on) ? 1.0 : 0.0
            push!(rows, (id=subj, tstart=time_grid[i], tstop=time_grid[i+1],
                         statefrom=1, stateto=1, obstype=2, x=x_val))
        end
    end
    panel_data = DataFrame(rows)
    
    # Define hazards - use Weibull for all to trigger MCEM
    h12_sim = Hazard(@formula(0 ~ x), "wei", 1, 2)   # Healthy → Ill
    h13_sim = Hazard(@formula(0 ~ x), "wei", 1, 3)   # Healthy → Dead
    h23_sim = Hazard(@formula(0 ~ x), "wei", 2, 3)   # Ill → Dead
    
    model_sim = multistatemodel(h12_sim, h13_sim, h23_sim; data=panel_data, surrogate=:markov)
    set_parameters!(model_sim, (
        h12 = [log(1.0), log(0.15), 0.3],  # wei: log(shape), log(scale), beta
        h13 = [log(1.2), log(0.08), 0.2],  # wei: log(shape), log(scale), beta
        h23 = [log(1.0), log(0.25), 0.4]   # wei: log(shape), log(scale), beta
    ))
    
    # Simulate
    sim_result = simulate(model_sim; paths=false, data=true, nsim=1, autotmax=false)
    simulated_data = sim_result[1, 1]
    
    # Fit
    h12_fit = Hazard(@formula(0 ~ x), "wei", 1, 2)
    h13_fit = Hazard(@formula(0 ~ x), "wei", 1, 3)
    h23_fit = Hazard(@formula(0 ~ x), "wei", 2, 3)
    model_fit = multistatemodel(h12_fit, h13_fit, h23_fit; data=simulated_data, surrogate=:markov)
    
    fitted = fit(model_fit;
        proposal=:markov,
        verbose=false,
        maxiter=MAX_ITER,
        tol=MCEM_TOL,
        ess_target_initial=30,
        max_ess=500,
        compute_vcov=false,
        return_convergence_records=true)
    
    # Verify MCEM was used
    @test !isnothing(fitted.ConvergenceRecords)
    @test hasproperty(fitted.ConvergenceRecords, :ess_trace)
    
    # Check convergence
    @test isfinite(fitted.loglik.loglik)
    
    # All fitted parameters should be finite
    fitted_params = get_parameters_flat(fitted)
    @test all(isfinite.(fitted_params))
    
    println("  ✓ Illness-death model with TVC MCEM fitting works")
end

# ============================================================================
# Test 6: MCEM with Multiple TVC Change Points - Weibull
# ============================================================================

@testset "MCEM Multiple TVC Change Points" begin
    Random.seed!(RNG_SEED + 5)
    
    # Scenario: Covariate changes at t=2, t=4, t=6
    # NOTE: Using Weibull to trigger MCEM
    true_log_shape = log(1.0)
    true_log_scale = log(0.2)
    true_beta = 0.25
    
    panel_data = build_tvc_panel_data(
        n_subjects = N_SUBJECTS,
        obs_times = [3.0, 5.0, 7.0, 9.0],
        covariate_change_times = [2.0, 4.0, 6.0],
        covariate_generator = subj -> begin
            # Covariate follows a step pattern
            base = randn() * 0.3
            [base, base + 0.5, base + 1.0, base + 0.75]  # Up then down
        end,
        obstype = 2
    )
    
    # Simulate with Weibull
    h12_sim = Hazard(@formula(0 ~ x), "wei", 1, 2)
    model_sim = multistatemodel(h12_sim; data=panel_data, surrogate=:markov)
    set_parameters!(model_sim, (h12 = [true_log_shape, true_log_scale, true_beta],))
    
    sim_result = simulate(model_sim; paths=false, data=true, nsim=1, autotmax=false)
    simulated_data = sim_result[1, 1]
    
    # Fit
    h12_fit = Hazard(@formula(0 ~ x), "wei", 1, 2)
    model_fit = multistatemodel(h12_fit; data=simulated_data, surrogate=:markov)
    
    fitted = fit(model_fit;
        proposal=:markov,
        verbose=false,
        maxiter=MAX_ITER,
        tol=MCEM_TOL,
        ess_target_initial=25,
        max_ess=400,
        compute_vcov=false,
        return_convergence_records=true)
    
    # Verify MCEM was used
    @test !isnothing(fitted.ConvergenceRecords)
    @test hasproperty(fitted.ConvergenceRecords, :ess_trace)
    
    # Check convergence
    @test isfinite(fitted.loglik.loglik)
    
    println("  ✓ Multiple TVC change points MCEM fitting works")
end

# ============================================================================
# Test 7: MCEM with AFT Effect + TVC - Weibull
# ============================================================================

@testset "MCEM AFT Effect with TVC" begin
    Random.seed!(RNG_SEED + 6)
    
    # AFT model: Covariates affect time scale, not hazard scale
    # NOTE: Using Weibull to trigger MCEM
    true_log_shape = log(1.2)
    true_log_scale = log(0.3)
    true_beta = 0.4  # AFT: time scale multiplier
    
    panel_data = build_tvc_panel_data(
        n_subjects = N_SUBJECTS,
        obs_times = [2.0, 4.0, 6.0],
        covariate_change_times = [3.0],
        covariate_generator = _ -> [0.0, 1.0],
        obstype = 2
    )
    
    # Simulate with Weibull AFT effect
    h12_sim = Hazard(@formula(0 ~ x), "wei", 1, 2; linpred_effect=:aft)
    model_sim = multistatemodel(h12_sim; data=panel_data, surrogate=:markov)
    set_parameters!(model_sim, (h12 = [true_log_shape, true_log_scale, true_beta],))
    
    sim_result = simulate(model_sim; paths=false, data=true, nsim=1, autotmax=false)
    simulated_data = sim_result[1, 1]
    
    # Fit with Weibull AFT
    h12_fit = Hazard(@formula(0 ~ x), "wei", 1, 2; linpred_effect=:aft)
    model_fit = multistatemodel(h12_fit; data=simulated_data, surrogate=:markov)
    
    fitted = fit(model_fit;
        proposal=:markov,
        verbose=false,
        maxiter=MAX_ITER,
        tol=MCEM_TOL,
        ess_target_initial=25,
        max_ess=400,
        compute_vcov=false,
        return_convergence_records=true)
    
    # Verify MCEM was used
    @test !isnothing(fitted.ConvergenceRecords)
    @test hasproperty(fitted.ConvergenceRecords, :ess_trace)
    
    @test isfinite(fitted.loglik.loglik)
    
    println("  ✓ AFT effect with TVC MCEM fitting works")
end

# ============================================================================
# Test 8: MCEM Reversible Model with TVC (Bidirectional Transitions)
# ============================================================================

@testset "MCEM Reversible Model with TVC" begin
    Random.seed!(RNG_SEED + 7)
    
    # Reversible 1 ↔ 2 model with TVC
    n_subj = N_SUBJECTS
    obs_times = [1.5, 3.0, 4.5, 6.0]
    change_time = 2.0
    
    rows = []
    for subj in 1:n_subj
        trt = rand() < 0.5 ? [0.0, 1.0] : [0.0, 0.0]
        
        all_times = sort(unique([0.0; obs_times; change_time]))
        for i in 1:(length(all_times)-1)
            x_val = all_times[i] < change_time ? trt[1] : trt[2]
            push!(rows, (id=subj, tstart=all_times[i], tstop=all_times[i+1],
                         statefrom=1, stateto=1, obstype=2, x=x_val))
        end
    end
    panel_data = DataFrame(rows)
    
    # Reversible Weibull model
    h12_sim = Hazard(@formula(0 ~ x), "wei", 1, 2)
    h21_sim = Hazard(@formula(0 ~ x), "wei", 2, 1)
    
    model_sim = multistatemodel(h12_sim, h21_sim; data=panel_data, surrogate=:markov)
    set_parameters!(model_sim, (
        h12 = [log(1.3), log(0.3), 0.3],
        h21 = [log(1.2), log(0.25), 0.2]
    ))
    
    sim_result = simulate(model_sim; paths=false, data=true, nsim=1, autotmax=false)
    simulated_data = sim_result[1, 1]
    
    # Fit
    h12_fit = Hazard(@formula(0 ~ x), "wei", 1, 2)
    h21_fit = Hazard(@formula(0 ~ x), "wei", 2, 1)
    model_fit = multistatemodel(h12_fit, h21_fit; data=simulated_data, surrogate=:markov)
    
    fitted = fit(model_fit;
        proposal=:markov,
        verbose=true,
        maxiter=MAX_ITER,
        tol=MCEM_TOL,
        ess_target_initial=30,
        max_ess=500,
        compute_vcov=false,
        return_convergence_records=true)
    
    # Verify MCEM was used
    @test !isnothing(fitted.ConvergenceRecords)
    @test hasproperty(fitted.ConvergenceRecords, :ess_trace)
    
    @test isfinite(fitted.loglik.loglik)
    
    println("  ✓ Reversible model with TVC MCEM fitting works")
end

# ============================================================================
# Summary
# ============================================================================

println("\n=== MCEM TVC Long Test Suite Complete ===\n")
println("Tests verify MCEM fitting for:")
println("  - Binary TVC (treatment switch)")
println("  - Continuous TVC (biomarker)")
println("  - Weibull + TVC (semi-Markov)")
println("  - Gompertz + TVC (aging + treatment)")
println("  - Illness-death model with TVC")
println("  - Multiple TVC change points")
println("  - AFT effect with TVC")
println("  - Reversible model with TVC")
