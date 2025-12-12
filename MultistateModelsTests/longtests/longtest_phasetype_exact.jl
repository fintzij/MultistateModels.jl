"""
Long test suite for phase-type HAZARD MODELS with exact (continuous-time) data.

This test suite validates inference when the TARGET MODEL has Coxian phase-type
hazard structure. Since phase-type models on the expanded state space are Markov
(exponential hazards), we use direct MLE fitting - no MCEM required.

Test workflow:
1. Build phase-type model (expanded state space with exponential hazards)
2. Simulate exact data from the phase-type model
3. Fit using standard Markov MLE
4. Verify parameter recovery

Test matrix:
- Number of phases: 2, 3 phases per transient state
- Model structure: illness-death (1→2→3, 1→3, 2→3)
- Observation type: exact (obstype=1)

The key insight: phase-type sojourn time distributions can approximate
semi-Markov processes (Weibull, Gompertz) while remaining tractable for
inference. This is an alternative to using phase-type as MCEM proposals.

References:
- Titman & Sharples (2010) Biometrics - phase-type semi-Markov approximations
- Asmussen et al. (1996) Coxian phase-type distributions
"""

using Test
using MultistateModels
using DataFrames
using Random
using Statistics
using LinearAlgebra

# Import internal functions
import MultistateModels: Hazard, multistatemodel, fit, set_parameters!, simulate,
    get_parameters_flat, get_parameters, SamplePath, @formula,
    PhaseTypeConfig, build_phasetype_model, build_phasetype_surrogate,
    phasetype_parameters_to_Q, build_phasetype_hazards

const RNG_SEED = 0xABCD0001
const N_SUBJECTS = 1000           # Large sample for MLE precision
const N_SIM_TRAJ = 5000           # Trajectories for distributional comparison
const MAX_TIME = 10.0             # Maximum follow-up time
const PARAM_TOL_REL = 0.15        # Relative tolerance (15% for direct MLE)

println("\n" * "="^70)
println("Phase-Type Hazard Models: Exact Data Long Tests")
println("="^70)
println("Testing inference for models with Coxian phase-type hazard structure.")
println("These models are Markov on the expanded state space → direct MLE.")
println("Default sample size: n=$N_SUBJECTS")

# Shared helper functions (compute_state_prevalence, compute_state_prevalence_phasetype, etc.) 
# are loaded from longtest_helpers.jl by the test runner.

# ============================================================================
# Helper Functions
# ============================================================================

"""
    check_parameter_recovery(fitted_params, true_params; tol_rel)

Verify fitted parameters are close to true values.
Works with log-scale parameters (baseline rates).
"""
function check_parameter_recovery(fitted_params::Vector{Float64}, true_params::Vector{Float64}; 
                                   tol_rel=PARAM_TOL_REL)
    @assert length(fitted_params) == length(true_params) "Parameter length mismatch"
    
    all_pass = true
    for i in eachindex(fitted_params)
        true_val = true_params[i]
        fitted_val = fitted_params[i]
        
        # Use absolute tolerance for values near zero
        if abs(true_val) < 0.1
            if abs(fitted_val - true_val) > 0.2
                println("  Parameter $i: true=$(round(true_val, digits=4)), fitted=$(round(fitted_val, digits=4)), diff=$(round(fitted_val - true_val, digits=4))")
                all_pass = false
            end
        else
            rel_err = abs(fitted_val - true_val) / abs(true_val)
            if rel_err > tol_rel
                println("  Parameter $i: true=$(round(true_val, digits=4)), fitted=$(round(fitted_val, digits=4)), rel_err=$(round(rel_err, digits=3))")
                all_pass = false
            end
        end
    end
    return all_pass
end

"""
    generate_exact_data_template(n_subj, max_time)

Create a DataFrame template for simulating exact data.
One row per subject, obstype=1 for exact observation.
"""
function generate_exact_data_template(n_subj::Int, max_time::Float64)
    return DataFrame(
        id = 1:n_subj,
        tstart = zeros(n_subj),
        tstop = fill(max_time, n_subj),
        statefrom = ones(Int, n_subj),
        stateto = ones(Int, n_subj),
        obstype = ones(Int, n_subj)
    )
end

# ============================================================================
# TEST SECTION 1: 2-PHASE COXIAN ILLNESS-DEATH
# ============================================================================

@testset "Phase-Type Hazard: 2-Phase Illness-Death (Exact Data)" begin
    Random.seed!(RNG_SEED)
    
    # Original 3-state illness-death: 1 → 2 → 3, with 1 → 3 direct
    tmat_obs = [0 1 1; 0 0 1; 0 0 0]
    
    # Configure 2 phases for each transient state
    # State 1: phases 1-2
    # State 2: phases 3-4  
    # State 3 (absorbing): phase 5
    config = PhaseTypeConfig(n_phases=[2, 2, 1])
    
    println("\n--- 2-Phase Coxian Illness-Death Model ---")
    println("Observed states: 3, Phases per transient: 2")
    println("Expanded states: 5 (2 + 2 + 1)")
    
    # Build surrogate to understand structure
    surrogate = build_phasetype_surrogate(tmat_obs, config)
    
    @testset "State space expansion" begin
        @test surrogate.n_observed_states == 3
        @test surrogate.n_expanded_states == 5
        @test surrogate.state_to_phases[1] == 1:2
        @test surrogate.state_to_phases[2] == 3:4
        @test surrogate.state_to_phases[3] == 5:5
    end
    
    # Create data template for simulation
    template = generate_exact_data_template(N_SUBJECTS, MAX_TIME)
    
    # Build phase-type model
    result = build_phasetype_model(tmat_obs, config; data=template, verbose=true)
    model = result.model
    
    @testset "Model structure" begin
        # Should have exponential hazards on expanded space
        @test length(model.hazards) > 0
        # All hazards should be MarkovHazard (exponential on expanded space)
        for haz in model.hazards
            @test haz isa MultistateModels.MarkovHazard
            @test haz.family == :exp
        end
    end
    
    # Set true parameters (log-scale rates)
    # Rates for transitions in expanded space
    # We'll set these directly on the model
    n_hazards = length(model.hazards)
    println("  Number of hazards in expanded model: $n_hazards")
    
    # True rates (positive scale)
    # Progression through phases: λ₁ (phase 1→2 within state 1), etc.
    # Exit rates: μ (phase → absorbing)
    true_rates = [0.5, 0.3, 0.4, 0.2, 0.25, 0.35, 0.3, 0.15]  # Adjust to match n_hazards
    
    # Truncate or pad if needed
    if length(true_rates) > n_hazards
        true_rates = true_rates[1:n_hazards]
    elseif length(true_rates) < n_hazards
        true_rates = vcat(true_rates, fill(0.25, n_hazards - length(true_rates)))
    end
    
    true_params_log = log.(true_rates)
    
    # Set parameters
    pars_dict = Dict{Symbol, Vector{Float64}}()
    for (i, haz) in enumerate(model.hazards)
        pars_dict[haz.hazname] = [true_params_log[i]]
    end
    set_parameters!(model, NamedTuple(pars_dict))
    
    println("  True rates: $(round.(true_rates, digits=3))")
    
    # Simulate exact data
    println("\nSimulating exact data...")
    sim_result = simulate(model; paths=false, data=true, nsim=1)
    exact_data = sim_result[1]
    
    # Count transitions in expanded space
    println("  Simulated $(nrow(exact_data)) observations")
    
    # For fitting, we create a NEW model using the SIMULATED DATA
    # The simulated data is already in expanded state space (phases 1-5)
    # We rebuild the hazard specifications to create a fresh model
    hazards_for_fit = build_phasetype_hazards(tmat_obs, config, surrogate)
    model_fit = multistatemodel(hazards_for_fit...; data=exact_data)
    
    @testset "Parameter recovery" begin
        println("\nFitting phase-type model with direct MLE...")
        fitted = fit(model_fit; verbose=false)
        
        fitted_params = get_parameters_flat(fitted)
        println("True params (log): $(round.(true_params_log, digits=3))")
        println("Fitted params (log): $(round.(fitted_params, digits=3))")
        
        @test check_parameter_recovery(fitted_params, true_params_log)
    end
end

# ============================================================================
# TEST SECTION 2: FUTURE - PHASE-TYPE APPROXIMATION QUALITY
# ============================================================================
# 
# The section below tests how well phase-type distributions can approximate
# semi-Markov (Weibull) sojourn times. This is a separate question from
# parameter recovery and requires careful analysis of:
# 1. How to initialize phase-type parameters to match Weibull moments
# 2. Whether the fitted phase-type faithfully approximates the target distribution
#
# This is commented out pending proper implementation of Weibull → phase-type
# moment matching or other approximation methods.
#
# @testset "Phase-Type Approximates Weibull" begin
#     ...
# end

# ============================================================================
# TEST SECTION 3: DISTRIBUTIONAL FIDELITY
# ============================================================================

@testset "Distributional Fidelity: Phase-Type Model" begin
    Random.seed!(RNG_SEED + 200)
    
    println("\n--- Distributional Fidelity Check ---")
    println("Verify fitted phase-type model reproduces data-generating distribution")
    
    # Simple 2-state model: 1 → 2
    tmat = [0 1; 0 0]
    config = PhaseTypeConfig(n_phases=[2, 1])
    
    # Build surrogate to understand phase structure
    surrogate = build_phasetype_surrogate(tmat, config)
    absorbing_phase = first(surrogate.state_to_phases[2])  # Phase for absorbing state
    
    # Generate template
    template = generate_exact_data_template(N_SUBJECTS, MAX_TIME)
    
    # Build and set true parameters
    result = build_phasetype_model(tmat, config; data=template, verbose=false)
    model_true = result.model
    
    # Set true rates: progression λ₁=0.4, exit μ₁=0.2, exit μ₂=0.5
    true_rates = [0.4, 0.2, 0.5]
    true_log = log.(true_rates)
    
    pars = Dict{Symbol, Vector{Float64}}()
    for (i, haz) in enumerate(model_true.hazards)
        pars[haz.hazname] = [true_log[i]]
    end
    set_parameters!(model_true, NamedTuple(pars))
    
    # Simulate data
    sim_true = simulate(model_true; paths=false, data=true, nsim=1)
    exact_data = sim_true[1]
    
    # Fit model - rebuild hazards for the simulated data (already in expanded space)
    hazards_for_fit = build_phasetype_hazards(tmat, config, surrogate)
    model_fit = multistatemodel(hazards_for_fit...; data=exact_data)
    fitted = fit(model_fit; verbose=false)
    
    # To compare distributions fairly, we must simulate from fitted using a FRESH template
    # with proper tstop=MAX_TIME (not the event times from exact_data).
    # Otherwise subjects who absorbed early have short observation windows in re-simulation.
    template_for_compare = generate_exact_data_template(N_SUBJECTS, MAX_TIME)
    hazards_for_compare = build_phasetype_hazards(tmat, config, surrogate)
    model_compare = multistatemodel(hazards_for_compare...; data=template_for_compare)
    
    # Set the fitted parameters on the comparison model
    fitted_params = get_parameters_flat(fitted)
    pars_compare = Dict{Symbol, Vector{Float64}}()
    for (i, haz) in enumerate(model_compare.hazards)
        pars_compare[haz.hazname] = [fitted_params[i]]
    end
    set_parameters!(model_compare, NamedTuple(pars_compare))
    
    # Simulate from fitted parameters with proper template
    sim_fitted = simulate(model_compare; paths=false, data=true, nsim=1)
    
    @testset "Sojourn time distribution" begin
        # Compare sojourn time CDFs
        # Note: simulated data uses expanded phases, absorbing state is phase 3
        sojourn_true = exact_data[exact_data.stateto .== absorbing_phase, :tstop]
        sojourn_fit = sim_fitted[1][sim_fitted[1].stateto .== absorbing_phase, :tstop]
        
        if length(sojourn_true) > 10 && length(sojourn_fit) > 10
            # Compare quantiles
            qs = [0.25, 0.5, 0.75, 0.9]
            q_true = quantile(sojourn_true, qs)
            q_fit = quantile(sojourn_fit, qs)
            
            println("  Quantiles (true → fitted):")
            for (q, qt, qf) in zip(qs, q_true, q_fit)
                println("    $(Int(q*100))th: $(round(qt, digits=3)) → $(round(qf, digits=3))")
            end
            
            # Median should match well
            @test abs(q_true[2] - q_fit[2]) / q_true[2] < 0.15
        else
            @test_skip "Insufficient events"
        end
    end
end

println("\n" * "="^70)
println("Phase-Type Hazard Model Tests Complete")
println("="^70)
