# =============================================================================
# Phase-Type Simulation Long Tests
# =============================================================================
#
# Tests that validate phase-type simulation by comparing:
# 1. A model using :pt hazard specifications  
# 2. A manually-expanded Markov model with explicit exponential hazards
#
# Both models should produce equivalent cumulative incidence and state 
# prevalence curves when the parameters correspond.
#
# Model structure: 1 → 2 → 3 (progressive, state 3 absorbing)
# Phase-type on both transitions with 2 phases each.
#
# Expanded state space:
#   Original: 1 → 2 → 3
#   Expanded: 1a → 1b → 2a → 2b → 3
#             ↘      ↘
#              → 2a   → 3
#
# Where:
#   - 1a, 1b are phases of state 1 (Coxian for 1→2 transition)
#   - 2a, 2b are phases of state 2 (Coxian for 2→3 transition)
#   - λ₁ is progression 1a → 1b
#   - μ₁₁₂ is exit 1a → 2a (first phase exit to state 2)
#   - μ₂₁₂ is exit 1b → 2a (second phase exit to state 2)
#   - λ₂ is progression 2a → 2b
#   - μ₁₂₃ is exit 2a → 3
#   - μ₂₂₃ is exit 2b → 3
#
# =============================================================================

# =============================================================================
# Simulation Test Configuration
# =============================================================================

# Use same general config but can override for simulation-specific needs
const N_SIM_SUBJECTS = 5000     # Large N for stable prevalence estimates
const SIM_MAX_TIME = 15.0
const SIM_EVAL_TIMES = collect(0.0:0.25:SIM_MAX_TIME)
const SIM_RNG_SEED = 38472910

# Tolerance for prevalence/cumincid comparisons
const PREV_TOLERANCE = 0.03     # 3% absolute difference acceptable
const CUMINCID_TOLERANCE = 0.03 # 3% absolute difference acceptable

# =============================================================================
# SimulationTestResult Structure
# =============================================================================

"""
    SimulationTestResult

Container for simulation comparison test results.
"""
mutable struct SimulationTestResult
    name::String
    description::String
    
    # Comparison metrics
    max_prevalence_diff::Float64
    max_cumincid_diff::Float64
    passed::Bool
    
    # Diagnostic data
    eval_times::Vector{Float64}
    
    # Prevalence from each model (n_times × n_observed_states)
    prevalence_pt::Union{Nothing, Matrix{Float64}}
    prevalence_manual::Union{Nothing, Matrix{Float64}}
    
    # Cumulative incidence for each transition
    cumincid_12_pt::Union{Nothing, Vector{Float64}}
    cumincid_12_manual::Union{Nothing, Vector{Float64}}
    cumincid_23_pt::Union{Nothing, Vector{Float64}}
    cumincid_23_manual::Union{Nothing, Vector{Float64}}
    
    function SimulationTestResult(;
            name::String,
            description::String = "",
            max_prevalence_diff::Float64 = NaN,
            max_cumincid_diff::Float64 = NaN,
            passed::Bool = false,
            eval_times::Vector{Float64} = Float64[],
            prevalence_pt::Union{Nothing, Matrix{Float64}} = nothing,
            prevalence_manual::Union{Nothing, Matrix{Float64}} = nothing,
            cumincid_12_pt::Union{Nothing, Vector{Float64}} = nothing,
            cumincid_12_manual::Union{Nothing, Vector{Float64}} = nothing,
            cumincid_23_pt::Union{Nothing, Vector{Float64}} = nothing,
            cumincid_23_manual::Union{Nothing, Vector{Float64}} = nothing)
        new(name, description, max_prevalence_diff, max_cumincid_diff, passed,
            eval_times, prevalence_pt, prevalence_manual, 
            cumincid_12_pt, cumincid_12_manual, cumincid_23_pt, cumincid_23_manual)
    end
end

# Global results storage for simulation tests
const SIM_TEST_RESULTS = SimulationTestResult[]

# =============================================================================
# Helper: Build Manually Expanded Markov Model
# =============================================================================

"""
    build_manual_expanded_model(n_phases_12, n_phases_23, data_template)

Build a Markov model with explicitly expanded state space corresponding to
phase-type hazards on transitions 1→2 and 2→3.

For n_phases_12=2 and n_phases_23=2:
- States: 1a=1, 1b=2, 2a=3, 2b=4, 3=5
- Transitions: 
  - λ_12: 1→2 (progression in state 1)
  - μ1_12: 1→3 (exit from phase 1 of state 1 to state 2)
  - μ2_12: 2→3 (exit from phase 2 of state 1 to state 2)
  - λ_23: 3→4 (progression in state 2)
  - μ1_23: 3→5 (exit from phase 1 of state 2 to state 3)
  - μ2_23: 4→5 (exit from phase 2 of state 2 to state 3)

Returns model and a mapping Dict from expanded state to observed state.
"""
function build_manual_expanded_model(n_phases_12::Int, n_phases_23::Int, 
                                      data_template::DataFrame)
    @assert n_phases_12 == 2 && n_phases_23 == 2 "Only 2-phase Coxian supported"
    
    # Expanded state space: 1a=1, 1b=2, 2a=3, 2b=4, 3=5
    n_expanded = 5
    
    # Phase to observed state mapping
    phase_to_state = Dict(1 => 1, 2 => 1, 3 => 2, 4 => 2, 5 => 3)
    
    # Prepare data for expanded model - start in phase 1a (expanded state 1)
    expanded_data = copy(data_template)
    # All subjects start in state 1 (phase 1a of observed state 1)
    
    # Define hazards on expanded space (all exponential)
    # State 1 (phases 1a, 1b): 
    #   - Progression: 1a → 1b (λ_12)
    #   - Exit: 1a → 2a (μ1_12), 1b → 2a (μ2_12)
    # State 2 (phases 2a, 2b):
    #   - Progression: 2a → 2b (λ_23)
    #   - Exit: 2a → 3 (μ1_23), 2b → 3 (μ2_23)
    
    h_prog_1 = Hazard(@formula(0 ~ 1), "exp", 1, 2)   # λ_12: 1a → 1b
    h_exit_1a = Hazard(@formula(0 ~ 1), "exp", 1, 3)  # μ1_12: 1a → 2a  
    h_exit_1b = Hazard(@formula(0 ~ 1), "exp", 2, 3)  # μ2_12: 1b → 2a
    h_prog_2 = Hazard(@formula(0 ~ 1), "exp", 3, 4)   # λ_23: 2a → 2b
    h_exit_2a = Hazard(@formula(0 ~ 1), "exp", 3, 5)  # μ1_23: 2a → 3
    h_exit_2b = Hazard(@formula(0 ~ 1), "exp", 4, 5)  # μ2_23: 2b → 3
    
    model = multistatemodel(h_prog_1, h_exit_1a, h_exit_1b, 
                            h_prog_2, h_exit_2a, h_exit_2b; 
                            data=expanded_data)
    
    return model, phase_to_state
end

"""
    set_manual_parameters!(model, λ_12, μ1_12, μ2_12, λ_23, μ1_23, μ2_23)

Set parameters on the manually expanded Markov model.
Parameters are on natural scale (rates).
"""
function set_manual_parameters!(model, λ_12, μ1_12, μ2_12, λ_23, μ1_23, μ2_23)
    # Hazard keys in order: h12 (prog), h13 (exit 1a), h23 (exit 1b), h34 (prog), h35 (exit 2a), h45 (exit 2b)
    set_parameters!(model, (
        h12 = [λ_12],    # Progression 1a → 1b
        h13 = [μ1_12],   # Exit 1a → 2a
        h23 = [μ2_12],   # Exit 1b → 2a
        h34 = [λ_23],    # Progression 2a → 2b
        h35 = [μ1_23],   # Exit 2a → 3
        h45 = [μ2_23]    # Exit 2b → 3
    ))
end

# =============================================================================
# Helper: Compute Prevalence with Phase-to-State Mapping
# =============================================================================

"""
    compute_prevalence_mapped(paths, eval_times, n_observed_states, phase_to_state)

Compute state prevalence, mapping expanded phase states to observed states.
"""
function compute_prevalence_mapped(paths::Vector, eval_times::Vector{Float64},
                                   n_observed_states::Int, 
                                   phase_to_state::Dict{Int,Int})
    n_times = length(eval_times)
    prevalence = zeros(Float64, n_times, n_observed_states)
    n_paths = length(paths)
    
    for path in paths
        for (t_idx, t) in enumerate(eval_times)
            state_idx = searchsortedlast(path.times, t)
            if state_idx >= 1
                expanded_state = path.states[state_idx]
                observed_state = get(phase_to_state, expanded_state, expanded_state)
                if observed_state <= n_observed_states
                    prevalence[t_idx, observed_state] += 1.0
                end
            end
        end
    end
    
    prevalence ./= n_paths
    return prevalence
end

"""
    compute_cumincid_mapped(paths, eval_times, from_obs, to_obs, phase_to_state)

Compute cumulative incidence for a transition from observed state from_obs 
to observed state to_obs, using phase_to_state mapping for expanded paths.
"""
function compute_cumincid_mapped(paths::Vector, eval_times::Vector{Float64},
                                 from_obs::Int, to_obs::Int,
                                 phase_to_state::Dict{Int,Int})
    n_times = length(eval_times)
    cumincid = zeros(Float64, n_times)
    n_paths = length(paths)
    
    for path in paths
        transition_time = Inf
        
        for i in 1:(length(path.states) - 1)
            state_from = get(phase_to_state, path.states[i], path.states[i])
            state_to = get(phase_to_state, path.states[i+1], path.states[i+1])
            
            # First time we see transition from_obs → to_obs
            if state_from == from_obs && state_to == to_obs
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
    collapse_path(path::MultistateModels.SamplePath, phase_to_state::Dict{Int,Int})

Collapse a path from expanded state space to observed state space.
Merges consecutive intervals in the same observed state.

Returns a new SamplePath with collapsed states.
"""
function collapse_path(path::MultistateModels.SamplePath, phase_to_state::Dict{Int,Int})
    if isempty(path.states)
        return path
    end
    
    # Map all states to observed states
    observed_states = [get(phase_to_state, s, s) for s in path.states]
    
    # Merge consecutive intervals in the same observed state
    new_times = Float64[path.times[1]]
    new_states = Int[observed_states[1]]
    
    for i in 2:length(observed_states)
        if observed_states[i] != new_states[end]
            push!(new_times, path.times[i])
            push!(new_states, observed_states[i])
        end
    end
    
    return MultistateModels.SamplePath(path.subj, new_times, new_states)
end

"""
    paths_equivalent(path1::MultistateModels.SamplePath, path2::MultistateModels.SamplePath; time_tol=1e-10)

Check if two paths are equivalent (same states at same times).
"""
function paths_equivalent(path1::MultistateModels.SamplePath, 
                          path2::MultistateModels.SamplePath; 
                          time_tol::Float64=1e-10)
    if length(path1.states) != length(path2.states)
        return false
    end
    if path1.states != path2.states
        return false
    end
    for (t1, t2) in zip(path1.times, path2.times)
        if abs(t1 - t2) > time_tol
            return false
        end
    end
    return true
end

# =============================================================================
# Test PT-SIM-1: 2-Phase Coxian, No Covariates
# =============================================================================

"""
    run_ptsim_2phase_nocov()

Compare simulation output from:
1. PhaseType model with :pt hazards (2-phase Coxian on each transition)
2. Manually expanded Markov model with equivalent exponential hazards

Both should produce statistically equivalent cumulative incidence and 
prevalence curves.
"""
function run_ptsim_2phase_nocov()
    test_name = "ptsim_2phase_nocov"
    @info "Running $test_name"
    
    Random.seed!(SIM_RNG_SEED)
    
    # ==========================================================================
    # Parameters (natural scale)
    # ==========================================================================
    # Transition 1→2 (Coxian 2-phase)
    λ_12 = 0.5     # Progression rate through phases of state 1
    μ1_12 = 0.3    # Exit rate from phase 1 of state 1
    μ2_12 = 0.4    # Exit rate from phase 2 of state 1
    
    # Transition 2→3 (Coxian 2-phase)
    λ_23 = 0.4     # Progression rate through phases of state 2
    μ1_23 = 0.25   # Exit rate from phase 1 of state 2
    μ2_23 = 0.35   # Exit rate from phase 2 of state 2
    
    # ==========================================================================
    # Model 1: PhaseType model using :pt hazard specification
    # ==========================================================================
    @info "Building PhaseType model..."
    
    dat_pt = DataFrame(
        id = 1:N_SIM_SUBJECTS,
        tstart = zeros(N_SIM_SUBJECTS),
        tstop = fill(SIM_MAX_TIME, N_SIM_SUBJECTS),
        statefrom = ones(Int, N_SIM_SUBJECTS),
        stateto = ones(Int, N_SIM_SUBJECTS),
        obstype = ones(Int, N_SIM_SUBJECTS)
    )
    
    h12_pt = Hazard(@formula(0 ~ 1), "pt", 1, 2; n_phases=2, coxian_structure=:unstructured)
    h23_pt = Hazard(@formula(0 ~ 1), "pt", 2, 3; n_phases=2, coxian_structure=:unstructured)
    
    model_pt = multistatemodel(h12_pt, h23_pt; data=dat_pt)
    
    # Set parameters: phase-type format is [λ, μ₁, μ₂] for 2-phase
    set_parameters!(model_pt, (
        h12 = [λ_12, μ1_12, μ2_12],
        h23 = [λ_23, μ1_23, μ2_23]
    ))
    
    # ==========================================================================
    # Model 2: Manually expanded Markov model
    # ==========================================================================
    @info "Building manually expanded Markov model..."
    
    dat_manual = DataFrame(
        id = 1:N_SIM_SUBJECTS,
        tstart = zeros(N_SIM_SUBJECTS),
        tstop = fill(SIM_MAX_TIME, N_SIM_SUBJECTS),
        statefrom = ones(Int, N_SIM_SUBJECTS),  # Start in expanded state 1 (=phase 1a)
        stateto = ones(Int, N_SIM_SUBJECTS),
        obstype = ones(Int, N_SIM_SUBJECTS)
    )
    
    model_manual, phase_to_state = build_manual_expanded_model(2, 2, dat_manual)
    set_manual_parameters!(model_manual, λ_12, μ1_12, μ2_12, λ_23, μ1_23, μ2_23)
    
    # ==========================================================================
    # Simulate from both models
    # ==========================================================================
    @info "Simulating from PhaseType model..."
    Random.seed!(SIM_RNG_SEED)
    paths_pt = simulate(model_pt; paths=true, data=false)[1]
    
    @info "Simulating from manually expanded model..."
    Random.seed!(SIM_RNG_SEED)
    paths_manual = simulate(model_manual; paths=true, data=false)[1]
    
    # ==========================================================================
    # Collapse manual paths and check path-level equivalence
    # ==========================================================================
    @info "Checking path-level equivalence..."
    paths_manual_collapsed = [collapse_path(p, phase_to_state) for p in paths_manual]
    
    n_equivalent = sum(paths_equivalent(p1, p2) 
                       for (p1, p2) in zip(paths_pt, paths_manual_collapsed))
    path_equivalence_rate = n_equivalent / length(paths_pt)
    @info "Path equivalence" n_equivalent total=length(paths_pt) rate=path_equivalence_rate
    
    # ==========================================================================
    # Compute prevalence curves
    # ==========================================================================
    @info "Computing prevalence..."
    n_observed_states = 3
    
    # PhaseType model returns paths on observed state space by default
    prev_pt = compute_state_prevalence(paths_pt, SIM_EVAL_TIMES, n_observed_states)
    
    # Manual model needs mapping from expanded states to observed
    prev_manual = compute_prevalence_mapped(paths_manual, SIM_EVAL_TIMES, 
                                            n_observed_states, phase_to_state)
    
    # ==========================================================================
    # Compute cumulative incidence curves
    # ==========================================================================
    @info "Computing cumulative incidence..."
    
    # For PhaseType model (observed states)
    cumincid_12_pt = compute_cumulative_incidence(paths_pt, SIM_EVAL_TIMES, 1, 2)
    cumincid_23_pt = compute_cumulative_incidence(paths_pt, SIM_EVAL_TIMES, 2, 3)
    
    # For manual model (need mapping)
    cumincid_12_manual = compute_cumincid_mapped(paths_manual, SIM_EVAL_TIMES, 
                                                  1, 2, phase_to_state)
    cumincid_23_manual = compute_cumincid_mapped(paths_manual, SIM_EVAL_TIMES, 
                                                  2, 3, phase_to_state)
    
    # ==========================================================================
    # Compare curves
    # ==========================================================================
    @info "Comparing curves..."
    
    # Maximum absolute difference in prevalence (across all times and states)
    prev_diff = abs.(prev_pt .- prev_manual)
    max_prev_diff = maximum(prev_diff)
    
    # Maximum absolute difference in cumulative incidence
    cumincid_12_diff = maximum(abs.(cumincid_12_pt .- cumincid_12_manual))
    cumincid_23_diff = maximum(abs.(cumincid_23_pt .- cumincid_23_manual))
    max_cumincid_diff = max(cumincid_12_diff, cumincid_23_diff)
    
    # Pass criteria: curves match AND all paths are equivalent
    # (since same RNG seed, paths should be identical)
    passed = (max_prev_diff <= PREV_TOLERANCE) && 
             (max_cumincid_diff <= CUMINCID_TOLERANCE) &&
             (path_equivalence_rate == 1.0)
    
    @info "$test_name completed" passed max_prev_diff max_cumincid_diff path_equivalence_rate
    
    # Store result
    result = SimulationTestResult(
        name = test_name,
        description = "2-phase Coxian, no covariates, compare PT model to manual expansion",
        max_prevalence_diff = max_prev_diff,
        max_cumincid_diff = max_cumincid_diff,
        passed = passed,
        eval_times = SIM_EVAL_TIMES,
        prevalence_pt = prev_pt,
        prevalence_manual = prev_manual,
        cumincid_12_pt = cumincid_12_pt,
        cumincid_12_manual = cumincid_12_manual,
        cumincid_23_pt = cumincid_23_pt,
        cumincid_23_manual = cumincid_23_manual
    )
    
    push!(SIM_TEST_RESULTS, result)
    
    @testset "$test_name" begin
        @test max_prev_diff <= PREV_TOLERANCE
        @test max_cumincid_diff <= CUMINCID_TOLERANCE
        @test path_equivalence_rate == 1.0  # All paths should be identical
    end
    
    return result
end

# =============================================================================
# Test PT-SIM-2: 3-Phase Coxian, No Covariates
# =============================================================================

"""
    build_manual_expanded_model_3phase(data_template)

Build a Markov model with 3-phase Coxian on 1→2 and 2-phase on 2→3.

States: 1a=1, 1b=2, 1c=3, 2a=4, 2b=5, 3=6
"""
function build_manual_expanded_model_3phase(data_template::DataFrame)
    # Expanded state space: 1a=1, 1b=2, 1c=3 (state 1), 2a=4, 2b=5 (state 2), 3=6
    n_expanded = 6
    
    # Phase to observed state mapping
    phase_to_state = Dict(1 => 1, 2 => 1, 3 => 1, 4 => 2, 5 => 2, 6 => 3)
    
    expanded_data = copy(data_template)
    
    # Hazards for 3-phase on 1→2:
    #   λ1: 1a → 1b
    #   λ2: 1b → 1c
    #   μ1: 1a → 2a
    #   μ2: 1b → 2a
    #   μ3: 1c → 2a
    # Hazards for 2-phase on 2→3:
    #   λ3: 2a → 2b
    #   μ4: 2a → 3
    #   μ5: 2b → 3
    
    h_prog_1a = Hazard(@formula(0 ~ 1), "exp", 1, 2)   # λ1: 1a → 1b
    h_prog_1b = Hazard(@formula(0 ~ 1), "exp", 2, 3)   # λ2: 1b → 1c
    h_exit_1a = Hazard(@formula(0 ~ 1), "exp", 1, 4)   # μ1: 1a → 2a
    h_exit_1b = Hazard(@formula(0 ~ 1), "exp", 2, 4)   # μ2: 1b → 2a
    h_exit_1c = Hazard(@formula(0 ~ 1), "exp", 3, 4)   # μ3: 1c → 2a
    h_prog_2a = Hazard(@formula(0 ~ 1), "exp", 4, 5)   # λ3: 2a → 2b
    h_exit_2a = Hazard(@formula(0 ~ 1), "exp", 4, 6)   # μ4: 2a → 3
    h_exit_2b = Hazard(@formula(0 ~ 1), "exp", 5, 6)   # μ5: 2b → 3
    
    model = multistatemodel(h_prog_1a, h_prog_1b, h_exit_1a, h_exit_1b, h_exit_1c,
                            h_prog_2a, h_exit_2a, h_exit_2b; 
                            data=expanded_data)
    
    return model, phase_to_state
end

"""
    run_ptsim_3phase_nocov()

3-phase Coxian on 1→2, 2-phase on 2→3, no covariates.
"""
function run_ptsim_3phase_nocov()
    test_name = "ptsim_3phase_nocov"
    @info "Running $test_name"
    
    Random.seed!(SIM_RNG_SEED + 1)
    
    # Parameters
    # Transition 1→2 (3-phase Coxian): [λ1, λ2, μ1, μ2, μ3]
    λ1_12 = 0.6    # Progression 1a → 1b
    λ2_12 = 0.5    # Progression 1b → 1c
    μ1_12 = 0.2    # Exit from phase 1
    μ2_12 = 0.25   # Exit from phase 2
    μ3_12 = 0.3    # Exit from phase 3
    
    # Transition 2→3 (2-phase Coxian): [λ, μ1, μ2]
    λ_23 = 0.4
    μ1_23 = 0.25
    μ2_23 = 0.35
    
    # PhaseType model
    dat_pt = DataFrame(
        id = 1:N_SIM_SUBJECTS,
        tstart = zeros(N_SIM_SUBJECTS),
        tstop = fill(SIM_MAX_TIME, N_SIM_SUBJECTS),
        statefrom = ones(Int, N_SIM_SUBJECTS),
        stateto = ones(Int, N_SIM_SUBJECTS),
        obstype = ones(Int, N_SIM_SUBJECTS)
    )
    
    h12_pt = Hazard(@formula(0 ~ 1), "pt", 1, 2; n_phases=3, coxian_structure=:unstructured)
    h23_pt = Hazard(@formula(0 ~ 1), "pt", 2, 3; n_phases=2, coxian_structure=:unstructured)
    
    model_pt = multistatemodel(h12_pt, h23_pt; data=dat_pt)
    
    # 3-phase format: [λ1, λ2, μ1, μ2, μ3]
    # 2-phase format: [λ, μ1, μ2]
    set_parameters!(model_pt, (
        h12 = [λ1_12, λ2_12, μ1_12, μ2_12, μ3_12],
        h23 = [λ_23, μ1_23, μ2_23]
    ))
    
    # Manual expanded model
    dat_manual = copy(dat_pt)
    model_manual, phase_to_state = build_manual_expanded_model_3phase(dat_manual)
    
    # Set parameters: hazards in order from build function
    set_parameters!(model_manual, (
        h12 = [λ1_12],    # λ1: 1a → 1b
        h23 = [λ2_12],    # λ2: 1b → 1c
        h14 = [μ1_12],    # μ1: 1a → 2a
        h24 = [μ2_12],    # μ2: 1b → 2a
        h34 = [μ3_12],    # μ3: 1c → 2a
        h45 = [λ_23],     # λ3: 2a → 2b
        h46 = [μ1_23],    # μ4: 2a → 3
        h56 = [μ2_23]     # μ5: 2b → 3
    ))
    
    # Simulate
    Random.seed!(SIM_RNG_SEED + 1)
    paths_pt = simulate(model_pt; paths=true, data=false)[1]
    
    Random.seed!(SIM_RNG_SEED + 1)
    paths_manual = simulate(model_manual; paths=true, data=false)[1]
    
    # Collapse manual paths and check path-level equivalence
    paths_manual_collapsed = [collapse_path(p, phase_to_state) for p in paths_manual]
    n_equivalent = sum(paths_equivalent(p1, p2) 
                       for (p1, p2) in zip(paths_pt, paths_manual_collapsed))
    path_equivalence_rate = n_equivalent / length(paths_pt)
    
    # Compute metrics
    n_observed_states = 3
    prev_pt = compute_state_prevalence(paths_pt, SIM_EVAL_TIMES, n_observed_states)
    prev_manual = compute_prevalence_mapped(paths_manual, SIM_EVAL_TIMES, 
                                            n_observed_states, phase_to_state)
    
    cumincid_12_pt = compute_cumulative_incidence(paths_pt, SIM_EVAL_TIMES, 1, 2)
    cumincid_23_pt = compute_cumulative_incidence(paths_pt, SIM_EVAL_TIMES, 2, 3)
    cumincid_12_manual = compute_cumincid_mapped(paths_manual, SIM_EVAL_TIMES, 
                                                  1, 2, phase_to_state)
    cumincid_23_manual = compute_cumincid_mapped(paths_manual, SIM_EVAL_TIMES, 
                                                  2, 3, phase_to_state)
    
    # Compare
    max_prev_diff = maximum(abs.(prev_pt .- prev_manual))
    cumincid_12_diff = maximum(abs.(cumincid_12_pt .- cumincid_12_manual))
    cumincid_23_diff = maximum(abs.(cumincid_23_pt .- cumincid_23_manual))
    max_cumincid_diff = max(cumincid_12_diff, cumincid_23_diff)
    
    passed = (max_prev_diff <= PREV_TOLERANCE) && 
             (max_cumincid_diff <= CUMINCID_TOLERANCE) &&
             (path_equivalence_rate == 1.0)
    
    @info "$test_name completed" passed max_prev_diff max_cumincid_diff path_equivalence_rate
    
    result = SimulationTestResult(
        name = test_name,
        description = "3-phase on 1→2, 2-phase on 2→3, no covariates",
        max_prevalence_diff = max_prev_diff,
        max_cumincid_diff = max_cumincid_diff,
        passed = passed,
        eval_times = SIM_EVAL_TIMES,
        prevalence_pt = prev_pt,
        prevalence_manual = prev_manual,
        cumincid_12_pt = cumincid_12_pt,
        cumincid_12_manual = cumincid_12_manual,
        cumincid_23_pt = cumincid_23_pt,
        cumincid_23_manual = cumincid_23_manual
    )
    
    push!(SIM_TEST_RESULTS, result)
    
    @testset "$test_name" begin
        @test max_prev_diff <= PREV_TOLERANCE
        @test max_cumincid_diff <= CUMINCID_TOLERANCE
        @test path_equivalence_rate == 1.0
    end
    
    return result
end

# =============================================================================
# Test PT-SIM-3: 2-Phase with allequal constraint
# =============================================================================

"""
    run_ptsim_2phase_allequal()

2-phase Coxian with allequal constraint (μ1 = μ2) on both transitions.

Note: The parameter format is still [λ, μ₁, μ₂] even with allequal structure.
The structure constraint affects fitting, not the parameter count.
For simulation with equal exit rates, we set μ₁ = μ₂ explicitly.
"""
function run_ptsim_2phase_allequal()
    test_name = "ptsim_2phase_allequal"
    @info "Running $test_name"
    
    Random.seed!(SIM_RNG_SEED + 2)
    
    # Parameters - with allequal intent, μ1 = μ2 for each transition
    λ_12 = 0.5
    μ_12 = 0.3      # Single exit rate (used for both phases)
    
    λ_23 = 0.4
    μ_23 = 0.25     # Single exit rate
    
    # PhaseType model with allequal structure
    # Note: still provide [λ, μ₁, μ₂] format but with μ₁ = μ₂
    dat_pt = DataFrame(
        id = 1:N_SIM_SUBJECTS,
        tstart = zeros(N_SIM_SUBJECTS),
        tstop = fill(SIM_MAX_TIME, N_SIM_SUBJECTS),
        statefrom = ones(Int, N_SIM_SUBJECTS),
        stateto = ones(Int, N_SIM_SUBJECTS),
        obstype = ones(Int, N_SIM_SUBJECTS)
    )
    
    h12_pt = Hazard(@formula(0 ~ 1), "pt", 1, 2; n_phases=2, coxian_structure=:allequal)
    h23_pt = Hazard(@formula(0 ~ 1), "pt", 2, 3; n_phases=2, coxian_structure=:allequal)
    
    model_pt = multistatemodel(h12_pt, h23_pt; data=dat_pt)
    
    # Still [λ, μ₁, μ₂] format, but with equal exit rates
    set_parameters!(model_pt, (
        h12 = [λ_12, μ_12, μ_12],    # μ₁ = μ₂
        h23 = [λ_23, μ_23, μ_23]     # μ₁ = μ₂
    ))
    
    # Manual model with μ1 = μ2
    dat_manual = copy(dat_pt)
    model_manual, phase_to_state = build_manual_expanded_model(2, 2, dat_manual)
    set_manual_parameters!(model_manual, λ_12, μ_12, μ_12, λ_23, μ_23, μ_23)
    
    # Simulate
    Random.seed!(SIM_RNG_SEED + 2)
    paths_pt = simulate(model_pt; paths=true, data=false)[1]
    
    Random.seed!(SIM_RNG_SEED + 2)
    paths_manual = simulate(model_manual; paths=true, data=false)[1]
    
    # Collapse manual paths and check path-level equivalence
    paths_manual_collapsed = [collapse_path(p, phase_to_state) for p in paths_manual]
    n_equivalent = sum(paths_equivalent(p1, p2) 
                       for (p1, p2) in zip(paths_pt, paths_manual_collapsed))
    path_equivalence_rate = n_equivalent / length(paths_pt)
    
    # Compute metrics
    n_observed_states = 3
    prev_pt = compute_state_prevalence(paths_pt, SIM_EVAL_TIMES, n_observed_states)
    prev_manual = compute_prevalence_mapped(paths_manual, SIM_EVAL_TIMES, 
                                            n_observed_states, phase_to_state)
    
    cumincid_12_pt = compute_cumulative_incidence(paths_pt, SIM_EVAL_TIMES, 1, 2)
    cumincid_23_pt = compute_cumulative_incidence(paths_pt, SIM_EVAL_TIMES, 2, 3)
    cumincid_12_manual = compute_cumincid_mapped(paths_manual, SIM_EVAL_TIMES, 
                                                  1, 2, phase_to_state)
    cumincid_23_manual = compute_cumincid_mapped(paths_manual, SIM_EVAL_TIMES, 
                                                  2, 3, phase_to_state)
    
    # Compare
    max_prev_diff = maximum(abs.(prev_pt .- prev_manual))
    cumincid_12_diff = maximum(abs.(cumincid_12_pt .- cumincid_12_manual))
    cumincid_23_diff = maximum(abs.(cumincid_23_pt .- cumincid_23_manual))
    max_cumincid_diff = max(cumincid_12_diff, cumincid_23_diff)
    
    passed = (max_prev_diff <= PREV_TOLERANCE) && 
             (max_cumincid_diff <= CUMINCID_TOLERANCE) &&
             (path_equivalence_rate == 1.0)
    
    @info "$test_name completed" passed max_prev_diff max_cumincid_diff path_equivalence_rate
    
    result = SimulationTestResult(
        name = test_name,
        description = "2-phase Coxian with allequal constraint",
        max_prevalence_diff = max_prev_diff,
        max_cumincid_diff = max_cumincid_diff,
        passed = passed,
        eval_times = SIM_EVAL_TIMES,
        prevalence_pt = prev_pt,
        prevalence_manual = prev_manual,
        cumincid_12_pt = cumincid_12_pt,
        cumincid_12_manual = cumincid_12_manual,
        cumincid_23_pt = cumincid_23_pt,
        cumincid_23_manual = cumincid_23_manual
    )
    
    push!(SIM_TEST_RESULTS, result)
    
    @testset "$test_name" begin
        @test max_prev_diff <= PREV_TOLERANCE
        @test max_cumincid_diff <= CUMINCID_TOLERANCE
        @test path_equivalence_rate == 1.0
    end
    
    return result
end

# =============================================================================
# Runner Function
# =============================================================================

"""
    run_all_phasetype_simulation_tests()

Run all phase-type simulation comparison tests.
"""
function run_all_phasetype_simulation_tests()
    @info "Running all Phase-Type SIMULATION tests"
    empty!(SIM_TEST_RESULTS)
    
    @testset "Phase-Type Simulation Tests" begin
        run_ptsim_2phase_nocov()
        run_ptsim_3phase_nocov()
        run_ptsim_2phase_allequal()
    end
    
    # Print summary
    println("\n" * "="^80)
    println("PHASE-TYPE SIMULATION TESTS SUMMARY")
    println("="^80)
    for r in SIM_TEST_RESULTS
        status = r.passed ? "✓ PASS" : "✗ FAIL"
        prev_diff = @sprintf("%.4f", r.max_prevalence_diff)
        cumincid_diff = @sprintf("%.4f", r.max_cumincid_diff)
        println("$(status) | $(r.name) | prev_diff=$prev_diff | cumincid_diff=$cumincid_diff")
    end
    passed = count(r -> r.passed, SIM_TEST_RESULTS)
    println("\nTotal: $passed/$(length(SIM_TEST_RESULTS)) passed")
    println("="^80)
    
    return SIM_TEST_RESULTS
end

# Run if executed directly
if abspath(PROGRAM_FILE) == @__FILE__
    using Test
    using MultistateModels
    using DataFrames
    using Random
    using Printf
    using Statistics
    
    include("../longtest_config.jl")
    include("../longtest_helpers.jl")
    
    run_all_phasetype_simulation_tests()
end
