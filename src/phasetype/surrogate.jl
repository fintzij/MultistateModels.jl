# =============================================================================
# Phase-Type Surrogate Building
# =============================================================================
#
# Functions for constructing phase-type surrogates used in MCEM importance sampling.
# - build_phasetype_surrogate: Main entry point
# - build_coxian_intensity: Construct Coxian Q matrix
# - build_expanded_Q: Construct expanded intensity matrix
#
# CONSTRUCTION PATHS:
# There are two ways to build a PhaseTypeSurrogate:
#
# 1. build_phasetype_surrogate (THIS FILE)
#    - Direct construction from transition matrix and PhaseTypeConfig
#    - Used for testing and standalone phase-type model creation
#    - Entry point when user explicitly constructs a phase-type surrogate
#    - Parameters initialized from heuristics or BIC selection
#
# 2. _build_phasetype_from_markov (src/surrogate/markov.jl)
#    - Builds phase-type from a fitted Markov surrogate's estimated rates
#    - Production path in MCEM: first fits Markov surrogate, then expands to phase-type
#    - Transition rates come from Markov surrogate parameters
#    - Called via _build_phasetype_surrogate_from_model
#
# Both paths ultimately construct the same PhaseTypeSurrogate struct, just with
# different initialization for the transition rates.
#
# =============================================================================

# =============================================================================
# Coxian Phase-Type Construction
# =============================================================================

"""
    build_coxian_intensity(λ::Vector{Float64}, μ::Vector{Float64})

Build (p+1)×(p+1) intensity matrix Q for p-phase Coxian distribution.

# Arguments
- `λ`: Progression rates between phases (length p-1)
- `μ`: Absorption rates from each phase (length p)

# Returns
Intensity matrix Q with absorbing state in last row/column.
"""
function build_coxian_intensity(λ::Vector{Float64}, μ::Vector{Float64})
    p = length(μ)
    length(λ) == p - 1 || throw(DimensionMismatch("λ must have length $(p-1)"))
    
    Q = zeros(p + 1, p + 1)
    
    for i in 1:p
        if i < p
            Q[i, i] = -(λ[i] + μ[i])
            Q[i, i+1] = λ[i]
        else
            Q[i, i] = -μ[i]
        end
        Q[i, p+1] = μ[i]
    end
    
    return Q
end

# =============================================================================
# Building Phase-Type Surrogates
# =============================================================================

"""
    build_phasetype_surrogate(tmat, config; data=nothing, hazards=nothing, verbose=true)

Construct a PhaseTypeSurrogate from transition matrix and configuration.

# Arguments
- `tmat::Matrix{Int64}`: Transition matrix (tmat[i,j] > 0 means i→j allowed)
- `config::PhaseTypeConfig`: Phase count and structure specification
- `data`: Required for `n_phases=:auto` (BIC selection)
- `hazards`: Required for `n_phases=:heuristic`
- `verbose::Bool`: Print selection results

# Returns
`PhaseTypeSurrogate` with expanded state space and Q matrix.

# Example
```julia
config = PhaseTypeConfig(n_phases=:auto)
surrogate = build_phasetype_surrogate(tmat, config; data=my_data)
```

See also: [`PhaseTypeConfig`](@ref), [`PhaseTypeSurrogate`](@ref)
"""
function build_phasetype_surrogate(tmat::Matrix{Int64}, config::PhaseTypeConfig;
                                   data::Union{Nothing, DataFrame}=nothing,
                                   hazards::Union{Nothing, Vector}=nothing,
                                   verbose::Bool=true)
    n_states = size(tmat, 1)
    
    # Handle :auto - deprecated, fall back to heuristic
    if config.n_phases === :auto
        @warn "n_phases=:auto is deprecated in build_phasetype_surrogate. Using :heuristic instead. " *
              "For BIC-based selection, use select_surrogate() at model construction time." maxlog=1
        if isnothing(hazards)
            throw(ArgumentError("hazards is required when n_phases=:auto (falling back to :heuristic)"))
        end
        n_phases_vec = _compute_default_n_phases(tmat, hazards)
        n_phases = Dict{Int,Int}(i => n_phases_vec[i] for i in eachindex(n_phases_vec))
        config = PhaseTypeConfig(n_phases=n_phases, constraints=config.constraints, 
                                 max_phases=config.max_phases)
        if verbose
            println("Heuristic n_phases per state: $n_phases")
        end
    # Handle :heuristic - use smart defaults based on hazard types
    elseif config.n_phases === :heuristic
        if isnothing(hazards)
            throw(ArgumentError("hazards is required when n_phases=:heuristic"))
        end
        n_phases_vec = _compute_default_n_phases(tmat, hazards)
        # Convert vector to Dict{Int,Int} as required by PhaseTypeConfig
        n_phases = Dict{Int,Int}(i => n_phases_vec[i] for i in eachindex(n_phases_vec))
        config = PhaseTypeConfig(n_phases=n_phases, constraints=config.constraints, 
                                 max_phases=config.max_phases)
        if verbose
            println("Heuristic n_phases per state: $n_phases")
        end
    end
    
    # Identify transient vs absorbing states
    is_absorbing = [all(tmat[s, :] .== 0) for s in 1:n_states]
    transient_states = findall(.!is_absorbing)
    
    # Determine number of phases per state
    n_phases_per_state = _get_n_phases_per_state(n_states, is_absorbing, config)
    
    # Build state mappings
    state_to_phases, phase_to_state, n_expanded = _build_state_mappings(
        n_states, n_phases_per_state)
    
    # Initialize PH distributions for transient states
    phasetype_dists = Dict{Int, PhaseTypeDistribution}()
    for s in transient_states
        phasetype_dists[s] = _build_default_phasetype(n_phases_per_state[s])
    end
    
    # Build the expanded Q matrix
    expanded_Q = build_expanded_Q(tmat, n_phases_per_state, state_to_phases, 
                                  phase_to_state, phasetype_dists, n_expanded)
    
    return PhaseTypeSurrogate(
        phasetype_dists, n_states, n_expanded, state_to_phases,
        phase_to_state, expanded_Q, config
    )
end

# =============================================================================
# Helper Functions
# =============================================================================

"""Determine number of phases for each state based on config."""
function _get_n_phases_per_state(n_states::Int, is_absorbing::Vector{Bool}, 
                                 config::PhaseTypeConfig)
    n_phases_per_state = zeros(Int, n_states)
    
    if config.n_phases isa Symbol
        for s in 1:n_states
            n_phases_per_state[s] = is_absorbing[s] ? 1 : 2
        end
    elseif config.n_phases isa Int
        for s in 1:n_states
            n_phases_per_state[s] = is_absorbing[s] ? 1 : config.n_phases
        end
    elseif config.n_phases isa Dict{Int,Int}
        for s in 1:n_states
            if is_absorbing[s]
                n_phases_per_state[s] = 1
            else
                n_phases_per_state[s] = get(config.n_phases, s, 1)
            end
        end
    end
    
    return n_phases_per_state
end

"""Build bidirectional mappings between observed states and expanded phases."""
function _build_state_mappings(n_states::Int, n_phases_per_state::Vector{Int})
    state_to_phases = Vector{UnitRange{Int}}(undef, n_states)
    phase_to_state = Int[]
    
    current_phase = 1
    for s in 1:n_states
        n_phases = n_phases_per_state[s]
        state_to_phases[s] = current_phase:(current_phase + n_phases - 1)
        for _ in 1:n_phases
            push!(phase_to_state, s)
        end
        current_phase += n_phases
    end
    
    return state_to_phases, phase_to_state, current_phase - 1
end

"""Build a default Coxian phase-type distribution with unit mean."""
function _build_default_phasetype(n_phases::Int)
    if n_phases == 1
        Q = [-1.0 1.0; 0.0 0.0]
        return PhaseTypeDistribution(1, Q, [1.0])
    else
        base_rate = Float64(n_phases)
        λ = Vector{Float64}(undef, n_phases - 1)
        μ = Vector{Float64}(undef, n_phases)
        
        for i in 1:(n_phases - 1)
            λ[i] = base_rate * (n_phases - i) / n_phases
            μ[i] = base_rate * i / (2 * n_phases)
        end
        μ[n_phases] = base_rate
        
        Q = build_coxian_intensity(λ, μ)
        initial = zeros(n_phases)
        initial[1] = 1.0
        
        return PhaseTypeDistribution(n_phases, Q, initial)
    end
end

# =============================================================================
# Log-likelihood for Expanded Paths
# =============================================================================

"""
    loglik_expanded_path(path::SamplePath, surrogate::PhaseTypeSurrogate)

Compute the log-likelihood of an expanded sample path under the phase-type surrogate.

For a continuous-time Markov chain path with transitions at times t₀, t₁, ..., tₙ
through states (phases) s₀, s₁, ..., sₙ, the log-likelihood is:

    log L = Σᵢ [Q[sᵢ, sᵢ] × (tᵢ₊₁ - tᵢ) + log(Q[sᵢ, sᵢ₊₁])]

This function computes:
- Sojourn time contribution: diagonal element times sojourn duration
- Transition contribution: log of off-diagonal rate for each jump

# Arguments
- `path::SamplePath`: Sample path with times and states on the expanded (phase) space
- `surrogate::PhaseTypeSurrogate`: Phase-type surrogate containing the expanded Q matrix

# Returns
- `Float64`: Log-likelihood of the path under the surrogate Q matrix.
  Returns 0.0 for paths with no transitions (length(states) ≤ 1).
  Returns -Inf for impossible transitions (rate = 0).

# Example
```julia
# 2-state model with 1 → 2 (absorbing)
tmat = [0 1; 0 0]
config = PhaseTypeConfig(n_phases=1)
surrogate = build_phasetype_surrogate(tmat, config)

# Set rate = 2.0 for transition 1→2
surrogate.expanded_Q[1, 1] = -2.0
surrogate.expanded_Q[1, 2] = 2.0

# Path: state 1 for time 0.5, then to state 2
path = SamplePath(1, [0.0, 0.5], [1, 2])

ll = loglik_expanded_path(path, surrogate)
# ≈ -2.0 * 0.5 + log(2.0) = -1.0 + 0.693 = -0.307
```
"""
function loglik_expanded_path(path::SamplePath, surrogate::PhaseTypeSurrogate)
    n_transitions = length(path.states) - 1
    
    # No transitions → log-likelihood is 0
    n_transitions >= 1 || return 0.0
    
    Q = surrogate.expanded_Q
    ll = 0.0
    
    for i in 1:n_transitions
        state_from = path.states[i]
        state_to = path.states[i + 1]
        sojourn_time = path.times[i + 1] - path.times[i]
        
        # Sojourn time contribution: diagonal element (negative rate) × time
        ll += Q[state_from, state_from] * sojourn_time
        
        # Transition contribution: log of off-diagonal rate
        # Skip if state_from == state_to (pseudo-transition at observation time, no actual jump)
        if state_from != state_to
            rate = Q[state_from, state_to]
            if rate <= 0.0
                return -Inf  # Impossible transition
            end
            ll += log(rate)
        end
    end
    
    return ll
end

# =============================================================================
# Expanded Q Matrix Construction
# =============================================================================

"""
    build_expanded_Q(tmat, n_phases_per_state, state_to_phases, phase_to_state,
                     phasetype_dists, n_expanded; transition_rates=nothing)

Construct expanded intensity matrix Q for phase-type augmented model.

Combines within-state phase transitions and between-state absorption transitions.

# Returns
`Matrix{Float64}`: n_expanded × n_expanded intensity matrix.
"""
function build_expanded_Q(tmat::Matrix{Int64}, 
                          n_phases_per_state::Vector{Int},
                          state_to_phases::Vector{UnitRange{Int}},
                          phase_to_state::Vector{Int},
                          phasetype_dists::Dict{Int, PhaseTypeDistribution},
                          n_expanded::Int;
                          transition_rates::Union{Nothing, Dict{Tuple{Int,Int}, Float64}}=nothing)
    
    n_states = size(tmat, 1)
    Q = zeros(Float64, n_expanded, n_expanded)
    
    for s in 1:n_states
        phases = state_to_phases[s]
        n_phases = n_phases_per_state[s]
        dest_states = findall(tmat[s, :] .> 0)
        
        if isempty(dest_states) || !haskey(phasetype_dists, s)
            continue
        end
        
        ph = phasetype_dists[s]
        abs_rates = absorption_rates(ph)
        S = subintensity(ph)
        
        # Fill in within-state phase transitions from S matrix
        for i in 1:n_phases
            phase_i = phases[i]
            
            for j in 1:n_phases
                if i != j && S[i, j] > 0
                    Q[phase_i, phases[j]] = S[i, j]
                end
            end
            
            # Between-state transitions (absorption from this phase)
            abs_rate = abs_rates[i]
            if abs_rate > 0 && !isempty(dest_states)
                if isnothing(transition_rates)
                    rate_per_dest = abs_rate / length(dest_states)
                    for d in dest_states
                        Q[phase_i, first(state_to_phases[d])] = rate_per_dest
                    end
                else
                    total_rate_out = sum(get(transition_rates, (s, d), 0.0) for d in dest_states)
                    if total_rate_out > 0
                        for d in dest_states
                            rate_sd = get(transition_rates, (s, d), 0.0)
                            if rate_sd > 0
                                prop = rate_sd / total_rate_out
                                Q[phase_i, first(state_to_phases[d])] = abs_rate * prop
                            end
                        end
                    else
                        rate_per_dest = abs_rate / length(dest_states)
                        for d in dest_states
                            Q[phase_i, first(state_to_phases[d])] = rate_per_dest
                        end
                    end
                end
            end
        end
    end
    
    # Set diagonal elements
    for i in 1:n_expanded
        Q[i, i] = -sum(Q[i, j] for j in 1:n_expanded if j != i)
    end
    
    return Q
end
