# =============================================================================
# Phase-Type Surrogate Building
# =============================================================================
#
# Functions for constructing phase-type surrogates used in MCEM importance sampling.
# - build_phasetype_surrogate: Main entry point
# - build_coxian_intensity: Construct Coxian Q matrix
# - build_expanded_Q: Construct expanded intensity matrix
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
    
    # Handle :auto - select via BIC
    if config.n_phases === :auto
        if data === nothing
            throw(ArgumentError("data is required when n_phases=:auto for BIC-based selection"))
        end
        n_phases = _select_n_phases_bic(tmat, data; max_phases=config.max_phases, verbose=verbose)
        config = PhaseTypeConfig(n_phases=n_phases, constraints=config.constraints, 
                                 max_phases=config.max_phases)
    # Handle :heuristic - use smart defaults based on hazard types
    elseif config.n_phases === :heuristic
        if hazards === nothing
            throw(ArgumentError("hazards is required when n_phases=:heuristic"))
        end
        n_phases = _compute_default_n_phases(tmat, hazards)
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
