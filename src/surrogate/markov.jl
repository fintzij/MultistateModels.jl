# =============================================================================
# Markov Surrogate Model Construction
# =============================================================================
#
# This file handles Markov surrogate construction for MCEM importance sampling.
#
# MARKOV SURROGATE:
# The MarkovSurrogate is an exponential (time-homogeneous) approximation to the
# target model used as an importance sampling proposal. It's fitted to the data
# or initialized from user parameters.
#
# PHASE-TYPE EXTENSION:
# For improved proposal quality (especially when target hazards have non-exponential
# sojourn time distributions), the Markov surrogate can be extended to a phase-type
# surrogate via `_build_phasetype_from_markov`.
#
# The construction path is:
#   Target Model → Markov Surrogate (fit) → Phase-Type Surrogate (expand)
#
# This differs from direct phase-type construction in src/phasetype/surrogate.jl,
# which builds the phase-type directly from heuristics without a Markov fit.
# The Markov-based approach provides better initialized transition rates.
#
# =============================================================================

"""
    make_surrogate_model(model)

Create a Markov surrogate model from a multistate model.

If the model already has a `markovsurrogate`, uses its hazards and parameters.
Otherwise, builds surrogate hazards from scratch using the model's hazard specifications.

# Arguments
- `model`: multistate model object

# Returns
- `MultistateModel` suitable for fitting as a Markov model
"""
function make_surrogate_model(model::MultistateProcess)
    if isnothing(model.markovsurrogate)
        # Build surrogate hazards from scratch
        # Ensure hazards are sorted to match model.hazards order (which determines tmat indices)
        hazards = collect(model.modelcall.hazards)
        sort!(hazards, by = h -> (h.statefrom, h.stateto))
        
        surrogate_haz, surrogate_pars, _ = build_hazards(hazards...; data = model.data, surrogate = true)
        markov_surrogate = MarkovSurrogate(surrogate_haz, surrogate_pars)
    else
        markov_surrogate = model.markovsurrogate
    end
    
    # Generate bounds for the surrogate model
    surrogate_bounds = MultistateModels._generate_package_bounds_from_components(
        markov_surrogate.parameters.flat, 
        markov_surrogate.hazards, 
        model.hazkeys
    )
    
    MultistateModels.MultistateModel(
        model.data,
        markov_surrogate.parameters,  # Use surrogate's parameters
        surrogate_bounds,
        markov_surrogate.hazards,
        model.totalhazards,
        model.tmat,
        model.emat,
        model.hazkeys,
        model.subjectindices,
        model.SubjectWeights,
        model.ObservationWeights,
        model.CensoringPatterns,
        markov_surrogate,  # Unified surrogate field
        model.modelcall,
        nothing)  # No phasetype_expansion for surrogate
end


"""
    fit_surrogate(model::MultistateProcess; type=:markov, method=:mle, ...)

Fit a Markov or phase-type surrogate model for importance sampling.

# Arguments
- `model`: Multistate model object
- `type::Symbol = :markov`: Surrogate type (`:markov` or `:phasetype`)
- `method::Symbol = :mle`: Fitting method (`:mle` or `:heuristic`)
- `n_phases = 2`: For phase-type: number of phases (Int, Dict{Int,Int}, or :auto/:heuristic)
- `surrogate_parameters = nothing`: Optional fixed parameters (skips fitting)
- `surrogate_constraints = nothing`: Optional constraints for MLE fitting
- `verbose = true`: Print progress information

# Returns
- `MarkovSurrogate` if `type=:markov`
- `PhaseTypeSurrogate` if `type=:phasetype`

# Examples
```julia
# Fit Markov surrogate via MLE (default)
surrogate = fit_surrogate(model)

# Fit Markov surrogate with heuristic (faster, no optimization)
surrogate = fit_surrogate(model; method=:heuristic)

# Fit phase-type surrogate with 3 phases
surrogate = fit_surrogate(model; type=:phasetype, n_phases=3)
```

See also: [`initialize_surrogate!`](@ref), [`MarkovSurrogate`](@ref), [`PhaseTypeSurrogate`](@ref)
"""
function fit_surrogate(model::MultistateProcess; 
    surrogate_parameters = nothing, 
    surrogate_constraints = nothing, 
    verbose = true,
    type::Symbol = :markov,
    method::Symbol = :mle,
    n_phases::Union{Int, Dict{Int,Int}, Symbol} = 2)
    
    _validate_surrogate_inputs(type, method)
    
    if type === :markov
        return _fit_markov_surrogate(model; 
            method = method,
            surrogate_parameters = surrogate_parameters,
            surrogate_constraints = surrogate_constraints,
            verbose = verbose)
    else  # :phasetype
        return _fit_phasetype_surrogate(model;
            method = method,
            n_phases = n_phases,
            surrogate_parameters = surrogate_parameters,
            surrogate_constraints = surrogate_constraints,
            verbose = verbose)
    end
end

"""
    _validate_surrogate_inputs(type, method)

Validate surrogate fitting input parameters.

# Throws
- `ArgumentError` for invalid type or method
"""
function _validate_surrogate_inputs(type::Symbol, method::Symbol)
    type in (:markov, :phasetype) || 
        throw(ArgumentError("type must be :markov or :phasetype, got :$type"))
    method in (:mle, :heuristic) || 
        throw(ArgumentError("method must be :mle or :heuristic, got :$method"))
end

# =============================================================================
# Internal: Markov Surrogate Fitting
# =============================================================================

"""
    _fit_markov_surrogate(model; method, surrogate_parameters, surrogate_constraints, verbose)

Internal function to fit a Markov surrogate.

- `:mle` method: Fits exponential hazards via maximum likelihood
- `:heuristic` method: Uses crude transition rates from data
"""
function _fit_markov_surrogate(model;
    method::Symbol = :mle,
    surrogate_parameters = nothing,
    surrogate_constraints = nothing,
    verbose = true)
    
    # Build surrogate model structure
    surrogate_model = make_surrogate_model(model)
    
    # If parameters provided directly, use them
    if !isnothing(surrogate_parameters)
        set_parameters!(surrogate_model, surrogate_parameters)
        markov_surrogate = MarkovSurrogate(surrogate_model.hazards, surrogate_model.parameters; fitted=true)
        if verbose
            println("Markov surrogate built with provided parameters.\n")
        end
        return markov_surrogate
    end
    
    if method === :heuristic
        # Heuristic: use crude rates
        set_crude_init!(surrogate_model)
        markov_surrogate = MarkovSurrogate(surrogate_model.hazards, surrogate_model.parameters; fitted=true)
        if verbose
            println("Markov surrogate built with crude rate heuristic.\n")
        end
        return markov_surrogate
        
    else  # :mle
        # MLE: optimize
        if verbose
            println("Fitting Markov surrogate via MLE ...\n")
        end
        
        # Set crude inits as starting point
        set_crude_init!(surrogate_model)
        
        # Validate constraints if provided
        if !isnothing(surrogate_constraints)
            consfun_surrogate = parse_constraints(surrogate_constraints.cons, surrogate_model.hazards; 
                                                   consfun_name = :consfun_surrogate)
            initcons = consfun_surrogate(zeros(length(surrogate_constraints.cons)), 
                                          get_parameters_flat(surrogate_model), nothing)
            badcons = findall(initcons .< surrogate_constraints.lcons .|| 
                             initcons .> surrogate_constraints.ucons)
            if length(badcons) > 0
                throw(ArgumentError("Constraints $badcons are violated at the initial parameter values for the Markov surrogate."))
            end
        end
        
        # Fit via Markov model MLE
        # Note: Disable all variance computation for surrogate - we only need point estimates
        surrogate_fitted = fit(surrogate_model; constraints = surrogate_constraints, 
                               compute_vcov = false, compute_ij_vcov = false, compute_jk_vcov = false,
                               verbose = false)
        
        markov_surrogate = MarkovSurrogate(surrogate_fitted.hazards, surrogate_fitted.parameters; fitted=true)
        
        if verbose
            println("Markov surrogate MLE complete.\n")
        end
        
        return markov_surrogate
    end
end

# =============================================================================
# Internal: Phase-Type Surrogate Fitting  
# =============================================================================

"""
    _fit_phasetype_surrogate(model; method, n_phases, surrogate_parameters, surrogate_constraints, verbose, fit_tau)

Internal function to fit a phase-type surrogate.

- `:mle` method: First fits Markov surrogate via MLE, then optionally fits all rates via MLE
- `:heuristic` method: Uses crude rates divided by n_phases

# Arguments
- `model`: The target multistate model
- `method::Symbol=:mle`: Fitting method
- `n_phases`: Number of phases per state
- `surrogate_parameters`: Optional fixed parameters
- `surrogate_constraints`: Optional constraints
- `verbose::Bool=true`: Print progress
- `fit_tau::Bool=true`: Estimate all phase-type rates via MLE (progression and absorption)

# Returns
- `PhaseTypeSurrogate` with fitted rates stored in `fitted_rates` field
"""
function _fit_phasetype_surrogate(model;
    method::Symbol = :mle,
    n_phases::Union{Int, Dict{Int,Int}, Symbol} = 2,
    surrogate_parameters = nothing,
    surrogate_constraints = nothing,
    verbose = true,
    fit_tau::Bool = true)
    
    # First, get or fit the Markov surrogate
    markov_surrogate = _fit_markov_surrogate(model;
        method = method,
        surrogate_parameters = surrogate_parameters,
        surrogate_constraints = surrogate_constraints,
        verbose = verbose)
    
    # Build phase-type configuration
    config = ProposalConfig(type = :phasetype, n_phases = n_phases)
    
    # Determine n_phases_vec based on config
    n_states = size(model.tmat, 1)
    is_absorbing = [all(model.tmat[s, :] .== 0) for s in 1:n_states]
    transient_states = findall(.!is_absorbing)
    
    n_phases_resolved = config.n_phases
    if n_phases_resolved === :auto
        n_phases_vec = _select_n_phases_bic(model.tmat, model.data; 
                                            max_phases = config.max_phases, verbose = verbose)
    elseif n_phases_resolved === :heuristic
        n_phases_vec = _compute_default_n_phases(model.tmat, model.hazards)
    elseif n_phases_resolved isa Int
        n_phases_vec = [is_absorbing[s] ? 1 : n_phases_resolved for s in 1:n_states]
    elseif n_phases_resolved isa Dict{Int,Int}
        n_phases_vec = zeros(Int, n_states)
        for s in 1:n_states
            if is_absorbing[s]
                n_phases_vec[s] = 1
            else
                n_phases_vec[s] = get(n_phases_resolved, s, 1)
            end
        end
    end
    
    # Fit all rates via MLE if requested and we have phases > 1
    fitted_rates = nothing
    if fit_tau && method == :mle && any(n_phases_vec[s] > 1 for s in transient_states)
        if verbose
            println("Fitting phase-type rates via MLE...")
        end
        fitted_rates = _fit_phasetype_mle(model, markov_surrogate, n_phases_vec;
                                           verbose = verbose)
    end
    
    # Build phase-type surrogate with fitted rates
    phasetype_surrogate = _build_phasetype_from_markov(model, markov_surrogate; 
                                                        config = config,
                                                        fitted_rates = fitted_rates,
                                                        verbose = verbose)
    
    return phasetype_surrogate
end

"""
    _build_phasetype_from_markov(model, markov_surrogate; config, fitted_rates, verbose)

Build a phase-type surrogate from a fitted/initialized Markov surrogate.

This is the production path for phase-type surrogate construction in MCEM:
- If `fitted_rates` is provided (from MLE), uses optimized progression/absorption rates directly
- Otherwise, falls back to Markov surrogate rates with Erlang-like structure

# Arguments
- `model`: The target multistate model
- `markov_surrogate::MarkovSurrogate`: Fitted Markov surrogate with estimated rates
- `config::ProposalConfig`: Phase-type configuration (n_phases, structure, etc.)
- `fitted_rates::Union{Nothing, Dict{Symbol, Any}}`: Fitted rates from `_fit_phasetype_mle`:
  - `:progression` → Dict{Int, Vector{Float64}} of λ rates
  - `:absorption` → Dict{Int, Matrix{Float64}} of μ rates (phases × destinations)
  - `:destinations` → Dict{Int, Vector{Int}} of destination states
  If `nothing`, falls back to Markov rates.
- `verbose::Bool`: Print construction details

# Returns
`PhaseTypeSurrogate` with expanded state space and Q matrix.

# See Also
- `build_phasetype_surrogate` (src/phasetype/surrogate.jl): Direct construction path
- `_fit_phasetype_mle`: Fits progression and absorption rates via MLE
"""
function _build_phasetype_from_markov(model, markov_surrogate::MarkovSurrogate;
                                       config::ProposalConfig, 
                                       fitted_rates::Union{Nothing, Dict{Symbol, Any}} = nothing,
                                       verbose::Bool = true)
    
    n_states = size(model.tmat, 1)
    is_absorbing = [all(model.tmat[s, :] .== 0) for s in 1:n_states]
    transient_states = findall(.!is_absorbing)
    
    # Determine number of phases per state
    n_phases = config.n_phases
    if n_phases === :auto
        n_phases_vec = _select_n_phases_bic(model.tmat, model.data; 
                                            max_phases = config.max_phases, verbose = verbose)
    elseif n_phases === :heuristic
        n_phases_vec = _compute_default_n_phases(model.tmat, model.hazards)
    elseif n_phases isa Int
        n_phases_vec = [is_absorbing[s] ? 1 : n_phases for s in 1:n_states]
    elseif n_phases isa Dict{Int,Int}
        # Per-state specification via Dict - states not in Dict default to 1 phase
        n_phases_vec = zeros(Int, n_states)
        for s in 1:n_states
            if is_absorbing[s]
                n_phases_vec[s] = 1
            else
                n_phases_vec[s] = get(n_phases, s, 1)
            end
        end
    end
    
    if verbose
        println("Phase-type surrogate configuration:")
        println("  n_phases per state: $n_phases_vec")
        if !isnothing(fitted_rates)
            println("  Using MLE-fitted progression/absorption rates")
        else
            println("  Using Markov surrogate rates (Erlang-like structure)")
        end
    end
    
    # Build state mappings
    state_to_phases, phase_to_state, n_expanded = _build_state_mappings(n_states, n_phases_vec)
    
    # Extract transition rates from Markov surrogate (used for fallback or PhaseTypeDistribution)
    transition_rates = Dict{Tuple{Int,Int}, Float64}()
    for (haz_idx, h) in enumerate(markov_surrogate.hazards)
        s, d = h.statefrom, h.stateto
        hazname = h.hazname
        rate = markov_surrogate.parameters.nested[hazname].baseline[Symbol("$(hazname)_rate")]
        transition_rates[(s, d)] = rate
    end
    
    # Build expanded Q matrix
    # If we have fitted rates, use them directly; otherwise use PhaseTypeDistribution approach
    expanded_Q = zeros(Float64, n_expanded, n_expanded)
    phasetype_dists = Dict{Int, PhaseTypeDistribution}()
    structure = config.structure
    
    if !isnothing(fitted_rates)
        # Build Q matrix directly from fitted progression/absorption rates
        progression = fitted_rates[:progression]
        absorption = fitted_rates[:absorption]
        destinations = fitted_rates[:destinations]
        
        for s in transient_states
            n_ph = n_phases_vec[s]
            phases_s = state_to_phases[s]
            dests = destinations[s]
            
            prog_rates = get(progression, s, Float64[])
            abs_rates = get(absorption, s, zeros(n_ph, length(dests)))
            
            for (j, phase_j) in enumerate(phases_s)
                total_out = 0.0
                
                # Progression to next phase (if not last phase)
                if j < n_ph && j <= length(prog_rates)
                    next_phase = phases_s[j + 1]
                    expanded_Q[phase_j, next_phase] = prog_rates[j]
                    total_out += prog_rates[j]
                end
                
                # Absorption to each destination
                for (d_idx, dest) in enumerate(dests)
                    dest_first_phase = first(state_to_phases[dest])
                    rate = j <= size(abs_rates, 1) && d_idx <= size(abs_rates, 2) ? abs_rates[j, d_idx] : 0.0
                    expanded_Q[phase_j, dest_first_phase] = rate
                    total_out += rate
                end
                
                # Diagonal
                expanded_Q[phase_j, phase_j] = -total_out
            end
            
            # Also build PhaseTypeDistribution for storage (informational)
            total_rate = sum(get(transition_rates, (s, d), 0.0) for d in 1:n_states if d != s)
            if total_rate > 0
                phasetype_dists[s] = _build_coxian_from_rate(n_ph, total_rate; structure=:sctp)
            else
                phasetype_dists[s] = _build_default_phasetype(n_ph)
            end
        end
    else
        # Fallback: use Markov rates with Erlang-like PhaseTypeDistribution
        for s in transient_states
            n_ph = n_phases_vec[s]
            total_rate = sum(get(transition_rates, (s, d), 0.0) for d in 1:n_states if d != s)
            
            if total_rate <= 0
                phasetype_dists[s] = _build_default_phasetype(n_ph)
            else
                phasetype_dists[s] = _build_coxian_from_rate(n_ph, total_rate; structure=:sctp)
            end
        end
        
        # Build Q matrix from PhaseTypeDistributions
        expanded_Q = build_expanded_Q(model.tmat, n_phases_vec, state_to_phases, 
                                      phase_to_state, phasetype_dists, n_expanded;
                                      transition_rates = transition_rates)
    end
    
    # Create PhaseTypeConfig for storage
    n_phases_dict = Dict{Int,Int}(s => n_phases_vec[s] for s in 1:n_states if n_phases_vec[s] > 1)
    ph_config = PhaseTypeConfig(n_phases = n_phases_dict, structure = structure,
                                max_phases = config.max_phases)
    
    # PhaseTypeSurrogate is self-contained: store hazards and parameters directly
    surrogate = PhaseTypeSurrogate(
        phasetype_dists,
        n_states,
        n_expanded,
        state_to_phases,
        phase_to_state,
        expanded_Q,
        ph_config,
        markov_surrogate.hazards,      # Store hazards for covariate definitions
        markov_surrogate.parameters;   # Store parameters for covariate coefficients (β)
        fitted_rates = fitted_rates
    )
    
    if verbose
        println("  Total expanded states: $n_expanded")
        println("  Phase-type surrogate built successfully.\n")
    end
    
    return surrogate
end


"""
    is_surrogate_fitted(model::MultistateProcess) -> Bool

Check if the model's surrogate has been fitted.

Returns `true` if the model has a surrogate and it has been fitted via MLE or
with user-provided parameters. Returns `false` if there is no surrogate or
if the surrogate has only default (placeholder) parameters.

# Example
```julia
model = multistatemodel(h12; data=data, surrogate=:markov)  # fitted by default
is_surrogate_fitted(model)  # true

model2 = multistatemodel(h12; data=data, surrogate=:markov, fit_surrogate=false)
is_surrogate_fitted(model2)  # false
```

See also: [`initialize_surrogate!`](@ref), [`MarkovSurrogate`](@ref)
"""
function is_surrogate_fitted(model::MultistateProcess)
    isnothing(model.surrogate) && return false
    return model.surrogate.fitted
end


"""
    initialize_surrogate!(model; type=:markov, method=:mle, ...)

Initialize and fit a surrogate for MCEM importance sampling.

This is the unified entry point for surrogate creation. It builds and fits
a Markov or phase-type surrogate, storing it in the model's `surrogate` field.

# Arguments
- `model::MultistateProcess`: A mutable multistate model

# Keywords
- `type::Symbol = :markov`: Surrogate type
  - `:markov`: Exponential hazard approximation (fast, good for most cases)
  - `:phasetype`: Phase-type approximation (better for non-exponential sojourns)
- `method::Symbol = :mle`: Fitting method
  - `:mle`: Maximum likelihood (recommended)
  - `:heuristic`: Fast initialization without optimization
- `n_phases::Union{Int, Dict{Int,Int}, Symbol} = 2`: For phase-type: number of phases
  - `Int`: Same number of phases for all transient states
  - `Dict{Int,Int}`: Per-state phase counts
  - `:heuristic`: Automatic selection based on hazard types
- `surrogate_parameters = nothing`: Optional fixed parameters (skips fitting)
- `surrogate_constraints = nothing`: Optional constraints for MLE fitting
- `verbose::Bool = true`: Print progress information

# Returns
- The modified model (also modifies in-place)

# Examples
```julia
# Create a semi-Markov model without surrogate
model = multistatemodel(h12_wei, h21_wei; data=data)

# Add and fit a Markov surrogate via MLE
initialize_surrogate!(model)

# Use heuristic (faster, no optimization)
initialize_surrogate!(model; method=:heuristic)

# Use phase-type surrogate with 3 phases
initialize_surrogate!(model; type=:phasetype, n_phases=3)

# Check if surrogate is fitted
is_surrogate_fitted(model)  # true

# Fit model using the surrogate
fitted = fit(model)
```

# Notes
MCEM fitting via `fit()` requires a fitted surrogate. If no surrogate is present,
`fit()` will error with a helpful message directing you to call this function.

See also: [`is_surrogate_fitted`](@ref), [`fit_surrogate`](@ref), [`multistatemodel`](@ref)
"""
function initialize_surrogate!(model::MultistateProcess; 
    type::Symbol = :markov,
    method::Symbol = :mle,
    n_phases::Union{Int, Dict{Int,Int}, Symbol} = 2,
    surrogate_parameters = nothing, 
    surrogate_constraints = nothing, 
    verbose = true)
    
    _validate_surrogate_inputs(type, method)
    
    if type === :markov
        surrogate = _fit_markov_surrogate(model;
            method = method,
            surrogate_parameters = surrogate_parameters,
            surrogate_constraints = surrogate_constraints,
            verbose = verbose)
    else  # :phasetype
        surrogate = _fit_phasetype_surrogate(model;
            method = method,
            n_phases = n_phases,
            surrogate_parameters = surrogate_parameters,
            surrogate_constraints = surrogate_constraints,
            verbose = verbose)
    end
    
    # Set the unified surrogate field
    model.surrogate = surrogate
    
    return model
end


"""
    _build_coxian_from_rate(n_phases::Int, total_rate::Float64; 
                            structure::Symbol=:unstructured) -> PhaseTypeDistribution

Build a Coxian phase-type distribution that has approximately the same mean
sojourn time as an exponential with the given rate.

A Coxian distribution has subintensity matrix S of the form (for 3 phases):
```
S = [-(r₁ + a₁)    r₁           0     ]
    [    0      -(r₂ + a₂)     r₂     ]
    [    0          0         -a₃    ]
```
where rᵢ is the progression rate to the next phase and aᵢ is the absorption rate.
The absorption rates are computed as: aᵢ = -sum(S[i,:]) (negative row sums).

The full intensity matrix Q (including absorbing state) would be:
```
Q = [-(r₁ + a₁)    r₁           0        a₁  ]
    [    0      -(r₂ + a₂)     r₂        a₂  ]
    [    0          0         -a₃        a₃  ]
    [    0          0           0         0  ]
```

# Arguments
- `n_phases::Int`: Number of phases in the Coxian distribution
- `total_rate::Float64`: Target total exit rate (mean sojourn time ≈ 1/total_rate)
- `structure::Symbol`: Coxian structure constraint (default: `:sctp`)
  - `:unstructured`: No constraints on absorption rates. Initialized with τ = 1
    (uniform absorption μᵢ = μ). Rates are free parameters during optimization.
  - `:sctp`: SCTP (Stationary Conditional Transition Probability) constraint from
    Titman & Sharples (2010). Requires τⱼ = 1 for all j, meaning μⱼ = μ₁ (uniform).
    This ensures P(dest | leaving) is constant across phases.
  - `:erlang`: Erlang structure with only last phase absorbing (μᵢ = 0 for i < n).
    Gives CV = 1/√n, good for approximating Weibull with shape > 1.
  - `Function`: Custom constraint function `f(n_phases, total_rate) -> Q` returning the 
    (n_phases + 1) × (n_phases + 1) intensity matrix

# Returns
- `PhaseTypeDistribution`: Coxian distribution with the specified structure

# Arguments
- `n_phases::Int`: Number of phases in the Coxian distribution
- `total_rate::Float64`: Total exit rate (sum of all destination rates)
- `structure::Union{Symbol, Function}`: Constraint structure
  - `:sctp`: SCTP constraint with provided τ values (no ordering)
  - `:sctp_increasing`: SCTP with τ₁ ≤ τ₂ ≤ ... ≤ τₙ constraint
  - `:sctp_decreasing`: SCTP with τ₁ ≥ τ₂ ≥ ... ≥ τₙ constraint
  - `:erlang`: Only last phase absorbs (τⱼ = 0 for j < n)
  - `:unstructured`: No constraints (free τⱼ)
  - `Function`: Custom function (n_phases, total_rate) -> Q matrix
- `tau::Union{Nothing, Vector{Float64}}`: Absorption rate multipliers τⱼ.
  - If `nothing`, defaults to τⱼ = 1 for all j
  - Length must equal n_phases
  - All values must be non-negative
  - For `:sctp_increasing`: must satisfy τ₁ ≤ τ₂ ≤ ... ≤ τₙ
  - For `:sctp_decreasing`: must satisfy τ₁ ≥ τ₂ ≥ ... ≥ τₙ
"""
function _build_coxian_from_rate(n_phases::Int, total_rate::Float64; 
                                  structure::Union{Symbol, Function} = :unstructured,
                                  tau::Union{Nothing, Vector{Float64}} = nothing)
    if n_phases == 1
        # Single phase = exponential with the given rate
        # Q is 2×2: phase + absorbing state
        Q = [-total_rate  total_rate;
              0.0         0.0]
        initial = [1.0]
        return PhaseTypeDistribution(1, Q, initial)
    end
    
    # Handle custom constraint function - expects user to return full Q matrix
    if structure isa Function
        Q = structure(n_phases, total_rate)
        expected_size = n_phases + 1
        @assert size(Q) == (expected_size, expected_size) "Custom constraint must return $(expected_size)×$(expected_size) matrix (phases + absorbing)"
        @assert all(diag(Q)[1:n_phases] .< 0) "Diagonal elements (transient states) must be negative"
        @assert all(Q[end, :] .== 0) "Last row (absorbing state) must be zeros"
        initial = zeros(Float64, n_phases)
        initial[1] = 1.0
        return PhaseTypeDistribution(n_phases, Q, initial)
    end
    
    # Handle tau parameter
    if isnothing(tau)
        # Default: uniform τⱼ = 1 for all j
        tau_vec = ones(Float64, n_phases)
    else
        tau_vec = tau
        # Validate tau
        length(tau_vec) == n_phases || 
            throw(ArgumentError("tau must have length $n_phases, got $(length(tau_vec))"))
        all(τ >= 0 for τ in tau_vec) || 
            throw(ArgumentError("all tau values must be non-negative"))
    end
    
    # Validate ordering constraint for :sctp
    if structure === :sctp
        for i in 1:(n_phases-1)
            tau_vec[i] <= tau_vec[i+1] || 
                throw(ArgumentError(":sctp requires τ₁ ≤ τ₂ ≤ ... ≤ τₙ (eigenvalue ordering for identifiability), but τ[$i]=$(tau_vec[i]) > τ[$(i+1)]=$(tau_vec[i+1])"))
        end
    end
    
    # Build Q matrix: (n_phases + 1) × (n_phases + 1)
    Q = zeros(Float64, n_phases + 1, n_phases + 1)
    
    if structure in (:sctp, :unstructured)
        # SCTP (Stationary Conditional Transition Probability) constraint
        # from Titman & Sharples (2010).
        #
        # The SCTP constraint ensures that P(dest = s | leaving state r) is
        # CONSTANT across all phases. This is achieved by:
        #   μⱼₛ = τⱼ × μ₁ₛ  for all destinations s
        #
        # The non-exponential behavior comes from varying τⱼ values:
        # - τⱼ = 1 for all j → exponential distribution
        # - τⱼ increasing → increasing hazard  
        # - τⱼ decreasing → decreasing hazard
        #
        # Structure:
        #   progression λ = n * total_rate (uniform across phases)
        #   absorption μⱼ = τⱼ × base_absorption
        #
        # We build at unit scale first, then rescale to match target mean.
        
        # Build at unit scale
        Q_unit = zeros(Float64, n_phases + 1, n_phases + 1)
        progression_unit = Float64(n_phases)
        base_absorption_unit = 1.0
        
        for i in 1:n_phases
            absorption_unit = tau_vec[i] * base_absorption_unit
            
            if i < n_phases
                Q_unit[i, i+1] = progression_unit
                Q_unit[i, n_phases+1] = absorption_unit
                Q_unit[i, i] = -(progression_unit + absorption_unit)
            else
                # Last phase: absorption only, no progression
                Q_unit[i, n_phases+1] = absorption_unit
                Q_unit[i, i] = -absorption_unit
            end
        end
        
        # Compute mean at unit scale
        S_unit = Q_unit[1:n_phases, 1:n_phases]
        α = zeros(Float64, n_phases); α[1] = 1.0
        e = ones(Float64, n_phases)
        unit_mean = -(α' * (S_unit \ e))
        
        # Scale to match target mean: scaling Q by c divides mean by c
        target_mean = 1.0 / total_rate
        scale_factor = unit_mean / target_mean
        
        Q .= Q_unit .* scale_factor
        Q[n_phases+1, :] .= 0.0  # Ensure absorbing row stays zero
        
    elseif structure == :erlang
        # Erlang structure: only last phase absorbs
        #
        # CRITICAL: For phase-type expansion to provide benefit over Markov,
        # the marginal survival function must be NON-EXPONENTIAL.
        #
        # Erlang structure:
        # - Phases 1 to n-1: only progression (rate = n * total_rate), no absorption
        # - Phase n: only absorption (rate = n * total_rate)
        #
        # This gives an Erlang(n, n*total_rate) distribution with:
        #   Mean = n / (n * total_rate) = 1/total_rate  ✓
        #   Var = n / (n * total_rate)^2 = 1/(n * total_rate^2)
        #   CV = 1/√n (decreasing with n, unlike exponential CV=1)
        #
        # The lower CV makes this better at approximating Weibull with shape > 1.
        
        erlang_rate = n_phases * total_rate  # Rate per phase
        
        for i in 1:n_phases
            if i < n_phases
                # Intermediate phases: progress only, no absorption
                Q[i, i+1] = erlang_rate
                Q[i, n_phases+1] = 0.0
                Q[i, i] = -erlang_rate
            else
                # Last phase: absorb only
                Q[i, n_phases+1] = erlang_rate
                Q[i, i] = -erlang_rate
            end
        end
        
    else
        throw(ArgumentError("Unknown structure: $structure. Use :sctp, :unstructured, or :erlang"))
    end
    # Last row (absorbing state) is already zeros
    
    initial = zeros(Float64, n_phases)
    initial[1] = 1.0  # Start in first phase
    
    return PhaseTypeDistribution(n_phases, Q, initial)
end


# =============================================================================
# Markov Surrogate Marginal Log-Likelihood
# =============================================================================

"""
    compute_markov_marginal_loglik(model, surrogate::MarkovSurrogate)

Compute the marginal log-likelihood of the observed data under the Markov surrogate.

This is used as the normalizing constant r(Y|θ') in importance sampling:
    log f̂(Y|θ) = log r(Y|θ') + Σᵢ log(mean(νᵢ))

where νᵢ are the importance weights for subject i.

# Arguments
- `model`: The multistate model containing the data
- `surrogate::MarkovSurrogate`: The fitted Markov surrogate

# Returns
- `Float64`: The marginal log-likelihood under the surrogate
"""
function compute_markov_marginal_loglik(model::MultistateProcess, surrogate::MarkovSurrogate)
    # Build a temporary Markov model with the surrogate's hazards and parameters
    # Generate bounds for the surrogate model
    surrogate_bounds = MultistateModels._generate_package_bounds_from_components(
        surrogate.parameters.flat,
        surrogate.hazards,
        model.hazkeys
    )
    
    surrogate_model = MultistateModel(
        model.data,
        surrogate.parameters,
        surrogate_bounds,
        surrogate.hazards,
        model.totalhazards,
        model.tmat,
        model.emat,
        model.hazkeys,
        model.subjectindices,
        model.SubjectWeights,
        model.ObservationWeights,
        model.CensoringPatterns,
        surrogate,               # unified surrogate field
        model.modelcall,
        nothing                  # phasetype_expansion (not needed for surrogate)
    )
    
    # Build the bookkeeping structure for MPanelData
    books = build_tpm_mapping(surrogate_model.data)
    
    # Create MPanelData for the Markov likelihood
    data = MPanelData(surrogate_model, books)
    
    # Compute log-likelihood using the surrogate's flat parameters
    ll = loglik_markov(surrogate.parameters.flat, data; neg = false)
    
    return ll
end


# =============================================================================
# Phase-Type Surrogate τ Estimation via MLE
# =============================================================================

"""
    phasetype_marginal_loglik(theta, model, markov_rates, n_phases_vec, 
                               state_to_phases, phase_to_state, n_expanded, tmat; 
                               neg=true, structure=:sctp)

Compute the marginal log-likelihood of panel data under the phase-type model.

This function is used for optimizing τ (absorption rate multipliers) in the
phase-type surrogate. Given a vector of τ values, it:
1. Builds Coxian phase-type distributions with the given τ for each transient state
2. Constructs the expanded Q matrix
3. Computes the forward algorithm likelihood on the expanded state space

# Arguments
- `theta::Vector{Float64}`: τ values for all phases across all transient states
  (concatenated: [τ₁¹, τ₂¹, ..., τₙ₁¹, τ₁², τ₂², ..., τₙ₂², ...])
- `model::MultistateProcess`: The target multistate model with panel data
- `markov_rates::Dict{Tuple{Int,Int}, Float64}`: Transition rates from Markov surrogate
- `n_phases_vec::Vector{Int}`: Number of phases per observed state
- `state_to_phases::Vector{UnitRange{Int}}`: Observed state → phase indices
- `phase_to_state::Vector{Int}`: Phase index → observed state
- `n_expanded::Int`: Total number of expanded phases
- `tmat::Matrix{Int}`: Original transition matrix
- `neg::Bool=true`: Return negative log-likelihood (for minimization)
- `structure::Symbol=:sctp`: Coxian structure constraint

# Returns
- `Float64`: (Negative) marginal log-likelihood under the phase-type model

# Mathematical Details
The SCTP constraint ensures P(dest | leaving) is constant across phases:
  μⱼₛ = τⱼ × μ₁ₛ  for all destinations s

Non-exponential behavior comes from varying τⱼ values:
- τⱼ = 1 for all j → exponential (Markov) distribution
- τⱼ increasing → increasing marginal hazard
- τⱼ decreasing → decreasing marginal hazard

See also: [`_fit_phasetype_mle`](@ref), [`_build_coxian_from_rate`](@ref)
"""
function phasetype_marginal_loglik(theta::AbstractVector{T}, 
                                    model::MultistateProcess,
                                    n_phases_vec::Vector{Int},
                                    state_to_phases::Vector{UnitRange{Int}},
                                    phase_to_state::Vector{Int},
                                    n_expanded::Int,
                                    tmat::Matrix{Int},
                                    destinations::Dict{Int, Vector{Int}};
                                    neg::Bool = true) where T
    
    n_states = size(tmat, 1)
    is_absorbing = [all(tmat[s, :] .== 0) for s in 1:n_states]
    transient_states = findall(.!is_absorbing)
    
    # Build expanded Q matrix directly from theta (AD-compatible)
    expanded_Q = zeros(T, n_expanded, n_expanded)
    
    offset = 0
    for s in transient_states
        n_ph = n_phases_vec[s]
        dests = destinations[s]
        n_dests = length(dests)
        phases_s = state_to_phases[s]
        
        # Extract parameters for this state
        # Progression rates: (n_ph - 1) parameters
        if n_ph > 1
            prog_rates = theta[(offset + 1):(offset + n_ph - 1)]
            offset += n_ph - 1
        else
            prog_rates = T[]
        end
        
        # Absorption rates: n_ph × n_dests parameters (by phase, then by destination)
        abs_rates = reshape(theta[(offset + 1):(offset + n_ph * n_dests)], n_ph, n_dests)
        offset += n_ph * n_dests
        
        # Fill Q matrix for this state's phases
        for (j, phase_j) in enumerate(phases_s)
            total_out = zero(T)
            
            # Progression to next phase (if not last phase)
            if j < n_ph
                next_phase = phases_s[j + 1]
                expanded_Q[phase_j, next_phase] = prog_rates[j]
                total_out += prog_rates[j]
            end
            
            # Absorption to each destination (enter first phase of destination)
            for (d_idx, dest) in enumerate(dests)
                dest_first_phase = first(state_to_phases[dest])
                expanded_Q[phase_j, dest_first_phase] = abs_rates[j, d_idx]
                total_out += abs_rates[j, d_idx]
            end
            
            # Diagonal: negative sum of outgoing rates
            expanded_Q[phase_j, phase_j] = -total_out
        end
    end
    
    # Compute likelihood using forward algorithm on expanded space
    ll = _compute_phasetype_panel_loglik_ad(model, expanded_Q, state_to_phases, 
                                             phase_to_state, n_expanded)
    
    return neg ? -ll : ll
end

"""
    _compute_phasetype_panel_loglik_ad(model, expanded_Q, state_to_phases, 
                                        phase_to_state, n_expanded)

Compute the panel data log-likelihood under a phase-type expanded model.
AD-compatible version that works with ForwardDiff dual numbers.

Uses the forward algorithm on the expanded state space with matrix exponential.
"""
function _compute_phasetype_panel_loglik_ad(model::MultistateProcess,
                                             expanded_Q::AbstractMatrix{T},
                                             state_to_phases::Vector{UnitRange{Int}},
                                             phase_to_state::Vector{Int},
                                             n_expanded::Int) where T
    
    n_obs_states = size(model.tmat, 1)
    nsubj = length(model.subjectindices)
    
    # Build expanded emission matrix: phase → observed state
    expanded_emat = zeros(T, size(model.emat, 1), n_expanded)
    for i in axes(model.emat, 1)
        for obs_state in 1:n_obs_states
            if model.emat[i, obs_state] > 0
                for phase in state_to_phases[obs_state]
                    expanded_emat[i, phase] = T(model.emat[i, obs_state])
                end
            end
        end
    end
    
    # Compute log-likelihood via forward algorithm
    ll = zero(T)
    
    for subj in 1:nsubj
        subj_inds = model.subjectindices[subj]
        
        # Forward probabilities
        α_prev = zeros(T, n_expanded)
        α_curr = zeros(T, n_expanded)
        
        # Initialize: start in phase 1 of the initial state
        first_state = model.data.statefrom[subj_inds[1]]
        α_prev[first(state_to_phases[first_state])] = one(T)
        
        # Forward pass
        for i in subj_inds
            dt = model.data.tstop[i] - model.data.tstart[i]
            
            # Compute TPM: P(dt) = exp(Q * dt)
            # Use ExponentialUtilities.exp_generic for AD compatibility
            if dt > 0
                Qt = expanded_Q * dt
                tpm = ExponentialUtilities.exp_generic(Qt)
            else
                # dt = 0: identity matrix
                tpm = Matrix{T}(I, n_expanded, n_expanded)
            end
            
            # Forward step: α_curr[j] = Σᵢ α_prev[i] * P[i,j] * e[j]
            fill!(α_curr, zero(T))
            for j in 1:n_expanded
                emission_j = expanded_emat[i, j]
                if emission_j > 0
                    for k in 1:n_expanded
                        α_curr[j] += α_prev[k] * tpm[k, j] * emission_j
                    end
                end
            end
            
            # Normalize to prevent underflow
            scale = sum(α_curr)
            if scale <= 0
                # Return very negative log-likelihood to signal invalid model
                return T(-1e10)
            end
            ll += log(scale)
            α_curr ./= scale
            
            # Swap
            α_prev, α_curr = α_curr, α_prev
        end
    end
    
    return ll
end

"""
    _fit_phasetype_mle(model, markov_surrogate, n_phases_vec; 
                       verbose=true, maxiter=500, structure=:sctp)

Fit all phase-type rates via maximum likelihood estimation.

Optimizes ALL progression (λ) and absorption (μ) rates to maximize the marginal 
likelihood of panel data under the phase-type model. Uses Optimization.jl with 
Ipopt and automatic differentiation via ForwardDiff.

# Arguments
- `model::MultistateProcess`: The target multistate model
- `markov_surrogate::MarkovSurrogate`: Fitted Markov surrogate (used for initialization)
- `n_phases_vec::Vector{Int}`: Number of phases per observed state
- `verbose::Bool=true`: Print optimization progress
- `maxiter::Int=500`: Maximum optimization iterations
- `structure::Symbol=:sctp`: Constraint structure for identifiability:
  - `:sctp`: Stationary Conditional Transition Probability only (no ordering)
  - `:sctp_decreasing`: SCTP + eigenvalue ordering ν₁ ≥ ν₂ ≥ ... ≥ νₙ
  - `:sctp_increasing`: SCTP + eigenvalue ordering ν₁ ≤ ν₂ ≤ ... ≤ νₙ
  - `:unstructured`: No constraints (may find non-identifiable solutions)

# Constraints
- **SCTP** (multi-destination only): P(dest | leaving state) constant across phases
  Constraint: μⱼ,d₁/μ₁,d₁ = μⱼ,d₂/μ₁,d₂ for all phases j and destination pairs d₁, d₂
- **Eigenvalue ordering** (only with `:sctp_decreasing` or `:sctp_increasing`):
  `:decreasing` → ν₁ ≥ ν₂ ≥ ... ≥ νₙ
  `:increasing` → ν₁ ≤ ν₂ ≤ ... ≤ νₙ
  where νⱼ = λⱼ + Σμⱼ,d is total exit rate from phase j

# Returns
- `Dict{Symbol, Any}`: Fitted parameters containing:
  - `:progression`: Dict{Int, Vector{Float64}} - λ rates per state
  - `:absorption`: Dict{Int, Matrix{Float64}} - μ rates per state (phases × destinations)
  - `:theta`: Vector{Float64} - raw parameter vector

# Parameter Layout
For each transient state s with n_s phases and d_s destinations:
- Progression rates: λ₁, ..., λₙₛ₋₁ (n_s - 1 parameters)
- Absorption rates: μ₁,d₁, ..., μₙₛ,d₁, μ₁,d₂, ..., μₙₛ,dₛ (n_s × d_s parameters)

Total parameters for state s: (n_s - 1) + n_s × d_s

# Example
```julia
fitted = _fit_phasetype_mle(model, markov_surrogate, n_phases_vec; verbose=true)
prog_rates = fitted[:progression]  # Dict of progression rates
abs_rates = fitted[:absorption]    # Dict of absorption rate matrices
```
"""
function _fit_phasetype_mle(model::MultistateProcess,
                             markov_surrogate::MarkovSurrogate,
                             n_phases_vec::Vector{Int};
                             verbose::Bool = true,
                             structure::Symbol = :sctp,
                             maxiter::Int = 500)
    
    n_states = size(model.tmat, 1)
    is_absorbing = [all(model.tmat[s, :] .== 0) for s in 1:n_states]
    transient_states = findall(.!is_absorbing)
    
    # Build destinations for each transient state
    destinations = Dict{Int, Vector{Int}}()
    for s in transient_states
        destinations[s] = findall(model.tmat[s, :] .!= 0)
    end
    
    # Count total parameters
    # For each transient state s: (n_phases - 1) progression + n_phases × n_dests absorption
    n_params = 0
    param_layout = Dict{Int, NamedTuple{(:prog_start, :prog_end, :abs_start, :abs_end, :n_phases, :n_dests), 
                                         Tuple{Int, Int, Int, Int, Int, Int}}}()
    offset = 0
    for s in transient_states
        n_ph = n_phases_vec[s]
        n_dests = length(destinations[s])
        n_prog = max(0, n_ph - 1)
        n_abs = n_ph * n_dests
        
        param_layout[s] = (prog_start = offset + 1,
                           prog_end = offset + n_prog,
                           abs_start = offset + n_prog + 1,
                           abs_end = offset + n_prog + n_abs,
                           n_phases = n_ph,
                           n_dests = n_dests)
        offset += n_prog + n_abs
    end
    n_params = offset
    
    if n_params == 0
        if verbose
            println("  No parameters to optimize (all single-phase absorbing states).")
        end
        return Dict{Symbol, Any}(
            :progression => Dict{Int, Vector{Float64}}(),
            :absorption => Dict{Int, Matrix{Float64}}(),
            :theta => Float64[]
        )
    end
    
    # Build state mappings
    state_to_phases, phase_to_state, n_expanded = _build_state_mappings(n_states, n_phases_vec)
    
    # Extract transition rates from Markov surrogate for initialization
    markov_rates = Dict{Tuple{Int,Int}, Float64}()
    for h in markov_surrogate.hazards
        s, d = h.statefrom, h.stateto
        hazname = h.hazname
        rate = markov_surrogate.parameters.nested[hazname].baseline[Symbol("$(hazname)_rate")]
        markov_rates[(s, d)] = rate
    end
    
    # Initialize parameters from Markov rates
    theta0 = zeros(Float64, n_params)
    lb = fill(SURROGATE_PARAM_MIN, n_params)  # Lower bound: small positive
    ub = fill(100.0, n_params)  # Upper bound: prevent extreme rates
    
    for s in transient_states
        layout = param_layout[s]
        n_ph = layout.n_phases
        dests = destinations[s]
        n_dests = layout.n_dests
        
        # Total exit rate from Markov surrogate
        total_rate = sum(get(markov_rates, (s, d), 0.1) for d in dests)
        
        # Initialize progression rates: λⱼ ≈ n_phases × total_rate (Erlang-like)
        if n_ph > 1
            prog_rate = n_ph * total_rate
            theta0[layout.prog_start:layout.prog_end] .= prog_rate
        end
        
        # Initialize absorption rates: for SCTP/Coxian structure, concentrate absorption in later phases
        # This is consistent with the model fitting convention where μ₁ ≈ 0 and μₙ dominates
        for (d_idx, dest) in enumerate(dests)
            dest_rate = get(markov_rates, (s, dest), 0.1)
            for j in 1:n_ph
                # Parameter index: absorption rates are stored [phase1_dest1, phase2_dest1, ..., phase1_dest2, ...]
                idx = layout.abs_start + (j - 1) + (d_idx - 1) * n_ph
                if j == n_ph
                    # Last phase gets most of the absorption
                    theta0[idx] = dest_rate
                else
                    # Earlier phases get minimal absorption (Coxian-like structure)
                    theta0[idx] = 1e-5
                end
            end
        end
    end
    
    if verbose
        println("Fitting phase-type surrogate via MLE...")
        println("  Parameters: $n_params ($(length(transient_states)) transient states)")
        println("  Structure: $structure")
        for s in transient_states
            layout = param_layout[s]
            println("    State $s: $(layout.n_phases) phases, $(layout.n_dests) destinations")
        end
    end
    
    # Build constraints based on structure
    # We use Ipopt's constraint interface: cons_lb ≤ g(x) ≤ cons_ub
    cons_fns = Function[]      # Constraint functions
    cons_lb = Float64[]        # Lower bounds
    cons_ub = Float64[]        # Upper bounds
    
    # Determine if we need SCTP constraints (includes eigenvalue ordering for identifiability)
    use_sctp = structure === :sctp
    
    if use_sctp
        # Add SCTP constraints (for multi-destination states only)
        # SCTP: μⱼ,d₁ * μ₁,d₂ - μⱼ,d₂ * μ₁,d₁ = 0
        for s in transient_states
            layout = param_layout[s]
            n_ph = layout.n_phases
            n_dests = layout.n_dests
            dests = destinations[s]
            
            # SCTP only applies when n_dests ≥ 2 and n_phases ≥ 2
            if n_dests >= 2 && n_ph >= 2
                # Reference destination is first one
                for d_other in 2:n_dests
                    for j in 2:n_ph
                        # μⱼ,d₁ * μ₁,d_other - μⱼ,d_other * μ₁,d₁ = 0
                        # Parameter indices: abs[j, d] is at abs_start + (j-1) + (d-1)*n_ph
                        idx_1_1 = layout.abs_start + (1-1) + (1-1)*n_ph      # μ₁,d₁
                        idx_j_1 = layout.abs_start + (j-1) + (1-1)*n_ph      # μⱼ,d₁
                        idx_1_other = layout.abs_start + (1-1) + (d_other-1)*n_ph  # μ₁,d_other
                        idx_j_other = layout.abs_start + (j-1) + (d_other-1)*n_ph  # μⱼ,d_other
                        
                        # Constraint: μⱼ,d₁ * μ₁,d_other - μⱼ,d_other * μ₁,d₁ = 0
                        push!(cons_fns, theta -> theta[idx_j_1] * theta[idx_1_other] - theta[idx_j_other] * theta[idx_1_1])
                        push!(cons_lb, 0.0)
                        push!(cons_ub, 0.0)
                    end
                end
            end
        end
    end
    
    if use_sctp
        # Add eigenvalue ordering constraints (ν₁ ≤ ν₂ ≤ ... ≤ νₙ)
        # Eigenvalue ordering ν₁ ≤ ν₂ ≤ ... ≤ νₙ (constraint: νⱼ₊₁ - νⱼ ≥ 0)
        # where νⱼ = λⱼ + Σ_d μⱼ,d (total exit rate from phase j)
        for s in transient_states
            layout = param_layout[s]
            n_ph = layout.n_phases
            n_dests = layout.n_dests
            
            if n_ph >= 2
                for j in 1:(n_ph - 1)
                    function make_ordering_constraint(layout, j, n_ph, n_dests)
                        return function(theta)
                            # Compute νⱼ
                            nu_j = zero(eltype(theta))
                            if j < n_ph
                                # Add progression rate λⱼ
                                nu_j += theta[layout.prog_start + j - 1]
                            end
                            # Add absorption rates for phase j
                            for d in 1:n_dests
                                idx = layout.abs_start + (j-1) + (d-1)*n_ph
                                nu_j += theta[idx]
                            end
                            
                            # Compute νⱼ₊₁
                            j_next = j + 1
                            nu_j_next = zero(eltype(theta))
                            if j_next < n_ph
                                # Add progression rate λⱼ₊₁
                                nu_j_next += theta[layout.prog_start + j_next - 1]
                            end
                            # Add absorption rates for phase j+1
                            for d in 1:n_dests
                                idx = layout.abs_start + (j_next-1) + (d-1)*n_ph
                                nu_j_next += theta[idx]
                            end
                            
                            # Return constraint value: ν₁ ≤ ν₂ (increasing order)
                            return nu_j_next - nu_j  # Should be ≥ 0
                        end
                    end
                    
                    push!(cons_fns, make_ordering_constraint(layout, j, n_ph, n_dests))
                    push!(cons_lb, 0.0)   # constraint ≥ 0
                    push!(cons_ub, Inf)   # No upper bound
                end
            end
        end
    end
    
    if verbose && !isempty(cons_fns)
        constraint_desc = use_sctp ? "SCTP + eigenvalue ordering (ν₁ ≤ ν₂ ≤ ... ≤ νₙ)" : ""
        println("  Constraints: $(length(cons_fns)) ($constraint_desc)")
    end
    
    # Define objective function
    function objective(theta, p)
        return phasetype_marginal_loglik(theta, model, n_phases_vec,
                                          state_to_phases, phase_to_state, n_expanded,
                                          model.tmat, destinations; neg=true)
    end
    
    # Set up optimization with or without constraints
    if isempty(cons_fns)
        # Unconstrained optimization
        opt_func = OptimizationFunction(objective, Optimization.AutoForwardDiff())
        prob = OptimizationProblem(opt_func, theta0, nothing; lb=lb, ub=ub)
    else
        # Constrained optimization - must use inplace form for Ipopt
        n_cons = length(cons_fns)
        local_cons_fns = cons_fns  # Capture in local scope
        
        function constraints(res, theta, p)
            for (i, f) in enumerate(local_cons_fns)
                res[i] = f(theta)
            end
            return nothing
        end
        
        opt_func = OptimizationFunction(objective, Optimization.AutoForwardDiff();
                                         cons = constraints)
        prob = OptimizationProblem(opt_func, theta0, nothing; 
                                    lb=lb, ub=ub,
                                    lcons=Vector{Float64}(cons_lb), 
                                    ucons=Vector{Float64}(cons_ub))
    end
    
    # Solve with Ipopt
    sol = solve(prob, IpoptOptimizer();
                print_level = verbose ? 3 : 0,
                max_iter = maxiter,
                tol = 1e-6)
    
    # Extract fitted parameters
    theta_fit = sol.u
    
    progression = Dict{Int, Vector{Float64}}()
    absorption = Dict{Int, Matrix{Float64}}()
    
    for s in transient_states
        layout = param_layout[s]
        n_ph = layout.n_phases
        n_dests = layout.n_dests
        dests = destinations[s]
        
        # Progression rates
        if n_ph > 1
            progression[s] = theta_fit[layout.prog_start:layout.prog_end]
        else
            progression[s] = Float64[]
        end
        
        # Absorption rates: reshape to (n_phases, n_dests)
        abs_vec = theta_fit[layout.abs_start:layout.abs_end]
        absorption[s] = reshape(abs_vec, n_ph, n_dests)
    end
    
    if verbose
        println("  Phase-type MLE complete.")
        for s in transient_states
            dests = destinations[s]
            println("    State $s:")
            if length(progression[s]) > 0
                println("      Progression λ: $(round.(progression[s], digits=4))")
            end
            for (d_idx, dest) in enumerate(dests)
                println("      Absorption μ→$dest: $(round.(absorption[s][:, d_idx], digits=4))")
            end
        end
    end
    
    return Dict{Symbol, Any}(
        :progression => progression,
        :absorption => absorption,
        :theta => theta_fit,
        :destinations => destinations,
        :param_layout => param_layout
    )
end