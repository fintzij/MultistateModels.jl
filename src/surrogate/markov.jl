# =============================================================================
# Markov Surrogate Model Construction
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
        markov_surrogate,
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

See also: [`set_surrogate!`](@ref), [`MarkovSurrogate`](@ref), [`PhaseTypeSurrogate`](@ref)
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
    _fit_phasetype_surrogate(model; method, n_phases, surrogate_parameters, surrogate_constraints, verbose)

Internal function to fit a phase-type surrogate.

- `:mle` method: First fits Markov surrogate via MLE, then builds phase-type
- `:heuristic` method: Uses crude rates divided by n_phases
"""
function _fit_phasetype_surrogate(model;
    method::Symbol = :mle,
    n_phases::Union{Int, Dict{Int,Int}, Symbol} = 2,
    surrogate_parameters = nothing,
    surrogate_constraints = nothing,
    verbose = true)
    
    # First, get or fit the Markov surrogate
    markov_surrogate = _fit_markov_surrogate(model;
        method = method,
        surrogate_parameters = surrogate_parameters,
        surrogate_constraints = surrogate_constraints,
        verbose = verbose)
    
    # Build phase-type from Markov surrogate
    config = ProposalConfig(type = :phasetype, n_phases = n_phases)
    
    phasetype_surrogate = _build_phasetype_from_markov(model, markov_surrogate; 
                                                        config = config, 
                                                        verbose = verbose)
    
    return phasetype_surrogate
end

"""
    _build_phasetype_from_markov(model, markov_surrogate; config, verbose)

Build a phase-type surrogate from a fitted/initialized Markov surrogate.
This is the core phase-type construction logic.
"""
function _build_phasetype_from_markov(model, markov_surrogate::MarkovSurrogate;
                                       config::ProposalConfig, verbose::Bool = true)
    
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
    end
    
    # Build state mappings
    state_to_phases, phase_to_state, n_expanded = _build_state_mappings(n_states, n_phases_vec)
    
    # Extract transition rates from Markov surrogate (use nested parameters)
    transition_rates = Dict{Tuple{Int,Int}, Float64}()
    
    for (haz_idx, h) in enumerate(markov_surrogate.hazards)
        s, d = h.statefrom, h.stateto
        # Get rate from nested structure
        hazname = h.hazname
        rate = markov_surrogate.parameters.nested[hazname].baseline[Symbol("$(hazname)_rate")]
        transition_rates[(s, d)] = rate
    end
    
    # Build phase-type distributions for each transient state
    phasetype_dists = Dict{Int, PhaseTypeDistribution}()
    structure = config.structure
    for s in transient_states
        n_ph = n_phases_vec[s]
        total_rate = sum(get(transition_rates, (s, d), 0.0) for d in 1:n_states if d != s)
        
        if total_rate <= 0
            phasetype_dists[s] = _build_default_phasetype(n_ph)
        else
            phasetype_dists[s] = _build_coxian_from_rate(n_ph, total_rate; structure = structure)
        end
    end
    
    # Build the expanded Q matrix
    expanded_Q = build_expanded_Q(model.tmat, n_phases_vec, state_to_phases, 
                                  phase_to_state, phasetype_dists, n_expanded;
                                  transition_rates = transition_rates)
    
    # Create PhaseTypeConfig for storage (convert n_phases_vec back to Dict)
    # Only include states with n_phases > 1 (absorbing states always have 1)
    n_phases_dict = Dict{Int,Int}(s => n_phases_vec[s] for s in 1:n_states if n_phases_vec[s] > 1)
    ph_config = PhaseTypeConfig(n_phases = n_phases_dict, structure = structure,
                                max_phases = config.max_phases)
    
    surrogate = PhaseTypeSurrogate(
        phasetype_dists,
        n_states,
        n_expanded,
        state_to_phases,
        phase_to_state,
        expanded_Q,
        ph_config
    )
    
    if verbose
        println("  Total expanded states: $n_expanded")
        println("  Phase-type surrogate built successfully.\n")
    end
    
    return surrogate
end


"""
    is_surrogate_fitted(model::MultistateProcess) -> Bool

Check if the model's Markov surrogate has been fitted.

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

See also: [`set_surrogate!`](@ref), [`MarkovSurrogate`](@ref)
"""
function is_surrogate_fitted(model::MultistateProcess)
    isnothing(model.markovsurrogate) && return false
    return model.markovsurrogate.fitted
end


"""
    set_surrogate!(model; type=:markov, method=:mle, ...)

Build and fit a Markov or phase-type surrogate for a multistate model,
populating the model's `markovsurrogate` field in-place.

This function always fits the surrogate (via MLE or heuristic), marking it
as `fitted=true`. For models created with `surrogate=:markov` and `fit_surrogate=true`
(the default), the surrogate is already fitted and calling this function will refit it.

# Arguments
- `model`: A mutable multistate model (MultistateModel, MultistateModel, etc.)

# Keywords
- `type::Symbol = :markov`: Surrogate type (:markov or :phasetype)
- `method::Symbol = :mle`: Fitting method (:mle or :heuristic)
- `n_phases = 2`: For phase-type: number of phases
- `surrogate_parameters = nothing`: Optional fixed parameters (skips fitting)
- `surrogate_constraints = nothing`: Optional constraints for MLE fitting
- `verbose::Bool = true`: Print progress information

# Returns
- The modified model (also modifies in-place)

# Examples
```julia
# Create a semi-Markov model without surrogate
model = multistatemodel(h12, h21; data=data)

# Add and fit a Markov surrogate via MLE
set_surrogate!(model)

# Use heuristic (faster, no optimization)
set_surrogate!(model; method=:heuristic)

# Use phase-type surrogate
set_surrogate!(model; type=:phasetype, n_phases=3)

# Check if surrogate is fitted
is_surrogate_fitted(model)  # true
```

See also: [`is_surrogate_fitted`](@ref), [`fit_surrogate`](@ref), [`multistatemodel`](@ref)
"""
function set_surrogate!(model::MultistateProcess; 
    type::Symbol = :markov,
    method::Symbol = :mle,
    n_phases::Union{Int, Dict{Int,Int}, Symbol} = 2,
    surrogate_parameters = nothing, 
    surrogate_constraints = nothing, 
    verbose = true)
    
    _validate_surrogate_inputs(type, method)
    
    if type === :markov
        markov_surrogate = _fit_markov_surrogate(model;
            method = method,
            surrogate_parameters = surrogate_parameters,
            surrogate_constraints = surrogate_constraints,
            verbose = verbose)
        model.markovsurrogate = markov_surrogate
    else
        # Phase-type: also need to set Markov surrogate for infrastructure
        markov_surrogate = _fit_markov_surrogate(model;
            method = method,
            surrogate_parameters = surrogate_parameters,
            surrogate_constraints = surrogate_constraints,
            verbose = verbose)
        model.markovsurrogate = markov_surrogate
        # Note: Phase-type surrogate is built in fit() when needed
    end
    
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
- `structure::Symbol`: Coxian structure constraint (default: `:unstructured`)
  - `:unstructured`: All rates are independent free parameters (for initialization, uses
    uniform rates across all parameters)
  - `:sctp`: SCTP (Stationary Conditional Transition Probability) constraint - uses
    proportional exit rates that maintain constant P(dest | leaving) across phases
  - `Function`: Custom constraint function `f(n_phases, total_rate) -> Q` returning the 
    (n_phases + 1) × (n_phases + 1) intensity matrix

# Returns
- `PhaseTypeDistribution`: Coxian distribution with the specified structure
"""
function _build_coxian_from_rate(n_phases::Int, total_rate::Float64; 
                                  structure::Union{Symbol, Function} = :unstructured)
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
    
    # Build Q matrix: (n_phases + 1) × (n_phases + 1)
    Q = zeros(Float64, n_phases + 1, n_phases + 1)
    
    if structure == :sctp
        # SCTP constraint: exit rates proportional to progression rates
        # This ensures P(dest | leaving state) is constant across phases
        # Use aᵢ = c × rᵢ for i < n (proportionality)
        #
        # For mean = 1/total_rate with proportionality constant c = 0.5:
        c = 0.5  # proportionality constant
        progression_rate = total_rate / c  # so that c × r = total_rate
        absorption_rate_intermediate = c * progression_rate  # = total_rate for phases 1 to n-1
        absorption_rate_final = total_rate  # for last phase
        
        for i in 1:n_phases
            if i < n_phases
                Q[i, i+1] = progression_rate
                Q[i, n_phases+1] = absorption_rate_intermediate
                Q[i, i] = -(progression_rate + absorption_rate_intermediate)
            else
                Q[i, n_phases+1] = absorption_rate_final
                Q[i, i] = -absorption_rate_final
            end
        end
        
    elseif structure == :unstructured
        # Unstructured: all rates are free parameters
        # For initialization, use uniform rates: total_rate / (2n-1) for all
        n_params = 2 * n_phases - 1
        uniform_rate = total_rate * n_phases / n_params  # Adjust to get mean ≈ 1/total_rate
        
        for i in 1:n_phases
            if i < n_phases
                Q[i, i+1] = uniform_rate
                Q[i, n_phases+1] = uniform_rate
                Q[i, i] = -2 * uniform_rate
            else
                # Last phase: only absorption
                Q[i, n_phases+1] = uniform_rate
                Q[i, i] = -uniform_rate
            end
        end
    else
        throw(ArgumentError("Unknown structure: $structure. Use :unstructured or :sctp"))
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
        surrogate,
        model.modelcall,
        nothing  # No phasetype_expansion for surrogate
    )
    
    # Build the bookkeeping structure for MPanelData
    books = build_tpm_mapping(surrogate_model.data)
    
    # Create MPanelData for the Markov likelihood
    data = MPanelData(surrogate_model, books)
    
    # Compute log-likelihood using the surrogate's flat parameters
    ll = loglik_markov(surrogate.parameters.flat, data; neg = false)
    
    return ll
end