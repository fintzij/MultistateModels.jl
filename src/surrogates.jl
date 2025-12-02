

"""
    make_surrogate_model(model)

Create a Markov surrogate model from a multistate model.

If the model already has a `markovsurrogate`, uses its hazards and parameters.
Otherwise, builds surrogate hazards from scratch using the model's hazard specifications.

# Arguments
- `model`: multistate model object

# Returns
- `MultistateMarkovModel` or `MultistateMarkovModelCensored` suitable for fitting as a Markov model
"""
function make_surrogate_model(model::Union{MultistateModel, MultistateMarkovModel, MultistateSemiMarkovModel})
    if isnothing(model.markovsurrogate)
        # Build surrogate hazards from scratch
        surrogate_haz, surrogate_pars, _ = build_hazards(model.modelcall.hazards...; data = model.data, surrogate = true)
        markov_surrogate = MarkovSurrogate(surrogate_haz, surrogate_pars)
    else
        markov_surrogate = model.markovsurrogate
    end
    
    MultistateModels.MultistateMarkovModel(
        model.data,
        markov_surrogate.parameters,  # Use surrogate's parameters
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
        model.modelcall)
end

"""
    make_surrogate_model(model::Union{MultistateMarkovModelCensored, MultistateSemiMarkovModelCensored})

Create a Markov surrogate model with censored states.

# Arguments
- `model`: multistate model object with censored observations

# Returns
- `MultistateMarkovModelCensored` suitable for fitting as a Markov model
"""
function make_surrogate_model(model::Union{MultistateMarkovModelCensored,MultistateSemiMarkovModelCensored})
    if isnothing(model.markovsurrogate)
        # Build surrogate hazards from scratch
        surrogate_haz, surrogate_pars, _ = build_hazards(model.modelcall.hazards...; data = model.data, surrogate = true)
        markov_surrogate = MarkovSurrogate(surrogate_haz, surrogate_pars)
    else
        markov_surrogate = model.markovsurrogate
    end
    
    MultistateModels.MultistateMarkovModelCensored(
        model.data,
        markov_surrogate.parameters,  # Use surrogate's parameters
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
        model.modelcall)
end


"""
fit_surrogate(model::MultistateSemiMarkovModelCensored)

Fit a Markov surrogate model.

# Arguments

- model: multistate model object
"""
function fit_surrogate(model; surrogate_parameters = nothing, surrogate_constraints = nothing, crude_inits = true, verbose = true)

    # initialize the surrogate
    surrogate_model = make_surrogate_model(model)

    # set parameters to supplied or crude inits
    if !isnothing(surrogate_parameters) 
        set_parameters!(surrogate_model, surrogate_parameters)
    elseif crude_inits
        set_crude_init!(surrogate_model)
    end

    # generate the constraint function and test at initial values
    if !isnothing(surrogate_constraints)
        # create the function
        consfun_surrogate = parse_constraints(surrogate_constraints.cons, surrogate_model.hazards; consfun_name = :consfun_surrogate)

        # test the initial values
        # Phase 3: Use ParameterHandling.jl flat parameters for constraint check
        initcons = consfun_surrogate(zeros(length(surrogate_constraints.cons)), get_parameters_flat(surrogate_model), nothing)
        
        badcons = findall(initcons .< surrogate_constraints.lcons .|| initcons .> surrogate_constraints.ucons)

        if length(badcons) > 0
            error("Constraints $badcons are violated at the initial parameter values for the Markov surrogate. Consider manually setting surrogate parameters.")
        end
    end

    # optimize the Markov surrogate
    if verbose
        println("Obtaining the MLE for the Markov surrogate model ...\n")
    end
    
    
    surrogate_fitted = fit(surrogate_model; constraints = surrogate_constraints, compute_vcov = false)

    return surrogate_fitted
end


"""
    set_surrogate!(model; surrogate_parameters=nothing, surrogate_constraints=nothing, 
                   crude_inits=true, optimize=true, verbose=true)

Build and optionally fit a Markov surrogate for a multistate model, populating 
the model's `markovsurrogate` field in-place.

This function is useful when you want to add a surrogate to an existing model
without recreating it, or when you want to control surrogate fitting separately
from model fitting.

# Arguments
- `model`: A mutable multistate model (MultistateModel, MultistateSemiMarkovModel, etc.)

# Keywords
- `surrogate_parameters=nothing`: Optional parameters to set on the surrogate. 
   If provided, these are used directly instead of fitting.
- `surrogate_constraints=nothing`: Optional constraints for surrogate fitting.
- `crude_inits::Bool=true`: Use crude initializations based on data if fitting.
- `optimize::Bool=true`: If true, fit the surrogate to the data. If false, 
   just build the surrogate structure with initial/provided parameters.
- `verbose::Bool=true`: Print progress information.

# Returns
- The modified model (also modifies in-place).

# Examples
```julia
# Create a semi-Markov model without surrogate
model = multistatemodel(h12, h21; data=data)  # surrogate=:none by default

# Later, add and fit a surrogate
set_surrogate!(model)

# Or just build the structure without fitting
set_surrogate!(model; optimize=false)

# Or use specific parameters
set_surrogate!(model; surrogate_parameters=my_params, optimize=false)
```

See also: [`fit_surrogate`](@ref), [`multistatemodel`](@ref)
"""
function set_surrogate!(model::MultistateProcess; 
                        surrogate_parameters = nothing, 
                        surrogate_constraints = nothing, 
                        crude_inits = true,
                        optimize = true,
                        verbose = true)
    
    if optimize
        # Fit the surrogate using existing fit_surrogate function
        surrogate_fitted = fit_surrogate(model; 
                                         surrogate_parameters = surrogate_parameters, 
                                         surrogate_constraints = surrogate_constraints, 
                                         crude_inits = crude_inits, 
                                         verbose = verbose)
        
        # Create MarkovSurrogate from fitted model
        markov_surrogate = MarkovSurrogate(surrogate_fitted.hazards, surrogate_fitted.parameters)
        
    else
        # Just build the surrogate structure without fitting
        surrogate_haz, surrogate_pars, _ = build_hazards(model.modelcall.hazards...; 
                                                          data = model.data, 
                                                          surrogate = true)
        
        if !isnothing(surrogate_parameters)
            # Use provided parameters - need to rebuild with them
            # surrogate_pars is a NamedTuple, we need to set the flat values
            unflatten = surrogate_pars.unflatten
            new_transformed = unflatten(surrogate_parameters)
            new_natural = NamedTuple{keys(new_transformed)}(
                Tuple(ParameterHandling.value(v) for v in values(new_transformed))
            )
            surrogate_pars = (
                flat = surrogate_parameters,
                transformed = new_transformed,
                natural = new_natural,
                unflatten = unflatten
            )
        elseif crude_inits
            # Build a temporary model to use set_crude_init!
            temp_model = make_surrogate_model(model)
            set_crude_init!(temp_model)
            surrogate_pars = temp_model.parameters
            surrogate_haz = temp_model.hazards
        end
        
        markov_surrogate = MarkovSurrogate(surrogate_haz, surrogate_pars)
    end
    
    # Populate the model's markovsurrogate field
    model.markovsurrogate = markov_surrogate
    
    if verbose && !optimize
        println("Markov surrogate built (not fitted).\n")
    end
    
    return model
end


"""
    fit_phasetype_surrogate(model, markov_surrogate; config, verbose)

Build a phase-type surrogate from a fitted Markov surrogate.

The phase-type surrogate expands each transient state into multiple phases
to better approximate non-exponential sojourn times. This function:
1. Takes the fitted Markov surrogate parameters
2. Builds phase-type distributions for each transient state
3. Constructs the expanded intensity matrix Q
4. Returns a PhaseTypeSurrogate ready for FFBS sampling

# Arguments
- `model`: The semi-Markov model being fitted
- `markov_surrogate::MarkovSurrogate`: Fitted Markov surrogate
- `config::ProposalConfig`: Proposal configuration with n_phases specification
- `verbose::Bool`: Print progress information

# Returns
- `PhaseTypeSurrogate`: Expanded surrogate for importance sampling
"""
function fit_phasetype_surrogate(model, markov_surrogate::MarkovSurrogate; 
                                  config::ProposalConfig, verbose::Bool=true)
    
    n_states = size(model.tmat, 1)
    is_absorbing = [all(model.tmat[s, :] .== 0) for s in 1:n_states]
    transient_states = findall(.!is_absorbing)
    
    # Determine number of phases per state
    n_phases = config.n_phases
    if n_phases === :auto
        # BIC-based selection
        n_phases_vec = _select_n_phases_bic(model.tmat, model.data; 
                                            max_phases=config.max_phases, verbose=verbose)
    elseif n_phases === :heuristic
        # Heuristic based on hazard types
        n_phases_vec = _compute_default_n_phases(model.tmat, model.hazards)
    elseif n_phases isa Int
        # Same for all transient states
        n_phases_vec = [is_absorbing[s] ? 1 : n_phases for s in 1:n_states]
    else
        # Per-state specification (Vector{Int})
        n_phases_vec = zeros(Int, n_states)
        transient_idx = 1
        for s in 1:n_states
            if is_absorbing[s]
                n_phases_vec[s] = 1
            else
                n_phases_vec[s] = n_phases[transient_idx]
                transient_idx += 1
            end
        end
    end
    
    if verbose
        println("Phase-type surrogate configuration:")
        println("  n_phases per state: $n_phases_vec")
    end
    
    # Build state mappings
    state_to_phases, phase_to_state, n_expanded = _build_state_mappings(n_states, n_phases_vec)
    
    # Extract transition rates from Markov surrogate
    # Build mapping from (from_state, to_state) -> rate
    transition_rates = Dict{Tuple{Int,Int}, Float64}()
    
    # Get parameters as tuple for integer indexing
    surrogate_pars = values(markov_surrogate.parameters.natural)
    
    for (haz_idx, h) in enumerate(markov_surrogate.hazards)
        s, d = h.statefrom, h.stateto
        # Get the rate parameter on natural scale
        # parameters.natural already contains the actual rates (not log-rates)
        rate = surrogate_pars[haz_idx][1]
        transition_rates[(s, d)] = rate
    end
    
    # Build phase-type distributions for each transient state
    phasetype_dists = Dict{Int, PhaseTypeDistribution}()
    for s in transient_states
        n_ph = n_phases_vec[s]
        
        # Compute total exit rate from state s
        total_rate = sum(get(transition_rates, (s, d), 0.0) for d in 1:n_states if d != s)
        
        if total_rate <= 0
            # No outgoing transitions, use default
            phasetype_dists[s] = _build_default_phasetype(n_ph)
        else
            # Build Coxian PH with appropriate rates
            # For now, use simple equal-rate phases that match total sojourn rate
            phasetype_dists[s] = _build_coxian_from_rate(n_ph, total_rate)
        end
    end
    
    # Build the expanded Q matrix
    expanded_Q = build_expanded_Q(model.tmat, n_phases_vec, state_to_phases, 
                                  phase_to_state, phasetype_dists, n_expanded;
                                  transition_rates=transition_rates)
    
    # Create PhaseTypeConfig for storage
    ph_config = PhaseTypeConfig(n_phases=n_phases_vec, max_phases=config.max_phases)
    
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
    _build_coxian_from_rate(n_phases::Int, total_rate::Float64) -> PhaseTypeDistribution

Build a Coxian phase-type distribution that has approximately the same mean
sojourn time as an exponential with the given rate.

For n phases with equal rates λ, the mean is n/λ.
For exponential with rate r, mean is 1/r.
So we set λ = n * r to match means.
"""
function _build_coxian_from_rate(n_phases::Int, total_rate::Float64)
    if n_phases == 1
        # Single phase = exponential
        S = reshape([-total_rate], 1, 1)
        initial = [1.0]
        absorption = [total_rate]
        return PhaseTypeDistribution(1, S, initial, absorption)
    end
    
    # Scale rate so mean matches: phase_rate = n_phases * total_rate
    phase_rate = n_phases * total_rate
    
    # Build Coxian: equal rates, some probability of early absorption
    # S[i,i] = -(phase_rate), S[i,i+1] = phase_rate * (1 - p_absorb)
    # absorption[i] = phase_rate * p_absorb
    # Use p_absorb = 1/n_phases to balance phase progression and absorption
    
    p_absorb = 1.0 / n_phases
    progression_rate = phase_rate * (1 - p_absorb)
    absorption_rate = phase_rate * p_absorb
    
    S = zeros(Float64, n_phases, n_phases)
    absorption = zeros(Float64, n_phases)
    
    for i in 1:n_phases
        if i < n_phases
            S[i, i+1] = progression_rate
        end
        absorption[i] = absorption_rate
        S[i, i] = -(absorption[i] + (i < n_phases ? progression_rate : 0.0))
    end
    
    # Last phase can only absorb
    S[n_phases, n_phases] = -absorption_rate
    absorption[n_phases] = absorption_rate
    
    initial = zeros(Float64, n_phases)
    initial[1] = 1.0  # Start in first phase
    
    return PhaseTypeDistribution(n_phases, S, initial, absorption)
end

