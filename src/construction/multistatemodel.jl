# multistatemodel.jl - Main entry point for model construction
#
# This file orchestrates the model construction pipeline by including:
# - hazard_constructors.jl: User-facing Hazard() constructor
# - hazard_builders.jl: Internal hazard building infrastructure
# - spline_builder.jl: Spline hazard building and utilities  
# - model_assembly.jl: Model component assembly
#
# The multistatemodel() function below is the main entry point.

# Include component files in dependency order
include("hazard_constructors.jl")
include("hazard_builders.jl")
include("spline_builder.jl")
include("model_assembly.jl")

"""
    multistatemodel(hazards::HazardFunction...; data::DataFrame, surrogate = :none, ...)

Construct a full multistate model from a collection of hazards defined via `Hazard`
or `@hazard`. Hazards without covariates can omit a `@formula` entirely; the helper
will insert the intercept-only design automatically as described in `Hazard`'s docs.

# Keywords
- `data`: long-format `DataFrame` with at least `:subject`, `:statefrom`, `:stateto`, `:time`, `:obstype`.
- `constraints`: optional parameter constraints (see `make_constraints`). When provided at model creation,
  constraints are validated at parameter setting time and used as defaults in `fit()`.
- `initialize::Bool = true`: whether to initialize parameters at model creation. Uses `:crude` method
  for Markov and phase-type models (crude rates from data), `:surrogate` method for semi-Markov panel
  models (simulate paths, fit to exact data). If `false`, parameters remain at defaults (all rates = 1).
  Set to `false` if you want to manually set parameters before fitting.
- `surrogate::Symbol = :none`: surrogate model for importance sampling in MCEM.
  - `:none` (default): no surrogate created (for Markov models or exact data)
  - `:auto`: automatically select surrogate type via BIC comparison. Fits multiple
    candidate surrogates (Markov, phase-type with 2 and 3 phases) and selects the
    one with lowest BIC. If phase-type wins on BIC despite having more parameters,
    the observed sojourn times are non-exponential, and phase-type will be a better
    importance sampling proposal. See `select_surrogate` for manual control.
  - `:markov`: create a Markov surrogate (exponential approximation)
  - `:phasetype`: create a phase-type surrogate (better for non-exponential hazards)
- `fit_surrogate::Bool = true`: if `surrogate != :none`, fit the surrogate via MLE at model creation time.
  If `false`, surrogate parameters remain at default values and will be fitted when `fit()` is called.
  Setting to `true` (default) is recommended as it avoids redundant fitting during initialization.
- `surrogate_constraints`: optional constraints for surrogate optimization (only used if `fit_surrogate = true`).
- `surrogate_n_phases::Union{Int, Dict{Int,Int}, Symbol} = :heuristic`: number of phases for phase-type surrogate.
  - `:heuristic` (default): use smart defaults based on hazard types
  - `:auto`: select via BIC
  - `Int`: same number of phases for all states
  - `Dict{Int,Int}`: per-state phase counts
- `n_phases::Union{Nothing, Dict{Int,Int}} = nothing`: number of phases per state for phase-type hazards.
  Only states with `:pt` hazards should be specified. Example: `Dict(1 => 3, 2 => 2)` means state 1 has
  3 phases and state 2 has 2 phases. If a state has `:pt` hazards but is not in the dict, an error is thrown.
  If `n_phases[s] == 1`, the phase-type is coerced to exponential internally.
- `coxian_structure::Symbol = :sctp`: constraint structure for phase-type hazards.
  - `:sctp` (default): SCTP (Stationary Conditional Transition Probability) constraint with
    eigenvalue ordering ν₁ ≤ ν₂ ≤ ... ≤ νₙ. Provides identifiability for phase-type models.
  - `:unstructured`: no constraints on progression and absorption rates (not recommended,
    may have identifiability issues).
- `ordering_at::Union{Symbol, NamedTuple} = :mean`: where to enforce eigenvalue ordering constraints.
  - `:mean` (default): enforce ordering at the mean covariate values (computed from data).
    This provides a more interpretable reference point when covariates are present.
  - `:reference`: enforce νⱼ ≥ νⱼ₊₁ at reference (x=0). Produces linear constraints.
  - `:median`: enforce ordering at the median covariate values (computed from data).
  - `NamedTuple`: enforce at explicit covariate values, e.g., `(age=50.0, treatment=0.5)`.
  
  When `ordering_at` is not `:reference`, nonlinear constraints are generated (AD-compatible for Ipopt).
  With `:homogeneous` covariate constraints (C1), the exp(β'x̄) factors cancel, simplifying back to linear.
- `SubjectWeights`: optional per-subject weights (length = number of subjects). Mutually exclusive with `ObservationWeights`.
- `ObservationWeights`: optional per-observation weights (length = number of rows in data). Mutually exclusive with `SubjectWeights`.
- `CensoringPatterns`: optional matrix describing which states are compatible with each censoring code. Values in [0,1].
- `EmissionMatrix`: optional matrix of emission probabilities (nrow(data) × nstates). Values are P(observation|state).
- `verbose`: print additional validation output.

# Examples
```julia
# Markov model (no surrogate needed)
model = multistatemodel(h12, h21; data = df)

# Model with parameter constraints
cons = make_constraints(
    cons = [:(log_λ_12 == log_λ_21)],  # Equal rates
    lcons = [0.0],
    ucons = [0.0]
)
model = multistatemodel(h12, h21; data = df, constraints = cons)

# Semi-Markov model - auto-select surrogate type (phase-type for non-exponential)
model = multistatemodel(h12_wei, h21_wei; data = df, surrogate = :auto)

# Semi-Markov model - explicitly use Markov surrogate (exponential approximation)
model = multistatemodel(h12_wei, h21_wei; data = df, surrogate = :markov)

# Semi-Markov model - explicitly use phase-type surrogate
model = multistatemodel(h12_wei, h21_wei; data = df, surrogate = :phasetype)

# Semi-Markov model - defer surrogate fitting to fit() call
model = multistatemodel(h12_wei, h21_wei; data = df, surrogate = :auto, fit_surrogate = false)

# Model without automatic initialization (set parameters manually)
model = multistatemodel(h12, h21; data = df, initialize = false)
set_parameters!(model, (h12 = [log(0.5)], h21 = [log(0.3)]))

# Phase-type model with 3 phases on state 1
h12 = Hazard(@formula(0 ~ 1 + x), "pt", 1, 2)
h13 = Hazard(@formula(0 ~ 1 + x), "pt", 1, 3)
model = multistatemodel(h12, h13; data = df, n_phases = Dict(1 => 3))

# Phase-type model with SCTP constraints
model = multistatemodel(h12, h13; data = df, n_phases = Dict(1 => 3), coxian_structure = :sctp)

# Phase-type model with eigenvalue ordering at mean covariate values
model = multistatemodel(h12, h13; data = df, n_phases = Dict(1 => 3), 
                        coxian_structure = :sctp, ordering_at = :mean)

# Phase-type model with eigenvalue ordering at explicit covariate values
model = multistatemodel(h12, h13; data = df, n_phases = Dict(1 => 3),
                        ordering_at = (x = 0.5,))
```
"""
function multistatemodel(hazards::HazardFunction...; 
                        data::DataFrame, 
                        constraints = nothing,
                        initialize::Bool = true,
                        surrogate::Symbol = :none,
                        fit_surrogate::Bool = true,
                        surrogate_constraints = nothing,
                        surrogate_n_phases::Union{Int, Dict{Int,Int}, Symbol} = :heuristic,
                        n_phases::Union{Nothing, Dict{Int,Int}} = nothing,
                        coxian_structure::Symbol = :sctp,
                        ordering_at::Union{Symbol, NamedTuple} = :mean,
                        SubjectWeights::Union{Nothing,Vector{Float64}} = nothing, 
                        ObservationWeights::Union{Nothing,Vector{Float64}} = nothing,
                        CensoringPatterns::Union{Nothing,Matrix{<:Real}} = nothing, 
                        EmissionMatrix::Union{Nothing,Matrix{Float64}} = nothing,
                        verbose = false) 

    # Validate surrogate option
    if surrogate ∉ (:none, :auto, :markov, :phasetype)
        throw(ArgumentError("surrogate must be :none, :auto, :markov, or :phasetype, got :$surrogate"))
    end
    
    # Validate coxian_structure
    if coxian_structure ∉ (:unstructured, :sctp)
        throw(ArgumentError("coxian_structure must be :unstructured or :sctp, got :$coxian_structure"))
    end
    
    # Validate ordering_at
    if ordering_at isa Symbol && ordering_at ∉ (:reference, :mean, :median)
        throw(ArgumentError("ordering_at must be :reference, :mean, :median, or a NamedTuple, got :$ordering_at"))
    end
    
    # Validate inputs
    isempty(hazards) && throw(ArgumentError("At least one hazard must be provided"))

    # Check for phase-type hazards and route accordingly
    if any(h -> h isa PhaseTypeHazard, hazards)
        return _build_phasetype_model_from_hazards(hazards;
            data = data,
            constraints = constraints,
            initialize = initialize,
            n_phases = n_phases,
            coxian_structure = coxian_structure,
            ordering_at = ordering_at,
            SubjectWeights = SubjectWeights,
            ObservationWeights = ObservationWeights,
            CensoringPatterns = CensoringPatterns,
            EmissionMatrix = EmissionMatrix,
            verbose = verbose
        )
    end
    
    # Validate n_phases not specified for non-phase-type models
    if !isnothing(n_phases) && !isempty(n_phases)
        throw(ArgumentError("n_phases specified but no :pt hazards found. n_phases only applies to phase-type models."))
    end

    # catch the model call (includes constraints for use by fit())
    modelcall = (hazards = hazards, data = data, constraints = constraints, SubjectWeights = SubjectWeights, ObservationWeights = ObservationWeights, CensoringPatterns = CensoringPatterns, EmissionMatrix = EmissionMatrix)

    # Expand smooth term basis columns into data (make a copy to avoid mutating user data)
    data = copy(data)
    expand_all_smooth_terms!(data, hazards)

    # get indices for each subject in the dataset
    subjinds, nsubj = get_subjinds(data)

    # enumerate the hazards and reorder 
    hazinfo = enumerate_hazards(hazards...)

    # reorder hazards and pop the order column in hazinfo
    hazards = hazards[hazinfo.order]
    select!(hazinfo, Not(:order))

    # compile matrix enumerating instantaneous state transitions
    tmat = create_tmat(hazinfo)
    
    # Validate that at least one absorbing state exists (L10_P2)
    # An absorbing state has no outgoing transitions (all zeros in its row)
    n_states = size(tmat, 1)
    has_absorbing = false
    for s in 1:n_states
        if all(tmat[s, :] .== 0)
            has_absorbing = true
            break
        end
    end
    if !has_absorbing
        @warn "Model has no absorbing states. All states have at least one outgoing transition. " *
              "This may cause simulation to run indefinitely or likelihood computation issues. " *
              "Consider adding at least one absorbing state (a state with no outgoing transitions)." maxlog=1
    end

    # Handle weight exclusivity and defaults
    SubjectWeights, ObservationWeights = check_weight_exclusivity(SubjectWeights, ObservationWeights, nsubj)
    
    # Prepare patterns
    CensoringPatterns = _prepare_censoring_patterns(CensoringPatterns, size(tmat, 1))

    _validate_inputs!(data, tmat, CensoringPatterns, SubjectWeights, ObservationWeights; verbose = verbose)
    emat = build_emat(data, CensoringPatterns, EmissionMatrix, tmat)

    _hazards, parameters, hazkeys = build_hazards(hazards...; data = data, surrogate = false)
    _totalhazards = build_totalhazards(_hazards, tmat)

    # Resolve surrogate option
    # - :none → no surrogate
    # - :markov/:phasetype → explicit surrogate type (built but not fitted here)
    # - :auto → defer BIC-based selection to fit_surrogate step
    resolved_surrogate = surrogate  # Preserve :auto for later BIC-based selection
    
    # Build initial surrogate structure if requested (initially unfitted)
    # For :auto, we build a Markov surrogate as placeholder - select_surrogate() will replace it
    # The surrogate field can hold either MarkovSurrogate or PhaseTypeSurrogate
    model_surrogate::Union{Nothing, AbstractSurrogate} = nothing
    if resolved_surrogate in (:markov, :phasetype, :auto)
        # Always build the Markov surrogate first (needed as placeholder or for Markov case)
        # The MarkovSurrogate stores the exponential hazards and covariate coefficients
        surrogate_haz, surrogate_pars_ph, _ = build_hazards(hazards...; data = data, surrogate = true)
        markov_surrogate = MarkovSurrogate(surrogate_haz, surrogate_pars_ph; fitted=false)
        model_surrogate = markov_surrogate  # Default to Markov
        
        # For :phasetype and :auto, the actual surrogate is fitted later via initialize_surrogate!
        # which will call select_surrogate() for :auto
    end

    components = (
        data = data,
        parameters = parameters,
        hazards = _hazards,
        totalhazards = _totalhazards,
        tmat = tmat,
        emat = emat,
        hazkeys = hazkeys,
        subjinds = subjinds,
        SubjectWeights = SubjectWeights,
        ObservationWeights = ObservationWeights,
        CensoringPatterns = CensoringPatterns,
    )

    mode = _observation_mode(data)
    process = _process_class(_hazards)

    model = _assemble_model(mode, process, components, model_surrogate, modelcall)
    
    # NOTE ON CONSTRUCTION ATOMICITY (C6_P2):
    # The following initialization steps are NOT transactional. If initialize_parameters!
    # succeeds but initialize_surrogate! fails, the model will be returned with initialized
    # parameters but no surrogate. This is by design - a partially initialized model is still
    # usable for some operations (e.g., simulation with manual parameter setting).
    # 
    # If strict atomicity is required, wrap the multistatemodel() call in try-catch and
    # discard the model on any failure.
    
    # Initialize parameters (default: true)
    # Uses :auto method which selects :crude for Markov/phase-type, :surrogate for semi-Markov
    if initialize
        try
            initialize_parameters!(model; constraints = constraints)
        catch e
            @warn "Parameter initialization failed. Model returned with default parameters." exception=(e, catch_backtrace())
            rethrow()  # Still throw, but user gets warning first
        end
    end
    
    # Fit surrogate at model creation time (default: true)
    if fit_surrogate && resolved_surrogate in (:markov, :phasetype, :auto)
        if verbose
            println("Fitting surrogate at model creation time...")
        end
        # Use initialize_surrogate! to fit the appropriate surrogate type
        # For :auto, this will use BIC-based selection via select_surrogate()
        try
            initialize_surrogate!(model; 
                type = resolved_surrogate,
                method = :mle,
                n_phases = surrogate_n_phases,
                surrogate_constraints = surrogate_constraints,
                verbose = verbose)
        catch e
            @warn "Surrogate initialization failed. Model returned with parameters initialized but no fitted surrogate. " *
                  "You may need to call initialize_surrogate!() manually before fitting." exception=(e, catch_backtrace())
            rethrow()  # Still throw, but user gets warning first
        end
    end
    
    return model
end
