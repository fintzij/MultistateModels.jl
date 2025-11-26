const _DEFAULT_HAZARD_FORMULA = StatsModels.@formula(0 ~ 1)

@inline function _hazard_formula_has_intercept(rhs_term)
    rhs_term isa StatsModels.ConstantTerm && return true
    rhs_term isa StatsModels.InterceptTerm && return true
    if rhs_term isa StatsModels.MatrixTerm
        return any(_hazard_formula_has_intercept, rhs_term.terms)
    end
    return false
end

"""
        Hazard(family::Union{AbstractString,Symbol}, statefrom::Integer, stateto::Integer; kwargs...)
        Hazard(hazard::StatsModels.FormulaTerm, family::Union{AbstractString,Symbol}, statefrom::Integer, stateto::Integer; kwargs...)

Construct a parametric or semi-parametric cause-specific hazard specification to be
consumed by `multistatemodel`. Provide a `StatsModels` formula only when the hazard
has covariates; when you omit it, the constructor automatically supplies the
intercept-only design `@formula(0 ~ 1)` so you never have to write `+ 1` yourself.

# Positional arguments
- `family`: "exp", "wei", "gom", or "sp" (string or symbol, case-insensitive).
- `statefrom` / `stateto`: integers describing the transition.
- `hazard` *(optional)*: a `StatsModels.FormulaTerm` describing covariates that act
    multiplicatively on the baseline hazard. Skip this argument for intercept-only hazards.

# Keyword arguments
- `degree`, `knots`, `boundaryknots`, `natural_spline`, `extrapolation`, `monotone`:
    spline controls used only when `family == "sp"`. See the BSplineKit docs for details.
- `monotone`: `0` (default) leaves the spline unconstrained, `1` enforces an increasing
    hazard, and `-1` enforces a decreasing hazard.
- `time_transform::Bool`: enable Tang-style shared-trajectory caching for this transition.
- `linpred_effect::Symbol`: `:ph` (default) for proportional hazards or `:aft` for
    accelerated-failure-time behaviour.

# Examples
```julia
julia> Hazard("exp", 1, 2)                      # intercept only
julia> Hazard(@formula(0 ~ age + trt), "wei", 1, 3)
julia> @hazard begin                             # macro front-end uses the same rules
                     family = :gom
                     transition = 2 => 4
                     formula = @formula(0 ~ stage)
             end
```
"""
function Hazard(
    hazard::StatsModels.FormulaTerm,
    family::Union{AbstractString,Symbol},
    statefrom::Int64,
    stateto::Int64;
    degree::Int64 = 3,
    knots::Union{Vector{Float64}, Float64, Nothing} = nothing,
    boundaryknots::Union{Vector{Float64}, Nothing} = nothing,
    natural_spline = true,
    extrapolation = "linear",
    monotone = 0,
    time_transform::Bool = false,
    linpred_effect::Symbol = :ph)

    metadata = HazardMetadata(time_transform = time_transform, linpred_effect = linpred_effect)
    family_str = family isa String ? family : String(family)
    family_key = lowercase(family_str)

    if family_key != "sp"
        h = ParametricHazard(hazard, family_key, statefrom, stateto, metadata)
    else 
        if natural_spline & (monotone != 0)
            @info "Natural boundary conditions are not currently compatible with monotone splines. The restrictions on second derivatives at the spline boundaries will be removed."
            natural_spline = false
        end

        # change extrapolation to flat if degree = 0
        extrapolation = degree > 0 ? extrapolation : "flat"
        natural_spline = degree < 2 ? false : natural_spline

        h = SplineHazard(hazard, family_key, statefrom, stateto, degree, knots, boundaryknots, extrapolation, natural_spline, sign(monotone), metadata)
    end

    return h
end

function Hazard(
    family::Union{AbstractString,Symbol},
    statefrom::Integer,
    stateto::Integer;
    kwargs...)
    family_str = family isa String ? family : String(family)
    return Hazard(_DEFAULT_HAZARD_FORMULA, family_str, Int(statefrom), Int(stateto); kwargs...)
end

"""
    enumerate_hazards(hazards::HazardFunction...)

Standardise a collection of `Hazard`/`@hazard` definitions. The result is a
`DataFrame` with columns `statefrom`, `stateto`, `trans`, and `order`, sorted by
origin and destination so downstream helpers (e.g. `create_tmat`) can rely on a
stable ordering. Duplicate transitions (same origin/destination pair) raise an error.
"""
function enumerate_hazards(hazards::HazardFunction...)

    n_haz = length(hazards)

    # initialize state space information
    hazinfo = 
        DataFrames.DataFrame(
            statefrom = zeros(Int64, n_haz),
            stateto = zeros(Int64, n_haz),
            trans = zeros(Int64, n_haz),
            order = collect(1:n_haz))

    # grab the origin and destination states for each hazard
    for i in eachindex(hazards)
        hazinfo.statefrom[i] = hazards[i].statefrom
        hazinfo.stateto[i] = hazards[i].stateto
    end

    # check for duplicate transitions
    transition_pairs = [(hazinfo.statefrom[i], hazinfo.stateto[i]) for i in 1:n_haz]
    if length(unique(transition_pairs)) != n_haz
        duplicates = [tp for tp in unique(transition_pairs) if count(==(tp), transition_pairs) > 1]
        error("Duplicate transitions detected: $(duplicates). Each transition (statefrom → stateto) should be specified only once.")
    end

    # enumerate and sort hazards
    sort!(hazinfo, [:statefrom, :stateto])
    hazinfo[:,:trans] = collect(1:n_haz)

    # return the hazard information
    return hazinfo
end

"""
    create_tmat(hazinfo::DataFrame)

Create the familiar transition matrix that `multistatemodel` expects. Rows are
origin states, columns are destination states, zeros mark impossible transitions,
and the positive entries are the transition numbers assigned by `enumerate_hazards`.
"""
function create_tmat(hazinfo::DataFrame)
    
    # initialize the transition matrix
    statespace = sort(unique([hazinfo[:,:statefrom] hazinfo[:, :stateto]]))
    n_states = length(statespace)

    # initialize transition matrix
    tmat = zeros(Int64, n_states, n_states)

    for i in axes(hazinfo, 1)
        tmat[hazinfo.statefrom[i], hazinfo.stateto[i]] = 
            hazinfo.trans[i]
    end

    return tmat
end


# mutable structs

#=============================================================================
PHASE 2: Consolidated build_hazards with Runtime Functions
=============================================================================# 

"""
        build_hazards(hazards::HazardFunction...; data::DataFrame, surrogate::Bool = false)

Instantiate runtime hazard objects from the symbolic specifications produced by
`Hazard` or `@hazard`. This attaches the relevant design matrices, builds the
baseline parameter containers, and returns everything needed by `multistatemodel`.

# Arguments
- `hazards`: one or more hazard specifications. Formula arguments are optional for
    intercept-only transitions, matching the `Hazard` constructor semantics.
- `data`: `DataFrame` containing the covariates referenced by the hazard formulas.
- `surrogate`: when `true`, force exponential baselines (useful for MCEM surrogates).

# Returns
1. `_hazards`: callable hazard objects used internally by the simulator/likelihood.
2. `parameters`: legacy nested vector of parameters for backwards compatibility.
3. `parameters_ph`: the ParameterHandling.jl view (flat, natural, transforms, etc.).
4. `hazkeys`: dictionary mapping hazard names (e.g. `:h12`) to indices in `_hazards`.
"""
function build_hazards(hazards::HazardFunction...; data::DataFrame, surrogate = false)
    
    # Initialize arrays
    _hazards = Vector{_Hazard}(undef, length(hazards))
    parameters = VectorOfVectors{Float64}()
    hazkeys = Dict{Symbol, Int64}()

    # Build each hazard
    for h in eachindex(hazards) 

        # Hazard name
        hazname = "h" * string(hazards[h].statefrom) * string(hazards[h].stateto)
        merge!(hazkeys, Dict(Symbol(hazname) => h))

        metadata = hazards[h].metadata

        # Parse formula and create design matrix
        hazschema = apply_schema(hazards[h].hazard, StatsModels.schema(hazards[h].hazard, data))
        has_explicit_intercept = _hazard_formula_has_intercept(hazschema.rhs)
        hazdat = modelcols(hazschema, data)[2]

        if !has_explicit_intercept
            n_obs = size(hazdat, 1)
            coltype = eltype(hazdat)
            intercept_col = coltype === Any ? ones(n_obs) : ones(coltype, n_obs)
            hazdat = hcat(intercept_col, hazdat)
        end

        coef_names = StatsModels.coefnames(hazschema.rhs)
        coef_names_vec = coef_names isa AbstractVector ? collect(coef_names) : [coef_names]
        rhs_names_vec = has_explicit_intercept ? coef_names_vec : vcat("(Intercept)", coef_names_vec)
        
        # Determine family (use surrogate exponential if requested)
        family = surrogate ? "exp" : hazards[h].family 
        
        # Number of covariates and parameters
        has_covariates = size(hazdat, 2) > 1
        ncovar = size(hazdat, 2) - 1
        
        shared_key = shared_baseline_key(hazards[h], family)

        # Build hazard struct based on family
        if family == "exp"
            # EXPONENTIAL: Markov process
            npar_baseline = 1
            npar_total = size(hazdat, 2)
            
            # Initialize parameters (zeros for now)
            hazpars = zeros(Float64, npar_total)
            push!(parameters, hazpars)
            
            # Parameter names - extract from formula terms, not from data
            parnames = replace.(hazname * "_" .* rhs_names_vec, "(Intercept)" => "Intercept")
            parnames = Symbol.(parnames)
            
            # Generate runtime functions with name-based covariate matching
            hazard_fn, cumhaz_fn = generate_exponential_hazard(parnames, metadata.linpred_effect)
            
            # Create consolidated struct
            haz_struct = MarkovHazard(
                Symbol(hazname),
                hazards[h].statefrom,
                hazards[h].stateto,
                family,
                parnames,
                npar_baseline,
                npar_total,
                hazard_fn,
                cumhaz_fn,
                has_covariates,
                metadata,
                shared_key
            )
            
        elseif family == "wei"
            # WEIBULL: Semi-Markov process
            npar_baseline = 2  # shape + scale
            npar_total = npar_baseline + ncovar
            
            # Initialize parameters
            hazpars = zeros(Float64, npar_total)
            push!(parameters, hazpars)
            
            # Parameter names - extract from formula terms directly
            if !has_covariates
                parnames = replace.(vec(hazname * "_" .* ["shape" "scale"] .* "_" .* rhs_names_vec), "_(Intercept)" => "")
                parnames = replace.(parnames, "_Intercept" => "")
            else
                parnames = vcat(
                    hazname * "_shape",
                    hazname * "_scale",
                    hazname * "_" .* rhs_names_vec[Not(1)])
            end
            parnames = Symbol.(parnames)
            
            # Generate runtime functions with name-based covariate matching
            hazard_fn, cumhaz_fn = generate_weibull_hazard(parnames, metadata.linpred_effect)
            
            # Create consolidated struct
            haz_struct = SemiMarkovHazard(
                Symbol(hazname),
                hazards[h].statefrom,
                hazards[h].stateto,
                family,
                parnames,
                npar_baseline,
                npar_total,
                hazard_fn,
                cumhaz_fn,
                has_covariates,
                metadata,
                shared_key
            )
            
        elseif family == "gom"
            # GOMPERTZ: Semi-Markov process
            npar_baseline = 2  # shape + scale
            npar_total = npar_baseline + ncovar
            
            # Initialize parameters
            hazpars = zeros(Float64, npar_total)
            push!(parameters, hazpars)
            
            # Parameter names - extract from formula terms directly
            if !has_covariates
                parnames = replace.(vec(hazname * "_" .* ["shape" "scale"] .* "_" .* rhs_names_vec), "_(Intercept)" => "")
                parnames = replace.(parnames, "_Intercept" => "")
            else
                parnames = vcat(
                    hazname * "_shape",
                    hazname * "_scale",
                    hazname * "_" .* rhs_names_vec[Not(1)])
            end
            parnames = Symbol.(parnames)
            
            # Generate runtime functions with name-based covariate matching
            hazard_fn, cumhaz_fn = generate_gompertz_hazard(parnames, metadata.linpred_effect)
            
            # Create consolidated struct
            haz_struct = SemiMarkovHazard(
                Symbol(hazname),
                hazards[h].statefrom,
                hazards[h].stateto,
                family,
                parnames,
                npar_baseline,
                npar_total,
                hazard_fn,
                cumhaz_fn,
                has_covariates,
                metadata,
                shared_key
            )
            
        elseif family == "sp"
            # B-SPLINES: Semi-Markov with spline basis
            # TODO: Implement spline function generation in Phase 2.x
            error("Spline hazards not yet implemented in Phase 2 - coming soon!")
            
        else
            error("Unknown hazard family: $family")
        end

        _hazards[h] = haz_struct
    end

    # Phase 3: Build parameters_ph structure from parameters
    # Note: parameters are on LOG scale, but ParameterHandling.positive() expects NATURAL scale
    # So we need to convert log → natural before wrapping with positive()
    params_transformed_pairs = [
        hazname => ParameterHandling.positive(exp.(Vector{Float64}(parameters[idx])))
        for (hazname, idx) in sort(collect(hazkeys), by = x -> x[2])
    ]
    
    params_transformed = NamedTuple(params_transformed_pairs)
    params_flat, unflatten_fn = ParameterHandling.flatten(params_transformed)
    params_natural = ParameterHandling.value(params_transformed)
    
    parameters_ph = (
        flat = params_flat,
        transformed = params_transformed,
        natural = params_natural,
        unflatten = unflatten_fn
    )

    # Phase 3: Now return parameters_ph as fourth output
    return _hazards, parameters, parameters_ph, hazkeys
end

#=============================================================================
OLD build_hazards (DEPRECATED - Phase 1 version)
=============================================================================# 
# Commented out - will be removed after Phase 2 complete

"""
    build_parameters_ph(parameters::VectorOfVectors, hazkeys::Dict{Symbol, Int64})

Create a ParameterHandling.jl structure from legacy VectorOfVectors parameters.

Returns a NamedTuple with fields:
- `flat`: Vector{Float64} of all parameters in flattened form (log scale)
- `transformed`: NamedTuple of positive() transformations for each hazard
- `natural`: NamedTuple of Vectors for each hazard (natural scale)
- `unflatten`: Function to reconstruct from flat vector

Parameters are stored on the log scale using ParameterHandling.positive().
"""
function build_parameters_ph(parameters::VectorOfVectors, hazkeys::Dict{Symbol, Int64})
    # Create a NamedTuple of Vectors for each hazard using positive() transformation
    # This will store parameters on log scale internally
    params_transformed_pairs = [
        hazname => ParameterHandling.positive(Vector{Float64}(parameters[idx]))
        for (hazname, idx) in sort(collect(hazkeys), by = x -> x[2])
    ]
    
    params_transformed = NamedTuple(params_transformed_pairs)
    
    # Flatten to get the flat vector and unflatten function
    params_flat, unflatten_fn = ParameterHandling.flatten(params_transformed)
    
    # Get natural scale parameters
    params_natural = ParameterHandling.value(params_transformed)
    
    # Return nested structure
    return (
        flat = params_flat,
        transformed = params_transformed,
        natural = params_natural,
        unflatten = unflatten_fn
    )
end

### Total hazards
"""
    build_totalhazards(_hazards, tmat)

 Return a vector of _TotalHazard objects for each origin state, which may be of subtype `_TotalHazardAbsorbing` or `_TotalHazardTransient`. 

 Accepts the internal array _hazards corresponding to allowable transitions, and the transition matrix tmat
"""
function build_totalhazards(_hazards, tmat)

    # initialize a vector for total hazards
    _totalhazards = Vector{_TotalHazard}(undef, size(tmat, 1))

    # populate the vector of total hazards
    for h in eachindex(_totalhazards) 
        if sum(tmat[h,:]) == 0
            _totalhazards[h] = 
                _TotalHazardAbsorbing()
        else
            _totalhazards[h] = 
                _TotalHazardTransient(tmat[h, findall(tmat[h,:] .!= 0)])
        end
    end

    return _totalhazards
end

"""
    build_emat(data::DataFrame, CensoringPatterns::Matrix{Int64}, tmat::Matrix{Int64})

Create the emission matrix used by the forward–backward routines. Each row
corresponds to an observation; columns correspond to latent states. The helper
marks which states are compatible with each observation or censoring code using
`CensoringPatterns` (if provided) and the transition structure `tmat`.
"""
function build_emat(data::DataFrame, CensoringPatterns::Matrix{Int64}, tmat::Matrix{Int64})
    
    # initialize the emission matrix
    n_obs = nrow(data)
    n_states = size(tmat, 1)
    emat = zeros(Int64, n_obs, n_states)

    for i in 1:n_obs
        if data.obstype[i] ∈ [1, 2] # observation not censored
            emat[i,data.stateto[i]] = 1
        elseif data.obstype[i] == 0 # observation censored, all state are possible
            emat[i,:] = ones(n_states)
        else
            emat[i,:] .= CensoringPatterns[data.obstype[i] - 2, 2:n_states+1]
        end 
    end

    return emat
end

"""
    multistatemodel(hazards::HazardFunction...; data::DataFrame, SamplingWeights = nothing, CensoringPatterns = nothing, verbose = false)

Construct a full multistate model from a collection of hazards defined via `Hazard`
or `@hazard`. Hazards without covariates can omit a `@formula` entirely; the helper
will insert the intercept-only design automatically as described in `Hazard`'s docs.

# Keywords
- `data`: long-format `DataFrame` with at least `:subject`, `:statefrom`, `:stateto`, `:time`, `:obstype`.
- `SamplingWeights`: optional per-subject weights.
- `CensoringPatterns`: optional matrix describing which states are compatible with each censoring code.
- `verbose`: print additional validation output.
"""
function multistatemodel(hazards::HazardFunction...; data::DataFrame, SamplingWeights::Union{Nothing,Vector{Float64}} = nothing, CensoringPatterns::Union{Nothing,Matrix{Int64}} = nothing, verbose = false) 

    # catch the model call
    modelcall = (hazards = hazards, data = data, SamplingWeights = SamplingWeights, CensoringPatterns = CensoringPatterns)

    # get indices for each subject in the dataset
    subjinds, nsubj = get_subjinds(data)

    # enumerate the hazards and reorder 
    hazinfo = enumerate_hazards(hazards...)

    # reorder hazards and pop the order column in hazinfo
    hazards = hazards[hazinfo.order]
    select!(hazinfo, Not(:order))

    # compile matrix enumerating instantaneous state transitions
    tmat = create_tmat(hazinfo)

    # initialize SamplingWeights if none supplied
    if isnothing(SamplingWeights)
        SamplingWeights = ones(Float64, nsubj)
    end

    # initialize censoring patterns if none supplied
    if isnothing(CensoringPatterns)
        CensoringPatterns = Matrix{Int64}(undef, 0, size(tmat, 1))
    end

    # check data formatting
    check_data!(data, tmat, CensoringPatterns; verbose = verbose)

    # check SamplingWeights
    check_SamplingWeights(SamplingWeights, data)

    # check CensoringPatterns and build emission matrix
    if any(data.obstype .∉ Ref([1,2]))
        check_CensoringPatterns(CensoringPatterns, tmat)
    end

    # build emission matrix
    emat = build_emat(data, CensoringPatterns, tmat)

    # generate tuple for compiled hazard functions
    # _hazards is a tuple of _Hazard objects
    _hazards, parameters, parameters_ph, hazkeys = build_hazards(hazards...; data = data, surrogate = false)

    # generate vector for total hazards 
    _totalhazards = build_totalhazards(_hazards, tmat)  

    # build exponential surrogate hazards (discard parameters_ph from surrogate)
    surrogate_haz, surrogate_pars, _, _ = build_hazards(hazards...; data = data, surrogate = true)

    # construct multistate mode
    # exactly observed data
    if all(data.obstype .== 1)        
        model = MultistateModel(
        data,
        parameters,
        parameters_ph,
        _hazards,
        _totalhazards,
        tmat,
        emat,
        hazkeys,
        subjinds,
        SamplingWeights,
        CensoringPatterns,
        MarkovSurrogate(surrogate_haz, surrogate_pars),
        modelcall)

    # panel data and/or exactly observed data
    elseif all(data.obstype .∈ Ref([1,2]))
        # Markov model
        if all(isa.(_hazards, _MarkovHazard))
            model = MultistateMarkovModel(
                data,
                parameters,
                parameters_ph,
                _hazards,
                _totalhazards,
                tmat,
                emat,
                hazkeys,
                subjinds,
                SamplingWeights,
                CensoringPatterns,
                MarkovSurrogate(surrogate_haz, surrogate_pars),
                modelcall)
        # Semi-Markov model
        elseif any(isa.(_hazards, _SemiMarkovHazard))
            model = MultistateSemiMarkovModel(
                data,
                parameters,
                parameters_ph,
                _hazards,
                _totalhazards,
                tmat,
                emat,
                hazkeys,
                subjinds,
                SamplingWeights,
                CensoringPatterns,
                MarkovSurrogate(surrogate_haz, surrogate_pars),
                modelcall)
        end

    # censored states and/or panel data and/or exactly observed data
    elseif !all(data.obstype .∈ Ref([1,2]))
        # Markov model
        if all(isa.(_hazards, _MarkovHazard))
            model = MultistateMarkovModelCensored(
                data,
                parameters,
                parameters_ph,
                _hazards,
                _totalhazards,
                tmat,
                emat,
                hazkeys,
                subjinds,
                SamplingWeights,
                CensoringPatterns,
                MarkovSurrogate(surrogate_haz, surrogate_pars),
                modelcall)
        # semi-Markov model
        elseif any(isa.(_hazards, _SemiMarkovHazard))
            model = MultistateSemiMarkovModelCensored(
                data,
                parameters,
                parameters_ph,
                _hazards,
                _totalhazards,
                tmat,
                emat,
                hazkeys,
                subjinds,
                SamplingWeights,
                CensoringPatterns,
                MarkovSurrogate(surrogate_haz, surrogate_pars),
                modelcall)
        end        
    end

    return model
end