"""
    Hazard(hazard::StatsModels.FormulaTerm, family::String, statefrom::Int64, stateto::Int64; df::Union{Int64,Nothing} = nothing, degree::Int64 = 3, knots::Union{Vector{Float64}, Float64, Nothing} = nothing, boundaryknots::Union{Vector{Float64}, Nothing}, periodic::Bool = false, monotonic::Bool = false)

Specify a parametric or semi-parametric baseline cause-specific hazard function. 

# Arguments
- `hazard`: StatsModels.jl FormulaTerm for the log-hazard. Covariates have a multiplicative effect on the baseline cause specific hazard. Must be specified with "0 ~" on the left hand side. 
- `family`: one of "exp", "wei", or "gom" for exponential, Weibull, or Gompertz cause-specific baseline hazard functions, or "sp" for a semi-parametric spline basis up to degree 3 for the baseline hazard.
- `statefrom`: integer specifying the origin state.
- `stateto`: integer specifying the destination state.

# Additional arguments for semiparametric baseline hazards. Splines up to degree 3 (cubic polynomials) are supported . Spline bases are constructed via a call to the BSplineKit.jl. See [the BSplineKit.jl documentation](https://jipolanco.github.io/BSplineKit.jl/stable/) for additional details. 
- `degree`: Degree of the spline polynomial basis, defaults to 3 for a cubic polynomial basis.
- `knots`: Vector of interior knots.
- `boundaryknots`: Optional vector of boundary knots, defaults to the range of possible sojourn times if not supplied.
- `extrapolation`: Either "linear" or "flat", see the BSplineKit.jl package. 
- `natural_spline`: Restrict the second derivative to zero at the boundaries, defaults to true.
- `knots` argument is interpreted as interior knots. 
- `monotone`; 0, -1, or 1 for non-monotone, monotone decreasing, or monotone increasing.
"""
function Hazard(hazard::StatsModels.FormulaTerm, family::String, statefrom::Int64, stateto::Int64; degree::Int64 = 3, knots::Union{Vector{Float64}, Float64, Nothing} = nothing, boundaryknots::Union{Vector{Float64}, Nothing} = nothing, natural_spline = true, extrapolation = "linear", monotone = 0)

    if family != "sp"
        h = ParametricHazard(hazard, family, statefrom, stateto)
    else 
        if natural_spline & (monotone != 0)
            @info "Natural boundary conditions are not currently compatible with monotone splines. The restrictions on second derivatives at the spline boundaries will be removed."
            natural_spline = false
        end

        # change extrapolation to flat if degree = 0
        extrapolation = degree > 0 ? extrapolation : "flat"
        natural_spline = degree < 2 ? false : natural_spline

        h = SplineHazard(hazard, family, statefrom, stateto, degree, knots, boundaryknots, extrapolation, natural_spline, sign(monotone))
    end

    return h
end

"""
    enumerate_hazards(hazards::Hazard...)

Generate a matrix whose columns record the origin state, destination state, and transition number for a collection of hazards. The hazards are reordered by origin state, then by destination state. `hazards::Hazard...` is an iterable collection of user-supplied `Hazard` objects.
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

Generate a matrix enumerating instantaneous transitions. Origin states correspond to rows, destination states to columns, and zero entries indicate that an instantaneous state transition is not possible. Transitions are enumerated in non-zero elements of the matrix. `hazinfo` is the output of a call to `enumerate_hazards`.
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
    build_hazards(hazards::HazardFunction...; data::DataFrame, surrogate = false)

**PHASE 2 VERSION** - Build consolidated hazard structs with runtime-generated functions.

Creates one of 3 hazard types (MarkovHazard, SemiMarkovHazard, SplineHazard) instead of 8.
Hazard structs no longer contain data - they store runtime-generated functions instead.
Parameters managed via ParameterHandling.jl with positive() transformation.

# Returns
- `_hazards`: Vector of consolidated hazard structs (callable)
- `parameters`: Legacy VectorOfVectors (for compatibility)
- `parameters_ph`: NamedTuple with (flat, transformed, natural, unflatten) - **Phase 3 addition**
- `hazkeys`: Dict mapping hazard names to indices

# Changes from Phase 1
- No data stored in hazard structs
- Runtime-generated functions for hazard/cumulative hazard
- Consolidated 8 types → 3 types
- Direct callable interface: hazard(t, pars, covars)

# Phase 3 Update
- Now returns `parameters_ph` as third output for ParameterHandling integration
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

        # Parse formula and create design matrix
        hazschema = apply_schema(hazards[h].hazard, StatsModels.schema(hazards[h].hazard, data))
        hazdat = modelcols(hazschema, data)[2]
        
        # Determine family (use surrogate exponential if requested)
        family = surrogate ? "exp" : hazards[h].family 
        
        # Number of covariates and parameters
        has_covariates = size(hazdat, 2) > 1
        ncovar = size(hazdat, 2) - 1
        
        # Build hazard struct based on family
        if family == "exp"
            # EXPONENTIAL: Markov process
            npar_baseline = 1
            npar_total = size(hazdat, 2)
            
            # Initialize parameters (zeros for now)
            hazpars = zeros(Float64, npar_total)
            push!(parameters, hazpars)
            
            # Parameter names - extract from formula terms, not from data
            # Get the right-hand side terms from the formula
            formula_rhs = hazards[h].hazard.rhs
            if formula_rhs isa StatsModels.ConstantTerm
                # Intercept-only model
                rhs_names_vec = ["Intercept"]
            else
                # Extract term names from the formula (before data is applied)
                rhs_terms = StatsModels.termvars(formula_rhs)
                if isempty(rhs_terms)
                    # Just intercept
                    rhs_names_vec = ["Intercept"]
                else
                    # Has covariates
                    rhs_names_vec = ["Intercept"; String.(rhs_terms)]
                end
            end
            parnames = replace.(hazname * "_" .* rhs_names_vec, "(Intercept)" => "Intercept")
            parnames = Symbol.(parnames)
            
            # Generate runtime functions with name-based covariate matching
            hazard_fn, cumhaz_fn = generate_exponential_hazard(has_covariates, parnames)
            
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
                has_covariates
            )
            
        elseif family == "wei"
            # WEIBULL: Semi-Markov process
            npar_baseline = 2  # shape + scale
            npar_total = npar_baseline + ncovar
            
            # Initialize parameters
            hazpars = zeros(Float64, npar_total)
            push!(parameters, hazpars)
            
            # Parameter names - extract from formula terms directly
            formula_rhs = hazards[h].hazard.rhs
            if formula_rhs isa StatsModels.ConstantTerm
                rhs_names_vec = ["Intercept"]
            else
                rhs_terms = StatsModels.termvars(formula_rhs)
                if isempty(rhs_terms)
                    rhs_names_vec = ["Intercept"]
                else
                    rhs_names_vec = ["Intercept"; String.(rhs_terms)]
                end
            end
            
            if !has_covariates
                parnames = replace.(vec(hazname * "_" .* ["shape" "scale"] .* "_" .* rhs_names_vec), "_(Intercept)" => "")
            else
                parnames = vcat(
                    hazname * "_shape",
                    hazname * "_scale",
                    hazname * "_" .* rhs_names_vec[Not(1)])
            end
            parnames = Symbol.(parnames)
            
            # Generate runtime functions with name-based covariate matching
            hazard_fn, cumhaz_fn = generate_weibull_hazard(has_covariates, parnames)
            
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
                has_covariates
            )
            
        elseif family == "gom"
            # GOMPERTZ: Semi-Markov process
            npar_baseline = 2  # shape + scale
            npar_total = npar_baseline + ncovar
            
            # Initialize parameters
            hazpars = zeros(Float64, npar_total)
            push!(parameters, hazpars)
            
            # Parameter names - extract from formula terms directly
            formula_rhs = hazards[h].hazard.rhs
            if formula_rhs isa StatsModels.ConstantTerm
                rhs_names_vec = ["Intercept"]
            else
                rhs_terms = StatsModels.termvars(formula_rhs)
                if isempty(rhs_terms)
                    rhs_names_vec = ["Intercept"]
                else
                    rhs_names_vec = ["Intercept"; String.(rhs_terms)]
                end
            end
            
            if !has_covariates
                parnames = replace.(vec(hazname * "_" .* ["shape" "scale"] .* "_" .* rhs_names_vec), "_(Intercept)" => "")
            else
                parnames = vcat(
                    hazname * "_shape",
                    hazname * "_scale",
                    hazname * "_" .* rhs_names_vec[Not(1)])
            end
            parnames = Symbol.(parnames)
            
            # Generate runtime functions with name-based covariate matching
            hazard_fn, cumhaz_fn = generate_gompertz_hazard(has_covariates, parnames)
            
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
                has_covariates
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
    build_emat(data::DataFrame, CensoringPatterns::Matrix{Int64})

Generate a matrix enumerating instantaneous transitions. Origin states correspond to rows, destination states to columns, and zero entries indicate that an instantaneous state transition is not possible. Transitions are enumerated in non-zero elements of the matrix. `hazinfo` is the output of a call to `enumerate_hazards`.
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

Constructs a multistate model from cause specific hazards. Parses the supplied hazards and dataset and returns an object of type `MultistateModel` that can be used for simulation and inference. Optional keyword arguments specified in kwargs may include sampling weights and censoring patterns.
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