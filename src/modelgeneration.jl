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

"""
    build_hazards(hazards:Hazard...; data:DataFrame, surrogate = false)

Return internal array of internal _Hazard subtypes called _hazards.

Accept iterable collection of `Hazard` objects, plus data. 

_hazards[1] corresponds to the first allowable transition enumerated in a transition matrix (in row major order), _hazards[2] to the second and so on... So _hazards will have length equal to number of allowable transitions.
"""
function build_hazards(hazards::HazardFunction...; data::DataFrame, surrogate = false)
    
    # initialize the arrays of hazards
    _hazards = Vector{_Hazard}(undef, length(hazards))

    # named tuple for parameters
    parameters = VectorOfVectors{Float64}()

    # initialize a dictionary for indexing into the vector of hazards
    hazkeys = Dict{Symbol, Int64}()

    # assign a hazard function
    for h in eachindex(hazards) 

        # name for the hazard
        hazname = "h"*string(hazards[h].statefrom)*string(hazards[h].stateto)

        # save index in the dictionary
        merge!(hazkeys, Dict(Symbol(hazname) => h))

        # # generate the model matrix
        # hazschema = 
        #     apply_schema(
        #         hazards[h].hazard, 
        #          StatsModels.schema(
        #              hazards[h].hazard, 
        #              data))

        # # grab the design matrix 
        # hazdat = modelcols(hazschema, data)[2]

        # # get the family
        # family = surrogate ? "exp" : hazards[h].family 

        # # now we get the functions and other objects for the mutable struct
        # if family == "exp"

        #     # number of parameters
        #     npars = size(hazdat)[2]

        #     # vector for parameters
        #     hazpars = zeros(Float64, npars)

        #     # append to model parameters
        #     push!(parameters, hazpars)

        #     # get names
        #     parnames = replace.(hazname*"_".*coefnames(hazschema)[2], "(Intercept)" => "Intercept")

        #     # generate hazard struct
        #     if npars == 1
        #         haz_struct = 
        #             _Exponential(
        #                 Symbol(hazname),
        #                 hazdat,
        #                 [Symbol.(parnames)],
        #                 hazards[h].statefrom,
        #                 hazards[h].stateto,
        #                 size(hazdat, 2) - 1) # make sure this is a vector
        #     else
        #         haz_struct = 
        #             _ExponentialPH(
        #                 Symbol(hazname),
        #                 hazdat,
        #                 Symbol.(parnames),
        #                 hazards[h].statefrom,
        #                 hazards[h].stateto,
        #                 size(hazdat, 2) - 1)
        #     end

        # elseif family == "wei" 

        #     # number of parameters
        #     npars = size(hazdat, 2)

        #     # vector for parameters
        #     hazpars = zeros(Float64, 1 + npars)

        #     # append to model parameters
        #     push!(parameters, hazpars)

        #     # generate hazard struct
        #     if npars == 1
                
        #         # parameter names
        #         parnames = replace.(vec(hazname*"_".*["shape" "scale"].*"_".*coefnames(hazschema)[2]), "_(Intercept)" => "")

        #         haz_struct = 
        #             _Weibull(
        #                 Symbol(hazname),
        #                 hazdat, 
        #                 Symbol.(parnames),
        #                 hazards[h].statefrom,
        #                 hazards[h].stateto,
        #                 size(hazdat, 2) - 1)
                        
        #     else
                
        #         # parameter names
        #         parnames = vcat(
        #                 hazname * "_shape",
        #                 hazname * "_scale",
        #                 hazname*"_".*coefnames(hazschema)[2][Not(1)])

        #         haz_struct = 
        #             _WeibullPH(
        #                 Symbol(hazname),
        #                 hazdat,
        #                 Symbol.(parnames),
        #                 hazards[h].statefrom,
        #                 hazards[h].stateto,
        #                 size(hazdat, 2) - 1)
        #     end

        # elseif family == "gom"

        #     # number of parameters
        #     npars = size(hazdat, 2)

        #     # vector for parameters
        #     hazpars = zeros(Float64, 1 + npars)

        #     # append to model parameters
        #     push!(parameters, hazpars)

        #     # generate hazard struct
        #     if npars == 1
                
        #         # parameter names
        #         parnames = replace.(vec(hazname*"_".*["shape" "scale"].*"_".*coefnames(hazschema)[2]), "_(Intercept)" => "")

        #         haz_struct = 
        #             _Gompertz(
        #                 Symbol(hazname),
        #                 hazdat, 
        #                 Symbol.(parnames),
        #                 hazards[h].statefrom,
        #                 hazards[h].stateto,
        #                 size(hazdat, 2) - 1)
                        
        #     else
                
        #         # parameter names
        #         parnames = vcat(
        #                 hazname * "_shape",
        #                 hazname * "_scale",
        #                 hazname*"_".*coefnames(hazschema)[2][Not(1)])

        #         haz_struct = 
        #             _GompertzPH(
        #                 Symbol(hazname),
        #                 hazdat,
        #                 Symbol.(parnames),
        #                 hazards[h].statefrom,
        #                 hazards[h].stateto,
        #                 size(hazdat, 2) - 1)
        #     end
        # elseif family == "sp" # B-splines

        #     # grab hazard object from splines2
        #     hazard, cumulative_hazard, rmat, knots, timespan = spline_hazards(hazards[h], data)

        #     # number of parameters
        #     npars = size(rmat, 2) + size(hazdat, 2) - 1

        #     # generate hazard struct
        #     ### no covariates
        #     if(size(hazdat, 2) == 1) 
                
        #         # parameter names
        #         parnames = replace.(vec(hazname*"_".*"splinecoef".*"_".*string.(collect(1:length(hazard.spline.basis)))), "(Intercept)" => "Intercept")

        #         # vector for parameters
        #         hazpars = zeros(Float64, npars)

        #         # append to model parameters
        #         push!(parameters, hazpars)

        #         haz_struct = _Spline(Symbol(hazname),
        #                                 hazdat, 
        #                                 Symbol.(parnames),
        #                                 hazards[h].statefrom,
        #                                 hazards[h].stateto,
        #                                 hazards[h].degree,
        #                                 knots,
        #                                 hazard,
        #                                 cumulative_hazard,
        #                                 hazards[h].natural_spline,
        #                                 hazards[h].monotone,
        #                                 timespan,
        #                                 timespan,
        #                                 length(hazard.spline.basis),
        #                                 size(hazdat, 2) - 1)            
        #     else
        #         ### proportional hazards
        #         # parameter names
        #         parnames = replace.(vcat(vec(hazname*"_".*"splinecoef".*"_".*string.(collect(1:length(hazard.spline.basis)))), hazname*"_".*coefnames(hazschema)[2][Not(1)]))
               
        #         # vector for parameters
        #         hazpars = zeros(Float64, npars)

        #         # append to model parameters
        #         push!(parameters, hazpars)

        #         # hazard struct
        #         haz_struct = _SplinePH(Symbol(hazname),
        #                                 hazdat[:,Not(1)], 
        #                                 Symbol.(parnames),
        #                                 hazards[h].statefrom,
        #                                 hazards[h].stateto,
        #                                 hazards[h].degree,
        #                                 knots,
        #                                 hazard,
        #                                 cumulative_hazard,
        #                                 hazards[h].natural_spline,
        #                                 hazards[h].monotone,
        #                                 timespan,
        #                                 timespan,
        #                                 length(hazard.spline.basis),
        #                                 size(hazdat, 2) - 1)                             
        #     end
        # end

        # note: want a symbol that names the hazard + vector of symbols for parameters
        _hazards[h] = haz_struct
    end

    return _hazards, parameters, hazkeys
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
    _hazards, parameters, hazkeys = build_hazards(hazards...; data = data, surrogate = false)

    # generate vector for total hazards 
    _totalhazards = build_totalhazards(_hazards, tmat)  

    # build exponential surrogate hazards
    surrogate = build_hazards(hazards...; data = data, surrogate = true)

    # construct multistate mode
    # exactly observed data
    if all(data.obstype .== 1)        
        model = MultistateModel(
        data,
        parameters,
        _hazards,
        _totalhazards,
        tmat,
        emat,
        hazkeys,
        subjinds,
        SamplingWeights,
        CensoringPatterns,
        MarkovSurrogate(surrogate[1], surrogate[2]),
        modelcall)

    # panel data and/or exactly observed data
    elseif all(data.obstype .∈ Ref([1,2]))
        # Markov model
        if all(isa.(_hazards, _MarkovHazard))
            model = MultistateMarkovModel(
                data,
                parameters,
                _hazards,
                _totalhazards,
                tmat,
                emat,
                hazkeys,
                subjinds,
                SamplingWeights,
                CensoringPatterns,
                MarkovSurrogate(surrogate[1], surrogate[2]),
                modelcall)
        # Semi-Markov model
        elseif any(isa.(_hazards, _SemiMarkovHazard))
            model = MultistateSemiMarkovModel(
                data,
                parameters,
                _hazards,
                _totalhazards,
                tmat,
                emat,
                hazkeys,
                subjinds,
                SamplingWeights,
                CensoringPatterns,
                MarkovSurrogate(surrogate[1], surrogate[2]),
                modelcall)
        end

    # censored states and/or panel data and/or exactly observed data
    elseif !all(data.obstype .∈ Ref([1,2]))
        # Markov model
        if all(isa.(_hazards, _MarkovHazard))
            model = MultistateMarkovModelCensored(
                data,
                parameters,
                _hazards,
                _totalhazards,
                tmat,
                emat,
                hazkeys,
                subjinds,
                SamplingWeights,
                CensoringPatterns,
                MarkovSurrogate(surrogate[1], surrogate[2]),
                modelcall)
        # semi-Markov model
        elseif any(isa.(_hazards, _SemiMarkovHazard))
            model = MultistateSemiMarkovModelCensored(
                data,
                parameters,
                _hazards,
                _totalhazards,
                tmat,
                emat,
                hazkeys,
                subjinds,
                SamplingWeights,
                CensoringPatterns,
                MarkovSurrogate(surrogate[1], surrogate[2]),
                modelcall)
        end        
    end

    return model
end