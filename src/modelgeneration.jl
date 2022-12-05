"""
    enumerate_hazards(hazards::Hazard...)

Generate a matrix whose columns record the origin state, destination state, and transition number for a collection of hazards. The hazards are reordered by origin state, then by destination state. `hazards::Hazard...` is an iterable collection of user-supplied `Hazard` objects.
"""
function enumerate_hazards(hazards::Hazard...)

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
function build_hazards(hazards::Hazard...; data::DataFrame, surrogate = false)
    
    # initialize the arrays of hazards
    _hazards = Vector{_Hazard}(undef, length(hazards))

    # initialize vector of parameters
    parameters = VectorOfVectors{Float64}()

    # initialize a dictionary for indexing into the vector of hazards
    hazkeys = Dict{Symbol, Int64}()

    # assign a hazard function
    for h in eachindex(hazards) 

        # name for the hazard
        hazname = "h"*string(hazards[h].statefrom)*string(hazards[h].stateto)

        # save index in the dictionary
        merge!(hazkeys, Dict(Symbol(hazname) => h))

        # generate the model matrix
        hazschema = 
            apply_schema(
                hazards[h].hazard, 
                 StatsModels.schema(
                     hazards[h].hazard, 
                     data))

        # grab the design matrix 
        hazdat = modelcols(hazschema, data)[2]

        # get the family
        family = surrogate ? "exp" : hazards[h].family 

        # now we get the functions and other objects for the mutable struct
        if family == "exp"

            # number of parameters
            npars = size(hazdat)[2]

            # vector for parameters
            hazpars = zeros(Float64, npars)

            # append to model parameters
            push!(parameters, hazpars)

            # get names
            parnames = hazname*"_".*coefnames(hazschema)[2]

            # generate hazard struct
            if npars == 1
                haz_struct = 
                    _Exponential(
                        Symbol(hazname),
                        hazdat,
                        [Symbol.(parnames)],
                        hazards[h].statefrom,
                        hazards[h].stateto) # make sure this is a vector
            else
                haz_struct = 
                    _ExponentialPH(
                        Symbol(hazname),
                        hazdat,
                        Symbol.(parnames),
                        hazards[h].statefrom,
                        hazards[h].stateto)
            end

        elseif family == "wei" 

            # number of parameters
            npars = size(hazdat, 2)

            # vector for parameters
            hazpars = zeros(Float64, 1 + npars)

            # append to model parameters
            push!(parameters, hazpars)

            # generate hazard struct
            if npars == 1
                
                # parameter names
                parnames = vec(hazname*"_".*["shape" "scale"].*"_".*coefnames(hazschema)[2])

                haz_struct = 
                    _Weibull(
                        Symbol(hazname),
                        hazdat, 
                        Symbol.(parnames),
                        hazards[h].statefrom,
                        hazards[h].stateto)
                        
            else
                
                # parameter names
                parnames = 
                    vcat(
                        hazname * "_shape_(Intercept)",
                        hazname * "_scale_(Intercept)",
                        hazname*"_".*coefnames(hazschema)[2][Not(1)])

                haz_struct = 
                    _WeibullPH(
                        Symbol(hazname),
                        hazdat,
                        Symbol.(parnames),
                        hazards[h].statefrom,
                        hazards[h].stateto)
            end

        elseif family == "gom"
        elseif family == "gg"
        else # semi-parametric family
        end

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

### Cumulative hazards
"""
    build_cumulativehazards(_totalhazard::_TotalHazard)

"""
function build_cumulativehazards(_totalhazards::_TotalHazard)

    # initialize a vector of cumulative hazards
    _cumulativehazards = Vector{_CumulativeHazard}(undef, length(_totalhazards))

    for h in eachindex(_cumulativehazards)
        if(isa(_totalhazards[h], _TotalHazardAbsorbing))
        else

        end
    end
    
end

"""
    multistatemodel(hazards::Hazard...; data::DataFrame)

Constructs a multistate model from cause specific hazards. Parses the supplied hazards and dataset and returns an object of type `MultistateModel` that can be used for simulation and inference.
"""
function multistatemodel(hazards::Hazard...; data::DataFrame)

    # catch the model call
    modelcall = (hazards = hazards, data = data)

    # get indices for each subject in the dataset
    subjinds = get_subjinds(data)

    # enumerate the hazards and reorder 
    hazinfo = enumerate_hazards(hazards...)

    # reorder hazards and pop the order column in hazinfo
    hazards = hazards[hazinfo.order]
    select!(hazinfo, Not(:order))

    # compile matrix enumerating instantaneous state transitions
    tmat = create_tmat(hazinfo)

    # function to check data formatting
    check_data!(data, tmat)

    # generate tuple for compiled hazard functions
    # _hazards is a tuple of _Hazard objects
    _hazards, parameters, hazkeys = build_hazards(hazards...; data = data, surrogate = false)

    # generate vector for total hazards 
    _totalhazards = build_totalhazards(_hazards, tmat)  

    # build exponential surrogate hazards
    surrogate = build_hazards(hazards...; data = data, surrogate = true)

    # return the multistate model
    model = MultistateModel(
        data,
        parameters,
        _hazards,
        _totalhazards,
        tmat,
        hazkeys,
        subjinds,
        MarkovSurrogate(surrogate[1], surrogate[2]),
        modelcall = modelcall)

    return model
end