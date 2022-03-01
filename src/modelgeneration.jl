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
    build_hazards(hazards:Hazard...; data:DataFrame)

Return internal array of internal _Hazard subtypes called _hazards.

Accept iterable collection of `Hazard` objects, plus data. 

_hazards[1] corresponds to the first allowable transition enumerated in a transition matrix (in row major order), _hazards[2] to the second and so on... So _hazards will have length equal to number of allowable transitions.
"""
function build_hazards(hazards::Hazard...; data::DataFrame)
    
    # initialize the arrays of hazards
    _hazards = Vector{_Hazard}(undef, length(hazards))

    # initialize vector of parameters
    parameters = Vector{Float64}(undef, 0)
    
    # counter for tracking indices of views
    hazpars_start = 0

    # assign a hazard function
    for h in eachindex(hazards) 

        # name for the hazard
        hazname = "h"*string(hazards[h].statefrom)*string(hazards[h].stateto)

        # generate the model matrix
        hazschema = 
            StatsModels.apply_schema(
                hazards[h].hazard, 
                 StatsModels.schema(
                     hazards[h].hazard, 
                     data))

        # grab the design matrix 
        hazdat = StatsModels.modelcols(hazschema, data)[2]

        # now we get the functions and other objects for the mutable struct
        if hazards[h].family == "exp"

            # number of parameters
            npars = size(hazdat)[2]

            # vector for parameters
            hazpars = zeros(Float64, npars)

            # append to model parameters
            append!(parameters, hazpars)

            # get names
            parnames = hazname*"_".*coefnames(hazschema)[2]

            # generate hazard struct
            if npars == 1
                haz_struct = 
                    _Exponential(
                        Symbol(hazname),
                        hazdat,
                        # hazpars,
                        view(parameters, hazpars_start .+ eachindex(hazpars)),
                        [Symbol.(parnames)]) # make sure this is a vector
            else
                haz_struct = 
                    _ExponentialReg(
                        Symbol(hazname),
                        hazdat,
                        # hazpars,
                        view(parameters, hazpars_start .+ eachindex(hazpars)),
                        Symbol.(parnames))
            end

        elseif hazards[h].family == "wei" || hazards[h].family == "weiPH"

            # number of parameters
            npars = size(hazdat, 2)

            # generate hazard struct
            if npars == 1

                # vector for parameters
                hazpars = zeros(Float64, npars * 2)

                # append to model parameters
                append!(parameters, hazpars)
                
                # parameter names
                parnames = vec(hazname*"_".*["scale" "shape"].*"_".*coefnames(hazschema)[2])

                # create struct
                haz_struct = 
                    _Weibull(
                        Symbol(hazname),
                        hazdat,
                        # hazpars,
                        view(parameters, hazpars_start .+ eachindex(hazpars)),
                        Symbol.(parnames))
                        
            elseif hazards[h].family == "weiPH"

                # vector for parameters
                hazpars = zeros(Float64, 1 + npars)

                # append to model parameters
                append!(parameters, hazpars)
                
                # parameter names
                parnames = 
                    vcat(
                        hazname * "_scale",
                        hazname * "_shape",
                        hazname*"_".*coefnames(hazschema)[2][Not(1)])
                        

                haz_struct = 
                    _WeibullPH(
                        Symbol(hazname),
                        hazdat[:,Not(1)],
                        # hazpars,
                        view(parameters, hazpars_start .+ eachindex(hazpars)),
                        Symbol.(parnames))

            else

                # vector for parameters
                hazpars = zeros(Float64, npars * 2)

                # append to model parameters
                append!(parameters, hazpars)
                
                # parameter names
                parnames = vec(hazname*"_".*["scale" "shape"].*"_".*coefnames(hazschema)[2])

                # generate struct
                haz_struct = 
                    _WeibullReg(
                        Symbol(hazname),
                        hazdat,
                        # hazpars,
                        view(parameters, hazpars_start .+ eachindex(hazpars)),
                        Symbol.(parnames),
                        UnitRange(1, npars),
                        UnitRange(1 + npars, 2 * npars))
            end

        elseif hazards[h].family == "gam"
        elseif hazards[h].family == "gg"
        else # semi-parametric family
        end

        # note: want a symbol that names the hazard + vector of symbols for parameters
        _hazards[h] = haz_struct

        # increment parameter starting index
        hazpars_start += npars
    end

    return _hazards, parameters
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
                _TotalHazardTransient(
tmat[h, findall(tmat[h,:] .!= 0)]
                )
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
function multistatemodel(hazards::Hazard...;data::DataFrame)

    # function to check data formatting
    # checkdat()

    # get indices for each subject in the dataset
    subjinds = get_subjinds(data)

    # enumerate the hazards and reorder 
    hazinfo = enumerate_hazards(hazards...)

    # reorder hazards and pop the order column in hazinfo
    hazards = hazards[hazinfo.order]
    select!(hazinfo, Not(:order))

    # compile matrix enumerating instantaneous state transitions
    tmat = create_tmat(hazinfo)

    # generate tuple for compiled hazard functions
    # _hazards is a tuple of _Hazard objects
    _hazards, parameters = build_hazards(hazards...; data = data)

    # generate vector for total hazards 
    _totalhazards = build_totalhazards(_hazards, tmat)  

    # initialize vector of model parameters and set views in hazards
    ### update this when we deal with data sampling dists/censoring
    # parameters = collate_parameters(_hazards)
    # set_parameter_views!(parameters, _hazards)

    # return the multistate model
    model = 
        MultistateModel(
            data,
            parameters,
            _hazards,
            _totalhazards,
            tmat,
            subjinds
        )

    return model
end