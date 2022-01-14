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

Accept iterable collection of `Hazard` objects, plus data. Return internal array of internal _Hazard subtypes called _hazards.

_hazards[1] corresponds to the first allowable transition enumerated in a transition matrix (in row major order), _hazards[2] to the second and so on... So _hazards will have length equal to number of allowable transitions.
"""
function build_hazards(hazards::Hazard...; data::DataFrame)
    
    # initialize the arrays of hazards
    _hazards = Vector{_Hazard}(undef, length(hazards))

    # assign a hazard function
    for h in eachindex(hazards) 

        # name for the hazard
        hazname = "h"*string(hazards[h].statefrom)*string(hazards[h].stateto)

        # generate the model matrix
        hazschema = 
            apply_schema(hazards[h].hazard, 
                         schema(hazards[h].hazard, 
                                data))

        # grab the design matrix 
        hazdat = modelcols(hazschema, data)[2]

        # now we get the functions and other objects for the mutable struct
        if hazards[h].family == "exp"

            # number of parameters
            npars = size(hazdat)[2]

            # vector for parameters
            hazpars = zeros(Float64, npars)
            parnames = hazname*"_".*coefnames(hazschema)[2]

            # generate hazard struct
            if npars == 1
                haz_struct = 
                    _Exponential(
                        Symbol(hazname),
                        Symbol.(parnames),
                        hazdat,
                        hazpars)
            else
                haz_struct = 
                    _ExponentialReg(
                        Symbol(hazname),
                        Symbol.(parnames),
                        hazdat,
                        hazpars)
            end

        elseif hazards[h].family == "wei"

            # number of parameters
            npars = size(hazdat, 2)

            # vector for parameters
            hazpars = zeros(Float64, npars * 2)
            parnames = vec(hazname*"_".*["scale" "shape"].*"_".*coefnames(hazschema)[2])

            # generate hazard struct
            if npars == 1
                haz_struct = 
                    _Weibull(
                        Symbol(hazname),
                        Symbol.(parnames),
                        UnitRange(1, npars),
                        UnitRange(1 + npars, 2 * npars),
                        hazdat,
                        hazpars)
            else
                haz_struct = 
                    _WeibullReg(
                        Symbol(hazname),
                        Symbol.(parnames),
                        UnitRange(1, npars),
                        UnitRange(1 + npars, 2 * npars),
                        hazdat,
                        hazpars)
            end

        elseif hazards[h].family == "gam"
        elseif hazards[h].family == "gg"
        else # semi-parametric family
        end

        # note: want a symbol that names the hazard + vector of symbols for parameters
        _hazards[h] = haz_struct
    end

    return _hazards
end

### Total hazards
"""
    build_totalhazards(_hazards, tmat)

This function accepts the internal array _hazards corresponding to allowable transitions, and the transition matrix tmat

This function returns a vector of functions for the total hazard out of each state. The total hazard for each aborbing state always returns 0.
"""
# function build_totalhazards(_hazards, tmat)

#     # initialize a vector for total hazards
#     # _totalhazards = Vector{Function}(undef, size(tmat, 1))
# # or do we want this

#     _totalhazards = Vector{_TotalHazard}(undef, size(tmat, 1))

# # so that we call total hazards via
#     call_tothaz(t::Float64, statecur::Int64, _totalhazards::Vector{_TotalHazard})

#     # 

# end

### function to make a multistate model
function MultistateModel(hazards::Hazard...;data::DataFrame)

    # function to check data formatting
    # checkdat()

    # enumerate the hazards and reorder 
    hazinfo = enumerate_hazards(hazards...)

    # reorder hazards and pop the order column in hazinfo
    hazards = hazards[hazinfo.order]
    select!(hazinfo, Not(:order))

    # compile matrix enumerating instantaneous state transitions
    tmat = create_tmat(hazinfo)

    # generate tuple for compiled hazard functions
    # _hazards is a tuple of _Hazard objects
    _hazards = build_hazards(hazards...; data = data)

    # generate tuple for total hazards ? 
    # _totalhazards = build_totalhazards(_hazards, tmat)    

    # need:

    # Q: data + parameters separate from tuple of hazard functions?
    # OR
    # Q: data + parameters that are local to each hazard function?

    # simulate(model_object)
    # fit(model_object)

    # model_object.fit
    # model_object.sim

    # - wrappers for formula schema
    # - function to parse cause-specific hazards for each origin state and return total hazard

end
