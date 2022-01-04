"""
    get_hazinfo(hazards::Hazard...; enumerate = true)

Generate a matrix whose columns record the origin state, destination state, and transition number for a collection of hazards. Optionally, reorder the hazards by origin state, then by destination state.
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
    create_tmat(hazards::Hazard...)

Generate a matrix enumerating instantaneous transitions, used internally. Origin states correspond to rows, destination states to columns, and zero entries indicate that an instantaneous state transition is not possible. Transitions are enumerated in non-zero elements of the matrix. 
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

# if no covariate data
function build_hazards(hazards::Hazard..., data::DataFrame)
    
    # check for covariates in dataset
    # any_covariates = DataFrames.ncols(data) > 6

    # initialize the arrays of hazards
    _hazards = []

    # assign a hazard function
    for h in axes(hazards) 

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

            # hazard function
            _hazfun = MultistateModels.haz_exp

            # number of parameters
            npars = ncol(hazdat)

            # vector for parameters
            hazpars = Vector{Float64, npars}
            parnames = hazname*"_".*coefnames(hazschema)[2]

        elseif hazards[h].family == "wei"

            # hazard function
            _hazfun = MultistateModels.haz_wei

            # number of parameters
            npars = 2 * ncol(hazdat)

            # vector for parameters
        elseif hazards[h].family == "gam"
        elseif hazards[h].family == "gg"
        else # semi-parametric family
        end

        # note: want a symbol that names the hazard + vector of symbols for parameters
        _hazards[h] = 
            _Hazard(
                Symbol(haznames),
                Symbol.(parnames),
                hazards[h].statefrom,
                hazards[h].stateto,
                hazards[h].family,
                hazdat,
                )

        # and we push the mutable struct to the array
        push!(_hazards, _haz)
    end
end

### function to make a multistate model
function MultistateModel(hazards::Hazard...; data::DataFrame)

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
    _hazards = build_hazards(hazards, data)

    # generate tuple for total hazards ? 
    #_totalhazards, _tothazdat = build_tothazards(_hazards, tmat)

    

    # need:

    # Q: data + parameters separate from tuple of hazard functions?
    # OR
    # Q: data + parameters that are local to each hazard function?

    simulate(model_object)
    fit(model_object)

    model_object.fit
    model_object.sim

    # - wrappers for formula schema
    # - function to parse cause-specific hazards for each origin state and return total hazard

end
