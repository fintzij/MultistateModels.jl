"""
    haz(haz::StatsModels.FormulaTerm, family::string, statefrom::Int64, stateto::Int64)

Composite type for a cause-specific hazard function. Documentation to follow. 
"""
struct Hazard
    hazard::StatsModels.FormulaTerm   # StatsModels.jl formula
    family::String     # one of "exp", "wei", "gg", or "sp"
    statefrom::Int64   # starting state number
    stateto::Int64     # destination state number
end

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

"""
    parse_hazard(hazards::Hazard)

Takes the formula in a Hazard object and modifies it to have the specified baseline hazard function. 

For exponential hazards, maps @formula(lambda ~ trt) -> @formula(lambda ~ 1 + trt)

For Weibull hazards, maps @formula(lambda ~ trt) -> (@formula(shape ~ 1 + trt), @formula(scale ~ 1 + trt))

shapes = exp.(modelmatrix(weibull_formula_1) \beta_shapes)
scales = exp.(modelmatrix(weibull_formula_2) \beta_scales)

function weibull_hazard(t, shapes, scales) 
    shapes .* (scales .^ shapes) .* (t .^ (shapes .- 1))
end

# f1 is a hazard
function f1(t, x)
    ...
end

# cumulative hazard
function F1(f, t0, t1, x)
    integrate(f, t0, t1, x)
end 

# total hazard
function T(F..., t0, t1, x)
    sum(F...)
end

# survival function
function S(T)
    exp(-T)
end
"""

### function to make a multistate model
function MultistateModel(hazards::Hazard...; data = nothing)

    # enumerate the hazards and reorder 
    hazinfo = enumerate_hazards(hazards...)

    # reorder hazards and pop the order column in hazinfo
    hazards = hazards[hazinfo.order]
    select!(hazinfo, Not(:order))

    # compile matrix enumerating instantaneous state transitions
    tmat = create_tmat(hazinfo)

    # need:
    # - wrappers for formula schema
    # - function to parse cause-specific hazards for each origin state and return total hazard

end

