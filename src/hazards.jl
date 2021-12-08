"""
    haz(haz::StatsModels.FormulaTerm, family::string, statefrom::Int64, stateto::Int64, timescale::String)

Composite type for a cause-specific hazard function. Documentation to follow. 
"""
struct Hazard
    hazard::StatsModels.FormulaTerm   # StatsModels.jl formula
    family::String     # one of "exp", "wei", "gg", or "sp"
    statefrom::Int64   # starting state number
    stateto::Int64     # destination state number
    timescale::String   # either "cr" or "cf" for clock-reset or clock-forward
end

"""
    get_hazinfo(hazards::Hazard...; enumerate = true)

Generate a matrix whose columns record the origin state, destination state, and transition number for a collection of hazards. Optionally, reorder the hazards by origin state, then by destination state.
"""
function enumerate_hazards(hazards::Hazard...)

    n_haz = length(hazards);

    # initialize state space information
    hazinfo = 
        DataFrames.DataFrame(
            statefrom = zeros(Int64, n_haz),
            stateto = zeros(Int64, n_haz),
            trans = zeros(Int64, n_haz),
            order = collect(1:n_haz));

    # grab the origin and destination states for each hazard
    for i in eachindex(hazards)
        hazinfo.statefrom[i] = hazards[i].statefrom;
        hazinfo.stateto[i] = hazards[i].stateto;
    end

    # enumerate and sort hazards
    sort!(hazinfo, [:statefrom, :stateto]);
    hazinfo[:,:trans] = collect(1:n_haz);

    # return the hazard information
    return hazinfo
end

"""
    create_tmat(hazards::Hazard...)

Generate a matrix enumerating instantaneous transitions, used internally. Origin states correspond to rows, destination states to columns, and zero entries indicate that an instantaneous state transition is not possible. Transitions are enumerated in non-zero elements of the matrix. 
"""
function create_tmat(hazinfo::Array{Int64})
    
    # initialize the transition matrix
    statespace = unique()

end

### function to make a multistate model
function MultiStateModel(hazards::Hazard...)

    # enumerate the hazards and reorder 
    hazinfo = enumerate_hazards(hazards...);

    # reorder hazards and pop the order column in hazinfo
    hazards = hazards[hazinfo.order];
    select!(hazinfo, Not(:order));

    # compile matrix enumerating instantaneous state transitions
    tmat = create_tmat(hazards...); 

end

