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
    create_tmat(hazards::Hazard...)

Generate a matrix enumerating instantaneous transitions, used internally. Origin states correspond to rows, destination states to columns, and zero entries indicate that an instantaneous state transition is not possible. Transitions are enumerated in non-zero elements of the matrix. 
"""
function create_tmat(hazards::Hazard...)
    
end

### function to make a multistate model
function MultiStateModel(hazards::Hazard...)

    # compile matrix enumerating instantaneous state transitions
    tmat = create_tmat(hazards...); 

end

#### Minimal model
# cause specific hazards
# starting state initializer (fixed or random)
