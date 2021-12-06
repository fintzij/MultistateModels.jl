# type for defining hazards
struct hazard
    hazfun::StatsModels.FormulaTerm   # StatsModels.jl formula
    family::String     # one of "exp", "wei", "gg", or "sp"
    statefrom::Int64   # starting state number
    stateto::Int64     # destination state number
    timescale::String   # either "cr" or "cf" for clock-reset or clock-forward
end