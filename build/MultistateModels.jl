module MultistateModels

using AbstractMCMC
using Distributions
using StatsModels

# Write your package code here.
export
    @formula,
    Hazard,
    MultistateModel

include("hazards.jl")

end
