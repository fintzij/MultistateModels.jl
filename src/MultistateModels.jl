module MultistateModels

using AbstractMCMC
using Distributions
using StatsModels

# Write your package code here.
export
    @formula,
    hazard

include("hazards.jl")

end
