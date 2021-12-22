module MultistateModels

using DataFrames
using Distributions
using StatsFuns
using StatsModels

# Write your package code here.
export
    @formula,
    Hazard,
    MultistateModel

include("hazards.jl")
include("modelgeneration.jl")

end
